#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Generate the golden corpus (designs) and the scalar-reference results stored next to it.

    python tools/make_golden_corpus.py [--workers N]

Phase 1 builds designs with no igniter and no activation profile. Phase 2 derives the cases that need a
burn time (igniter and activation sources, the blowdown-truncation quirk, timeouts, solver failure) from
the Phase 1 reference results. The designs come from a fixed seed, so regenerating on the same reference
code reproduces the same corpus. Rerun it only when the scalar physics changes on purpose.
"""

import argparse
import copy
import json
import math
import os
import platform
import subprocess
import sys
import time
import warnings
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import scipy

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "tests")]

import golden_corpus as gc  # noqa: E402

SEED = 20261002
CORPUS_VERSION = 1
SIZES = {"small": (0.018, 0.03), "medium": (0.03, 0.05), "large": (0.05, 0.09)}
STEPS = (0.01, 0.02, 0.03)


def _r(value, digits=6):
    return round(float(value), digits)


def sample_grain(rng, outer, geometry, ends_burn, web=None, height_ratio=None):
    web = rng.uniform(0.35, 0.75) if web is None else web
    height_ratio = rng.uniform(1.8, 4.5) if height_ratio is None else height_ratio
    grain = {
        "outer_radius": _r(outer),
        "initial_inner_radius": _r(outer * (1.0 - web)),
        "initial_height": _r(outer * height_ratio),
        "geometry": geometry,
        "ends_burn": bool(ends_burn),
    }
    if geometry == "star":
        points = int(rng.integers(3, 9))
        grain.update(
            n_points=points,
            epsilon=_r(rng.uniform(0.08, 0.8 * math.pi / points), 5),
            slot_fraction=_r(rng.uniform(0.3, 0.9), 4),
        )
    return grain


def sample_stack(rng, geometries, size="medium", ends=False, identical=False, web=None, height_ratio=None):
    outer = rng.uniform(*SIZES[size])
    flags = list(ends) if isinstance(ends, (list, tuple)) else [ends] * len(geometries)
    if identical and len(set(geometries)) == 1:
        grain = sample_grain(rng, outer, geometries[0], flags[0], web, height_ratio)
        return [copy.deepcopy(grain) for _ in geometries]
    return [sample_grain(rng, outer, g, e, web, height_ratio) for g, e in zip(geometries, flags)]


def sample_motor(rng, grains, kn=(120.0, 350.0), replicate=None, angle="random"):
    count = replicate or len(grains)
    area = sum(gc.Grain(**grain).burn_area for grain in grains) * (replicate or 1)
    throat = math.sqrt(area / rng.uniform(*kn) / math.pi)
    exit_radius = throat * math.sqrt(rng.uniform(4.0, 12.0))
    separation = rng.choice([0.0, 0.0, 0.002, 0.005])
    stack = sum(grain["initial_height"] for grain in grains) * (replicate or 1) + separation * (count - 1)
    motor = {
        "chamber_inner_radius": _r(max(grain["outer_radius"] for grain in grains) + rng.uniform(0.0005, 0.004)),
        "chamber_length": _r(stack + rng.uniform(0.005, 0.03)),
        "nozzle_throat_radius": _r(throat),
        "nozzle_exit_radius": _r(exit_radius),
        "nozzle_angle": (None if rng.random() < 0.5 else 0.2618) if angle == "random" else angle,
        "grain_separation": _r(separation, 4),
    }
    if replicate:
        motor["grain_number"] = replicate
    return motor


def sample_propellant(rng, **override):
    propellant = {
        "specific_heat_ratio": _r(rng.uniform(1.10, 1.25), 4),
        "products_molecular_mass": _r(rng.uniform(0.035, 0.045), 5),
        "combustion_temperature": _r(rng.uniform(1500.0, 1900.0), 1),
        "density": _r(rng.uniform(1700.0, 1900.0), 1),
        "burn_rate_a": _r(rng.uniform(4.0, 9.0), 3),
        "burn_rate_n": _r(rng.uniform(0.2, 0.45), 4),
    }
    propellant.update(override)
    return propellant


def with_table(propellant, name):
    propellant = {k: v for k, v in propellant.items() if k not in ("burn_rate_a", "burn_rate_n")}
    propellant["interpolation_list"] = f"data/burnrate/{name}.csv"
    return propellant


def thermo_table(rng, propellant, rows):
    pressure = np.geomspace(1e5, 1.2e7 * rng.uniform(0.6, 1.0), rows)
    decade = np.log10(pressure / 1e6)
    return [
        [float(p), float(900.0 * (1.0 + 0.01 * d)), float(propellant["specific_heat_ratio"] * (1.0 - 0.01 * d)),
         float(propellant["combustion_temperature"] * (1.0 + 0.015 * d))]
        for p, d in zip(pressure, decade)
    ]


def sim(rng, steps=STEPS, **kwargs):
    simulation = {"max_step_size": float(rng.choice(steps))}
    simulation.update(kwargs)
    return simulation


def efficiency_kwargs(rng, mask):
    names = ("eta_c", "eta_Cf", "discharge_coefficient")
    return {name: _r(rng.uniform(0.85, 0.99), 4) for name, flag in zip(names, mask) if flag}


class Builder:
    def __init__(self):
        self.cases = []

    def add(self, family, grains, motor, propellant, simulation, environment=None, tags=(), expect=None):
        index = sum(1 for case in self.cases if case["family"] == family)
        case = {
            "id": f"{family}-{index:03d}",
            "family": family,
            "extra_tags": sorted(set(tags)),
            "grains": grains,
            "motor": motor,
            "propellant": propellant,
            "environment": environment or {},
            "simulation": simulation,
        }
        if expect:
            case["expect_termination_reason"] = expect
        case["tags"] = sorted(set(gc.derive_tags(case)) | set(tags))
        self.cases.append(case)


def phase1(rng):
    b = Builder()
    pick = lambda options: str(rng.choice(options))  # noqa: E731

    for geometry in ("tubular", "star"):
        for count in (1, 2, 4, 8):
            for i in range(8):
                grains = sample_stack(rng, [geometry] * count, pick(["small", "medium", "large"]), identical=i % 2 == 0)
                b.add(geometry, grains, sample_motor(rng, grains), sample_propellant(rng), sim(rng))

    for geometry in ("tubular", "star"):
        for i in range(10):
            count = int(rng.choice([1, 2, 4]))
            ends = [True] * count if i % 3 else [bool(x) for x in rng.integers(0, 2, count)]
            grains = sample_stack(rng, [geometry] * count, pick(["small", "medium"]), ends=ends)
            b.add("ends-" + geometry, grains, sample_motor(rng, grains), sample_propellant(rng), sim(rng))

    for i in range(16):
        count = int(rng.choice([2, 3, 4, 6]))
        first = i % 2
        geometries = [("tubular", "star")[(first + k) % 2] for k in range(count)]
        ends = [bool(x) for x in rng.integers(0, 2, count)]
        grains = sample_stack(rng, geometries, pick(["small", "medium"]), ends=ends)
        b.add("mixed", grains, sample_motor(rng, grains), sample_propellant(rng), sim(rng))

    for count, kind in [(24, "tubular")] * 3 + [(24, "star")] * 3 + [(24, "mixed")] * 2 + \
            [(16, "tubular"), (16, "star"), (16, "mixed"), (16, "tubular")] + \
            [(12, "tubular"), (12, "star"), (12, "mixed"), (12, "star")]:
        geometries = [("tubular", "star")[k % 2] for k in range(count)] if kind == "mixed" else [kind] * count
        grains = sample_stack(rng, geometries, "small", identical=kind != "mixed")
        b.add("many", grains, sample_motor(rng, grains), sample_propellant(rng), sim(rng, steps=(0.03,)))

    for i in range(24):
        count = int(rng.choice([1, 2, 4]))
        grains = sample_stack(rng, [("tubular", "star")[i % 2]] * count, pick(["small", "medium"]))
        propellant = sample_propellant(
            rng,
            erosive_burning_coefficient=float(f"{rng.uniform(2e-6, 2e-5):.4g}"),
            erosive_alpha=float(rng.choice([20.0, 35.0, 50.0])),
        )
        b.add("erosive", grains, sample_motor(rng, grains, kn=(80.0, 250.0)), propellant, sim(rng))

    masks = [(1, 0, 0), (0, 1, 0), (0, 0, 1), (1, 1, 0), (1, 0, 1), (0, 1, 1), (1, 1, 1), (1, 1, 1)]
    for i in range(24):
        count = int(rng.choice([1, 2, 4]))
        grains = sample_stack(rng, [("tubular", "star")[i % 2]] * count, pick(["small", "medium"]))
        b.add("efficiency", grains, sample_motor(rng, grains), sample_propellant(rng),
              sim(rng, **efficiency_kwargs(rng, masks[i % len(masks)])))

    for i in range(8):
        grains = sample_stack(rng, [("tubular", "star")[i % 2]] * int(rng.choice([1, 2])), "small", web=0.35)
        propellant = sample_propellant(rng, burn_rate_a=_r(rng.uniform(7.0, 9.0), 3), burn_rate_n=0.2)
        b.add("short", grains, sample_motor(rng, grains), propellant, sim(rng), tags=["short_burn"])
    for i in range(8):
        grains = sample_stack(rng, [("tubular", "star")[i % 2]] * int(rng.choice([1, 2])), "large", web=0.7)
        propellant = sample_propellant(rng, burn_rate_a=_r(rng.uniform(3.5, 5.0), 3), burn_rate_n=0.2)
        b.add("long", grains, sample_motor(rng, grains), propellant, sim(rng, steps=(0.03,)), tags=["long_burn"])

    for i in range(12):
        grains = sample_stack(rng, [("tubular", "star")[i % 2]] * int(rng.choice([1, 2])), pick(["small", "medium"]))
        b.add("altitude", grains, sample_motor(rng, grains), sample_propellant(rng), sim(rng),
              environment={"altitude": float(rng.choice([500.0, 1500.0, 3000.0, 5000.0, 8000.0]))})
    for i in range(8):
        grains = sample_stack(rng, [("tubular", "star")[i % 2]] * int(rng.choice([1, 2])), "medium")
        b.add("lowkn", grains, sample_motor(rng, grains, kn=(18.0, 30.0)), sample_propellant(rng),
              sim(rng, steps=(0.02, 0.03)), tags=["low_kn"])

    for i, name in enumerate(["KNSB3"] * 8 + ["KNSB"] * 6 + ["KNSB2"] * 4):
        grains = sample_stack(rng, [("tubular", "star")[i % 2]] * int(rng.choice([1, 2, 4])), pick(["small", "medium"]))
        b.add("ratetable", grains, sample_motor(rng, grains, kn=(120.0, 300.0)),
              with_table(sample_propellant(rng), name), sim(rng))

    for i, rows in enumerate([2, 3, 4, 4, 5, 6, 8, 10, 4, 6]):
        grains = sample_stack(rng, [("tubular", "star")[i % 2]] * int(rng.choice([1, 2])), pick(["small", "medium"]))
        propellant = sample_propellant(rng)
        propellant["thermo_table"] = thermo_table(rng, propellant, rows)
        extra = efficiency_kwargs(rng, (1, 0, 0)) if i in (3, 7) else {}
        b.add("thermotable", grains, sample_motor(rng, grains), propellant, sim(rng, **extra))

    for i in range(6):
        grains = sample_stack(rng, [("tubular", "star")[i % 2]] * int(rng.choice([1, 2])), "small")
        b.add("analytical", grains, sample_motor(rng, grains), sample_propellant(rng),
              sim(rng, tail_off_method="analytical"))
    for i in range(4):
        grains = sample_stack(rng, [("tubular", "star")[i % 2]] * int(rng.choice([1, 2])), "small")
        b.add("notail", grains, sample_motor(rng, grains), sample_propellant(rng),
              sim(rng, tail_off_evaluation=False))

    settings = [(1e-6, 1e-9, 0.05), (1e-7, 1e-9, 0.1), (1e-9, 1e-11, 0.01), (1e-9, 1e-11, 0.003),
                (1e-6, 1e-10, 0.003), (1e-7, 1e-11, 0.05), (1e-8, 1e-9, 0.1), (1e-8, 1e-11, 0.02)]
    for i in range(16):
        rtol, atol, step = settings[i % len(settings)]
        grains = sample_stack(rng, [("tubular", "star")[i % 2]] * int(rng.choice([1, 2])), "small")
        b.add("tolerance", grains, sample_motor(rng, grains), sample_propellant(rng),
              {"max_step_size": step, "rtol": rtol, "atol": atol})

    for count in (2, 3, 4, 4, 6, 6):
        grains = sample_stack(rng, ["tubular" if count % 2 else "star"], pick(["small", "medium"]))
        b.add("replicated", grains, sample_motor(rng, grains, replicate=count), sample_propellant(rng), sim(rng))

    for i in range(4):
        count = int(rng.choice([1, 2]))
        grains = sample_stack(rng, ["star"] * count, "small")
        for grain in grains:
            grain["epsilon"] = _r((math.pi - 2e-3) / grain["n_points"], 6)
        b.add("guard-nearpi", grains, sample_motor(rng, grains), sample_propellant(rng), sim(rng))
    for i, slot in enumerate([0.0, 0.0, 1.0, 1.0]):
        grains = sample_stack(rng, ["star"], "small")
        grains[0]["slot_fraction"] = slot
        b.add("guard-slot", grains, sample_motor(rng, grains), sample_propellant(rng), sim(rng))
    for i in range(6):
        grains = sample_stack(rng, [("tubular", "star")[i % 2]] * int(rng.choice([1, 2])), "medium",
                              web=0.7, height_ratio=float(rng.uniform(0.5, 0.9)))
        b.add("guard-axial", grains, sample_motor(rng, grains), sample_propellant(rng), sim(rng),
              tags=["axial_burnout_first"])
    for i in range(3):
        grains = sample_stack(rng, [("tubular", "star")[i % 2]], "small", web=0.04)
        b.add("guard-thinweb", grains, sample_motor(rng, grains, kn=(40.0, 120.0)), sample_propellant(rng),
              sim(rng), tags=["thin_web"])
    return b.cases


def eligible_bases(cases, records):
    out = []
    for case in cases:
        record = records[case["id"]]
        if not (case["family"] in ("tubular", "star") and len(case["grains"]) <= 4):
            continue
        if not record["status"]["completed"]:
            continue
        burnout = [t for t in record["metrics"]["grain_burnout_times_s"] if t is not None]
        if burnout and record["end_time_s"] - max(burnout) > 1e-3:
            out.append((case, max(burnout), record["end_time_s"]))
    return out


def derived(rng, cases, records):
    bases = eligible_bases(cases, records)
    if len(bases) < 12:
        raise SystemExit(f"only {len(bases)} usable base cases for the derived families")
    cursor = {"i": 0}

    def next_base():
        case, burn_end, end = bases[(cursor["i"] * 7) % len(bases)]
        cursor["i"] += 1
        return copy.deepcopy(case), burn_end, end

    b = Builder()

    def add(family, base, simulation_update=None, propellant_update=None, tags=(), expect=None):
        simulation = {**base["simulation"], **(simulation_update or {})}
        propellant = {**base["propellant"], **(propellant_update or {})}
        b.add(family, base["grains"], base["motor"], propellant, simulation, base["environment"], tags, expect)

    def times(count, burn_end, upper=0.7):
        return sorted(_r(t, 5) for t in rng.uniform(0.03 * burn_end, upper * burn_end, count))

    for i in range(6):
        base, burn_end, _ = next_base()
        add("igniter-scalar", base, {
            "igniter_mass_flow": _r(rng.uniform(0.001, 0.008), 5),
            "igniter_burn_time": _r(rng.uniform(0.1, 0.5) * burn_end, 4),
            **({"igniter_temperature": 3000.0} if i % 2 else {}),
        })
    for i in range(6):
        base, burn_end, _ = next_base()
        knots = [0.0] + times(3, burn_end)
        flows = [0.0] + [_r(x, 5) for x in rng.uniform(0.002, 0.01, 2)] + [0.0]
        add("igniter-table", base, {"igniter_mass_flow": [[t, m] for t, m in zip(knots, flows)]})
    for i in range(3):
        base, burn_end, _ = next_base()
        add("igniter-after", base, {
            "igniter_mass_flow": _r(rng.uniform(0.001, 0.004), 5),
            "igniter_burn_time": _r(burn_end + rng.uniform(0.5, 1.5), 4),
            "igniter_temperature": 3500.0,
        }, tags=["igniter_after_burnout"])
    for i in range(3):
        base, burn_end, _ = next_base()
        add("igniter-callable", base, {
            "igniter_mass_flow": {"callable": "igniter_decay"},
            "igniter_burn_time": _r(rng.uniform(0.2, 0.6) * burn_end, 4),
        })
    for i in range(5):
        base, _, _ = next_base()
        add("activation-scalar", base, {"burn_area_activation": _r(rng.uniform(0.3, 0.95), 4)})
    for i in range(6):
        base, burn_end, _ = next_base()
        knots = times(3, burn_end, upper=0.6)
        values = [_r(x, 4) for x in sorted(rng.uniform(0.1, 0.9, 2))] + [1.0]
        add("activation-table", base, {"burn_area_activation": [[0.0, 0.1]] + [[t, v] for t, v in zip(knots, values)]})
    for i in range(6):
        base, burn_end, _ = next_base()
        add("ramp", base, {"ignition_ramp_time": _r(rng.uniform(0.05, 0.3) * burn_end, 5)})
    for i in range(3):
        base, _, _ = next_base()
        add("activation-callable", base, {"burn_area_activation": {"callable": "activation_front"}})
    for i in range(3):
        base, burn_end, _ = next_base()
        add("combo-ramp", base, {
            "igniter_mass_flow": _r(rng.uniform(0.001, 0.006), 5),
            "igniter_burn_time": _r(rng.uniform(0.15, 0.4) * burn_end, 4),
            "ignition_ramp_time": _r(rng.uniform(0.05, 0.25) * burn_end, 5),
        })
    for i in range(3):
        base, burn_end, _ = next_base()
        knots = times(2, burn_end, upper=0.5)
        add("combo-table", base, {
            "igniter_mass_flow": [[0.0, 0.0], [knots[0], 0.006], [knots[1], 0.0]],
            "burn_area_activation": [[0.0, 0.2], [knots[1], 1.0]],
        })
    for i in range(4):
        base, burn_end, end = next_base()
        knot = _r(burn_end + 0.5 * (end - burn_end), 6)
        add("quirk", base, {"burn_area_activation": [[0.0, 1.0], [knot, 1.0], [knot + 10.0, 1.0]]},
            tags=["blowdown_truncated_by_breakpoint"], expect="blowdown_timeout")
    for i in range(5):
        base, burn_end, _ = next_base()
        add("timeout-burn", base, {"burn_timeout_s": _r(0.5 * burn_end, 5)}, expect="burn_timeout")
    for i in range(5):
        base, burn_end, end = next_base()
        add("timeout-blowdown", base, {"tail_off_timeout_s": _r(0.3 * (end - burn_end), 6)},
            expect="blowdown_timeout")
    for i in range(2):
        base, burn_end, _ = next_base()
        add("igniter-unknown", base, {"igniter_mass_flow": {"callable": "igniter_decay"}, "igniter_burn_time": 0.0},
            expect="unknown_igniter_duration")
    # An absurd erosive coefficient makes the pressure runaway fatal for the solver, but whether it fails
    # depends on the design, so try increasing coefficients per base and keep the first that really fails.
    found = 0
    while found < 3:
        base, _, _ = next_base()
        for coefficient in (5.0, 50.0, 500.0, 5000.0):
            update = {"erosive_burning_coefficient": coefficient, "erosive_alpha": 35.0}
            candidate = {**base, "propellant": {**base["propellant"], **update}}
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                reason = gc.simulate(candidate).result["status"]["termination_reason"]
            if reason == "solver_failure":
                add("solver-failure", base, propellant_update=update, expect="solver_failure")
                found += 1
                break
        if cursor["i"] > 4 * len(bases):
            raise SystemExit("could not find designs that make the reference solver fail")
    return b.cases


def run_case(case):
    start = time.perf_counter()
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        try:
            simulation = gc.simulate(case)
        except Exception as exc:  # recorded, then reported by the caller
            return case["id"], {"error": f"{type(exc).__name__}: {exc}"}
    record = gc.reference_record(simulation, time.perf_counter() - start)
    record["warnings"] = sorted({warning.category.__name__ for warning in caught})
    return case["id"], record


def run_all(cases, workers):
    order = sorted(cases, key=lambda c: -(c["motor"].get("grain_number") or len(c["grains"])))
    with ProcessPoolExecutor(max_workers=workers) as pool:
        return dict(pool.map(run_case, order, chunksize=1))


def check(cases, records):
    problems = []
    for case in cases:
        record = records[case["id"]]
        if "error" in record:
            problems.append(f"{case['id']}: {record['error']}")
            continue
        expected = case.get("expect_termination_reason")
        actual = record["status"]["termination_reason"]
        if expected and actual != expected:
            problems.append(f"{case['id']}: expected {expected}, reference gave {actual}")
        if not expected and case["family"] not in ("analytical", "notail", "thermotable", "ratetable") and \
                "callable" not in case["tags"] and not record["status"]["completed"]:
            problems.append(f"{case['id']}: plain case did not complete ({actual})")
    return problems


def write_lines(path, header, key, items):
    head = json.dumps(header, separators=(",", ":"), allow_nan=False)[:-1]
    body = ",\n".join(json.dumps(item, separators=(",", ":"), allow_nan=False) for item in items)
    path.write_text(f'{head},"{key}":[\n{body}\n]}}\n')


def git_sha():
    try:
        return subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True, text=True,
                              check=True).stdout.strip()
    except (OSError, subprocess.SubprocessError):
        return None


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--workers", type=int, default=os.cpu_count())
    args = parser.parse_args()

    rng = np.random.default_rng(SEED)
    started = time.perf_counter()
    cases = phase1(rng)
    print(f"phase 1: {len(cases)} designs", flush=True)
    records = run_all(cases, args.workers)
    problems = check(cases, records)
    if problems:
        print("\n".join(problems))
        raise SystemExit("phase 1 produced unexpected results; fix the generator before continuing")

    extra = derived(rng, cases, records)
    print(f"phase 2: {len(extra)} derived designs", flush=True)
    records.update(run_all(extra, args.workers))
    cases = cases + extra
    problems = check(cases, records)
    if problems:
        print("\n".join(problems))
        raise SystemExit("phase 2 produced unexpected results; fix the generator before continuing")

    gc.GOLDEN_DIR.mkdir(exist_ok=True)
    write_lines(gc.CORPUS_PATH, {"corpus_version": CORPUS_VERSION, "seed": SEED}, "cases", cases)
    manifest = {
        "corpus_version": CORPUS_VERSION,
        "generator": "tools/make_golden_corpus.py",
        "seed": SEED,
        "reference_sources": list(gc.REFERENCE_SOURCES),
        "reference_sources_sha256": gc.reference_sources_sha256(),
        "solidpy_git_sha": git_sha(),
        "python": platform.python_version(),
        "numpy": np.__version__,
        "scipy": scipy.__version__,
        "cases": len(cases),
        "curve_points": gc.CURVE_POINTS,
    }
    write_lines(gc.REFERENCE_PATH, manifest, "records", [{"id": c["id"], **records[c["id"]]} for c in cases])

    reasons = Counter(records[c["id"]]["status"]["termination_reason"] for c in cases)
    cpu_seconds = sum(records[c["id"]]["wall_seconds"] for c in cases)
    print(f"wrote {len(cases)} cases in {time.perf_counter() - started:.0f} s "
          f"({cpu_seconds:.0f} s of reference time, {cpu_seconds / len(cases):.2f} s per case)")
    print("termination reasons:", dict(reasons))
    print("families:", dict(Counter(c["family"] for c in cases)))


if __name__ == "__main__":
    main()
