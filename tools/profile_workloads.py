#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Measure where the scalar reference spends its time on the reference workloads and the share that is offloaded.

    python tools/profile_workloads.py                      # W1, W2, W3, printed
    python tools/profile_workloads.py --out benchmarks/results/offloaded_share.json

Workloads (docs/gpu_backend_architecture.md, section 7.1), all on the scalar code path:

* **W1** burn-only: ``BurnSimulation`` on a mix of golden-corpus designs.
* **W2** W1 plus the advanced physics: for four-grain design variants, ``run_detailed_ballistics`` and
  ``simulate_advanced_physics`` (thermal, structural, CFD proxies, ignition proxy, 1-D flight).
* **W3** robustness: ``run_robustness_analysis`` (nominal, default scenarios, Latin-hypercube samples).

The entry points listed in ``tools/tier_map.toml`` are timed with wall-clock wrappers (outermost call only) so that the
measurement does not inflate Python-heavy code the way a profiler does. ``offloaded_share`` is the time in entry points
marked ``batched = true`` over the whole workload. Needs Python 3.11 or newer (``tomllib``).
"""

import argparse
import copy
import importlib
import json
import sys
import time
import tomllib
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "tests")]

import golden_corpus as gc  # noqa: E402
from solidpy import (  # noqa: E402
    BurnSimulation, CasingMaterial, NozzleMaterial, geometry_from_components, run_detailed_ballistics,
    run_robustness_analysis, simulate_advanced_physics,
)
from test_robustness import make_motor_stack  # noqa: E402


class Timers:
    """Wall-clock time of the entry points of the tier map; only the outermost call of each is counted."""

    def __init__(self, entries):
        self.entries = entries
        self.seconds = {entry["path"]: 0.0 for entry in entries}
        self.calls = {entry["path"]: 0 for entry in entries}
        self._depth = 0
        self._originals = []

    def __enter__(self):
        for entry in self.entries:
            owner, name = self._resolve(entry["path"])
            original = owner.__dict__[name]
            self._originals.append((owner, name, original))
            setattr(owner, name, self._wrap(entry["path"], original))
        return self

    def __exit__(self, *exc):
        for owner, name, original in reversed(self._originals):
            setattr(owner, name, original)

    @staticmethod
    def _resolve(path):
        parts = path.split(".")
        for split in range(len(parts) - 1, 0, -1):
            try:
                owner = importlib.import_module(".".join(parts[:split]))
            except ImportError:
                continue
            for attribute in parts[split:-1]:
                owner = getattr(owner, attribute)
            return owner, parts[-1]
        raise ImportError(path)

    def _wrap(self, path, function):
        def timed(*args, **kwargs):
            if self._depth:
                return function(*args, **kwargs)
            self._depth += 1
            started = time.perf_counter()
            try:
                return function(*args, **kwargs)
            finally:
                self.seconds[path] += time.perf_counter() - started
                self.calls[path] += 1
                self._depth -= 1

        timed.__wrapped__ = function
        return timed


def w1_designs(count):
    corpus = gc.load_corpus()["cases"]
    reference = gc.load_reference()["records"]
    mixed = [c for c in corpus if reference[c["id"]]["history_points"] <= 300
             and reference[c["id"]]["status"]["termination_reason"] == "completed"
             and not {"igniter_callable", "activation_callable"} & set(c["tags"])]
    step = max(len(mixed) // count, 1)
    return mixed[::step][:count]


def four_grain_variants(count):
    """Variants of the test stack (grain count and throat) for W2 and W3; each needs dry hardware for the post-processing."""
    designs = []
    for i in range(count):
        grain, motor, propellant, environment = make_motor_stack()
        scaled = copy.deepcopy(motor)
        scaled.nozzle_throat_area *= (0.96 + 0.02 * (i % 5)) ** 2
        scaled.expansion_ratio = scaled.nozzle_exit_area / scaled.nozzle_throat_area
        designs.append((grain, scaled, propellant, environment))
    return designs


def run_w1(count):
    for case in w1_designs(count):
        gc.simulate(case)


def run_w2(count):
    for grain, motor, propellant, environment in four_grain_variants(count):
        curve = run_detailed_ballistics(grain, motor, propellant, environment, max_step_size=0.01, max_time_points=1000)
        geometry = geometry_from_components(grain, motor, propellant, casing_wall_thickness_m=0.004, dry_mass_kg=3.0)
        simulate_advanced_physics(geometry, curve, casing_material=CasingMaterial(), nozzle_material=NozzleMaterial(),
                                  flame_temp_k=propellant.combustion_temperature, r_specific=propellant.products_constant)


def run_w3(count):
    for grain, motor, propellant, environment in four_grain_variants(count):
        run_robustness_analysis(grain, motor, propellant, environment, monte_carlo_sample_count=4)


WORKLOADS = {"W1": (run_w1, 12), "W2": (run_w2, 6), "W3": (run_w3, 2)}


def measure(name, entries, size):
    run, default = WORKLOADS[name]
    with Timers(entries) as timers:
        started = time.perf_counter()
        run(size or default)
        wall = time.perf_counter() - started
    batched = sum(timers.seconds[e["path"]] for e in entries if e["batched"])
    listed = sum(timers.seconds.values())
    return {
        "workload": name, "size": size or default, "wall_s": wall, "offloaded_share": batched / wall,
        "listed_share": listed / wall, "other_s": wall - listed,
        "entries": {e["path"]: {"seconds": timers.seconds[e["path"]], "calls": timers.calls[e["path"]],
                                "share": timers.seconds[e["path"]] / wall, "tier": e["tier"], "batched": e["batched"]}
                    for e in entries},
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--workloads", default="W1,W2,W3")
    parser.add_argument("--size", type=int, default=None, help="designs per workload (default: 12, 6 and 2)")
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()
    entries = tomllib.loads((ROOT / "tools" / "tier_map.toml").read_text())["entry"]

    report = [measure(name, entries, args.size) for name in args.workloads.split(",")]
    for item in report:
        print(f"\n{item['workload']}: {item['size']} designs, {item['wall_s']:.1f} s on the scalar path, "
              f"offloaded_share = {item['offloaded_share']:.3f} (listed entry points {item['listed_share']:.3f})")
        for path, row in sorted(item["entries"].items(), key=lambda kv: -kv[1]["seconds"]):
            if row["calls"]:
                mark = "batched" if row["batched"] else "cpu    "
                print(f"   {row['share']:6.1%}  {row['seconds']:7.2f} s  {row['calls']:5d} calls  {row['tier']} {mark}  {path}")
        print(f"   {item['other_s'] / item['wall_s']:6.1%}  {item['other_s']:7.2f} s  other (copies, validation, glue)")
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps({"entries": entries, "workloads": report}, indent=2))
        print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
