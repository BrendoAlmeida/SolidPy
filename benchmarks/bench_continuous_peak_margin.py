#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Measure the 100x acceptance margin against refined scalar dense-output peaks.

The batched solver reports canonical, accepted-point metrics. With
``continuous_peak_diagnostics=True`` it also estimates peaks from its dense
DOP853 steps. This tool builds a high-accuracy scalar reference for every
supported golden-corpus design, refines all four peak metrics on the scalar
dense output, and compares those peaks plus the integral metrics through the
versioned numerical-acceptance policy.

Examples::

    python benchmarks/bench_continuous_peak_margin.py --backend cpu-vectorized
    XLA_PYTHON_CLIENT_PREALLOCATE=false python benchmarks/bench_continuous_peak_margin.py \\
        --backend jax --device cuda:0
"""

import argparse
import copy
import importlib
import json
import platform
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
from scipy.optimize import minimize_scalar

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "tests")]

from solidpy import backends  # noqa: E402
from solidpy.Acceptance import evaluate_numerical_acceptance  # noqa: E402
from solidpy.ensemble import ProblemBatch, simulate_burn  # noqa: E402
import golden_corpus as gc  # noqa: E402


PEAK_METRICS = (
    "max_generated_mass_flow_kg_s",
    "peak_chamber_pressure_pa",
    "peak_thrust_n",
    "max_nozzle_mass_flow_kg_s",
)
INTEGRAL_METRICS = (
    "total_impulse_ns",
    "generated_mass_integral_kg",
    "nozzle_mass_integral_kg",
)
MARGIN_LIMITS = {**{name: 2e-4 for name in PEAK_METRICS},
                 **{name: 1e-4 for name in INTEGRAL_METRICS}}


def _metric_value(quantities, name):
    if name == "max_generated_mass_flow_kg_s":
        return quantities["generated"]
    if name == "peak_chamber_pressure_pa":
        return quantities["pressure"]
    if name == "peak_thrust_n":
        return quantities["components"]["total_n"]
    return quantities["nozzle"]


def _scalar_dense_reference(case, max_step=None, rtol=None):
    grain, motor, propellant, environment, options = gc.build_objects(case)
    options = dict(options)
    if max_step is not None:
        options["max_step_size"] = max_step
    if rtol is not None:
        options["rtol"] = rtol
    burn_module = importlib.import_module("solidpy.Burn")
    original_solve_ivp = burn_module.solve_ivp
    dense_segments = []

    def solve_ivp_with_dense_output(*args, **kwargs):
        kwargs["dense_output"] = True
        result = original_solve_ivp(*args, **kwargs)
        dense_segments.append(result)
        return result

    burn_module.solve_ivp = solve_ivp_with_dense_output
    try:
        simulation = gc.BurnSimulation(grain, motor, propellant, environment, **options)
    finally:
        burn_module.solve_ivp = original_solve_ivp

    maxima = {name: -np.inf for name in PEAK_METRICS}
    cache = {}

    def quantities_at(time, segment):
        time = float(time)
        key = (id(segment), time)
        if key not in cache:
            cache[key] = simulation._state_quantities(time, segment.sol(time))
        return cache[key]

    for segment in dense_segments:
        for left, right in zip(segment.t[:-1], segment.t[1:]):
            if right <= left:
                continue
            for time in (left, np.nextafter(right, -np.inf), right):
                quantities = quantities_at(time, segment)
                for name in PEAK_METRICS:
                    maxima[name] = max(maxima[name], float(_metric_value(quantities, name)))
            for name in PEAK_METRICS:
                optimum = minimize_scalar(
                    lambda time, metric=name: -float(
                        _metric_value(quantities_at(time, segment), metric)
                    ),
                    bounds=(left, right),
                    method="bounded",
                    options={"xatol": 1e-13},
                )
                maxima[name] = max(maxima[name], -float(optimum.fun))

    refined = copy.deepcopy(simulation.result)
    refined["provenance"]["continuous_peak_oracle"] = {
        "method": "scipy_dop853_dense_bounded_per_step",
        "max_step_size_s": options.get("max_step_size"),
        "rtol": options.get("rtol"),
        "accepted_intervals": int(sum(max(len(s.t) - 1, 0) for s in dense_segments)),
    }
    for name, value in maxima.items():
        refined["metrics"][name] = value
    return refined


def _supported_corpus(backend_name, device, only_ids):
    capabilities = backends.get_backend(backend_name, device=device).capabilities()
    selected = []
    for case in gc.load_corpus()["cases"]:
        if only_ids is not None and case["id"] not in only_ids:
            continue
        grain, motor, propellant, environment, options = gc.build_objects(case)
        lane = ProblemBatch.from_objects(motor, propellant, environment, options)
        if not lane.unsupported(capabilities)[0]:
            selected.append((case, motor, propellant, environment, options))
    if only_ids is not None:
        found = {entry[0]["id"] for entry in selected}
        missing = sorted(set(only_ids) - found)
        if missing:
            raise ValueError(f"requested case(s) are missing or unsupported by {backend_name}: {missing}")
    if not selected:
        raise ValueError("the selected backend has no supported corpus cases")
    return selected


def _machine_info():
    try:
        gpu = subprocess.run(
            ["nvidia-smi", "--query-gpu=name,memory.total,driver_version", "--format=csv,noheader"],
            capture_output=True, text=True, timeout=10, check=True,
        ).stdout.strip()
    except (OSError, subprocess.SubprocessError):
        gpu = None
    import scipy

    return {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "numpy": np.__version__,
        "scipy": scipy.__version__,
        "gpu": gpu,
    }


def _margin_delta(report, metric):
    relative = report["convergence"][metric]["relative_delta"]
    return None if relative is None else relative / MARGIN_LIMITS[metric]


def run(args):
    started = time.perf_counter()
    selected = _supported_corpus(args.backend, args.device, args.case_ids)
    if args.max_cases is not None:
        selected = selected[:args.max_cases]

    cases = [entry[0] for entry in selected]
    batch = ProblemBatch.from_objects(
        [entry[1] for entry in selected],
        [entry[2] for entry in selected],
        [entry[3] for entry in selected],
        [entry[4] for entry in selected],
    )
    print(f"solving {len(cases)} supported designs on {args.backend}...", flush=True)
    batched = simulate_burn(
        batch,
        backend=args.backend,
        device=args.device,
        strict=True,
        continuous_peak_diagnostics=True,
        chunk_size=args.chunk_size,
    ).to_results()

    rows = []
    for index, (case, coarse_raw) in enumerate(zip(cases, batched), start=1):
        continuous = coarse_raw["provenance"]["execution"].get("continuous_peaks")
        if continuous is None:
            raise RuntimeError(f"{case['id']}: backend result has no continuous peak diagnostics")
        refined = _scalar_dense_reference(case, args.oracle_max_step, args.oracle_rtol)
        coarse = copy.deepcopy(coarse_raw)
        for name in PEAK_METRICS:
            coarse["metrics"][name] = continuous[name]
        acceptance = evaluate_numerical_acceptance(coarse, refined)
        margin_eligible = bool(coarse["status"]["completed"] and refined["status"]["completed"])
        margins = {
            name: (_margin_delta(acceptance, name) if margin_eligible else None)
            for name in MARGIN_LIMITS
        }
        relative_deltas = {
            name: (acceptance["convergence"][name]["relative_delta"] if margin_eligible else None)
            for name in MARGIN_LIMITS
        }
        rows.append({
            "case_id": case["id"],
            "case_family": case["family"],
            "acceptance_status": acceptance["status"],
            "acceptance_api_passed": acceptance["passed"],
            "outer_thresholds_passed": bool(
                margin_eligible
                and acceptance["mass_balance_passed"] is True
                and all(item["passed"] is True for item in acceptance["convergence"].values())
            ),
            "margin_eligible": margin_eligible,
            "incomplete_reasons": acceptance["incomplete_reasons"],
            "relative_deltas": relative_deltas,
            "margin_ratios": margins,
            "termination_match": (
                coarse["status"]["termination_reason"] == refined["status"]["termination_reason"]
            ),
            "completed_match": coarse["status"]["completed"] == refined["status"]["completed"],
            "backend_status": {
                key: coarse["status"].get(key)
                for key in ("completed", "termination_reason", "numerical_blowdown_completed", "burnout_completed")
            },
            "scalar_reference_status": {
                key: refined["status"].get(key)
                for key in ("completed", "termination_reason", "numerical_blowdown_completed", "burnout_completed")
            },
            "scalar_reference_solver_settings": refined["provenance"]["continuous_peak_oracle"],
            "backend_metrics": {name: coarse["metrics"].get(name) for name in MARGIN_LIMITS},
            "scalar_reference_metrics": {name: refined["metrics"].get(name) for name in MARGIN_LIMITS},
        })
        if index % 10 == 0 or index == len(cases):
            print(f"refined scalar oracle: {index}/{len(cases)}", flush=True)

    maxima = {}
    worst_cases = {}
    for metric in MARGIN_LIMITS:
        available = [(row["margin_ratios"][metric], row["case_id"]) for row in rows
                     if row["margin_ratios"][metric] is not None]
        if available:
            maxima[metric], worst_cases[metric] = max(available)
        else:
            maxima[metric], worst_cases[metric] = None, None
    all_margin_passed = all(value is not None and value <= 1.0 for value in maxima.values())
    eligible_rows = [row for row in rows if row["margin_eligible"]]
    all_outer_thresholds_passed = bool(eligible_rows) and all(
        row["outer_thresholds_passed"] for row in eligible_rows
    )
    all_acceptance_api_passed = bool(eligible_rows) and all(
        row["acceptance_api_passed"] for row in eligible_rows
    )
    same_outcomes = all(row["termination_match"] and row["completed_match"] for row in rows)
    output = {
        "schema": "solidpy_continuous_peak_margin_v2",
        "backend": args.backend,
        "device": args.device,
        "chunk_size": args.chunk_size,
        "design_count": len(rows),
        "oracle": {
            "method": "scipy_dop853_dense_bounded_per_step_at_corpus_settings",
            "max_step_size_s": args.oracle_max_step,
            "rtol": args.oracle_rtol,
        },
        "margin_eligible_design_count": len(eligible_rows),
        "margin_limits": MARGIN_LIMITS,
        "max_margin_ratios": maxima,
        "worst_cases": worst_cases,
        "all_100x_margins_passed": all_margin_passed,
        "all_outer_thresholds_passed": all_outer_thresholds_passed,
        "all_acceptance_api_passed": all_acceptance_api_passed,
        "termination_and_completion_match": same_outcomes,
        "elapsed_s": time.perf_counter() - started,
        "machine": _machine_info(),
        "cases": rows,
    }
    return output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backend", choices=("cpu-vectorized", "jax"), required=True)
    parser.add_argument("--device", default=None, help="JAX device, for example cuda:0")
    parser.add_argument("--case-ids", nargs="+", default=None, help="run selected corpus IDs")
    parser.add_argument("--max-cases", type=int, default=None, help="run only the first N supported cases")
    parser.add_argument("--chunk-size", type=int, default=32,
                        help="maximum lanes per backend call; defaults to 32 for bounded diagnostics")
    parser.add_argument("--oracle-max-step", type=float, default=None,
                        help="override each design's scalar max_step_size (defaults to corpus settings)")
    parser.add_argument("--oracle-rtol", type=float, default=None,
                        help="override each design's scalar rtol (defaults to corpus settings)")
    parser.add_argument("--out", type=Path, default=None, help="write the detailed JSON report here")
    args = parser.parse_args()
    if args.backend == "jax" and args.device is None:
        parser.error("--device is required for the JAX backend")
    if args.max_cases is not None and args.max_cases < 1:
        parser.error("--max-cases must be positive")
    if args.chunk_size < 1:
        parser.error("--chunk-size must be positive")
    if ((args.oracle_max_step is not None and args.oracle_max_step <= 0.0)
            or (args.oracle_rtol is not None and args.oracle_rtol <= 0.0)):
        parser.error("oracle tolerances must be positive")

    result = run(args)
    summary = {
        key: result[key]
        for key in (
            "backend", "device", "chunk_size", "design_count", "max_margin_ratios", "worst_cases",
            "margin_eligible_design_count", "all_100x_margins_passed",
            "all_outer_thresholds_passed", "all_acceptance_api_passed",
            "termination_and_completion_match", "elapsed_s",
        )
    }
    print(json.dumps(summary, indent=2, sort_keys=True))
    if args.out is not None:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
        print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
