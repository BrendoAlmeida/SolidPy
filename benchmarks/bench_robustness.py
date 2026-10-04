#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Throughput benchmark of the robustness analysis (workload W3): many designs, each nominal + scenarios.

A design is a variant of the tabulated four-grain test motor or of a two-grain power-law motor, with its throat and
density varied, run through ``run_robustness_ensemble`` (every design x scenario is one lane): the default scenarios plus
Latin-hypercube samples, 27 lanes per design with the default 16 samples. The baseline is the scalar path
(``run_robustness_analysis`` without a backend) in a process pool with one design per task, the best static schedule
the CPU has because every design is independent.

    python benchmarks/bench_robustness.py --backend cpu-scalar --workers 12,6 --designs 24 --out cpu.json
    python benchmarks/bench_robustness.py --backend jax --device cuda:0 --designs 16,64,152,304 --out gpu.json

The first call of an accelerator shape compiles; it is timed apart (``first_call_s``) and the throughput is the best of
``--repeat`` warm runs. ``timings`` splits a run into packing, burns, the batched detailed-ballistics service, scalar
fallback work and report assembly.
"""

import argparse
import copy
import json
import os
import platform
import subprocess
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "tests")]

from solidpy import Environment, Grain, Motor, Propellant, run_robustness_analysis  # noqa: E402
from solidpy.ensemble import run_robustness_ensemble  # noqa: E402
from test_robustness import make_motor_stack  # noqa: E402

MONTE_CARLO_SAMPLES = 16
MAX_STEP_SIZE = 0.01
MAX_TIME_POINTS = 1000


def machine_info():
    info = {"python": platform.python_version(), "platform": platform.platform(), "cpu_count": os.cpu_count()}
    try:
        cpu = [line for line in Path("/proc/cpuinfo").read_text().splitlines() if line.startswith("model name")]
        info["cpu"] = cpu[0].split(":", 1)[1].strip() if cpu else None
    except OSError:
        info["cpu"] = platform.processor()
    try:
        info["gpu"] = subprocess.run(["nvidia-smi", "--query-gpu=name,memory.total,driver_version", "--format=csv,noheader"],
                                     capture_output=True, text=True, timeout=10, check=True).stdout.strip()
    except (OSError, subprocess.SubprocessError):
        info["gpu"] = None
    import numpy
    import scipy

    info.update(numpy=numpy.__version__, scipy=scipy.__version__)
    try:
        import jax

        info["jax"] = jax.__version__
    except Exception:
        info["jax"] = None
    return info


def two_grain_design():
    grain = Grain(outer_radius=0.0305, initial_inner_radius=0.0115, initial_height=0.09)
    motor = Motor([grain, grain], chamber_inner_radius=0.032, chamber_length=0.2, nozzle_throat_radius=0.0085,
                  nozzle_exit_radius=0.021, nozzle_angle=0.2618, dry_mass_kg=1.5, dry_center_of_mass_position_m=0.0)
    propellant = Propellant(1.1308, 0.04197, 1720.0, density=1879.0, burn_rate_a=7.36, burn_rate_n=0.32)
    return grain, motor, propellant, Environment()


def designs(count):
    """``count`` designs: two base motors with the throat and density varied, so no two lanes are the same problem."""
    out = []
    for i in range(count):
        grain, motor, propellant, environment = copy.deepcopy(make_motor_stack() if i % 2 == 0 else two_grain_design())
        factor = 0.94 + 0.12 * ((i * 7919) % 101) / 100.0
        motor.nozzle_throat_area *= factor**2
        motor.expansion_ratio = motor.nozzle_exit_area / motor.nozzle_throat_area
        propellant.density *= 0.98 + 0.04 * ((i * 104729) % 97) / 96.0
        out.append((grain, motor, propellant, environment))
    return out


def _scalar_design(design):
    return run_robustness_analysis(*design, monte_carlo_sample_count=MONTE_CARLO_SAMPLES, max_step_size=MAX_STEP_SIZE,
                                   max_time_points=MAX_TIME_POINTS)["status"]


def lanes_per_design():
    return 1 + 10 + MONTE_CARLO_SAMPLES


def bench_scalar(count, workers):
    work = designs(count)
    started = time.perf_counter()
    if workers <= 1:
        for design in work:
            _scalar_design(design)
    else:
        with ProcessPoolExecutor(max_workers=workers) as pool:
            list(pool.map(_scalar_design, work, chunksize=1))
    seconds = time.perf_counter() - started
    lanes = count * lanes_per_design()
    return {"backend": "cpu-scalar", "workers": workers, "designs": count, "lanes": lanes, "seconds": seconds,
            "lanes_per_s": lanes / seconds}


def bench_ensemble(count, backend, device, repeat, max_steps, workers, keep_series, chunk_lanes):
    work = designs(count)
    options = dict(monte_carlo_sample_count=MONTE_CARLO_SAMPLES, max_step_size=MAX_STEP_SIZE,
                   max_time_points=MAX_TIME_POINTS, backend=backend, device=device, max_steps=max_steps, workers=workers,
                   keep_series=keep_series, chunk_lanes=chunk_lanes)
    timings, execution = {}, {}
    started = time.perf_counter()
    first = run_robustness_ensemble(work, timings=timings, execution=execution, **options)
    first_call = time.perf_counter() - started
    best, best_timings, best_execution = None, None, None
    for _ in range(repeat):
        timings, execution = {}, {}
        started = time.perf_counter()
        reports = run_robustness_ensemble(work, timings=timings, execution=execution, **options)
        seconds = time.perf_counter() - started
        if best is None or seconds < best:
            best, best_timings, best_execution = seconds, dict(timings), dict(execution)
    lanes = count * lanes_per_design()
    fallback = sum(
        1 for report in reports for r in [report["nominal"]] + report["scenarios"]
        if (r["provenance"].get("execution") or {}).get("fallback")
    )
    complete = sum(report["status"] == "completed" for report in reports)
    return {"backend": backend, "device": device, "designs": count, "lanes": lanes, "workers": workers,
            "chunk_lanes": chunk_lanes,
            "first_call_s": first_call,
            "seconds": best, "lanes_per_s": lanes / best, "timings": best_timings, "fallback_lanes": fallback,
            "completed_designs": complete, "max_steps": max_steps, "keep_series": keep_series,
            "execution": best_execution}


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--backend", required=True, help="cpu-scalar (the scalar path in a pool), cpu-reference, cpu-vectorized, jax")
    parser.add_argument("--device", default=None)
    parser.add_argument("--designs", default="16", help="comma-separated numbers of designs (27 lanes each)")
    parser.add_argument("--workers", default="1",
                        help="process counts for cpu-scalar; CPU post-processing workers for ensemble backends")
    parser.add_argument("--repeat", type=int, default=2)
    parser.add_argument("--max-steps", type=int, default=2000, help="history points per lane on a batched backend")
    parser.add_argument("--chunk-lanes", type=int, default=4096,
                        help="maximum burn lanes per solve chunk; reduce to measure W3 pipeline overlap")
    parser.add_argument("--keep-series", action="store_true",
                        help="keep the full series and history of every lane (about 300 kB each); default: scalars only")
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()

    results = []
    sizes = [int(n) for n in args.designs.split(",")]
    if args.backend == "cpu-scalar":
        for workers in (int(w) for w in args.workers.split(",")):
            for count in sizes:
                results.append(bench_scalar(count, workers))
                print(json.dumps(results[-1]), flush=True)
    else:
        workers = int(args.workers.split(",")[0])
        for count in sizes:
            results.append(bench_ensemble(count, args.backend, args.device, args.repeat, args.max_steps, workers,
                                          args.keep_series, args.chunk_lanes))
            print(json.dumps(results[-1]), flush=True)
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps({"machine": machine_info(), "monte_carlo_samples": MONTE_CARLO_SAMPLES,
                                        "max_step_size": MAX_STEP_SIZE, "results": results}, indent=2))
        print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
