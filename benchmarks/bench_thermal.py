#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Throughput benchmark of the thermal ablation (the part of workload W2 that costs most after the burn).

A lane is the wall conduction and throat ablation of one design: a bell-shaped burn curve, a casing with or without a
liner, a gas and a nozzle material. Two sets of lanes:

* ``typical``: the 15 cases of ``tests/thermal_cases.py`` (steel and aluminium casings, 4 to 11 wall cells, 50 to 200
  time steps) tiled to the batch size with the gas and the start temperature varied, so no two lanes are the same problem;
* ``wide``: random designs (4 to 18 wall cells, 30 to 400 time steps, metal and composite casings);
* ``advanced``: the whole advanced physics (thermal, structural, CFD and ignition proxies, 1-D flight) of designs whose curves
  come from the burn solver (the test motor with its throat varied, 0.01 s steps), tiled to the batch size. The scalar
  baseline is ``simulate_advanced_physics`` in a process pool and the batched run is ``run_advanced_physics_ensemble``,
  whose thermal ablation is one batch and whose other models run on the CPU, optionally in a process pool.

The baseline is the scalar model in a process pool, the best static schedule the CPU has because every lane is independent.

    python benchmarks/bench_thermal.py --backend cpu-scalar --workers 12,6 --lanes 1024 --out cpu.json
    python benchmarks/bench_thermal.py --backend cpu-vectorized --kind advanced --workers 1 --lanes 1024 --out workers1.json
    python benchmarks/bench_thermal.py --backend cpu-vectorized --kind advanced --workers 6 --lanes 1024 --out workers6.json
    python benchmarks/bench_thermal.py --backend jax --device cuda:0 --lanes 1024,4096,16384 --out gpu.json

The first call of an accelerator shape compiles; it is timed apart (``first_call_s``) and the throughput is the best of
``--repeat`` warm runs. ``pack_s`` is the host time to build the batch, which a user pays once per batch.
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

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "tests")]

from solidpy import (  # noqa: E402
    CasingMaterial, NozzleMaterial, backends, geometry_from_components, run_detailed_ballistics, simulate_advanced_physics,
)
from solidpy.backends import SolveOptions  # noqa: E402
from solidpy.ensemble import run_advanced_physics_ensemble  # noqa: E402
from test_robustness import make_motor_stack  # noqa: E402
from thermal_cases import CASES, case_lane, pack_lanes, random_lanes, scalar_thermal  # noqa: E402

DEGENERATE = ("single_point", "single_interval")


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


def lanes_for(kind, count):
    if kind == "wide":
        return random_lanes(count, seed=5)
    rng = np.random.default_rng(1)
    names = [name for name in CASES if name not in DEGENERATE]
    lanes = []
    for i in range(count):
        lane = copy.copy(case_lane(names[i % len(names)]))
        lane["flame_temp_k"] = float(lane["flame_temp_k"]) * (0.95 + 0.1 * rng.random())
        lane["initial_temperature_k"] = float(lane["initial_temperature_k"]) + float(rng.uniform(-20.0, 20.0))
        lanes.append(lane)
    return lanes


def _scalar_lane(lane):
    return scalar_thermal(lane)


def bench_scalar(kind, count, workers):
    lanes = lanes_for(kind, count)
    started = time.perf_counter()
    if workers <= 1:
        for lane in lanes:
            _scalar_lane(lane)
    else:
        with ProcessPoolExecutor(max_workers=workers) as pool:
            list(pool.map(_scalar_lane, lanes, chunksize=4))
    seconds = time.perf_counter() - started
    return {"backend": "cpu-scalar", "kind": kind, "workers": workers, "lanes": count, "seconds": seconds,
            "lanes_per_s": count / seconds}


def bench_backend(kind, count, name, device, repeat, workers):
    lanes = lanes_for(kind, count)
    started = time.perf_counter()
    batch = pack_lanes(lanes)
    pack_s = time.perf_counter() - started
    backend = backends.get_backend(name, device)
    options = SolveOptions(workers=workers)
    started = time.perf_counter()
    result = backend.thermal_ablation(batch, options)
    first_call = time.perf_counter() - started
    best, best_timings = None, None
    for _ in range(repeat):
        started = time.perf_counter()
        result = backend.thermal_ablation(batch, options)
        seconds = time.perf_counter() - started
        if best is None or seconds < best:
            best, best_timings = seconds, dict(getattr(backend, "last_timings", {}))
    return {"backend": name, "device": device, "kind": kind, "lanes": count, "nodes": batch.n_max, "steps": batch.t_max,
            "pack_s": pack_s, "first_call_s": first_call, "seconds": best, "lanes_per_s": count / best,
            "timings": best_timings, "failed_lanes": len(result.execution.get("failed_lanes", [])),
            "radau_steps": result.execution.get("radau_steps")}


def advanced_designs(count):
    """``count`` designs of the test motor with the throat varied: (geometry, curve, propellant) with real burn curves."""
    distinct = []
    for i in range(6):
        grain, motor, propellant, environment = make_motor_stack()
        motor.nozzle_throat_area *= (0.96 + 0.02 * (i % 5)) ** 2
        motor.expansion_ratio = motor.nozzle_exit_area / motor.nozzle_throat_area
        curve = run_detailed_ballistics(grain, motor, propellant, environment, max_step_size=0.01, max_time_points=1000)
        geometry = geometry_from_components(grain, motor, propellant, casing_wall_thickness_m=0.004, dry_mass_kg=3.0)
        distinct.append((geometry, curve, propellant))
    return [distinct[i % len(distinct)] for i in range(count)]


ADVANCED_CASING = CasingMaterial(liner_thickness_m=0.002)


def _advanced_lane(design):
    geometry, curve, propellant = design
    return simulate_advanced_physics(geometry, curve, casing_material=ADVANCED_CASING, nozzle_material=NozzleMaterial(),
                                     flame_temp_k=propellant.combustion_temperature, r_specific=propellant.products_constant)


def bench_advanced_scalar(count, workers):
    designs = advanced_designs(count)
    started = time.perf_counter()
    if workers <= 1:
        for design in designs:
            _advanced_lane(design)
    else:
        with ProcessPoolExecutor(max_workers=workers) as pool:
            list(pool.map(_advanced_lane, designs, chunksize=4))
    seconds = time.perf_counter() - started
    return {"backend": "cpu-scalar", "kind": "advanced", "workers": workers, "lanes": count, "seconds": seconds,
            "lanes_per_s": count / seconds}


def bench_advanced_backend(count, name, device, repeat, workers):
    designs = advanced_designs(count)
    geometries = [d[0] for d in designs]
    curves = [d[1] for d in designs]
    propellant = designs[0][2]
    options = dict(casing_material=ADVANCED_CASING, nozzle_material=NozzleMaterial(), flame_temp_k=propellant.combustion_temperature,
                   r_specific=propellant.products_constant, backend=name, device=device, workers=workers)
    timings = {}
    started = time.perf_counter()
    run_advanced_physics_ensemble(geometries, curves, timings=timings, **options)
    first_call = time.perf_counter() - started
    best, best_timings = None, None
    for _ in range(repeat):
        timings = {}
        started = time.perf_counter()
        run_advanced_physics_ensemble(geometries, curves, timings=timings, **options)
        seconds = time.perf_counter() - started
        if best is None or seconds < best:
            best, best_timings = seconds, dict(timings)
    return {"backend": name, "device": device, "kind": "advanced", "lanes": count, "workers": workers,
            "first_call_s": first_call, "seconds": best, "lanes_per_s": count / best, "timings": best_timings}


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--backend", required=True, help="cpu-scalar (the scalar model in a pool), cpu-reference, cpu-vectorized, jax")
    parser.add_argument("--device", default=None)
    parser.add_argument("--lanes", default="1024", help="comma-separated numbers of lanes")
    parser.add_argument("--kind", default="typical", choices=("typical", "wide", "advanced"))
    parser.add_argument("--workers", default="1",
                        help="process counts for cpu-scalar; count for cpu-reference or advanced-ensemble post-processing")
    parser.add_argument("--repeat", type=int, default=2)
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()

    results = []
    sizes = [int(n) for n in args.lanes.split(",")]
    worker_count = int(args.workers.split(",")[0])
    if args.kind != "advanced" and args.backend not in ("cpu-scalar", "cpu-reference") and worker_count != 1:
        parser.error("--workers above 1 applies to cpu-scalar/reference solves or --kind advanced post-processing")
    if args.backend == "cpu-scalar":
        for workers in (int(w) for w in args.workers.split(",")):
            for count in sizes:
                results.append(bench_advanced_scalar(count, workers) if args.kind == "advanced"
                               else bench_scalar(args.kind, count, workers))
                print(json.dumps(results[-1]), flush=True)
    else:
        workers = worker_count if args.backend == "cpu-reference" or args.kind == "advanced" else None
        for count in sizes:
            results.append(bench_advanced_backend(count, args.backend, args.device, args.repeat, workers)
                           if args.kind == "advanced"
                           else bench_backend(args.kind, count, args.backend, args.device, args.repeat, workers))
            print(json.dumps(results[-1]), flush=True)
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps({"machine": machine_info(), "results": results}, indent=2))
        print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
