#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Measure W4 structural Monte Carlo throughput for a few fixed designs.

The default is the plan's 100,000 peak-pressure samples per design. The first call
is timed separately so an optional JAX compile is visible; warm calls include host
sampling and report assembly, as users of ``StructuralMonteCarlo.run`` pay them.

    python benchmarks/bench_structural_monte_carlo.py --backend cpu-reference
    python benchmarks/bench_structural_monte_carlo.py --backend cpu-vectorized
    python benchmarks/bench_structural_monte_carlo.py --backend jax --device cuda:0
"""

import argparse
import gc
import json
import os
import platform
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from solidpy import CasingMaterial, MotorGeometry, StructuralMonteCarlo  # noqa: E402
from solidpy import backends  # noqa: E402


class FixedPeakPressure:
    """Picklable deterministic nominal pressure source for a design."""

    def __init__(self, pressure_pa):
        self.pressure_pa = float(pressure_pa)

    def __call__(self):
        return self.pressure_pa


def _design(index):
    return (
        MotorGeometry(
            motor_length_m=0.22 + 0.01 * (index % 3),
            motor_inner_diameter_m=0.074 + 0.002 * (index % 2),
            casing_wall_thickness_m=0.0035 + 0.0005 * (index % 3),
            grain_outer_diameter_m=0.066,
            grain_core_diameter_m=0.028,
            grain_gap_m=0.002,
            grain_length_each_m=0.09,
            grain_number=3,
            fill_length_m=0.20,
            throat_diameter_m=0.012,
            exit_diameter_m=0.026,
            free_volume_m3=0.0008,
            propellant_mass_kg=0.75,
            dry_mass_kg=2.8,
            motor_initial_mass_kg=3.55,
            motor_final_mass_kg=2.8,
        ),
        CasingMaterial(),
    )


def _machine_info():
    info = {"python": platform.python_version(), "platform": platform.platform(), "cpu_count": os.cpu_count()}
    try:
        names = [line for line in Path("/proc/cpuinfo").read_text().splitlines() if line.startswith("model name")]
        info["cpu"] = names[0].split(":", 1)[1].strip() if names else platform.processor()
    except OSError:
        info["cpu"] = platform.processor()
    try:
        info["gpu"] = subprocess.run(
            ["nvidia-smi", "--query-gpu=name,memory.total,driver_version", "--format=csv,noheader"],
            capture_output=True, text=True, timeout=10, check=True,
        ).stdout.strip()
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


def _run_one(index, iterations, backend, device, repeat):
    geometry, material = _design(index)
    model = StructuralMonteCarlo(
        geometry,
        material,
        peak_pressure_distribution=FixedPeakPressure(6.5e6 + index * 0.4e6),
        parameter_sigmas={"perturb_peak_pressure": {"mean": 0.0, "sigma": 8.0e5}},
        bolt_count=4,
        bolt_diameter_m=0.005,
        bolt_strength_mpa=600.0,
        random_seed=2026 + index,
    )

    started = time.perf_counter()
    first = model.run(iterations, backend=backend, device=device)
    first_call_s = time.perf_counter() - started
    complete = first["n_evaluated"]
    fallback = len(first["provenance"].get("execution", {}).get("fallback_lanes", []))
    del first
    gc.collect()

    warm_times = []
    for _ in range(repeat):
        started = time.perf_counter()
        result = model.run(iterations, backend=backend, device=device)
        warm_times.append(time.perf_counter() - started)
        complete = result["n_evaluated"]
        fallback = len(result["provenance"].get("execution", {}).get("fallback_lanes", []))
        del result
        gc.collect()
    best_warm = min(warm_times) if warm_times else None
    return {
        "design": index,
        "iterations": iterations,
        "first_call_s": first_call_s,
        "first_samples_per_s": iterations / first_call_s,
        "warm_times_s": warm_times,
        "best_warm_s": best_warm,
        "warm_samples_per_s": iterations / best_warm if best_warm else None,
        "n_evaluated": complete,
        "n_fallback": fallback,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--backend", choices=("cpu-reference", "cpu-vectorized", "jax"), required=True)
    parser.add_argument("--device", default=None)
    parser.add_argument("--iterations", type=int, default=100_000)
    parser.add_argument("--designs", type=int, default=4)
    parser.add_argument("--repeat", type=int, default=1)
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()
    if args.iterations < 1 or args.designs < 1 or args.repeat < 0:
        parser.error("iterations/designs must be positive and repeat must be non-negative")

    backend = backends.get_backend(args.backend, args.device)
    results = [
        _run_one(i, args.iterations, args.backend, args.device, args.repeat)
        for i in range(args.designs)
    ]
    summary = {
        "backend": args.backend,
        "device": args.device,
        "backend_provenance": backend.provenance(),
        "service": "structural_response",
        "workload": "W4",
        "machine": _machine_info(),
        "iterations_per_design": args.iterations,
        "designs": args.designs,
        "repeat": args.repeat,
        "results": results,
    }
    print(json.dumps(summary, indent=2))
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(summary, indent=2) + "\n")


if __name__ == "__main__":
    main()
