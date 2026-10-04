#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Break down the warmed W4 StructuralMonteCarlo.run wall time by host stage.

The backend service timer includes array preparation, dispatch, synchronization and
result conversion. The profiler also reads the JAX backend's internal device-region
timer, which includes device transfer and waiting for the result. These measurements
are diagnostic and do not replace the end-to-end throughput benchmark.

    python benchmarks/profile_structural_monte_carlo.py --backend cpu-vectorized
    python benchmarks/profile_structural_monte_carlo.py --backend jax --device cuda:0
"""

import argparse
import gc
import json
import os
import platform
import statistics
import subprocess
import sys
import time
from pathlib import Path

try:
    import resource
except ImportError:  # pragma: no cover - unavailable on Windows
    resource = None

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
import scipy

from benchmarks.bench_structural_monte_carlo import FixedPeakPressure, _design
from solidpy import StructuralMonteCarlo, backends


def _machine_info():
    info = {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "cpu_count": os.cpu_count(),
        "numpy": np.__version__,
        "scipy": scipy.__version__,
    }
    try:
        names = [line for line in Path("/proc/cpuinfo").read_text().splitlines()
                 if line.startswith("model name")]
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
    return info


def _peak_rss_mib():
    if resource is None:
        return None
    try:
        peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    except (AttributeError, OSError):
        return None
    # Linux reports KiB; macOS reports bytes.
    scale = 1024 if sys.platform.startswith("linux") else 1024 * 1024
    return peak / scale


def _run_profile(iterations, backend_name, device, design_index, repeat):
    geometry, material = _design(design_index)
    model = StructuralMonteCarlo(
        geometry,
        material,
        peak_pressure_distribution=FixedPeakPressure(6.5e6 + design_index * 0.4e6),
        parameter_sigmas={"perturb_peak_pressure": {"mean": 0.0, "sigma": 8.0e5}},
        bolt_count=4,
        bolt_diameter_m=0.005,
        bolt_strength_mpa=600.0,
        random_seed=2026 + design_index,
    )
    backend = backends.get_backend(backend_name, device)
    times = {
        "sample_parameters_s": 0.0,
        "structural_batch_s": 0.0,
        "backend_service_wall_s": 0.0,
    }
    original_sample = model._sample_parameters
    original_batch = model._run_structural_batch
    original_service = backend.structural_response

    def timed_sample(n_iterations, rng):
        start = time.perf_counter()
        try:
            return original_sample(n_iterations, rng)
        finally:
            times["sample_parameters_s"] += time.perf_counter() - start

    def timed_batch(samples, perturbations, selected_backend, strict):
        start = time.perf_counter()
        try:
            return original_batch(samples, perturbations, selected_backend, strict)
        finally:
            times["structural_batch_s"] += time.perf_counter() - start

    def timed_service(*positional, **keywords):
        start = time.perf_counter()
        try:
            return original_service(*positional, **keywords)
        finally:
            times["backend_service_wall_s"] += time.perf_counter() - start

    model._sample_parameters = timed_sample
    model._run_structural_batch = timed_batch
    backend.structural_response = timed_service

    try:
        warmup = model.run(iterations, backend=backend_name, device=device)
        del warmup
        gc.collect()
        measurements = []
        for _ in range(repeat):
            for key in times:
                times[key] = 0.0
            if hasattr(backend, "_device_s"):
                backend._device_s = 0.0

            started = time.perf_counter()
            result = model.run(iterations, backend=backend_name, device=device)
            total_s = time.perf_counter() - started
            batch_host_s = times["structural_batch_s"] - times["backend_service_wall_s"]
            post_batch_s = total_s - times["sample_parameters_s"] - times["structural_batch_s"]
            measurements.append({
                "total_s": total_s,
                **times,
                "structural_batch_host_remainder_s": batch_host_s,
                "post_batch_run_remainder_s": post_batch_s,
                "jax_device_region_s": getattr(backend, "_device_s", None),
                "n_evaluated": result["n_evaluated"],
                "n_fallback": len(result["provenance"].get("execution", {}).get("fallback_lanes", [])),
                # resource reports the process high-water mark, including the warm-up call.
                "process_peak_rss_mib_including_warmup": _peak_rss_mib(),
            })
            del result
        measured_fields = (
            "total_s", "sample_parameters_s", "structural_batch_s", "backend_service_wall_s",
            "structural_batch_host_remainder_s", "post_batch_run_remainder_s", "jax_device_region_s",
            "process_peak_rss_mib_including_warmup",
        )
        medians = {
            field: statistics.median(row[field] for row in measurements)
            for field in measured_fields
            if all(row[field] is not None for row in measurements)
        }
        median_total_run = min(
            measurements, key=lambda row: abs(row["total_s"] - medians["total_s"])
        )
        return {
            "design": design_index,
            "iterations": iterations,
            "repeat": repeat,
            "median": medians,
            "median_total_run": median_total_run,
            "measurements": measurements,
        }
    finally:
        backend.structural_response = original_service


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--backend", choices=("cpu-vectorized", "jax"), required=True)
    parser.add_argument("--device", default=None)
    parser.add_argument("--iterations", type=int, default=100_000)
    parser.add_argument("--designs", type=int, default=1)
    parser.add_argument("--repeat", type=int, default=3)
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()
    if args.iterations < 1 or args.designs < 1 or args.repeat < 1:
        parser.error("iterations, designs and repeat must be positive")

    backend = backends.get_backend(args.backend, args.device)
    results = [
        _run_profile(args.iterations, args.backend, args.device, i, args.repeat)
        for i in range(args.designs)
    ]
    summary = {
        "schema_version": "w4-stage-profile-v3",
        "backend": args.backend,
        "device": args.device,
        "backend_provenance": backend.provenance(),
        "service": "structural_response",
        "workload": "W4",
        "machine": _machine_info(),
        "iterations_per_design": args.iterations,
        "designs": args.designs,
        "warmups": 1,
        "repeat": args.repeat,
        "timing_notes": {
            "median": "Component-wise medians; values from different repeats may not add to median total_s.",
            "median_total_run": "Repeat closest to median total_s; its stage values add to its total_s.",
            "structural_batch_s": "Includes backend_service_wall_s.",
            "structural_batch_host_remainder_s": "Structural batch time minus backend service call time.",
            "post_batch_run_remainder_s": "Remaining run time after sampling and structural batch stages.",
            "jax_device_region_s": (
                "Internal JAX timer for device_put, compiled invocation, synchronization and device_get."
            ),
            "process_peak_rss_mib_including_warmup": "Process high-water RSS, including the warm-up call.",
        },
        "results": results,
    }
    rendered = json.dumps(summary, indent=2) + "\n"
    print(rendered, end="")
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(rendered)


if __name__ == "__main__":
    main()
