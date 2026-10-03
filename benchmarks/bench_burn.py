#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Throughput benchmark of the burn solvers on workload W1 (burn-only ensembles).

The workload is every golden-corpus design the batched backends support, tiled to the batch size. Heavy-tail
designs (up to ~23,000 accepted steps) are included on purpose: lockstep solvers pay for their slowest lane.

    python benchmarks/bench_burn.py --backend jax --device cuda:0 --sizes 256,1024,4096,8192 --out result.json
    python benchmarks/bench_burn.py --backend cpu-reference --workers 6,12 --out reference.json

For an accelerator the first call of each shape compiles; it is timed apart (``first_call_s``) and the
reported throughput is the best of ``--repeat`` warm runs. ``cpu-reference`` runs the corpus once per worker
count with the heavy lanes first, which is the best static schedule a process pool can have.
"""

import argparse
import json
import os
import platform
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "tests")]

import golden_corpus as gc  # noqa: E402
from solidpy import backends  # noqa: E402
from solidpy.backends import SolveOptions  # noqa: E402
from solidpy.backends.numpy_vectorized import NumpyBackend  # noqa: E402
from solidpy.ensemble import ProblemBatch, lane_cost, simulate_burn  # noqa: E402


def machine_info():
    info = {"python": platform.python_version(), "platform": platform.platform(), "cpu_count": os.cpu_count()}
    try:
        cpu = [line for line in Path("/proc/cpuinfo").read_text().splitlines() if line.startswith("model name")]
        info["cpu"] = cpu[0].split(":", 1)[1].strip() if cpu else None
    except OSError:
        info["cpu"] = platform.processor()
    try:
        gpu = subprocess.run(["nvidia-smi", "--query-gpu=name,memory.total,driver_version", "--format=csv,noheader"],
                             capture_output=True, text=True, timeout=10, check=True).stdout.strip()
        info["gpu"] = gpu
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


def workload(max_points=None):
    """The supported corpus designs as one ProblemBatch, plus their case ids.

    ``max_points`` keeps only designs whose reference run took at most that many accepted steps, which drops the
    heavy tail and leaves the "typical" ensemble.
    """
    supported = set(NumpyBackend.SUPPORTED_FEATURES)
    cases = gc.load_corpus()["cases"]
    stored = gc.load_reference()["records"]
    keep, built = [], []
    for case in cases:
        if max_points is not None and stored[case["id"]]["history_points"] > max_points:
            continue
        grain, motor, propellant, environment, kwargs = gc.build_objects(case)
        batch = ProblemBatch.from_objects(motor, propellant, environment, kwargs)
        if set(batch.lane_features[0]) <= supported:
            keep.append(case["id"])
            built.append((motor, propellant, environment, kwargs))
    base = ProblemBatch.from_objects([b[0] for b in built], [b[1] for b in built], [b[2] for b in built],
                                     [b[3] for b in built])
    return keep, base


def tile(base, size):
    return base.select(np.arange(size) % len(base))


def device_memory(backend):
    try:
        stats = backend._device.memory_stats()
        return stats.get("peak_bytes_in_use") if stats else None
    except Exception:
        return None


def bench_accelerator(args, base, ids):
    backend = backends.get_backend(args.backend, args.device)
    rows = []
    for size in args.sizes:
        batch = tile(base, size)
        start = time.perf_counter()
        simulate_burn(batch, backend=args.backend, device=args.device, history="metrics", tiers=args.tiers)
        first = time.perf_counter() - start
        warm, parts = [], []
        for _ in range(args.repeat):
            start = time.perf_counter()
            result = simulate_burn(batch, backend=args.backend, device=args.device, history="metrics",
                                   tiers=args.tiers)
            warm.append(time.perf_counter() - start)
            parts.append(dict(backend.last_timings))
        best = int(np.argmin(warm))
        failed = sum(1 for r in result.to_results() if not r["status"]["completed"])
        rows.append({
            "lanes": size, "first_call_s": round(first, 3), "warm_s": [round(w, 3) for w in warm],
            "best_warm_s": round(warm[best], 3), "lanes_per_s": round(size / warm[best], 2),
            "split_s": {k: round(v, 3) for k, v in parts[best].items()},
            "peak_device_bytes": device_memory(backend), "lanes_not_completed": failed,
            "tiers": result.execution.get("tiers"),
        })
        print(json.dumps(rows[-1]), flush=True)
    return rows


def bench_reference(args, base):
    """The scalar solver with a process pool, heavy lanes first, on the corpus tiled to each batch size.

    A pool cannot finish before its slowest lane: with one pass over the corpus the makespan is the time of the
    heaviest lane (~50 s) whatever the core count, so the throughput that matters for large ensembles is the
    one measured on a tiled batch, where the cores stay busy.
    """
    reference = backends.get_backend("cpu-reference")
    rows = []
    for size in args.sizes:
        tiled = tile(base, size)
        batch = tiled.select(np.argsort(-lane_cost(tiled), kind="stable"))
        for workers in args.workers:
            start = time.perf_counter()
            reference.solve_burn(batch, SolveOptions(workers=workers))
            elapsed = time.perf_counter() - start
            rows.append({"lanes": size, "workers": workers, "seconds": round(elapsed, 2),
                         "lanes_per_s": round(size / elapsed, 3)})
            print(json.dumps(rows[-1]), flush=True)
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--backend", default="jax")
    parser.add_argument("--device", default=None)
    parser.add_argument("--sizes", type=lambda s: [int(x) for x in s.split(",")], default=[256, 1024, 4096])
    parser.add_argument("--workers", type=lambda s: [int(x) for x in s.split(",")], default=[6, 12])
    parser.add_argument("--repeat", type=int, default=2)
    parser.add_argument("--tiers", type=lambda s: () if s == "none" else tuple(int(x) for x in s.split(",")),
                        default=None, help="iteration caps of the capped tiers, or 'none' for one uncapped solve")
    parser.add_argument("--max-points", type=int, default=None,
                        help="keep only designs with at most this many accepted steps in the reference run")
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()

    ids, base = workload(args.max_points)
    name = "W1" if args.max_points is None else f"W1 typical (<= {args.max_points} accepted steps)"
    record = {"workload": f"{name}: supported golden-corpus designs, tiled", "base_lanes": len(ids),
              "machine": machine_info(), "backend": args.backend, "device": args.device,
              "tiers": "default" if args.tiers is None else list(args.tiers)}
    print(f"{len(ids)} base lanes; backend {args.backend} device {args.device}", flush=True)
    if args.backend == "cpu-reference":
        record["results"] = bench_reference(args, base)
    else:
        record["results"] = bench_accelerator(args, base, ids)
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(record, indent=2) + "\n")
        print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
