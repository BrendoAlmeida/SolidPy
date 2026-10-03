# Accelerated backends: first benchmark and the Phase 3 gate

Measured 2026-10-03 with `benchmarks/bench_burn.py`; raw results are in `benchmarks/results/`. This is the
go/no-go gate of `docs/gpu_backend_architecture.md` (section 12, Phase 3): parity on the GPU **and** at least 5x
the throughput of the CPU reference on all cores at 4,096 lanes or more, in float64.

**Result: the gate is met.** At 4,096 lanes the JAX backend on the GPU runs 11.8x faster than the scalar
reference on all CPU cores (77.7 against 6.59 lanes/s) on the full corpus workload, and 54.7x faster on the
typical workload. Parity holds on the CPU device and on the GPU.

## Setup

| | |
|---|---|
| CPU | AMD Ryzen 5 3600, 6 cores / 12 threads |
| GPU | NVIDIA GeForce RTX 4060, 8 GiB, driver 610.57 (consumer card, float64 at 1/64 of float32 peak) |
| Software | Python 3.12.13, NumPy 2.5.3, SciPy 1.18.1, JAX 0.11.2 with the CUDA 12 plugin |
| Precision | float64 everywhere (`jax_enable_x64` scoped to each solve) |

## Workloads

Both are the golden-corpus designs that the batched kernels support (no igniter, activation, tabulated burn rate
or thermochemistry), tiled to the batch size. The tiling repeats the same designs, so the mix is fixed.

* **W1**: all 256 supported designs. It keeps a heavy tail on purpose. The median design needs 200 accepted steps
  and 90 % need at most 515, but 12 need more than 1,000 and two more than 10,000 (up to 23,307). The slowest are
  the `guard-nearpi` designs (star grains whose `n_points * epsilon` is within 2e-3 of pi) and the `lowkn`
  designs (low Kn, flow unchoked most of the burn), all with default tolerances. A pass over the 256 designs
  took 57 to 63 s even with 12 processes, because it cannot finish before its slowest lane.
* **W1 typical**: the 244 designs that need at most 1,000 accepted steps in the reference run.

The CPU baseline is `cpu-reference` (the unchanged `BurnSimulation`) in a process pool, heavy designs first, on the
workload tiled to 1,024 lanes, so that the cores stay busy and the figure is a steady-state throughput. A single
pass over the 256 designs gives 4.1 lanes/s with 6 processes and 4.5 with 12, limited by that one slow lane
(`w1_cpu_reference_single_pass.json`).

## Results

Throughput in lanes per second, best of two warm runs; the first call of a shape also compiles.

| Workload | Backend | Lanes | Lanes/s | vs CPU, 12 processes | vs CPU, 6 processes |
|---|---|---|---|---|---|
| W1 | CPU reference, 12 processes | 1,024 | 6.59 | 1x | 1.4x |
| W1 | CPU reference, 6 processes | 1,024 | 4.79 | 0.7x | 1x |
| W1 | JAX, GPU, one uncapped launch | 4,096 | 35.7 | 5.4x | 7.5x |
| W1 | JAX, GPU, one uncapped launch | 8,192 | 37.9 | 5.8x | 7.9x |
| W1 | **JAX, GPU, tiers** | 2,048 | 45.9 | 7.0x | 9.6x |
| W1 | **JAX, GPU, tiers** | 4,096 | **77.7** | **11.8x** | 16.2x |
| W1 | **JAX, GPU, tiers** | 8,192 | 115.8 | 17.6x | 24.2x |
| W1 | **JAX, GPU, tiers** | 16,384 | 182.9 | 27.8x | 38.2x |
| W1 typical | CPU reference, 12 processes | 1,024 | 11.49 | 1x | 1.3x |
| W1 typical | CPU reference, 6 processes | 1,024 | 8.76 | 0.8x | 1x |
| W1 typical | **JAX, GPU, tiers** | 4,096 | **628** | **54.7x** | 71.7x |
| W1 typical | **JAX, GPU, tiers** | 16,384 | 798 | 69.4x | 91.1x |
| W1 | `cpu-vectorized` (NumPy), tiers | 256 | 0.94 | 0.14x | 0.20x |
| W1 | `cpu-vectorized` (NumPy), chunks of 64, no tiers | 249 | 0.73 | 0.11x | 0.15x |

First-call times (compilation included): 45 to 255 s depending on the shape; later calls of the same shape do not
recompile. Peak device memory in the `metrics` history policy: 190 to 420 MB. The host remainder (building the
result mappings, the provider hash, the git SHA) was 0.3 s at 2,048 lanes and 1.7 to 2.0 s at 16,384, about 1 to
10 % of the run (the larger share is on the typical workload, where the device part is short).

`cpu-vectorized` is not an accelerator: in lockstep each launch pays for its slowest lane and NumPy adds a fixed cost
per iteration, so it is about as fast as one core of the scalar solver on this mix (the 256 designs take about 330 s
on one core of the reference, 273 s here). It stays useful as the oracle
for the kernel logic and as the portable backend where JAX cannot be imported.

## What the numbers say

1. **Where the time goes.** A lockstep launch costs *the iterations of its slowest lane x the lanes in the
   launch*, because finished lanes are masked but still computed. With one launch (rows "uncapped") the 23,307-step
   slowest lane set the pace for every lane and the throughput saturated at about 38 lanes/s beyond 4,096 lanes.
2. **Tiers fix most of it.** Every lane first runs with a cap of 2,048 loop iterations; the lanes not finished are
   rerun from the start in a smaller batch with a cap of 16,384, then without a cap. A finished lane is identical
   to an uncapped run because lanes are independent (tested on NumPy: identical metrics and status for every lane). This took 4,096 lanes from
   35.7 to 77.7 lanes/s and removed the saturation: the time grows slowly with the batch (52.7 s for 4,096 lanes,
   89.6 s for 16,384) because the heavy tail is paid once, by few lanes, at the low per-iteration cost of a small
   batch.
3. **What is left is latency.** After the first tier 9 of the 256 designs are still running, after the second 2.
   The last tier runs those few lanes for tens of thousands of iterations at the per-iteration cost of a tiny batch
   (an uncapped 256-lane launch, which pays the same tail, took 22.9 s). A GPU cannot go below that; those lanes
   are better placed on spare CPU cores while the GPU runs the bulk (the heterogeneous executor of the
   architecture document, section 6, which is not implemented yet).
4. **The corpus is harsher than the documented spike.** The spike measured 45 to 49x against 6 cores on a corpus
   whose slowest lane needed 3,195 iterations; this one needs tens of thousands. On the typical workload, where
   the tail is removed, the figures are 55 to 69x.

## Parity at the time of measurement

* NumPy backend against the stored reference on 160 completed lanes of the golden corpus, relative differences
  (maximum, median): final time 1.4e-6 (1e-9); grain burnout times 1.5e-6 (3e-10); peak pressure 1.6e-4 (3e-9);
  peak thrust 1.8e-4 (3e-9); maximum generated mass flow 2.0e-3 (3e-9); maximum nozzle mass flow 1.6e-4 (3e-9);
  integrals within 1e-5. The limits are in `solidpy/backends/_tolerances.py` (version 2).
* Every lane ends with the same termination reason as the reference, except the three `solver-failure` designs:
  their reference outcome is decided by rounding (identical grains burn out together and scipy snaps only one of
  them), and the batched solver completes them (`tests/test_batch_solver.py`).
* JAX on the CPU device and on the GPU match the reference with the same limits, agree with the NumPy backend, and
  `evaluate_numerical_acceptance(batched, reference)` is `passed` with an identical `physics_provider_hash`.

## Not covered yet

Igniter and activation sources, tabulated burn rate and thermochemistry (those lanes fall back to the reference),
the `uniform:N` and `decimated:N` history policies, the CPU+GPU executor, multi-GPU, the Tier 1 physics
(thermal, structural, robustness as lanes) and a data-center GPU. All numbers are for one machine; they say nothing
about other devices.

## Reproducing

```
# CPU baseline (scalar reference, process pool, workload tiled to 1,024 lanes)
python benchmarks/bench_burn.py --backend cpu-reference --sizes 1024 --workers 12,6 --out benchmarks/results/w1_cpu_reference.json
# GPU, full workload; add --tiers none for one uncapped launch, --max-points 1000 for the typical workload
python benchmarks/bench_burn.py --backend jax --device cuda:0 --sizes 2048,4096,8192,16384 --out benchmarks/results/w1_jax_cuda0.json
```

JAX comes from `pip install "solidpy[jax-cuda12]"`. The CPU measurements need no extra package. The
`w1_jax_cuda0_untiered.json` file was written from the console log of an earlier run (its 16,384-lane point was
not completed); the others are written by the script.
