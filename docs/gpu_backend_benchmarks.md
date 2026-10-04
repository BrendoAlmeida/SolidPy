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

The table records the original Phase 3 gate environment. The updated W2-W4 runs below used the project `.venv` with
Python 3.12.13, NumPy 2.4.4, SciPy 1.17.1 and JAX 0.11.2; each raw JSON file records its exact environment metadata.

## Workloads

Both are the 256 golden-corpus designs that the batched kernels supported when the gate was measured (no igniter,
activation, tabulated burn rate or thermochemistry; those kernels were added afterwards and are not in these
numbers), tiled to the batch size. The tiling repeats the same designs, so the mix is fixed.

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
   architecture document, section 6, added in Phase 5).
4. **The corpus is harsher than the documented spike.** The spike measured 45 to 49x against 6 cores on a corpus
   whose slowest lane needed 3,195 iterations; this one needs tens of thousands. On the typical workload, where
   the tail is removed, the figures are 55 to 69x.

## Parity at the time of measurement

* NumPy backend against the stored reference on 160 completed lanes of the golden corpus, relative differences
  (maximum, median): final time 1.4e-6 (1e-9); grain burnout times 1.5e-6 (3e-10); peak pressure 1.6e-4 (3e-9);
  peak thrust 1.8e-4 (3e-9); maximum generated mass flow 2.0e-3 (3e-9); maximum nozzle mass flow 1.6e-4 (3e-9);
  integrals within 1e-5. The limits were those of `solidpy/backends/_tolerances.py` version 2; the table and source
  lanes added later needed version 3 (generated mass-flow peak 1.5e-2, the stored reference being the inexact side).
* Every lane ends with the same termination reason as the reference, except the three `solver-failure` designs:
  their reference outcome is decided by rounding (identical grains burn out together and scipy snaps only one of
  them), and the batched solver completes them (`tests/test_batch_solver.py`).
* JAX on the CPU device and on the GPU match the reference with the same limits, agree with the NumPy backend, and
  `evaluate_numerical_acceptance(batched, reference)` is `passed` with an identical `physics_provider_hash`.

## Whole-corpus parity on the GPU (2026-10-03, after tables and sources)

`pytest tests/test_batch_parity.py --runslow -k jax` on the RTX 4060 (JAX 0.11.2, float64): all 323 corpus designs
the JAX backend supports, one `simulate_burn` call with `strict=True`, 267 s including compilation. Tiers: 326 lanes
launched with the 2,048-iteration cap, 9 rerun with the 16,384 cap, 2 uncapped. Every lane ends with the stored
termination reason and the same completion flags, and the grain burnout set matches. Relative differences against
the stored reference (maximum, median):

| Quantity | Max | Median | Limit |
|---|---|---|---|
| total impulse | 2.5e-06 | 7e-10 | 1e-5 |
| generated / nozzle mass integral | 1.4e-06 | 4e-10 / 5e-10 | 1e-5 |
| igniter mass injected | 4.6e-15 | 5e-16 | 1e-5 |
| peak chamber pressure, peak thrust, max nozzle flow | 3.8e-04, 4.4e-04, 3.8e-04 | 3e-09, 4e-09, 3e-09 | 5e-3 |
| gas mass at cutoff | 1.3e-04 | 4e-10 | 5e-3 |
| max generated mass flow | 7.9e-03 | 3e-09 | 1.5e-2 |
| grain burnout times | 6.3e-07 | 4e-10 | 5e-6 |
| nozzle flow end | 1.8e-06 | 5e-10 | 5e-6 |

The largest differences sit in the stored reference, not in the kernels: the reference's own error against a refined
run is of the same size (comments in `solidpy/backends/_tolerances.py`, version 3). The NumPy backend passed the same
test on 2026-10-03 (892 s).

## Offloaded share of the scalar reference (Phase 4)

`python tools/profile_workloads.py` times the entry points listed in `tools/tier_map.toml` on the scalar path of three
workloads (wall clock around the outermost call, so Python-heavy code is not inflated the way a profiler would). The
offloaded share is the time in entry points that have a batched implementation over the whole workload
(`docs/gpu_backend_architecture.md`, section 7.1). The latest profile was measured 2026-10-03 on the machine above after
the detailed-ballistics service was added (`benchmarks/results/offloaded_share_post_detail_batch.json`). The previous
profiles are preserved as `benchmarks/results/offloaded_share_post_proxy.json` and
`benchmarks/results/offloaded_share.json`; before thermal batching, W2 read 0.757:

| Workload | Designs | Scalar time | Burn (batched) | Thermal (batched) | Detailed ballistics (batched) | Structural/CFD/ignition proxies (batched) | CPU and other | **Offloaded share** |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| W1 burn only (corpus mix) | 12 | 3.7 s | 98.9 % | - | - | - | 1.1 % | **0.989** |
| W2 burn + detailed ballistics + advanced physics (four-grain variants) | 6 | 5.5 s | 75.8 % | 23.1 % | - | 0.2 % | 0.9 % | **0.990** |
| W3 robustness (nominal + 10 default + 4 Latin-hypercube scenarios) | 2 | 20.6 s | 99.4 % | - | 0.6 % | - | 0.1 % | **0.999** |

The gate of the architecture document is 0.8 on W1 to W3, and all three pass. W1 and W3 pass with burn batching; W2
also batches thermal ablation and transient structural, CFD and ignition proxies; W2 detailed ballistics is still scalar.
Detailed ballistics, flight and other host work account for about 0.9 % of W2's scalar time. The measured share can vary
with host load; these numbers measure available batched coverage. Actual JAX/GPU measurements are recorded below.

## Robustness ensembles (W3) on the GPU (Phase 4a)

Measured 2026-10-03 with `benchmarks/bench_robustness.py` on the machine above (`benchmarks/results/w3_cpu_scalar.json`,
`benchmarks/results/w3_jax_cuda0.json`). A design is a variant of the tabulated four-grain test motor or of a two-grain
power-law motor, with throat and density varied; each is run through nominal + the 10 default scenarios + 16
Latin-hypercube samples, 27 lanes per design, with `max_step_size = 0.01` and 1,000 time points. The CPU baseline is
the unchanged scalar path (`run_robustness_analysis` without a backend), one design per task in a process pool, which is
the best static schedule the CPU has because designs are independent, and it includes the post-processing. The GPU path is
`run_robustness_ensemble(..., backend="jax", device="cuda:0", keep_series=False)`, float64, `max_steps = 2000`.

| Path | Lanes | Warm time | Lanes/s | vs CPU, 12 processes | vs CPU, 6 processes |
|---|---|---|---|---|---|
| CPU scalar, 12 processes | 972 | 93.3 s | 10.4 | 1x | 1.1x |
| CPU scalar, 6 processes | 972 | 102.5 s | 9.5 | 0.9x | 1x |
| JAX, GPU | 432 | 3.85 s | 112.2 | 10.8x | 11.8x |
| JAX, GPU | 1,728 | 12.6 s | 136.7 | 13.1x | 14.4x |
| JAX, GPU | **4,104** | 30.0 s | **136.8** | **13.1x** | 14.4x |
| JAX, GPU | 8,208 | 62.4 s | 131.5 | 12.6x | 13.9x |

First calls, which also compile the full-history program of each shape: 74, 90, 207 and 154 s. No lane left the batched
backend (`fallback_lanes = 0`) and every design completed.

Before the `detailed_ballistics` service was added, the 4,104-lane warm run spent 1.4 s packing, 16.7 s on device burns
(246 lanes/s on their own), 11.8 s building detailed histories on one CPU core (2.9 ms per lane, 39 % of the run), and
0.08 s assembling reports. Those historical numbers met the 5x gate but were capped near 137 lanes/s by the serial
post-processing loop. CPU and updated GPU reruns after batching are recorded below.

Two things to know when using it:

* A lane of a report with its series and canonical history is about 195 kB for a four-grain design. The first attempt
  kept them for 8,208 lanes and was killed by the host's out-of-memory killer (exit 137) on this 15 GB machine, which was
  shared with other work; `keep_series=False` keeps only the scalar outputs of each lane and does not change the report,
  its statistics or its validity ratio, and is what the table above used.
* The parity of the reports (scalar path against the NumPy backend, JAX on the CPU device and JAX on the GPU) is in
  `tests/test_batch_robustness.py` and `tests/test_backend_jax.py`, with the limits of tolerances version 4. The maximum
  generated mass flow and the maximum pressure rise rate are the loosest (3.5e-2 and 6e-2, at least 1.5 times the worst
  difference measured on eight designs) because the scalar path itself is up to 2.4e-2 and 4.6e-2 from a refined run on
  them (`solidpy/backends/_tolerances.py`).

## Thermal ablation on the GPU (Phase 4b)

Measured 2026-10-03 with `benchmarks/bench_thermal.py` on the machine above (`benchmarks/results/thermal_*.json`). A lane
is the wall conduction and throat ablation of one design (`Multiphysics.simulate_thermal_ablation`). The baseline is the scalar
model in a process pool, the best static schedule the CPU has because lanes are independent. The batched backends take
the same Radau IIA(5) steps as scipy (tolerances version 5: the worst difference to the scalar model is 1.4e-12 on 1,814
lanes), so the comparison is of equal work. Lane sets (`--kind`): `typical`, 13 of the 15 cases of `tests/thermal_cases.py` (without the two degenerate curves; 4 to
11 wall cells, 50 to 200 time steps) tiled with the gas and start temperature varied; `wide`, random designs (4 to 18 cells, 30
to 400 steps); `advanced`, the whole advanced physics of designs with real burn curves (344 points) through
`run_advanced_physics_ensemble`, against `simulate_advanced_physics` in a pool. The `advanced` rows below predate the
`advanced_physics_proxies` service; updated whole-W2 CPU and GPU results after adding those proxies are recorded later in
this document.

| Lane set | Path | Lanes | Warm time | Lanes/s | vs CPU, 12 processes |
|---|---|---|---|---|---|
| typical | CPU scalar, 12 processes | 1,024 | 12.1 s | 84.6 | 1x |
| typical | CPU scalar, 6 processes | 1,024 | 13.2 s | 77.5 | 0.9x |
| typical | NumPy, 1 thread | 4,096 | 14.5 s | 282 | 3.3x |
| typical | JAX, GPU | 1,024 | 0.71 s | 1,441 | 17x |
| typical | JAX, GPU | 4,096 | 1.34 s | 3,053 | **36x** |
| typical | JAX, GPU | 16,384 | 5.18 s | 3,163 | **37x** |
| wide | CPU scalar, 12 processes | 1,024 | 21.8 s | 47.0 | 1x |
| wide | NumPy, 1 thread | 1,024 | 12.4 s | 82.3 | 1.8x |
| wide | JAX, GPU | 1,024 | 2.46 s | 416 | 8.8x |
| wide | JAX, GPU | 4,096 | 5.56 s | 737 | **15.7x** |
| advanced | CPU scalar, 12 processes | 1,024 | 35.1 s | 29.1 | 1x |
| advanced | NumPy, 1 thread | 1,024 | 7.56 s | 135 | 4.6x |
| advanced | JAX, GPU | 1,024 | 6.41 s | 160 | 5.5x |
| advanced | JAX, GPU | 4,096 | 24.2 s | 169 | **5.8x** |

First calls, which also compile the program of each shape: 4.0 to 8.9 s (typical), 5.9 and 9.0 s (wide), 10.3 and 27.5 s
(advanced, which includes the first CPU models). No lane left the batched backend (`failed_lanes = 0`).

What the numbers say:

* The thermal ablation alone is where the accelerator pays: 36x the scalar model on all 12 threads at 4,096 lanes of the
  typical walls, 15.7x on walls of up to 18 cells (the wall cells are a sequential recurrence, so the cost grows with
  them). One NumPy thread already beats 12 scalar processes by 3.3x, because the scalar call spends most of its time in
  scipy's per-call overhead, not in arithmetic.
* In the pre-proxy serial baseline, the whole advanced physics was limited by what stayed on the CPU. At 4,096 lanes the thermal batch
  takes 1.0 s and packing it 1.4 s, while the structural, CFD, ignition and flight models of the 4,096 lanes take
  **21.8 s**, 5.3 ms per lane on one core (90 % of the run). The structural, CFD and ignition share now runs through the
  batched proxy service; flight remains on the CPU. The updated CPU-only W2 measurement is below. Packing costs 0.3 to
  0.4 ms per lane on the host.
* Below about 2,000 lanes the latency of the sequential loops dominates (1,441 lanes/s at 1,024); from 4,096 lanes the
  device is saturated and the time grows with the lanes (16,384 lanes take 3.9x the time of 4,096). The device runs float64,
  which a consumer GPU does at 1/64 of its float32 rate.

## Not covered yet

Replacing finished lanes inside an active accelerator launch remains unimplemented. Detailed-ballistics, general
transient structural, CFD and ignition proxy kernels are implemented; W4's synthetic peak-pressure path is a separate
kernel. Completion rates are reported from real chunks, so a separate throughput-calibration solve is not used. The
heterogeneous executor and bounded W2/W3 post-processing pipeline are implemented. Single-GPU parity and updated W2,
W3 and W4 throughput have now been measured on the RTX 4060 below; concurrent real multi-GPU execution remains pending
because this host has one GPU. The intra-launch refill follow-up remains conditional on evidence of unused capacity.
All measured numbers are for one machine; they say nothing about other devices.

## CPU post-processing workers (W2 and W3)

`run_advanced_physics_ensemble` and `run_robustness_ensemble` accept `workers > 1` for scalar CPU fallback work after the
batched solve. The rows below predate the `advanced_physics_proxies` and `detailed_ballistics` services. Each row is one
timed warm CPU-vectorized call on the Ryzen 5 3600 host, after an initial call; process startup is included. No GPU or
JAX was available. W2 uses 256 or 1,024 curves. W3 uses 8 designs and 216 lanes, with 16 Latin-hypercube samples per
design and `keep_series=False`.

| Workload | Workers | Total (s) | Batched solve (s) | CPU post-processing (s) |
|---|---:|---:|---:|---:|
| W2 advanced physics, 256 lanes | 1 | 2.238 | 0.816 thermal | 1.333 models |
| W2 advanced physics, 256 lanes | 6 | 5.294 | 0.811 thermal | 4.390 models |
| W2 advanced physics, 1,024 lanes | 1 | 7.591 | 1.852 thermal | 5.367 models |
| W2 advanced physics, 1,024 lanes | 6 | 17.460 | 1.867 thermal | 15.222 models |
| W3 robustness, 216 lanes | 1 | 11.194 | 10.564 burn | 0.557 detailed ballistics |
| W3 robustness, 216 lanes | 6 | 12.358 | 10.740 burn | 1.542 detailed ballistics |

Six workers were slower on all three tested shapes: the W2 model stage took 3.29x as long at 256 lanes and 2.84x as long
at 1,024 lanes; the W3 detailed-ballistics stage took 2.77x as long. Pool startup and job serialization outweigh the
parallel work on this host, so the pool is opt-in and serial is the default. The rows above were measured before the
producer-consumer pipeline was added and do not include its overlap. A CPU-only post-pipeline sample follows; updated
GPU end-to-end W2/W3 throughput is recorded below. The benchmark CLIs expose `--chunk-size` for W2 and `--chunk-lanes` for W3,
and their JSON records whether multiple solve chunks overlapped process work. Raw pre-pipeline benchmark output is in
`benchmarks/results/postprocess_w2_numpy_workers{1,6}.json`,
`benchmarks/results/postprocess_w2_numpy_1024_workers{1,6}.json` and
`benchmarks/results/postprocess_w3_numpy_workers{1,6}.json`.

Reproduce the measured shapes with:

```
python benchmarks/bench_thermal.py --backend cpu-vectorized --kind advanced --workers 6 --lanes 1024 --chunk-size 256 --repeat 1
python benchmarks/bench_robustness.py --backend cpu-vectorized --workers 6 --designs 8 --chunk-lanes 54 --repeat 1
```

## Producer-consumer pipeline sample (CPU-only)

Measured on the current host with `cpu-vectorized`, six post-processing workers, and one warm repeat after the first call.
The chunks force more than one solve so the pipeline can overlap stages. These numbers validate the benchmark path and
record the CPU schedule; they do not estimate GPU throughput. `models_s` is the sum of worker CPU time and is not wall
time, while `thermal_s` and `solve_s` are producer wall time accumulated across chunks.

| Workload | Lanes / solve chunks | Total wall (s) | Solve wall (s) | Post-processing CPU (s) | Lanes/s | Overlap |
|---|---:|---:|---:|---:|---:|---|
| W2 advanced, chunk size 256 | 1,024 / 4 | 19.52 | 3.32 thermal | 5.81 models | 52.5 | yes |
| W3 robustness, chunk lanes 54 | 216 / 4 | 32.39 | 31.65 burn | 0.66 ballistics | 6.7 | yes |

Machine-readable results: `benchmarks/results/postpipeline_w2_cpu_vectorized_1024_workers6_chunk256.json` and
`benchmarks/results/postpipeline_w3_cpu_vectorized_8_workers6_chunk54.json`. The CPU-vectorized W3 burn dominates this
host run; the process pipeline does not turn that backend into an accelerator.

## W2 after batching transient proxies (CPU-only)

After `advanced_physics_proxies` was added, the same advanced workload was rerun at 1,024 lanes on the Ryzen 5 3600.
The serial run used NumPy batches for the transient structural, CFD and ignition proxies; the six-worker run used four
thermal chunks and the producer-consumer pipeline. Both rows have one warm repeat after the first call. The `models_s`
column is worker CPU time; it is wall time in the serial row and accumulated worker time in the pipeline row.

| Workers | Solve chunks | Total wall (s) | Thermal solve (s) | Proxy batch (s) | CPU models (s) | Lanes/s | Overlap |
|---:|---:|---:|---:|---:|---:|---:|---|
| 1 | 1 | 6.402 | 1.884 | 0.113 | 4.032 | 159.9 | no |
| 6 | 4 | 20.086 | 3.389 | 0.099 | 4.374 | 51.0 | yes |

All 1,024 lanes used the `cpu-vectorized` proxy service, with no scalar proxy fallbacks. The remaining CPU flight work and
spawned process startup dominate the total; six workers remain slower on this host. These are CPU measurements and do not
estimate GPU throughput. Raw results: `benchmarks/results/advanced_proxy_w2_cpu_vectorized_1024_workers1.json` and
`benchmarks/results/advanced_proxy_w2_cpu_vectorized_1024_workers6_chunk256.json`.

Reproduce with:

```
python benchmarks/bench_thermal.py --backend cpu-vectorized --kind advanced --workers 1 --lanes 1024 --repeat 1
python benchmarks/bench_thermal.py --backend cpu-vectorized --kind advanced --workers 6 --lanes 1024 --chunk-size 256 --repeat 1
```

## W2 on the RTX 4060 after batching transient proxies

Measured with one warm repeat on the Ryzen 5 3600 / RTX 4060 host, JAX 0.11.2, float64 and no scalar proxy fallbacks.
The NumPy and JAX runs used the same W2 advanced-physics workload, including thermal ablation, transient proxies and
CPU flight models:

| Lanes | NumPy warm (s) | JAX warm (s) | JAX lanes/s | JAX speedup |
|---:|---:|---:|---:|---:|
| 1,024 | 6.25 | 5.17 | 198.0 | 1.21x |
| 4,096 | 24.06 | 18.67 | 219.4 | 1.29x |

At 4,096 lanes, the JAX run spent 1.44 s packing, 1.01 s on thermal, 0.25 s on the batched proxies and 15.94 s on
CPU models/flight. The remaining host work limits the end-to-end gain. First calls including compilation were 9.38 s and
22.59 s. Raw results: `benchmarks/results/advanced_proxy_w2_cpu_vectorized_1024_4096_post_gpu.json` and
`benchmarks/results/advanced_proxy_w2_jax_cuda0_1024_4096.json`.

Reproduce with:

```
python benchmarks/bench_thermal.py --backend cpu-vectorized --kind advanced --lanes 1024,4096 --repeat 1 --out benchmarks/results/advanced_proxy_w2_cpu_vectorized_1024_4096_post_gpu.json
python benchmarks/bench_thermal.py --backend jax --device cuda:0 --kind advanced --lanes 1024,4096 --repeat 1 --out benchmarks/results/advanced_proxy_w2_jax_cuda0_1024_4096.json
```

## W2 process pipeline after compacting lane jobs

Measured on 2026-10-03 on the Ryzen 5 3600 / RTX 4060 with JAX 0.11.2. The workload used 4,096 advanced-physics lanes,
four thermal chunks of 1,024 lanes, and the overlapping thermal/post-processing pipeline. Before compaction, each
accelerated lane's process job carried the full ballistic curve, including unused histories. The bounded-worker path
now retains only `time_s`, `thrust_n`, optional `propellant_mass_kg` and `scenario_factors` for lanes whose transient
proxies were computed in a batch. Scalar fallback lanes retain the full curve.

| Workers | Previous wall time (s) | Compact wall time (s) | Compact lanes/s | Improvement |
|---:|---:|---:|---:|---:|
| 2 | 63.21 | 12.83 | 319.1 | 4.92x |
| 6 | 64.15 | 8.03 | 510.1 | 7.99x |

The compact runs processed all 4,096 proxy lanes in JAX and reported no scalar proxy lanes or fallback errors. Stage
`models_s` is cumulative worker time, whereas `seconds` is end-to-end wall time; worker overlap lets cumulative stage
time exceed wall time. Raw before/after results:
`benchmarks/results/advanced_proxy_w2_jax_cuda0_4096_workers2_chunk1024.json`,
`benchmarks/results/advanced_proxy_w2_jax_cuda0_4096_workers6_chunk1024.json`,
`benchmarks/results/advanced_proxy_w2_jax_cuda0_4096_compact_workers2_chunk1024.json` and
`benchmarks/results/advanced_proxy_w2_jax_cuda0_4096_compact_workers6_chunk1024.json`.

Reproduce the compact runs with:

```
XLA_PYTHON_CLIENT_PREALLOCATE=false python benchmarks/bench_thermal.py --backend jax --device cuda:0 --kind advanced --workers 2 --chunk-size 1024 --lanes 4096 --repeat 1 --out benchmarks/results/advanced_proxy_w2_jax_cuda0_4096_compact_workers2_chunk1024.json
XLA_PYTHON_CLIENT_PREALLOCATE=false python benchmarks/bench_thermal.py --backend jax --device cuda:0 --kind advanced --workers 6 --chunk-size 1024 --lanes 4096 --repeat 1 --out benchmarks/results/advanced_proxy_w2_jax_cuda0_4096_compact_workers6_chunk1024.json
```

## W3 detailed-ballistics service after batching (CPU-only)

Measured on 2026-10-03 on the same Ryzen 5 3600 host with eight designs and 216 lanes (27 per design), using
`cpu-vectorized`, one solve chunk, and `keep_series=False`. The warm call took 11.88 s (18.2 lanes/s): 0.069 s to pack,
11.422 s to solve burns, 0.363 s in the detailed-ballistics service, and 0.005 s to assemble reports. All 216 histories
used the batch service; there were no scalar fallbacks. This CPU run checks the updated stage accounting; updated GPU
throughput is recorded below. It is not directly comparable to the older six-worker, four-chunk pipeline sample above.

Raw result: `benchmarks/results/w3_cpu_vectorized_detail_batch.json`.

Reproduce with:

```
python benchmarks/bench_robustness.py --backend cpu-vectorized --designs 8 --workers 1 --repeat 1 --out benchmarks/results/w3_cpu_vectorized_detail_batch.json
```

## W3 on the RTX 4060 after batching detailed ballistics

Measured with one warm repeat, `keep_series=False`, 2,000 maximum history points and 4,096-lane burn chunks. All designs
completed, every lane used JAX for burn and detailed-ballistics services, and there were no scalar fallbacks:

| Designs | Lanes | First call (s) | Warm (s) | Lanes/s | Fallback lanes |
|---:|---:|---:|---:|---:|---:|
| 16 | 432 | 69.96 | 3.24 | 133.2 | 0 |
| 64 | 1,728 | 86.60 | 9.47 | 182.4 | 0 |
| 152 | 4,104 | 188.07 | 21.52 | 190.7 | 0 |
| 304 | 8,208 | 135.11 | 47.93 | 171.2 | 0 |

At 4,104 lanes, throughput increased from 136.8 to 190.7 lanes/s (1.39x) after detailed-ballistics batching; its service
took 3.20 s. First-call time includes compilation of the launch shapes. Raw result:
`benchmarks/results/w3_jax_cuda0_post_detail_batch.json`.

Reproduce with:

```
python benchmarks/bench_robustness.py --backend jax --device cuda:0 --designs 16,64,152,304 --repeat 1 --out benchmarks/results/w3_jax_cuda0_post_detail_batch.json
```

## Structural Monte Carlo (W4)

Measured on 2026-10-03 on the Ryzen 5 3600 host above, with one design and 100,000 peak-pressure samples. The numbers
include host sampling and assembly of the full `StructuralMonteCarlo` report. These CPU baselines were collected before
the JAX/RTX 4060 run below:

| Path | Samples | First call | Samples/s | Fallback lanes |
|---|---:|---:|---:|---:|
| CPU scalar reference | 100,000 | 14.43 s | 6,932 | 0 |
| CPU vectorized | 100,000 | 5.16 s | 19,398 | 0 |

The NumPy path is 2.8x faster in this single-process measurement. It does not establish a GPU speedup or compare against
the scalar model spread over all CPU cores. The one-design raw results are in
`benchmarks/results/w4_cpu_reference_100k.json` and `benchmarks/results/w4_cpu_vectorized_100k.json`.

The plan-shaped CPU baseline was also measured with four designs, each running 100,000 samples. The benchmark processes
designs sequentially and includes report assembly; throughput below uses total elapsed warm time across all four:

| Path | Total samples | Warm elapsed (s) | Aggregate samples/s | Fallback lanes |
|---|---:|---:|---:|---:|
| CPU scalar reference | 400,000 | 53.56 | 7,468 | 0 |
| CPU vectorized | 400,000 | 17.98 | 22,248 | 0 |

The vectorized CPU run is 2.98x faster for this workload. Raw results are in
`benchmarks/results/w4_cpu_reference_100k_4designs.json` and
`benchmarks/results/w4_cpu_vectorized_100k_4designs.json`.

The RTX 4060 ran the same four designs, each with 100,000 samples. Warm time includes host sampling and full report
assembly; all samples were evaluated, with no fallback:

| Path | Total samples | Warm elapsed (s) | Aggregate samples/s | Relative to JAX |
|---|---:|---:|---:|---:|
| CPU scalar reference | 400,000 | 53.56 | 7,468 | 0.33x |
| CPU vectorized | 400,000 | 17.98 | 22,248 | 0.97x |
| JAX, RTX 4060 | 400,000 | 17.48 | 22,886 | 1.00x |

The end-to-end JAX result is 1.03x the NumPy throughput. Host sampling and report assembly dominate enough that this
measurement does not show a material GPU advantage over vectorized CPU; it does not isolate the structural kernel.
This result predates the host-side result-validation optimization below. Raw result:
`benchmarks/results/w4_jax_cuda0_100k_4designs.json`.

Reproduce with:

```
python benchmarks/bench_structural_monte_carlo.py --backend jax --device cuda:0 --iterations 100000 --designs 4 --repeat 1 --out benchmarks/results/w4_jax_cuda0_100k_4designs.json
```

### W4 after vectorizing numeric result validation

Profiled on 2026-10-04 on the Ryzen 5 3600 / RTX 4060 host. The previous implementation checked Python types and
finiteness metric by metric for every sample. Standard numeric arrays now validate finiteness in a NumPy array pass;
unusual output dtypes retain per-lane conversion and fallback handling. In a warm cProfile run for 100,000 samples,
the instrumented call time fell from 13.10 s to 1.89 s. The benchmark below measures unprofiled calls.

Four designs each ran 100,000 samples, with three warm repetitions. Aggregate time is the sum of each design's median
warm time; all 400,000 samples were evaluated without fallback:

| Path | Total samples | Aggregate median warm (s) | Samples/s | Relative to CPU vectorized |
|---|---:|---:|---:|---:|
| CPU vectorized | 400,000 | 3.91 | 102,207 | 1.00x |
| JAX, RTX 4060 | 400,000 | 4.08 | 98,098 | 0.96x |

The CPU and GPU paths both improved substantially from the earlier 17.98 s and 17.48 s single-warm measurements because
both use the shared host-side validation. The W4 end-to-end path is now near parity, with NumPy slightly faster on this
machine; the GPU kernel itself is not the host-side bottleneck. Raw results:
`benchmarks/results/w4_cpu_vectorized_100k_4designs_post_validation.json` and
`benchmarks/results/w4_jax_cuda0_100k_4designs_post_validation.json`.

Reproduce with:

```
python benchmarks/bench_structural_monte_carlo.py --backend cpu-vectorized --iterations 100000 --designs 4 --repeat 3 --out benchmarks/results/w4_cpu_vectorized_100k_4designs_post_validation.json
XLA_PYTHON_CLIENT_PREALLOCATE=false python benchmarks/bench_structural_monte_carlo.py --backend jax --device cuda:0 --iterations 100000 --designs 4 --repeat 3 --out benchmarks/results/w4_jax_cuda0_100k_4designs_post_validation.json
```

### W4 scale check at one million samples per design

Measured on 2026-10-04 on the same Ryzen 5 3600 / RTX 4060 host. Four designs each ran one million samples, with
two warm repetitions per design. Aggregate time is the sum of each design's median warm time; the medians were
calculated from the recorded `warm_times_s` arrays. All four million samples completed without fallback on both paths.

| Path | Total samples | Aggregate median warm (s) | Samples/s | Relative to CPU vectorized |
|---|---:|---:|---:|---:|
| CPU vectorized | 4,000,000 | 38.36 | 104,275 | 1.00x |
| JAX, RTX 4060 | 4,000,000 | 39.85 | 100,380 | 0.96x |

Increasing the batch from 100,000 to one million samples per design does not produce an end-to-end GPU throughput
advantage. These timings include host sampling, backend dispatch, device transfers and construction of the complete
Python report; they do not isolate kernel time. A stage-level profile is needed before attributing the remaining gap
to transfer, compute or report assembly. Raw results:
`benchmarks/results/w4_cpu_vectorized_1m_4designs.json` and
`benchmarks/results/w4_jax_cuda0_1m_4designs.json`.

Reproduce with:

```
python benchmarks/bench_structural_monte_carlo.py --backend cpu-vectorized --iterations 1000000 --designs 4 --repeat 2 --out benchmarks/results/w4_cpu_vectorized_1m_4designs.json
XLA_PYTHON_CLIENT_PREALLOCATE=false python benchmarks/bench_structural_monte_carlo.py --backend jax --device cuda:0 --iterations 1000000 --designs 4 --repeat 2 --out benchmarks/results/w4_jax_cuda0_1m_4designs.json
```

### W4 host and backend stage profile

Profiled on 2026-10-04 on the Ryzen 5 3600 / RTX 4060 host with one design and 100,000 samples, after one warm-up
call and three measured repetitions. Table values come from the repeat closest to the median total `run()` time, so
the stage rows add to the total. The JSON also records each repeat and component-wise medians. The stages are parameter
sampling, the structural-batch host remainder, the backend service call, and the remaining report/provenance work.
The batch host remainder is the structural-batch time minus the service call. The service call includes array
conversion, dispatch, transfers, synchronization and result conversion; the JAX device region does not split transfer
time from device execution.

| Stage | CPU vectorized | JAX, RTX 4060 |
|---|---:|---:|
| Total `run()` | 0.989 s | 1.020 s |
| Parameter sampling | 0.073 s (7.4%) | 0.074 s (7.2%) |
| Structural-batch host remainder | 0.622 s (62.9%) | 0.589 s (57.8%) |
| Backend service call | 0.003 s (0.3%) | 0.017 s (1.7%) |
| Post-batch run remainder | 0.290 s (29.4%) | 0.340 s (33.4%) |

Both backends evaluated all 100,000 samples without fallback. More than 98% of the GPU end-to-end time is outside the
backend service call, so this W4 path is limited mainly by host sampling, lane preparation/validation and report
construction. The profile does not isolate the JAX kernel from transfers and conversion, and it does not show which
host operation dominates within those remainders. Process peak RSS, including warm-up, was 370 MiB for CPU and
1,084 MiB for JAX; this is host memory, not GPU VRAM. Raw results:
`benchmarks/results/w4_profile_cpu_vectorized_100k.json` and
`benchmarks/results/w4_profile_jax_cuda0_100k.json`.

Reproduce with:

```
python benchmarks/profile_structural_monte_carlo.py --backend cpu-vectorized --iterations 100000 --designs 1 --repeat 3 --out benchmarks/results/w4_profile_cpu_vectorized_100k.json
XLA_PYTHON_CLIENT_PREALLOCATE=false python benchmarks/profile_structural_monte_carlo.py --backend jax --device cuda:0 --iterations 100000 --designs 1 --repeat 3 --out benchmarks/results/w4_profile_jax_cuda0_100k.json
```

## Flight dispersion (W5; CPU reference, reported and non-gating)

Measured on 2026-10-04 with `benchmarks/bench_flight_dispersion.py`. The benchmark exercises the public
`DispersionAnalysis.run` and `simulate_flight_3dof` APIs with 16 sampled flights. It builds a four-grain KNSB
burn using `BurnSimulation`, derives a complete `MotorGeometry` with `geometry_from_components`, and passes
the burn's time, thrust and grain-regression histories to each flight. Remaining propellant mass is calculated
from those regression histories. Samples vary launch angle, azimuth, wind speed and direction, and drag factor.
Impact points come from the flight reports' `landing_*` fields; the separate `trajectory` arrays cover powered
flight and are checked for consistent shapes and finite values.

The run used 12 CPU process workers and returned 16 finite flights. Campaign timing includes process-pool
startup, flight simulation and report assembly; preparation of the shared burn took 0.350716752 s separately.
The fixed seed reproduced identical impact coordinates across campaigns:

| Measure | Time / throughput |
|---|---:|
| First call | 2.02 s |
| Repeat 1 | 2.11 s |
| Repeat 2 | 2.34 s |
| Median repeat | 2.22 s |
| Throughput at median repeat | 7.20 flights/s |

W5 remains a T2 CPU workload and is reported, not gating. This CPU reference measurement does not implement or
measure GPU flight execution and does not demonstrate a GPU speedup. The raw result, including machine metadata,
sampled inputs and impact coordinates, is in
`benchmarks/results/w5_cpu_reference_flight_dispersion.json`.

Reproduce with:

```
python benchmarks/bench_flight_dispersion.py --samples 16 --repeat 2 --seed 20261004 --out benchmarks/results/w5_cpu_reference_flight_dispersion.json
```

## Reproducing

```
# CPU baseline (scalar reference, process pool, workload tiled to 1,024 lanes)
python benchmarks/bench_burn.py --backend cpu-reference --sizes 1024 --workers 12,6 --out benchmarks/results/w1_cpu_reference.json
# GPU, full workload; add --tiers none for one uncapped launch, --max-points 1000 for the typical workload
python benchmarks/bench_burn.py --backend jax --device cuda:0 --sizes 2048,4096,8192,16384 --out benchmarks/results/w1_jax_cuda0.json
```

Robustness ensembles (W3):

```
python benchmarks/bench_robustness.py --backend cpu-scalar --workers 12,6 --designs 36 --out benchmarks/results/w3_cpu_scalar.json
python benchmarks/bench_robustness.py --backend jax --device cuda:0 --designs 16,64,152,304 --repeat 1 --out benchmarks/results/w3_jax_cuda0_post_detail_batch.json
python tools/profile_workloads.py --out benchmarks/results/offloaded_share.json
```

Structural Monte Carlo (W4):

```
python benchmarks/bench_structural_monte_carlo.py --backend cpu-reference --iterations 100000 --designs 4 --out benchmarks/results/w4_cpu_reference.json
python benchmarks/bench_structural_monte_carlo.py --backend jax --device cuda:0 --iterations 100000 --designs 4 --repeat 1 --out benchmarks/results/w4_jax_cuda0_100k_4designs.json
```

Thermal ablation (phase 4b):

```
python benchmarks/bench_thermal.py --backend cpu-scalar --kind typical --workers 12,6 --lanes 1024 --out benchmarks/results/thermal_typical_cpu_scalar.json
python benchmarks/bench_thermal.py --backend jax --device cuda:0 --kind typical --lanes 1024,4096,16384 --out benchmarks/results/thermal_typical_jax_cuda0.json
python benchmarks/bench_thermal.py --backend jax --device cuda:0 --kind advanced --lanes 1024,4096 --out benchmarks/results/thermal_advanced_jax_cuda0.json
```

JAX comes from `pip install "solidpy[jax-cuda12]"`. The CPU measurements need no extra package. The
`w1_jax_cuda0_untiered.json` file was written from the console log of an earlier run (its 16,384-lane point was
not completed); the others are written by the script.
