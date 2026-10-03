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
   architecture document, section 6, which is not implemented yet).
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
(`docs/gpu_backend_architecture.md`, section 7.1). Measured 2026-10-03 on the machine above
(`benchmarks/results/offloaded_share.json`, after phase 4b; before it the thermal ablation was not marked batched and W2
read 0.757):

| Workload | Designs | Scalar time | Burn (batched) | Thermal ablation (batched) | Other advanced + post-processing | **Offloaded share** |
|---|---|---|---|---|---|---|
| W1 burn only (corpus mix) | 12 | 3.5 s | 99.0 % | - | - | **0.990** |
| W2 burn + detailed ballistics + advanced physics (four-grain variants) | 6 | 5.2 s | 74.5 % | 24.3 % | 1.2 % | **0.988** |
| W3 robustness (nominal + 10 default + 4 Latin-hypercube scenarios) | 2 | 19.0 s | 99.4 % | - | 0.6 % | **0.994** |

The gate of the architecture document is 0.8 on W1 to W3, and all three pass. W1 and W3 pass with the burn alone (W3
through `run_robustness_ensemble`); W2 needed the thermal ablation, which was 24 % of the workload and 95 % of the advanced
physics (one scipy Radau solve per time step), and is now batched (`simulate_thermal`, `run_advanced_physics_ensemble`).
Structural, CFD and ignition proxies and the 1-D flight together are under 1 %.

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

Where the warm time goes (seconds, 4,104 lanes): packing 1.4, the burns on the device 16.7 (246 lanes/s on their own),
the detailed ballistics of every lane on **one CPU core** 11.8 (2.9 ms per lane, 39 % of the run), report assembly 0.08.
So the gate of the architecture document (at least 5x the CPU on all cores at 4,096 lanes or more) is met with a margin,
and the next limit is not the device: building the detailed ballistics on the host in a loop caps the run at about
137 lanes/s, and overlapping it with the next launch or spreading it over the other cores (the pipeline of the
architecture document, section 6.4, not implemented) would lift the ceiling towards the 246 lanes/s of the device.

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
`run_advanced_physics_ensemble`, against `simulate_advanced_physics` in a pool.

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
* The whole advanced physics is limited by what stays on the CPU. At 4,096 lanes the thermal batch takes 1.0 s and
  packing it 1.4 s, while the structural, CFD, ignition and flight models of the 4,096 lanes take **21.8 s**, 5.3 ms per
  lane on one core (90 % of the run). That loop is serial in this process; spreading it over the other cores, or
  batching those models, is what would lift the 169 lanes/s (the same observation as for the detailed ballistics of the
  robustness ensembles). Packing costs 0.3 to 0.4 ms per lane on the host.
* Below about 2,000 lanes the latency of the sequential loops dominates (1,441 lanes/s at 1,024); from 4,096 lanes the
  device is saturated and the time grows with the lanes (16,384 lanes take 3.9x the time of 4,096). The device runs float64,
  which a consumer GPU does at 1/64 of its float32 rate.

## Not covered yet

The `uniform:N` and `decimated:N` history policies, the CPU+GPU executor, multi-GPU, general transient structural
responses (W4's synthetic peak-pressure path is implemented), spreading the CPU post-processing (detailed ballistics,
the proxies after the thermal ablation) over cores, and a data-center GPU. W4's 100,000-sample harness is
`benchmarks/bench_structural_monte_carlo.py`; GPU measurements remain pending. All measured numbers are for one machine;
they say nothing about other devices.

## Structural Monte Carlo (W4)

Measured on 2026-10-03 on the Ryzen 5 3600 host above, with one design and 100,000 peak-pressure samples. The numbers
include host sampling and assembly of the full `StructuralMonteCarlo` report. This host has no working NVIDIA driver or
JAX installation, so the comparison is the scalar reference and NumPy vectorization on one process:

| Path | Samples | First call | Samples/s | Fallback lanes |
|---|---:|---:|---:|---:|
| CPU scalar reference | 100,000 | 14.43 s | 6,932 | 0 |
| CPU vectorized | 100,000 | 5.16 s | 19,398 | 0 |

The NumPy path is 2.8x faster in this single-process measurement. It does not establish a GPU speedup or compare against
the scalar model spread over all CPU cores. Raw results are in `benchmarks/results/w4_cpu_reference_100k.json` and
`benchmarks/results/w4_cpu_vectorized_100k.json`; the GPU and four-design run remain to be measured on an available JAX
CUDA host.

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
python benchmarks/bench_robustness.py --backend jax --device cuda:0 --designs 16,64,152,304 --repeat 1 --out benchmarks/results/w3_jax_cuda0.json
python tools/profile_workloads.py --out benchmarks/results/offloaded_share.json
```

Structural Monte Carlo (W4):

```
python benchmarks/bench_structural_monte_carlo.py --backend cpu-reference --iterations 100000 --designs 4 --out benchmarks/results/w4_cpu_reference.json
python benchmarks/bench_structural_monte_carlo.py --backend jax --device cuda:0 --iterations 100000 --designs 4 --out benchmarks/results/w4_jax_cuda0.json
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
