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
(`benchmarks/results/offloaded_share.json`):

| Workload | Designs | Scalar time | Burn (batched) | Thermal ablation | Other advanced + post-processing | **Offloaded share** |
|---|---|---|---|---|---|---|
| W1 burn only (corpus mix) | 12 | 3.6 s | 99.0 % | - | - | **0.990** |
| W2 burn + detailed ballistics + advanced physics (four-grain variants) | 6 | 5.3 s | 75.7 % | 23.1 % | 1.2 % | **0.757** |
| W3 robustness (nominal + 10 default + 4 Latin-hypercube scenarios) | 2 | 19.8 s | 99.3 % | - | 0.6 % | **0.993** |

The gate of the architecture document is 0.8 on W1 to W3. W1 and W3 are above it with the burn alone, W3 through
`run_robustness_ensemble`. **W2 is below it (0.757) and cannot pass without the thermal ablation**, which is 23 % of the
workload and 95 % of the advanced physics (one scipy Radau solve per time step); structural, CFD and ignition proxies and
the 1-D flight together are under 1 %. The thermal ablation is therefore the next piece of Tier 1 work (phase 4b); this
round stops before it.

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

## Not covered yet

The `uniform:N` and `decimated:N` history policies, the CPU+GPU executor, multi-GPU, the Tier 1 physics
(thermal, structural, robustness as lanes) and a data-center GPU. All numbers are for one machine; they say nothing
about other devices.

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

JAX comes from `pip install "solidpy[jax-cuda12]"`. The CPU measurements need no extra package. The
`w1_jax_cuda0_untiered.json` file was written from the console log of an earlier run (its 16,384-lane point was
not completed); the others are written by the script.
