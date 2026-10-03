### NEW: Q2 2023 Update
We have some interesting updates regarding the project.
* Amazing work by [@phmbressan](https://www.github.com/phmbressan) now allows SolidPy to have custom grain geometries (besides BATES), through the employment of fast marching numerical methods.
* A nice PyQT graphical interface is under development by [@caiogatinho](https://www.github.com/caiogatinho) inside the `GUI` branch. It can already reliably run sims with most of the features and will be merged to master soon. He also did some incredible work concerning structural analysis of bolted casings, aiding design by plotting safety factors for different parameters and failure modes.

These features will probably need a general code overhaul and review process to avoid bugs and conflicts, which should happen in the next few months (after which we'll probably have our first release!).

We also plan on adding combustion and nozzle efficiencies options to more closely match sims to data received from previous static fires. On the long term, tackling nozzle *erosion/slag* as well as modeling *erosive burning* is also something on our timeline.

---

# SolidPy

SolidPy is _Projeto Jupiter_ solid motor simulation code. This repository is under development and is a work in progress.

## Goals

This project aims to build an easy-to-use and versatile simulation tool for solid motor design and validation inside the project. It will follow an OOP design which allows for straightforward simulations of different combinations of propellants, motors and external conditions.

### Expected Outputs

- Total Impulse
- Specific Impulse
- Thrust (time)
- Pressure Chamber (time)
- Kn (time)

## Prerequisites

- Python 3
- NumPy >= 1.0
- SciPy >= 1.0
- Matplotlib >= 3.0

## Installation

The base install is pure Python and runs the reference (CPU) solver:

```
pip install .
```

Optional extras add accelerated batch backends and thermochemistry tools. They are independent, so install
only what you need:

| Extra | Installs | Use |
|---|---|---|
| `cea` | `rocketcea` | NASA CEA thermochemistry tables |
| `jax` | JAX (CPU or any device), Python 3.10+ | batched backend through JAX |
| `jax-cuda12` | JAX with CUDA 12 wheels, Python 3.10+ | NVIDIA GPU |
| `gpu` | alias of `jax-cuda12` | recommended GPU setup |
| `all` | `cea` and `jax-cuda12` | everything |

```
pip install ".[jax-cuda12]"      # or: uv sync --extra jax-cuda12
```

The JAX extras carry a `python_version >= '3.10'` marker: on Python 3.9 they install nothing, and the core
keeps working on the reference CPU path.

The reference CPU path is always the default and is never replaced. Accelerated backends are opt-in, with
`solidpy.set_backend(...)`, `with solidpy.use_backend(...)`, or the environment variables
`SOLIDPY_BACKEND` and `SOLIDPY_DEVICE` (the device applies to the backend that `SOLIDPY_BACKEND` names).
`solidpy.backends.available()` lists what the current environment can run and the install command for what
is missing. The accelerated backends are under development; the design is in
`docs/gpu_backend_architecture.md`.

## Simulating many motors

`BurnSimulation` runs one motor and is unchanged. To run thousands (sweeps, robustness scenarios, training data)
use the batch entry point, which can run on a GPU when the `jax-cuda12` extra is installed:

```python
from solidpy.ensemble import ProblemBatch, simulate_burn

batch = ProblemBatch.from_objects(motors, propellants, environments, settings)   # settings: BurnSimulation keywords
results = simulate_burn(batch, backend="jax", device="cuda:0").to_results()      # one canonical result per lane
```

`results[i]` has the same `history`, `metrics`, `status`, `efficiencies` and `provenance` as `BurnSimulation.result`
for lane `i`, with the same `physics_provider_hash`, plus `provenance["execution"]` (backend, device, versions).
Backends are `cpu-reference` (the scalar solver, the default), `cpu-vectorized` (NumPy) and `jax`. Lanes a backend
cannot run (Python callables as igniter or activation, analytical tail-off, subclasses, instance-level overrides,
a burn rate that can be negative or not finite)
are solved by the reference and flagged in `provenance["execution"]["fallback"]`, or raise `UnsupportedLane` with
`strict=True`. Igniter and activation profiles, tabulated burn rates and tabulated thermochemistry are supported.
Use `history="decimated:256"` to keep up to 256 native accepted points, or `history="uniform:256"` for fixed-grid
pressure, thrust, flow, burn-area and regression-rate curves with diagnostics from the native grid.

`simulate_burn` can also schedule a batch over several devices and CPU reference workers:

```python
results = simulate_burn(batch, backend=[("jax", "cuda:0"), ("cpu-reference", 6)]).to_results()
```

Each engine pulls chunks from a shared cost-ordered queue; results remain in input order, and unsupported or failed
accelerator lanes are retried on `cpu-reference` with the reason recorded in their provenance. Pass `strict=True` to
reject lanes no selected engine supports. `simulate_thermal` and `run_advanced_physics_ensemble` accept the same engine
list. With `workers > 1`, advanced-physics thermal chunks and CPU models flow through a bounded pipeline; robustness
does the same for burn and detailed-ballistics chunks. `reserved_cores` can reserve host cores for accelerator feeders.
CPU worker pools use `spawn`, so applications should call the API under an `if __name__ == "__main__":` guard when
process workers are enabled.

The speedup comes from batch size: on an RTX 4060 in float64 the JAX backend runs 11.8x faster than the scalar
solver on all 12 threads of the CPU at 4,096 lanes, and slower than it below a few hundred lanes. See
`docs/gpu_backend_benchmarks.md` for the method and numbers, and `docs/gpu_backend_architecture.md` for the design.
Robustness analysis uses the same machinery: `run_robustness_analysis(..., backend="cpu-vectorized")` solves the nominal
run and the scenarios as lanes of one batch, and `solidpy.ensemble.run_robustness_ensemble(designs, backend="jax")` does it
for many designs at once (pass `keep_series=False` to keep only the scalar outputs of each lane). With `workers > 1`,
burn/design chunks and detailed-ballistics post-processing flow through an ordered, bounded process pipeline; the default
remains serial. Without `backend` the analysis is the scalar one, unchanged. Process pools use `spawn`; scripts that pass
`workers > 1` must call these APIs inside an `if __name__ == "__main__":` guard.

The wall conduction and throat ablation of the advanced physics (`Multiphysics.simulate_thermal_ablation`, a Radau solve per
time step) is batched the same way: `solidpy.ensemble.run_advanced_physics_ensemble(geometries, curves, casing_material=...,
backend="jax")` returns what `simulate_advanced_physics` returns for each design, with the thermal ablation of all of them
as one batch (`simulate_thermal(ThermalBatch.from_objects(...))` is the thermal model alone). The batched integrator takes the
same steps as scipy's, so the results agree with the scalar ones to 1e-12; on the GPU it runs 36x the scalar model on all
CPU threads at 4,096 lanes. The whole advanced-physics 5.8x measurement predates the new chunk pipeline, whose GPU
throughput has not yet been remeasured. Pass `workers > 1` to overlap thermal chunks with structural, CFD, ignition and
flight models in a bounded process pool; the default is serial. See `docs/gpu_backend_benchmarks.md` for the measurements
and their dates.

`StructuralMonteCarlo.run(100_000, backend="jax", device="cuda:0")` batches the structural evaluation of its sampled
peak pressures; omitting `backend` preserves its scalar path. `compute_structural_features_vectorized(..., xp=jax.numpy)`
also keeps its numeric results on the selected JAX device. The W4 benchmark is
`benchmarks/bench_structural_monte_carlo.py`; general time-varying structural curves are still evaluated on the CPU.

The slow whole-corpus parity tests run with `pytest --runslow`. GPU parity tests use the `gpu` marker and skip without a
device; set `SOLIDPY_REQUIRE_GPU=1` to make a missing device fail the run. The optional workflow at
`.github/workflows/gpu-backend.yml` runs those tests on a self-hosted runner labelled `gpu` when manually dispatched.

## Authors

-
-

## Assumptions

- Unidirectional and isentropic flow
- Homogeneous combustion products
- No heat exchange with chamber walls (adiabatic)
- No shockwaves or discontinuities in nozzle
- Erosive burning is neglected
- BATES grain
- some others
