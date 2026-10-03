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


