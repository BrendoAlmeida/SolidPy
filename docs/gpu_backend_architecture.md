# SolidPy accelerator backends: architecture and implementation plan

Status: design proposal; Phases 0 to 3 are implemented (section 14.4). Audience: SolidPy maintainers and whoever implements this.
Scope: add GPU execution to SolidPy **without replacing or changing the existing CPU code path**.

## 0. Summary

SolidPy today simulates one motor at a time with `scipy.integrate.solve_ivp` (DOP853) over a scalar,
object-oriented model. It is accurate and well tested, but it is slow when a workload is an *ensemble*
(thousands to millions of independent motors: parameter sweeps, robustness scenarios, Monte Carlo,
training-data generation). A GPU cannot make a single scalar simulation faster; it can make a *batch* of
simulations much faster, because every motor is independent and runs the same equations.

The proposal:

1. Keep every existing class, method, result schema and test exactly as is. The current code becomes the
   `cpu-reference` backend and stays the numerical source of truth.
2. Add an optional, backend-neutral **batch layer**: a structure-of-arrays description of many motors
   (`ProblemBatch`), pure array kernels for the physics, a batched DOP853 integrator that reproduces the
   scipy step controller per lane, and a `BatchResult` that converts back to the existing canonical
   result mappings.
3. Provide several **execution backends** behind one registry: `cpu-reference` (existing code),
   `cpu-vectorized` (NumPy, same kernels as the GPU, also useful on its own), and one or more GPU backends
   (`jax` first). Backends are installed through pip extras, so a CPU-only install stays exactly as light
   as today.
4. Let CPU and GPU work **at the same time**: a heterogeneous executor splits one ensemble between
   CPU worker processes and one or more GPU feeders in proportion to measured throughput.
5. Guarantee equivalence with explicit, automated parity tests (kernel level, integrator level, whole
   simulation level), and record the backend in the provenance of every result.

What this document does not promise: a speedup number. Section 9 gives a back-of-envelope estimate with
its assumptions and a go/no-go gate (Phase 3) that decides, with measurements, whether the rest is worth
building.

## 1. Goals, non-goals, success criteria

### 1.1 Goals

* **G1, expand, do not replace.** `import solidpy` and every current call keep working and keep
  producing the same numbers. No new mandatory dependency. CPU remains the default.
* **G2, CPU and GPU both usable, selectable per call, usable together.**
* **G3, coverage.** The GPU path executes everything the CPU path executes, in tiers. The first release
  must cover at least the code that accounts for **80% of CPU time on the reference workloads**
  (section 7.3 defines how this is measured). Unsupported inputs fall back to the CPU path instead of
  failing, unless the caller asks for `strict=True`.
* **G4, equivalence you can audit.** Every accelerated result is traceable to its backend, device,
  dtype, library versions and integrator settings, and the backend passes a published parity suite.
* **G5, packaging like a modern scientific library.** `pip install solidpy` (CPU),
  `pip install "solidpy[jax-cuda12]"` (GPU), and so on. Missing optional libraries produce actionable
  error messages, never import-time crashes.

### 1.2 Non-goals

* Bit-for-bit identity between a GPU result and the scipy result. The target is tight numerical
  equivalence (section 8); bit identity is only attempted for the `cpu-vectorized` backend's integrator
  decisions (step sizes and event times), and only as a diagnostic.
* Making a single simulation faster on a GPU. Batch size is the lever.
* Replacing scipy/NumPy in the scalar API, or porting plotting/export code to the GPU.
* Single-precision production results. A float32 mode may exist as an explicit "fast, not acceptable
  for acceptance checks" profile (section 5.5); it is not part of the equivalence guarantee.
* Mandatory support for a specific GPU vendor. CUDA is the first target; the design must not preclude
  ROCm/Metal/TPU through the same array code.

### 1.3 Success criteria (release gates)

| Gate | Criterion |
|---|---|
| Parity | All parity tests of section 8 pass for every shipped backend; the numerical acceptance policy in `solidpy.Acceptance` reports `accepted` when a GPU result is compared with the CPU reference on the golden corpus, with margins at least 100x tighter than the policy limits on peaks and integrals. |
| Coverage | >= 80% of CPU time in each reference workload W1-W3 (7.3) runs on the accelerator (Tier 0 + Tier 1). |
| Throughput | On the benchmark ensemble (batch >= 4096 motors), one GPU delivers >= 5x the throughput of the CPU reference on all cores of the same machine, or the project stops at the Phase 3 gate. |
| Safety | CPU-only installs, CPU-only CI and every existing test are unaffected. |
| Provenance | Every result produced by a non-reference backend carries the backend block of section 9. |

## 2. Where the time goes today

Facts measured on the current code (CPython 3.12, NumPy 2.4, SciPy 1.17), profiling 12 star-grain
motors (tail-off to the 1% blowdown cutoff, `rtol=1e-8`, `atol=1e-10`), about 2.8 s per motor:

| Item (cumulative unless noted) | Share of wall time | Calls (12 motors) |
|---|---|---|
| everything inside one full attempt (design sampling, simulation, validation, record building) | 99.9% | 12 |
| the simulation entry point that wraps `BurnSimulation` | 97.3% | 12 |
| `solve_ivp` (all stages) | 88.2% | 6,809 |
| `BurnSimulation._conservative_rhs` | 62.7% | 111,660 |
| `BurnSimulation._state_quantities` | 65% | 143,478 |
| result assembly after the solve (`_public_solution`, per-point quantities and thrust) | 6.4% | 48 |
| axial-flux diagnostic | 0.5% | 12 |
| design sampling + validation + record building + curve features | ~2.7% together | - |
| `Grain.evaluate_port_area` (self time) | 5.5% | 1.7 M |
| builtin `max` / `min` (self time) | 6.1% / 2.4% | 13.0 M / 5.0 M |

Reading: the time is not in numerics, it is in Python-level scalar code executed about 9,300 times per
motor (one right-hand-side evaluation, per-grain loops, `max`/`min` clamps, small NumPy calls). That is
exactly the shape a batched array kernel removes. About 97% of the CPU time of a full attempt is inside
the simulation call (the solve plus its result assembly), which is Tier 0 in this plan; only about 3%
(sampling, validation, record building, diagnostics) is outside it.

Consequences that drive the architecture:

1. **Amdahl.** Accelerating only `solve_ivp` caps the end-to-end speedup near 1 / 0.12 ~ 8x. Accelerating
   the whole simulation call (solve + result assembly, Tier 0) caps it near 1 / 0.03 ~ 30x, and the
   remaining ~3% on the host must overlap with device solving (section 6.4) to approach that cap.
2. **Batching is everything.** One motor is about 10-60 state variables of light arithmetic; a GPU is
   only efficient with thousands of lanes.

## 3. Principles

1. **Single source of truth per backend family.** The physics kernels used by every vectorized backend
   (`cpu-vectorized`, `jax`, `torch`, ...) are written once, as pure functions over an array namespace.
   The existing scalar code is frozen and is *not* re-implemented through these kernels; it is the oracle.
2. **The scalar API is a thin client of the batch layer, never the reverse.** `BurnSimulation(...)` keeps
   its exact behaviour; an optional `backend=` argument can route one simulation through the batch
   engine (a batch of 1) for parity checking, with no speed claim.
3. **Capabilities, not assumptions.** Each backend declares what it supports; the router sends each
   lane to a backend that supports it and records the choice.
4. **Determinism is documented, not assumed.** Reductions over grains are fixed-size and ordered; any
   remaining device-dependent behaviour is listed in section 9.3.
5. **Failure is a result.** Per-lane failure (solver failure, timeout, unsupported feature) is reported in
   that lane's `status`, using the existing `termination_reason` vocabulary, never by aborting the batch.

## 4. Package architecture

### 4.1 Distribution and extras

One distribution, optional extras (the model used by `jax[cuda12]`, `dask[distributed]`,
`scikit-learn`'s optional stacks). The base install is unchanged.

```toml
[project.optional-dependencies]
cea         = ["rocketcea>=1.2"]                     # existing
dev         = ["pytest>=7"]                          # existing

# accelerator backends (each one independent; install only what you need)
jax         = ["jax>=0.4.30; python_version >= '3.10'"]            # CPU/any-device JAX
jax-cuda12  = ["jax[cuda12]>=0.4.30; python_version >= '3.10'"]    # NVIDIA GPU, CUDA 12 wheels
cupy        = ["cupy-cuda12x>=13"]
torch       = ["torch>=2.3; python_version >= '3.9'"]              # GPU wheels need the vendor index; see docs
warp        = ["warp-lang>=1.3"]

# convenience aliases
gpu         = ["solidpy[jax-cuda12]"]                # the recommended default accelerator
all         = ["solidpy[cea,jax-cuda12]"]
```

Notes:

* The wheel stays pure Python (`hatchling`, `packages = ["solidpy"]` already includes sub-packages).
* The Python floor of the core stays `>=3.9`. Accelerator extras carry their own environment markers.
* PyTorch/CuPy GPU wheels are published on vendor indexes; the docs must give the exact install
  commands, and the extras only declare the libraries.
* Never import an accelerator library at `import solidpy` time. Backends import lazily and raise
  `BackendUnavailable` with the install command when requested but missing.

### 4.2 Plugin registry (so backends can live elsewhere later)

Backends are discovered through an entry-point group, so the first-party extras and any third-party or
future split-out distribution (`solidpy-jax`, say) use the same mechanism.

```toml
[project.entry-points."solidpy.backends"]
cpu-reference  = "solidpy.backends.cpu_reference:ReferenceBackend"
cpu-vectorized = "solidpy.backends.numpy_vectorized:NumpyBackend"
jax            = "solidpy.backends.jax_backend:JaxBackend"       # lazy: imports jax on first use
```

Public control surface:

```python
import solidpy

solidpy.backends.available()              # {'cpu-reference': ok, 'cpu-vectorized': ok, 'jax': 'missing: pip install "solidpy[jax-cuda12]"'}
solidpy.backends.describe("jax")          # version, devices, dtype support, capability matrix
solidpy.set_backend("jax", device="cuda:0")        # process-wide default (default stays 'cpu-reference')
with solidpy.use_backend("cpu-vectorized"):        # scoped override
    ...
# environment override, no code change: SOLIDPY_BACKEND=jax SOLIDPY_DEVICE=cuda:0
```

Splitting into separate distributions later is a packaging change only; the entry-point group and the
`Backend` protocol (4.4) are the stable contract.

### 4.3 Module layout

```
solidpy/                         # existing flat modules: unchanged
  backends/
    __init__.py                  # registry, available(), describe(), set_backend(), use_backend()
    _protocol.py                 # Backend protocol, Capabilities, BackendUnavailable
    cpu_reference.py             # adapter: runs the existing scalar code one lane at a time
    numpy_vectorized.py          # NumPy implementation of the batch engine
    jax_backend.py               # jit + lax.while_loop driver (lazy imports)
    torch_backend.py             # optional, later
  batch/
    problem.py                   # ProblemBatch (structure of arrays), pack(), validate()
    result.py                    # BatchResult, to_results(), history policies
    kernels/                     # pure array functions, no loops over lanes, no mutation
      geometry.py                # tubular + star: burn area, port area, remaining volume, burnout depth
      propellant.py              # burn rate (power law, table), erosive term, thermo tables
      nozzle.py                  # mass flow, exit Mach (batched root find), exit state, thrust components
      rhs.py                     # conservative right-hand side for the burn and blowdown stages
      metrics.py                 # peaks, integrals, flow intervals, mass balance
      structural.py, thermal.py  # Tier 1, see section 7
    integrators/
      dop853.py                  # Butcher tableau, per-lane step control, dense output
      events.py                  # per-lane terminal events, restart policy
      stages.py                  # source breakpoints, burn stage, blowdown stage
    _xp.py                       # array-namespace helpers (array-api-compat), dtype policy
  ensemble.py                    # public high-level API (section 4.5)
  executor.py                    # heterogeneous CPU+GPU executor (section 6)
tests/
  backends/                      # parity, capability, packaging tests; `backend` fixture
  golden/                        # frozen corpus and reference results (section 8.3)
benchmarks/                      # reproducible benchmark suite (section 10)
```

### 4.4 The `Backend` protocol

A backend is an *execution engine for batches*, not a re-implementation of every method.

```python
class Backend(Protocol):
    name: str
    api_version: int                       # bumped on breaking protocol changes
    def capabilities(self) -> Capabilities: ...          # what a lane may contain (4.6)
    def devices(self) -> list[str]: ...
    def solve_burn(self, batch: ProblemBatch, options: SolveOptions) -> BatchResult: ...
    # Tier 1 services are optional and advertised through capabilities():
    def thermal_ablation(self, batch, histories, options): ...
    def structural_response(self, batch, histories, options): ...
    def provenance(self) -> dict: ...                   # versions, device, dtype, integrator
```

`cpu-reference` implements `solve_burn` by calling the existing `BurnSimulation` per lane (optionally with
a process pool), so every backend, including the legacy one, is reachable through the same call.

### 4.5 Public API

Existing API: unchanged. New, additive:

```python
from solidpy.ensemble import ProblemBatch, simulate_burn

batch = ProblemBatch.from_objects(motors, propellants, environments, **per_lane_settings)
res = simulate_burn(
    batch,
    backend="jax", device="cuda:0",          # or "auto", "cpu-vectorized", "cpu-reference"
    dtype="float64",
    rtol=1e-8, atol=1e-10, max_step=0.01,    # same defaults as BurnSimulation
    tail_off="numerical",
    history="metrics",                        # "full" | "decimated:N" | "uniform:N" | "metrics" (see 5.8)
    strict=False,                             # False: unsupported lanes fall back to CPU
)
results = res.to_results()                    # list of canonical result mappings (same schema as today)
```

Existing entry points gain an optional `backend=` keyword that defaults to the current behaviour:
`BurnSimulation(..., backend=None)`, `run_robustness_analysis(..., backend=None)`,
`StructuralMonteCarlo.run(..., backend=None)`, `compute_structural_features_vectorized(..., xp=None)`.
With `backend=None` the code path is byte-for-byte today's.

### 4.6 Capability model and routing

Every backend declares, per feature, `supported | partial | unsupported`. The packer annotates each lane
with the features it needs; the router assigns lanes accordingly.

| Lane feature | GPU first release | Notes |
|---|---|---|
| Tubular and star grains (`ends_burn` on/off), any N up to the padding bucket | supported | closed-form geometry, vectorizes with `where` |
| Mixed geometries within one motor | supported | per-grain flags |
| Burn rate: power law `a * p^n` | supported | |
| Burn rate: tabulated, cubic or linear interpolation with clamped ends | supported | same interpolant, see 5.6 |
| Lenoir-Robillard erosive term | supported | |
| Thermochemistry: scalar `k`, `Tc`, `c*` | supported | |
| Thermochemistry: pressure tables (`load_thermo_table`) | supported | batched interpolation |
| Thermochemistry: live `RocketCEA` objects | CPU only | pre-tabulation allowed only when explicitly requested, with a stated error bound |
| `igniter_mass_flow`: none, scalar, table | supported | |
| `igniter_mass_flow` / `burn_area_activation`: Python callables | CPU only | the packer can sample a callable into a table on request (`tabulate_callables=True`) and records the sampling error |
| `burn_area_activation`: none (ramp), scalar, table | supported | |
| Numerical tail-off to the blowdown cutoff | supported | |
| Analytical tail-off | supported | closed form |
| `eta_c`, `eta_Cf`, `discharge_coefficient` | supported | |
| Ambient pressure | supported | scalar per lane, pre-evaluated (the atmosphere lookup stays on CPU at pack time) |

`strict=False` (default): lanes the chosen backend cannot run are executed on `cpu-reference` and flagged in
`provenance.execution.fallback`. `strict=True`: raise `UnsupportedLane` listing lane indices and features.

## 5. Numerical design

### 5.1 State and padding

Per lane the state is the existing vector: `[gas_mass, thermal_inventory, regression_0..regression_{G-1},
generated_mass_integral, igniter_mass_integral, nozzle_mass_integral, thrust_integral,
pressure_throat_integral]`, size `G + 7`. A batch pads the grain axis to a bucket `G_max` (suggested
buckets 4, 8, 16, 24, 32) and carries a boolean `grain_valid[lane, g]`. Padded grains have zero area,
zero volume and are never active, so they contribute exactly zero to every sum and never raise events.
Batches are bucketed by `G_max` so a 2-grain motor does not pay for 24 grains.

### 5.2 Kernels (`batch/kernels`)

Pure functions with signature `f(xp, arrays...) -> arrays`, no Python loops over lanes or grains, no
in-place mutation (required for JAX), branches expressed with `xp.where`. They mirror the current scalar
semantics one-to-one, including every guard (`max(...,0)`, `np.nextafter` clamp before burnout,
`max(port_area, 1e-9)`, `np.finfo(float).tiny` floors, `pressure <= ambient` returns zero flow/thrust):

| Kernel | Scalar origin | Notes |
|---|---|---|
| `burn_area`, `port_area`, `remaining_volume`, `burnout_depth` | `Grain.*` | tubular and star, `ends_burn` variants, burned-through branch |
| `regression_rate` | `Propellant.evaluate_burn_rate` | power law or table, then erosive correction `k_e G^0.8 exp(-alpha_e r0/G)` when `G > 1e-3` |
| `thermo_at_pressure` | `Propellant.Tc_at_pressure`, `get_gamma`, `get_cstar` | scalar or table, `eta_c^2` applied to `Tc` |
| `nozzle_mass_flow` | `Burn.evaluate_nozzle_mass_flow` | choked/unchoked branches |
| `exit_mach` | `Burn.evaluate_exit_mach` | see 5.7 |
| `exit_state`, `thrust_components` | `Burn.evaluate_exit_*`, `evaluate_thrust_components` | momentum, pressure, `eta_Cf`, divergence factor |
| `activation`, `igniter_flow` | `evaluate_burn_area_activation`, `evaluate_igniter_mass_flow` | table interpolation, built-in smooth ramp |
| `conservative_rhs` | `BurnSimulation._conservative_rhs` | assembles the derivative vector |

Port flux detail to preserve: the erosive term uses `G = 0.5 * nozzle_flow / max(mean_over_grains(port_area), 1e-9)`
and the activation uses the mean regression over the *valid* grains. The batched mean must divide by the
number of valid grains, not by `G_max`.

### 5.3 Integrator: reproduce the scipy controller per lane

The strongest parity lever is to make each lane follow the *same controller scipy uses*. The batched
integrator therefore ports, per lane and with the same constants (a first implementation exists, see
Appendix D; it matches scipy's initial-step selection exactly and its integrals to ~1e-6, but, as
measured there, it does **not** reproduce scipy's accepted-step sequence step for step, and no
implementation using a different summation order will):

* the DOP853 Butcher tableau (A, B, C, E3, E5, D and the extra stages used by the dense output), taken
  from `scipy.integrate._ivp.dop853_coefficients`;
* `select_initial_step` (first-step heuristic), `rtol`/`atol` scaling `atol + rtol * max(|y|, |y_new|)`,
  the RMS error norm combining the 5th and 3rd order estimates, safety factor 0.9, factor clipping
  `[0.2, 10]`, error exponent `-1/8`, `max_step` clipping, and step rejection;
* the 7th-order dense output of DOP853 (needed for exact event localization and for the final state of an
  event-terminated step);
* scipy's `solve_ivp` restart behaviour: every stage and every event restarts with a fresh initial-step
  selection, exactly as the current code does by calling `solve_ivp` again.

Each lane owns its time `t`, step `h`, state `y` and flags; one loop iteration attempts one step for
every lane that is not finished. Accepted lanes advance, rejected lanes shrink `h`; finished lanes are
masked (their values are held, not recomputed). Loop count equals the largest per-lane attempt count in
the batch, so batches are built from lanes with similar expected cost (section 6.3).

Higher-level choice, to be confirmed by the Phase 2 numbers: implement DOP853 itself first (parity);
only if its 12 stages per step prove too costly on the accelerator, evaluate Dormand-Prince 5(4) at a
tighter tolerance as an optional integrator, validated against the same acceptance policy.

### 5.4 Events and stages

The current solver structure must be reproduced:

1. **Stage plan.** Source breakpoints (igniter end, ramp end, table knots inside the stage) split the
   integration into segments, as `_source_breakpoints` does. Breakpoint lists are padded per lane.
2. **Per-grain burnout (terminal, direction +1).** After every accepted step, a lane whose grain
   regression crossed `burnout_regression_m` is located on the dense output (root find with the same
   tolerance family scipy uses), the step is truncated to the event, the grain is snapped to its exact
   burnout depth, flagged inactive (`active[g] = False`, `burnout_time[g]` recorded), and the lane
   restarts. Because the active mask changes the right-hand side, "inactive" is a per-grain boolean input
   to the kernels, not a different code path.
3. **Blowdown cutoff (terminal, direction -1).** Pressure falling through
   `ambient + 0.01 * (reference_peak - ambient)`; terminates the lane successfully.
4. **Timeouts and failures.** `burn_timeout_s`, `tail_off_timeout_s`, non-finite state, step underflow:
   reported per lane with the existing `termination_reason` strings (`solver_failure`, `burn_timeout`,
   `blowdown_timeout`, ...).

The blowdown reference peak requires the maximum pressure over the burn-stage history; the batched
integrator therefore tracks `max(pressure)` on every accepted step *and* on event points, matching the
scalar code, which takes the maximum over all stored points.

### 5.5 Precision

* **Production dtype is float64.** Parity at `rtol=1e-8` needs it. Enable x64 explicitly in the JAX
  backend (`jax_enable_x64`); never rely on global user state.
* GPU double-precision throughput varies by two orders of magnitude between data-center parts and
  consumer or inference parts (roughly 1/2 to 1/3 of single-precision on the former, 1/32 to 1/64 on the
  latter). This workload is dominated by launch latency and memory traffic at moderate batch sizes
  (section 9), so consumer GPUs can still win, but it must be measured per device and reported by the
  benchmark suite.
* `dtype="float32"` may be exposed as an experimental profile for exploration (it will not meet the
  numerical acceptance policy and the results are tagged `precision: float32`). It is optional and last.

### 5.6 Interpolated properties

Burn-rate and thermochemistry tables use `scipy.interpolate.interp1d` with `kind="cubic"` for four or more
points (otherwise linear) and constant extrapolation (`fill_value=(y[0], y[-1])`). The batched version
must reproduce the same interpolant, not a different spline:

* at pack time, compute the cubic-spline coefficients on the CPU with scipy (`CubicSpline` with the
  not-a-knot boundary that `interp1d(kind="cubic")` uses, which is what the current code evaluates) and
  store per-lane padded breakpoint and coefficient arrays;
* on the device, evaluate by `searchsorted` + Horner evaluation, clamping to the end values outside the
  range;
* tables of different lengths are padded to a per-batch maximum with an explicit valid-length array.

This keeps scipy as the single place where the interpolant is defined and makes the device code a pure
evaluation.

### 5.7 Exit Mach (root find)

The scalar code solves the supersonic area-Mach relation with a bracketed `brentq` (`xtol=rtol=1e-10`) and
caches by `(round(k, 12), expansion_ratio)`. With a pressure-dependent `k` the batched kernel must solve
per evaluation. Use a safeguarded Newton iteration on the monotonic branch `M > 1` with a bracket
fallback (bisection step when Newton leaves the bracket), a fixed maximum iteration count (suggested 12),
per-lane convergence masks, and the same residual tolerance. When `k` is constant per lane (the common
case) solve once per lane at pack time and pass the value in, which removes the iteration from the inner
loop entirely. Both paths are validated against `brentq` over a dense grid of `(k, expansion_ratio)`.

### 5.8 Outputs and history policy

The CPU result stores the adaptive accepted-step history (plus event points). A device batch cannot hold
unbounded histories for every lane, so output is configurable:

| `history` | What is stored | Use |
|---|---|---|
| `"metrics"` | only the canonical `metrics`, `status`, `provenance` (computed on device from running reductions) | large ensembles; smallest memory |
| `"decimated:N"` | metrics plus up to N native accepted points, spread evenly by accepted-point index, with all history channels | surrogate/feature workflows |
| `"full"` | the accepted-step grid per lane, padded to `max_steps` with a per-lane length | parity tests, debugging, plots |
| `"uniform:N"` (recommended for ensemble consumers) | metrics, seven main channels resampled to N uniform time points, and the diagnostics below | consumers that persist fixed-grid curves |

**What a typical ensemble consumer actually needs.** An investigation of how a downstream consumer uses
the CPU result found that it never keeps the raw adaptive history. It uses (a) the scalar `metrics`;
(b) seven channels (chamber pressure, thrust, generated and nozzle mass flow, burn area, mean regression
rate, generated-mass integral) resampled with `np.interp` (linear interpolation of the accepted-step
grid, not the dense output) onto a uniform grid of `ceil(end_time / dt) + 1` points capped at a
maximum (typically a few hundred points, never more than a few thousand); and (c) a few diagnostics
computed **on the native adaptive grid**: the time of the thrust maximum, the maximum of
`|gradient(pressure, time)|` (second-order differences on the irregular grid), and the axial mass-flux
maximum, its time and its grain (the per-time, per-grain flux series is optional and large). `"uniform:N"`
therefore derives (c) from the native accepted-step buffer, resamples (b) with the same linear
interpolation, and returns only the uniform curves and the diagnostics. The current implementation copies the retained
accepted-step buffer to the host and computes the pressure derivative and axial-flow diagnostic there.

Two consequences: the device must keep the accepted-step buffer *transiently* (about `max_steps x
(7 + G)` values per lane, around 1 MB per lane in float64, so a 4,096-lane chunk needs ~4 GB; the output
contains only the uniform curves, tens of KB per lane, though host assembly temporarily receives the native grid too);
and the native-grid diagnostics (time of the maximum, maximum pressure-rise rate, axial-flux maximum) depend on the
accepted-step sequence, which is a second reason to port the scipy step controller exactly (5.3). They get their own parity limits
(Appendix B): looser than integrals, tight for the `cpu-vectorized` backend.

Memory budget (worked formula, float64): `lanes x max_steps x (G + 7 + extras) x 8 bytes`. Example: 4,096
lanes x 4,000 steps x 40 values x 8 B = about 5.2 GB for `"full"`; the same batch in `"metrics"` mode
needs only the running reductions (a few hundred bytes per lane). The executor chunks batches to fit device
memory (section 6.3). A per-lane `step_overflow` flag is raised if `max_steps` is hit.

Metrics that the scalar code derives from the history (peaks, integrals, flow start/end brackets,
mass-balance residual, burn duration, `grain_burnout_times_s`) are computed on device with running
reductions where possible; the few that need the whole history (`_flow_interval` brackets) use a
two-pass or masked-scan formulation. The kernel for each metric is listed in `kernels/metrics.py` with a
unit test against the scalar function.

## 6. Running on CPU and GPU together

### 6.1 Execution modes

| Mode | What runs where |
|---|---|
| `backend="cpu-reference"` | existing scalar code, optionally in a process pool (as `DispersionAnalysis` does today) |
| `backend="cpu-vectorized"` | batch engine on NumPy; one process per core with small batches; also the parity oracle for kernel logic |
| `backend="jax", device="cuda:0"` | batch engine on one GPU |
| `backend="auto"` | choose by lane capabilities, batch size and installed backends |
| `backend=[("jax","cuda:0"), ("cpu-reference", 6)]` | **heterogeneous**: several engines at once |

### 6.2 Heterogeneous executor (`solidpy.executor`)

The first scheduler is implemented by `solidpy.executor.HeterogeneousExecutor` and exposed as a list passed to
`simulate_burn`, for example `backend=[("jax", "cuda:0"), ("cpu-reference", 6)]`. Each backend/device gets one
feeder thread. Feeders pull compatible lanes from a shared queue ordered by the existing `lane_cost` estimate; a
feeder that finishes a chunk can claim another, so faster engines naturally take more work. Accelerator chunks
default to at most 2,048 lanes (and respect the backend's own memory cap); reference chunks default to 64 lanes per
worker. Callers can set `chunk_size` to bound either kind of chunk.

Multiple devices can run concurrently because each device is a distinct backend instance and feeder. There is no
cross-device data exchange. Lanes unsupported by all selected engines go to `cpu-reference` unless `strict=True`;
an engine exception, missing result or `step_overflow` is isolated and retried on the reference. Each lane records the
selected backend or fallback reason, and output order matches input order. The scheduler uses chunk completion to
adapt the split; it does not run a separate throughput-calibration solve.

Still open in this section: configurable core reservation and native-thread limits for worker processes. The CPU
reference worker pool currently follows the existing `workers` option and host process cap.

### 6.3 Batching policy

* The heterogeneous scheduler sorts lanes by the estimated cost from burn time (`~ web / burn_rate`) and grain count.
  Backend-specific grain and table padding still happens when the chunk is prepared.
* Accelerator chunk size respects the backend memory cap and defaults to at most 2,048 lanes; a caller may set a
  smaller `chunk_size` when memory or latency requires it.
* **Refill** (continuous batching), where a finished lane is replaced inside an active device batch, is not
  implemented. Feeders currently submit fixed chunks from the shared queue.

### 6.4 Overlapping accelerator and CPU work

After the batched ODE the remaining per-lane work (derived metrics that stay on the host, result assembly, user
callbacks, writing outputs) can dominate. The advanced-physics and robustness ensemble APIs accept `workers > 1` for
bounded CPU post-processing. The heterogeneous burn scheduler can overlap separate backend chunks, including a
device solve and a CPU reference chunk; the high-level W2/W3 APIs still wait for the complete batch result before
starting their CPU post-processing. There is not yet a producer/consumer pipeline that overlaps solve chunk `k`,
post-processes chunk `k-1` and packs chunk `k+1` as one flow.

## 7. Coverage map (what runs where)

Tiers: **T0** required for the first release; **T1** next, completes the 80% target on advanced-physics
workloads; **T2** later or CPU-only by design.

| Module / function | Pattern | Tier | Approach |
|---|---|---|---|
| `Grain` (tubular, star geometry) | closed-form arithmetic | T0 | geometry kernels |
| `Motor` (free volume, areas, Kn) | pack-time scalar | T0 | computed on CPU at pack time and passed as lane constants |
| `Propellant` burn rate, erosive term, thermo tables | arithmetic + interpolation | T0 | propellant kernels, spline coefficients precomputed |
| `Propellant` live RocketCEA | external Fortran | T2 (CPU) | optional pre-tabulation at pack time |
| `Burn` nozzle flow, exit state, thrust | arithmetic + small root find | T0 | nozzle kernels |
| `BurnSimulation` solve (burn + tail-off) | stiff-free ODE with events | T0 | batched DOP853 + events |
| `BurnSimulation._build_result` metrics | reductions over history | T0 | metrics kernels |
| `AxialFlow` (axial mass flux, diagnostics) | per-grain vector math over history | T0 | batched, memory-bound |
| `Robustness` (`run_robustness_analysis`, Latin hypercube) | many perturbed copies of one design | T0 | scenarios become extra lanes of one batch; scenario application stays on CPU at pack time; thrust rescale is vector math |
| `surrogate_physics` (static, structural features, vectorized variants) | NumPy vector code already | T0 | accept `xp=` and run on the device |
| `Multiphysics.simulate_structural_response` | algebra over time series | T1 | general transient curves remain scalar; the synthetic peak-pressure history used by `StructuralMonteCarlo` is batched |
| `Multiphysics.StructuralMonteCarlo` | many independent samples | T1 | W4 peak-pressure scenarios run as lanes on NumPy or JAX; callback sampling and report assembly stay on the host |
| `Multiphysics.simulate_thermal_ablation` | 1-D conduction (banded linear system per time step) | T1 | **done (4b)**: batched Radau IIA(5) with the scipy controller, tridiagonal factorizations, ragged node counts padded (section 14.6) |
| `Multiphysics.simulate_cfd_proxies`, `simulate_ignition_proxy` | algebraic proxies | T1 | vectorized |
| `Multiphysics.geometry_from_components` (mass, CG, bulkheads) | vector algebra (vectorized variants exist) | T1 | `xp=` |
| `DetailedBallistics` (`build_detailed_ballistics`, stability, nozzle ablation rate) | post-processing of histories | T1 | vectorized over lanes; interpolation of histories to uniform grids is a batched `interp` |
| `TwoPhaseFlow` | profile algebra | T1 | vectorized |
| `Acceptance.evaluate_numerical_acceptance` | compares two result mappings | T2 | stays on CPU (operates on results) |
| `Multiphysics.simulate_flight_1d`, `simulate_flight_3dof`, `evaluate_barrowman_stability`, `evaluate_cd_by_components` | separate ODE systems with events/stages | T2 | optional second integrator target after the burn solver is validated |
| `MonteCarlo.DispersionAnalysis` | flight Monte Carlo | T2 | depends on batched flight; the existing process pool remains |
| `Acoustics.CavityResonance` | modal algebra | T2 | vectorizable, low priority |
| `Rail`, `BurnEmpirical`, `Export`, `Environment` | small ODE / empirical / I/O / lookup | T2 (CPU) | stay on CPU by design; `Environment` is evaluated at pack time |

### 7.1 The 80% rule, made measurable

"80% of the heavy work" is defined by profiling, not by counting modules:

1. Fix reference workloads:
   * **W1** burn-only ensemble: >= 2,000 motors mixing tubular and star grains, 1-24 grains, erosive
     burning on and off, tabulated and power-law propellants, tail-off to cutoff.
   * **W2** W1 plus the advanced physics (thermal, structural, cfd proxies, detailed ballistics).
   * **W3** robustness: nominal + the default scenarios + Latin hypercube scenarios for a subset of W1.
   * **W4** structural Monte Carlo (10^5 samples on a few designs).
   * **W5** flight dispersion (reported, not gating).
2. Profile the CPU reference on each workload and map every function to a tier
   (`tools/tier_map.toml`, maintained with the code).
3. Report `offloaded_share = CPU seconds in T0 + T1 functions / total CPU seconds` per workload.
4. Release gate: `offloaded_share >= 0.80` for W1, W2, W3. Measured on a star-grain burn-only workload
   the Tier 0 functions already account for about 97% of CPU time (section 2), so W1 is expected to pass
   with Tier 0 alone; W2 and W3 are what Tier 1 is for.

### 7.2 What "fully executes" means for T0/T1 modules

For every function in T0/T1 the accelerated version returns the same keys, units and shapes as the
CPU function (batched along a leading lane axis) and is covered by the parity tests of section 8.
Features outside the capability matrix (4.6) are not silent degradations: they fall back or raise.

## 8. Equivalence and validation

### 8.1 Layers

1. **Kernel parity** (fast, exhaustive). Each kernel vs the scalar function it mirrors, on randomized
   inputs (property-based) including edge cases: pressure at/below ambient, regression at and past the
   burnout depth, `n_points * epsilon` near `pi`, `ends_burn`, zero port flux, table clamps, unchoked
   flow. Target: relative error <= 1e-12 in float64 (`cpu-vectorized`), <= 1e-10 on GPU.
2. **Integrator parity.** Against `solve_ivp` on the same right-hand side: the *first* accepted steps
   and the initial step (identical to rounding, checked directly), event times (within 1e-6 relative
   in time, measured), and the integral outputs. **Step-for-step identity is not a target**: the
   measured step counts agree on only ~12% of lanes (median difference 0.55%, maximum 6.8%) even
   though the right-hand side is identical to 1e-13, because summation-order differences in the stage
   combinations are amplified through the error estimate and the ignition transient (Appendix D).
3. **Whole-simulation parity.** Batched backend vs `BurnSimulation` on the golden corpus (8.3) and on
   large random ensembles. Compare the policy metrics through `solidpy.Acceptance.evaluate_numerical_acceptance`
   (reference as `refined`, backend as `coarse`) **and** with internal limits 100x tighter than the
   policy: **integrals** (impulse, generated and nozzle mass) <= 1e-5 relative (measured max 4.8e-6),
   **quantities sampled on the accepted-step grid** (peak pressure, peak thrust, peak flows) <= 2e-3
   relative (measured max 8e-4; this is the same order as the variation of the reference path between
   two CPU environments with different NumPy versions, so it is a property of grid-sampled maxima, not
   of the backend), mass-balance residual no worse than the reference, identical `termination_reason`
   classification, and `burnout_times` within 1e-6 relative. An optional improvement is to define peaks
   on the continuous solution (refine the maximum on the dense output or by parabolic refinement) so
   they stop depending on the step grid on every backend.
4. **Statistical parity** on 10^4-10^5 random lanes per release: report the distribution (max, 99.9th
   percentile, median) of each relative delta; fail on any lane above the limit, and require that
   failures (solver failure, timeout) occur for the same lanes in both paths.
5. **Existing test-suite reuse.** The current test files (burn, conservative ballistics, efficiencies,
   exit Mach, grain conservation, motor volume, multiphysics, robustness, structural contract,
   surrogate physics, numerical acceptance, solve cache, ...) are parametrized over a `backend` fixture so
   every behaviour that is specified for the CPU path is exercised on each backend that claims it.
   Tests that need a device carry `@pytest.mark.gpu` and skip cleanly when none is present.

### 8.2 Tolerance policy

Parity limits are versioned constants next to the code (`solidpy/backends/_tolerances.py`), never edited
to make a failing test pass. Any change needs a written cause. The existing numerical acceptance limits
(peaks 2%, integrals 1%, mass balance 1%) are the *outer* contract; the internal limits above are what
CI enforces.

### 8.3 Golden corpus

A frozen set of about 300 designs with stored reference results (all canonical metrics plus decimated
histories), generated by the reference backend and committed under `tests/golden/` with the SolidPy
revision that produced them. It must cover: tubular and star, 1/2/8/24 grains, `ends_burn` both ways,
power-law and tabulated burn rate, scalar and tabulated thermochemistry, erosive on/off, igniter none /
scalar / table, activation none / ramp / table, `eta_c < 1`, `discharge_coefficient < 1`, very short and
very long burns, unchoked blowdown, solver-failure and timeout cases, and designs at the edge of every
guard in the scalar code.

## 9. Provenance and reproducibility

The reference result already records `physics_provider_hash` (SHA-256 over the resolved inputs and the
bytes of `Burn.py`, `Grain.py`, `Propellant.py`), the git SHA and the solver settings. Accelerated results
extend this, without removing anything:

```python
provenance["execution"] = {
    "backend": "jax", "backend_api_version": 1,
    "device": "cuda:0", "device_name": "...", "dtype": "float64",
    "library_versions": {"jax": "...", "jaxlib": "...", "cuda": "...", "numpy": "...", "scipy": "..."},
    "integrator": {"name": "dop853_batched", "scipy_controller_port_version": 1,
                   "rtol": 1e-8, "atol": 1e-10, "max_step_s": 0.01},
    "fallback": None,                      # or {"lane_reason": "callable_igniter", "ran_on": "cpu-reference"}
    "history": "metrics",
    "kernel_source_hash": "sha256:...",    # hash of the kernel and integrator sources actually used
    "parity_certificate": {"suite_version": "...", "tolerances_version": "...", "passed": True},
}
```

* `physics_provider_hash` must change whenever the code that defines the physics changes. For
  accelerated results it covers the kernel and integrator sources; the reference hash keeps its current
  definition so existing stored results remain comparable.
* A shared **`physics_equivalence_class`** field is added: identical for the reference backend and for a
  backend that carries a valid parity certificate against it, different otherwise. Consumers that persist
  results (studies, archives) can then decide programmatically whether results from different backends
  may be mixed, instead of guessing.
* Results record the *requested* and the *effective* backend per lane when routing or fallback occurred.

### 9.3 Known sources of cross-device variation (documented, tested)

Fused multiply-add and operation reordering by the compiler, different libm implementations for
`exp/pow`, and parallel reductions. Reductions over grains are over a small fixed axis and are written
in a fixed order; XLA/Torch options that enable fast-math or reassociation are disabled in the
production profile. The parity suite runs on every supported device class and the tolerances in 8.1 are
the observed envelope with margin.

## 10. Performance engineering

### 10.1 Measured kernel cost and projection (assumptions are explicit)

**Measurement.** A batched right-hand side with the same operation structure as the scalar one (tubular and
star geometry with `where`, port flux, erosive burn rate, choked/unchoked nozzle flow, exit velocity and
thrust components; 12 grains per lane; float64) was timed on a consumer GPU (an 8 GB Ada-generation card,
PyTorch 2.12, CUDA 13), without any integrator or events. Eager mode launches ~100 kernels per evaluation;
"graph" replays the same operations as one captured CUDA graph (what a compiled whole-loop backend would
do). Reference: the CPU scalar code spends about 190 microseconds per lane per right-hand-side evaluation.

| lanes | dtype | eager ms/eval | graph ms/eval | graph microseconds per lane-eval |
|---|---|---|---|---|
| 512 | fp64 | 2.92 | 0.38 | 0.74 |
| 2,048 | fp64 | 2.14 | 0.44 | 0.22 |
| 8,192 | fp64 | 2.27 | 0.62 | 0.076 |
| 32,768 | fp64 | 2.20 | 2.08 | 0.064 |
| 8,192 | fp32 | 2.08 | 0.27 | 0.033 |
| 32,768 | fp32 | 2.14 | 0.72 | 0.022 |

Readings: (1) eager mode is launch-bound (about 2 ms regardless of batch size up to 32k lanes), so a
Python-driven loop is already 600-700x faster per lane-evaluation than the CPU at 8k lanes, and a
compiled loop 2,000x or more; (2) double precision costs only about 3x over single precision on this
kernel even on a card whose double-precision peak is 1/64 of single, because the kernel is bound by
launches and memory, not by arithmetic; (3) the crossover is around a few hundred lanes: below that the
launch cost is not amortized.

**Projection to a full simulation (planning numbers, not a promise).** One motor needs about 9,300
right-hand-side evaluations on the CPU reference, i.e. about 780 DOP853 attempts (12 stages each,
including rejections). In a lockstep batch the loop runs as many iterations as the slowest lane; assume
2,000 (a 2.5x imbalance margin) and a 1.5x integrator overhead on top of the right-hand-side cost. For a
batch of 8,192 lanes that is `2,000 x 12 x 1.5 = 36,000` evaluation-equivalents:

* eager (Python-driven loop): 36,000 x 2.27 ms = about 82 s per batch, 10 ms per motor;
* compiled loop / graph: 36,000 x 0.62 ms = about 22 s per batch, 2.7 ms per motor;
* the CPU reference needs 2.8 s per motor per core: about 0.47 s per motor on 6 physical cores.

That is roughly 47x (eager) to 170x (compiled) on the solve itself against 6 CPU cores, before the
remaining haircuts: padding to `G_max`, event/dense-output work, transfers, host remainder and compile
time (together another 1.5-3x). **Planning range: 15-60x on the simulation call for batches of at least
~4,000 lanes on a consumer GPU in float64; a data-center GPU removes the arithmetic limit but not the
launch limit, so the gain from compiling the whole loop remains the main lever.** End to end, Amdahl
(section 2) and the host remainder (~3% of a CPU attempt, about 85 ms per motor per core) bound the gain
to roughly 10-30x when CPU post-processing overlaps with the device (6.4), and less without overlap. On
a machine with few CPU cores the host remainder, not the GPU, is what limits throughput.

These numbers are projections from a kernel micro-benchmark. They must be replaced by measurements of the
real integrator at the Phase 3 gate. Small batches (< ~500 lanes) will not beat the CPU; `backend="auto"`
must know this.

### 10.2 Techniques, in order of expected value

1. Compile the whole integrator loop (`lax.while_loop` under `jit`), keep the state on device.
2. Shape bucketing (`G_max`, table lengths, `max_steps`) plus a persistent compilation cache to avoid
   recompiles; report compile time separately from run time.
3. Keep the structure-of-arrays batch resident across calls (pack once, reuse for several solves, e.g.
   scenario sweeps).
4. Lane sorting by estimated cost; optional refill (6.3).
5. Fuse the per-step derived-metric reductions into the loop (no history storage in `"metrics"` mode).
6. Pinned host memory and asynchronous transfers for the pipeline in 6.4.
7. Only if profiling shows launch or divergence limits: a hand-written per-lane kernel backend (one
   thread per motor, in-kernel adaptive loop, e.g. with Numba-CUDA, Warp or CuPy `RawKernel`). It
   duplicates numerics in another language, so it is gated behind the same parity suite and not planned
   unless the measurements justify it.

### 10.3 Benchmark suite (`benchmarks/`)

Reproducible, versioned, machine-readable output: lanes/second and seconds/lane for each reference
workload, each backend, batch sizes 64 to 65,536, plus compile time, peak device memory and the
profile-based `offloaded_share`. Results record hardware and library versions. A regression in lanes/second
beyond a stated threshold fails the (optional) performance CI job.

## 11. Technology choice for the first accelerator backend

| Criterion | JAX (+ optional Diffrax) | PyTorch | CuPy | Custom kernels (Numba-CUDA, Warp, RawKernel) |
|---|---|---|---|---|
| float64 on GPU | yes | yes | yes | yes |
| Whole-loop compilation (no per-op launch cost) | yes (`jit` + `while_loop`) | partial (CUDA graphs, `torch.compile`) | no (per-op launches) | yes (a single kernel) |
| Per-lane adaptive stepping and events | masked loop, natural | masked loop in Python or compiled | masked Python loop | per-thread control flow, natural |
| Same code on CPU, GPU, TPU | yes | CPU + GPU | GPU only | GPU only |
| Automatic differentiation (sensitivities, calibration) | yes, first class | yes | no | partial (Warp) |
| Development speed from the NumPy-style kernels | high (functional, `where`) | high | highest (near-NumPy) | lowest |
| Install and maintenance burden | moderate; CUDA wheels via extras; Python >= 3.10 | large wheels; vendor index | CUDA-version-specific wheels | compiler/toolchain issues |
| Debuggability | moderate | good | good | harder |

Recommendation: implement the kernels against an array namespace, ship `cpu-vectorized` (NumPy) first
as oracle and portable fallback, then **JAX as the first accelerator backend**. Evaluate Diffrax only as
an optional integrator implementation inside the JAX backend (its adaptive controller and event
machinery would need to match scipy's closely enough to pass section 8); the in-house DOP853 port is the
default because step-sequence parity is the strongest equivalence evidence. A PyTorch backend reusing the
same kernels is a good second target if the user environment is PyTorch-centric, because
the kernels are framework-agnostic; it will rely on `torch.compile`/CUDA graphs to avoid launch overhead.

**Environment constraint found while testing (affects the choice above).** One target machine is a
virtual machine whose emulated CPU model exposes neither AVX nor even SSE4/POPCNT (the generic
"QEMU Virtual CPU 2.5" model). On it JAX refuses to import (`jaxlib` is built with AVX), and NumPy 2.4+
refuses to import ("baseline optimizations X86_V2 not supported"); NumPy 1.26, PyTorch 2.11 (CUDA 12.8)
and NVIDIA Warp 1.17 do import and run (tested: a float64 kernel and a float64 matmul on the
data-center GPU of that machine). The proper fix is host-side (expose the host CPU model, or
`x86-64-v3`, to the VM), but a design that must work on such a machine should not assume JAX: this is
the practical argument for keeping the physics kernels framework-agnostic (section 4.3, `_xp.py`) and
for treating the PyTorch backend, and a per-lane kernel backend (Warp), as first-class citizens rather
than optional extras.

The array-namespace layer uses the Array API standard (`array-api-compat`) for NumPy/CuPy/Torch and
JAX's NumPy-compatible namespace. The loop driver is the only backend-specific piece
(`lax.while_loop` for JAX, a Python loop with masks otherwise).

## 12. Phased roadmap (rough effort, one experienced engineer)

| Phase | Deliverables | Exit criterion | Effort |
|---|---|---|---|
| 0. Baseline | profiling harness, reference workloads W1-W5, tier map, golden corpus, `ProblemBatch` schema, empty backend registry and extras, CI skeleton | baseline numbers recorded; corpus frozen | 1-2 wk |
| 1. Kernels + `cpu-vectorized` | geometry, propellant, nozzle kernels; fixed-step RK4 prototype to validate the right-hand side; kernel parity tests | kernel parity 1e-12 on the property tests | 2-3 wk |
| 2. Batched DOP853 + events + stages | per-lane controller port, dense output, burnout and cutoff events, stage plan, metrics kernels, `BatchResult.to_results()`; routing, capabilities, provenance | whole-simulation parity (8.1.3) on golden corpus and W1 ensembles with `cpu-vectorized` | 3-5 wk |
| 3. JAX backend and **go/no-go** | jit/`while_loop` driver, x64, device selection, memory chunking, bucketing; first benchmark | parity on GPU **and** >= 5x vs all CPU cores at B >= 4,096 (else stop and keep phases 0-2 as a CPU-vectorized improvement) | 2-3 wk |
| 4. Tier 1 coverage | robustness as lanes, axial flux, structural/thermal/cfd/ignition/detailed-ballistics batching, `xp=` for surrogate physics, structural Monte Carlo | `offloaded_share >= 0.8` on W1-W3 | 4-6 wk |
| 5. Heterogeneous executor and polish | CPU+GPU executor, multi-GPU, refill batching, overlap pipeline, docs, install guides, optional GPU CI | benchmark suite results published; documentation complete | 3-4 wk |
| 6. Optional | PyTorch backend, flight ODE (T2), Diffrax variant, autodiff APIs, float32 profile, custom-kernel backend | each justified by measurements | open |

**Status 2026-10-02:** a time-boxed spike of Phases 1-3 for the tubular/star, scalar-thermochemistry,
no-igniter case exists and passed its checks (Appendix D); the go/no-go numbers for a consumer GPU are
in. Igniter/activation profiles, tabulated properties, full-history modes, packaging and the scheduler
remain.

Effort figures are planning estimates, not commitments; Phase 2 carries most of the risk (event
localization and exact dense output) and Phase 3 decides the future of the rest.

## 13. Risks and mitigations

| Risk | Mitigation |
|---|---|
| GPU double-precision is slow on consumer hardware | measure per device in the benchmark suite; launch-bound regime still favours the GPU at large B; the go/no-go gate uses the real target hardware |
| Lane imbalance wastes the device | cost-based bucketing, sorting, refill batching |
| Event localization or dense output deviates from scipy | port scipy's algorithm and constants exactly; test event times against scipy directly; keep `cpu-vectorized` as an oracle |
| Two physics implementations drift apart | reference code frozen; kernels tested against it on every change; parity suite and golden corpus in CI; `physics_equivalence_class` makes drift visible downstream |
| Python callables in user inputs cannot run on a device | capability routing with CPU fallback; optional tabulation with recorded error |
| Compile time dominates small jobs | persistent compilation cache, shape bucketing, `auto` routing by batch size |
| Memory exhaustion on long histories | history policy (5.8), chunking by the memory formula, `step_overflow` flag |
| Optional-dependency breakage | lazy imports, extras, CPU-only CI job required, GPU CI optional |
| Python 3.9 core vs accelerator libraries needing newer Python | extras carry markers; core support policy unchanged |
| Results from different backends mixed silently in downstream storage | provenance block and `physics_equivalence_class` (section 9) |

## 14. Decisions and remaining open points

### 14.1 Decided

| # | Question | Decision |
|---|---|---|
| 1 | Target hardware | Known so far: a consumer 8 GB Ada-generation GPU (double-precision peak 1/64 of single, but see 10.1: the penalty measured on this kernel is ~3x) on the main workstation; a data-center Ampere-class 24 GB GPU (strong double precision) on a server with only 4 virtual CPU cores, where the host-side remainder and the feeder are the likely limit; cloud-notebook GPUs vary by session and must be recorded with `nvidia-smi` before the benchmark. |
| 2 | First accelerator framework | **JAX** (installing it is acceptable). PyTorch is already present on the workstation and remains the documented second backend with the same kernels. |
| 3 | Parity level | **Accepted as proposed:** acceptance-policy compliance with a 100x margin on peaks and integrals (8.1, Appendix B). |
| 4 | Scope | **Everything, in phases** (section 12): Tier 0, then Tier 1, then Tier 2 as justified by measurements. |
| 5 | History needs | **Investigated** (5.8): consumers need metrics, a handful of uniform-grid curves and a few native-grid diagnostics, never the raw adaptive history; the `"uniform:N"` policy is added. |
| 6 | Callable inputs | **Investigated:** a typical consumer passes no igniter and no activation profile at all (defaults), uses scalar thermochemistry only, and rejects live RocketCEA results. For that use the impact of tabulated-only GPU inputs is zero. Tables, scalars and the built-in ramp run on the device; arbitrary Python callables fall back to the CPU (or, in the JAX backend, are accepted when they are written with array operations and can be traced). |

### 14.2 Still open

1. **Typical batch sizes** in real use (decides how much of the heterogeneous executor is needed and
   whether refill batching in 6.3 is Phase 5 or earlier).
2. **Provenance policy.** Should results from different backends ever be stored together? This decides
   whether `physics_equivalence_class` must be strict or may be informational.
3. **Differentiability.** Is gradient access (calibration, sensitivity) a goal? It favours JAX and
   affects how the integrator is written (adjoint vs unrolled).
4. **CI and licensing.** Is GPU CI available (self-hosted or on demand), and are there constraints on
   optional dependencies for a public package?
5. **Cloud-notebook GPU model(s)** to record for the benchmark baseline.

### 14.3 Decisions taken at implementation start (2026-10-02)

Scope of the first implementation round: Phases 0 to 3 (up to the go/no-go gate), on the single branch
`feat/gpu-batch-backend`, one commit per implementation unit, one review per phase. The scalar code in
`Burn.py`, `Grain.py` and `Propellant.py` is not modified, because `physics_provider_hash` hashes the bytes
of those three files (`Burn.py`, resolved inputs plus source bytes) and any edit changes the hash of every
CPU result. The following points replace the corresponding text above.

| # | Section | Decision |
|---|---|---|
| D1 | 4.5 | `BurnSimulation` does not get a `backend=` argument. Accelerated execution goes through `solidpy.ensemble.simulate_burn` and `solidpy.backends`. `backend=` on `run_robustness_analysis` and `StructuralMonteCarlo.run` is deferred to Phase 4. |
| D2 | 9 | The `physics_provider_hash` of a result from any backend keeps the reference definition (resolved inputs plus the bytes of the three reference files). `Acceptance.evaluate_numerical_acceptance` requires equal hashes and otherwise reports `physical_inputs_or_provider_mismatch`, so a different hash would make 8.1.3 impossible. The hash of the kernel and integrator sources is recorded in `provenance.execution.kernel_source_hash`. |
| D3 | 4.2 | Built-in backends are registered in code and imported lazily. The `solidpy.backends` entry-point group is only for third-party backends. |
| D4 | 4.1 | Only the `jax`, `jax-cuda12`, `gpu` and `all` extras are declared. `cupy`, `torch` and `warp` extras are added together with their backends. |
| D5 | 4.3 | Kernels take `xp` as `numpy` or `jax.numpy`. `array-api-compat` is introduced with the first third array namespace. |
| D6 | 5.4 | The blowdown cutoff is `ambient + 0.01 * max(peak - ambient, 0)`, where `peak` is the maximum pressure over the stored points of the burn stage and of the optional source-only segment (`Burn.py`, `solve_numerical_tail_off_regime`). Points of the blowdown stage are not included. |
| D7 | 4.6 | RocketCEA is only called while a `Propellant` is constructed and produces a pressure table; the solver evaluates the table. Propellants with a thermochemistry table are therefore packable. The CPU path already labels their results `unsupported_thermochemistry`. |
| D8 | 5.8, 7.1 | First round: reference workload W1 plus the golden corpus; history policies `metrics` and `full` (main channels). `uniform:N`, `decimated:N`, W2 to W5, the tier map and Tier 1 are deferred. |

Reference-path behaviour that the batched solver reproduces on purpose (the reference is the oracle):

* In the blowdown stage the breakpoint loop of `_integrate_stage` stops after the first segment, because
  `active` is all false and `stop_after_burnout` is true. An activation-table knot or `ignition_ramp_time`
  between burnout and the cutoff therefore truncates the blowdown and gives `blowdown_timeout`. A fix
  belongs to the scalar code and would change `physics_provider_hash`.
* `Robustness` applies `burn_rate_factor` by replacing `evaluate_burn_rate` on the propellant instance.
  The packer treats an instance-level override as an unsupported feature instead of reading `burn_rate_a`
  and `burn_rate_n`.

### 14.4 Status after the first implementation round (2026-10-03)

Phases 0 to 3 are implemented on `feat/gpu-batch-backend`, with the gate of Phase 3 **met**
(`docs/gpu_backend_benchmarks.md`): 11.8x the scalar reference on all cores of the same machine at 4,096 lanes on
the full corpus workload, 54.7x on the typical workload, in float64 on an 8 GiB consumer GPU.

Implemented: the backend registry and extras, the golden corpus, `ProblemBatch`, the geometry, propellant, nozzle
and right-hand-side kernels, the batched DOP853 with burnout and blowdown events, igniter, activation and ramp
sources with their stage breakpoints, tabulated burn rate and thermochemistry (cubic splines computed at pack time,
evaluated by bisection and Horner), canonical results with the reference `physics_provider_hash`, the
`cpu-reference`, `cpu-vectorized` and `jax` backends, `simulate_burn` with capability routing and fallback,
iteration-capped tiers (a result of the measurements: section 6.3 expected cost-sorted chunks, but sorting cannot
help when the per-iteration cost is flat; capping iterations and rerunning the unfinished lanes in smaller batches
does), and the benchmark suite. Lanes with callables, analytical tail-off, custom classes, instance overrides or a
burn rate that can be negative or not finite (the scalar code raises `ValueError` for those, which a compiled loop
cannot do) still run on the reference (or raise with `strict=True`).

Parity is checked per kernel at 1e-12 and per simulation with the versioned limits in
`solidpy/backends/_tolerances.py` (version 4). The stored reference is the inexact side of those comparisons, which
was verified against a refined run; each limit carries the cause in a comment. The whole-corpus test
(`pytest tests/test_batch_parity.py --runslow`) takes minutes and is skipped by default.

Known cost, not yet removed: the running reductions re-evaluate the model at every accepted point, about one in
fourteen of the evaluations a step makes; the derivative evaluation at the end of the step could be reused.

Not implemented, in the order they matter: the `uniform:N` and `decimated:N` history policies; the
heterogeneous CPU+GPU executor (the last tier is latency bound and suits spare CPU cores, benchmarks section
"What the numbers say"); Tier 1 physics; a data-center GPU measurement. Open point 14.2.1 (typical batch sizes)
now has a first answer: the speedup is large from about 2,000 lanes up and is below the reference under a few
hundred lanes, so `backend="auto"` keeps the reference for small batches.

### 14.5 Status after Phase 4a: robustness as lanes (2026-10-03)

Measured before planning (`tools/profile_workloads.py`, `benchmarks/results/offloaded_share.json`): on the scalar path the
burn is 99.0 % of W1 and 99.3 % of W3 (robustness), and 75.7 % of W2 (burn plus the advanced physics), where the thermal
ablation is 23.1 % and the structural, CFD and ignition proxies together under 1 %. The offloaded share is therefore 0.990,
0.993 and 0.757: W1 and W3 pass the 0.8 gate of section 7.1, W2 does not until the thermal ablation is batched.

Implemented, on the same branch and with `Burn.py`, `Grain.py` and `Propellant.py` untouched:

* A per-lane `burn_rate_factor` (`ProblemBatch.from_objects(..., burn_rate_factor=)`) that multiplies the whole burn rate,
  erosive term included, as `Robustness` applies a scenario's factor by replacing `evaluate_burn_rate` on the instance
  (which the packer refuses). The factor 1.0 changes nothing; the reference backend reproduces the override on a copy of
  the lane's propellant. The `physics_provider_hash` stays the scalar one, so it does not tell scenarios that differ
  only in this factor apart (item 3 of `docs/pending_cpu_reference_changes.md`); the factor is in the lane's
  `provenance["execution"]["scenario_inputs"]` and in the report's `scenario_factors`.
* `SimulationView` (`solidpy/batch/simulation_view.py`): the five things the detailed-ballistics post-processing reads from a
  `BurnSimulation`, built from a batch lane's canonical result. Its detailed ballistics equals that of a real simulation
  bit for bit.
* `solidpy.ensemble.run_robustness_ensemble(designs, ...)`: every (design, scenario) pair is a lane of one batch, solved on
  any backend with the full history; the detailed ballistics of each lane is built on the CPU. `run_robustness_analysis`
  gains optional `backend`, `device` and `workers` (without `backend` it is the unchanged scalar path, and `device` or
  `workers` alone raise). Through `cpu-reference` the report equals the scalar one bit for bit; through the batched
  backends it agrees within the limits of tolerances version 4, each set from the worst difference measured on eight
  designs (the causes are in the module).
* Measured on the GPU (`docs/gpu_backend_benchmarks.md`): 4,104 lanes in 30.0 s, 136.8 lanes/s, 13.1x the scalar path on
  12 processes. A third of the time is the detailed ballistics of each lane on one CPU core, which now sets the ceiling
  (the device alone does 246 lanes/s).

Not done in 4a, in the order they matter: the thermal ablation as lanes (done in 4b, section 14.6; W2 stays at 0.757 without it; the CPU runs a
scipy Radau solve per time step, so a batched version is a new integrator and its parity is by tolerance); overlapping or
parallelising the post-processing of the lanes (section 6.4); the structural response and `StructuralMonteCarlo` as
vector code; `xp=` for `surrogate_physics`; a `decimated:N` history so that thousands of lanes do not carry full
histories.

### 14.6 Status after Phase 4b: the thermal ablation as lanes (2026-10-03)

The plan of section 14.5 left the thermal ablation for last because the scalar code runs `solve_ivp(method="Radau")` once per
time step of the curve and a batched version was a new integrator whose parity could only be by tolerance. Measured
before starting (`tests/thermal_cases.py`, scipy 1.17.1): Radau takes 1.0 to 1.9 steps per time step of 0.01 to 0.03 s, the
wall temperatures of the scalar run are within 1e-8 of a run at `rtol=1e-12`, and `heat_load_kj_m2`, a trapezoid over the
solver's own steps, is within 3e-5 to 1.2e-3 of it. A different integrator would therefore have differed from the scalar
model by the scalar model's own quadrature error; the decision was to port scipy's controller instead.

Implemented, on the same branch and with `Burn.py`, `Grain.py` and `Propellant.py` untouched:

* `solidpy/batch/integrators/radau.py`: Radau IIA(5) for a batch, with the controller of `scipy.integrate.Radau` ported
  step for step (initial step, simplified Newton on the transformed system with its convergence tests, the error estimate and
  its second pass, the two-step step predictor, reuse of the Jacobian and of the factorization across steps including a
  factorization made for another step size, the dense-output Newton start). It returns the trapezoid and the maximum of an
  observed scalar over the accepted points, which is what the scalar code does with the heat flux. In 160 random walls it takes
  the same number of steps as `solve_ivp` and ends at the same state and integral (`tests/test_batch_radau.py`; in the 240-lane
  sweep those lanes come from, a call took 1 to 43 steps and 28 % had rejected steps). Systems are factored by the Thomas algorithm, since the Jacobian of a wall is
  tridiagonal: dense inverses cost 29 ms for 4,096 complex 32x32 systems on the RTX 4060 against 0.6 ms for the tridiagonal
  factorization, and 4,096 lanes of 18-cell walls went from 47 s to 7.7 s. The real and complex systems are factored in one pass
  because the cost of a pass is its sequential steps, not its width.
* `solidpy/batch/thermal.py`, `kernels/thermal.py`, `integrators/thermal_solver.py`: `ThermalBatch` packs the wall of each
  lane on the host with the scalar code's own helpers (`_wall_layers` and the finite-volume operator, extracted from
  `simulate_thermal_ablation` with identical results) and, per time step, the step length and the Bartz coefficient; the throat
  ablation is a sum over the series and is finished at pack time. The solver is a loop over the time steps around the batched
  Radau call and keeps the extremes the scalar code keeps.
* The Backend protocol gains optional services advertised by `Capabilities.services`; `thermal_ablation(batch, options)` is
  provided by `cpu-reference` (the scalar call per lane, optionally in a pool), `cpu-vectorized` and `jax` (one jit-compiled
  program, lanes padded to a power of two, wall cells to a multiple of 4, steps to a quarter-octave bucket).
* `solidpy.ensemble.simulate_thermal(batch, backend, ...)` routes like `simulate_burn`: lanes the backend cannot take and lanes
  whose integration did not finish are rerun on the scalar reference. `run_advanced_physics_ensemble(geometries, curves, ...)`
  runs the thermal ablation of many designs as one batch and the other advanced models on the CPU for each lane; pass
  `workers > 1` to run those models in a process pool while preserving lane order. The default remains serial.
  `simulate_advanced_physics` was split unchanged into the thermal call and `_advanced_after_thermal`.
* Tolerances version 5 adds `THERMAL_RTOL = 1e-9`. The worst difference to the scalar model over 14 fixed cases and 1,800
  random lanes on NumPy, JAX on the CPU device and JAX on the GPU is 1.4e-12, every metric included (`heat_load_kj_m2` too, since
  the steps are the same).

The review of the phase (`/code-review high`) found nine points, all handled: a gas with `gamma - 1` below 1e-6 (the scalar
code only clamps cp there, the wall problem becomes extremely stiff and the pivot-free factorization takes other steps than
scipy's LU, moving the heat load by 4e-6) is left to the scalar model by a lane feature; the Radau constants are copied
instead of imported from scipy's private module and compared with it in a test; each JAX launch is padded to its own
shape and the launch budget is linear in the wall cells; the scenario inputs of a curve are read by one helper; the
summary of `simulate_thermal` keeps the backends' execution records; and the docstrings say that a lane whose integration did
not finish is rerun on the reference even with `strict=True`, that a batch keeps references to its input objects, and that the
CPU models after the thermal one run serially by default or in a process pool with `workers > 1`.

Measured (`docs/gpu_backend_benchmarks.md`): offloaded share W1 0.990, **W2 0.988**, W3 0.994, so the gate of 0.8 is met on all
three workloads. The thermal ablation alone runs 36x the scalar model on 12 threads at 4,096 lanes of typical walls (15.7x on
walls up to 18 cells); the whole advanced physics runs 5.8x, limited by the structural, CFD, ignition and flight models that
stay on one CPU core (5.3 ms per lane, 90 % of the batched run).

The CPU post-processing of advanced physics and robustness ensembles now accepts `workers > 1` and runs in a bounded,
ordered process pool; measurements are in `docs/gpu_backend_benchmarks.md`. The general transient structural response over
arbitrary curves remains unimplemented. W4 peak-pressure sampling and `xp=` for `surrogate_physics` are recorded in section 14.7.
Couplings between the batched thermal code and the scalar model, and the 1e-3 quadrature error of the scalar heat load, are
in `docs/pending_cpu_reference_changes.md` (items 5 and 6).

### 14.7 Status after the first W4 structural batch (2026-10-03)

`StructuralMonteCarlo.run(n, backend=...)` now evaluates its synthetic peak-pressure structural responses in one batch on
`cpu-vectorized` or JAX. With `backend=None`, it keeps the existing scalar code path. The sample callbacks remain on the
host; a callback failure is recorded against its sample, while unsupported services, malformed outputs and non-finite
lane metrics fall back to the scalar response unless `strict=True`. Numeric outputs are checked against the scalar schema
and every metric must have one value per lane. Accelerated reports carry backend/device/version provenance and the batch
kernel source hash without changing `physics_provider_hash` across backends.

Parity tests cover steel and composite cases, configured/unavailable bolts, optional thermal service metrics, callback
failures, backend fallback and the scalar reference. Optional JAX CPU and GPU tests are present. The W4 harness is
`benchmarks/bench_structural_monte_carlo.py` (100,000 samples per design by default); a 100,000-sample GPU result is
still pending because the current verification host has no working NVIDIA driver or installed JAX package.

This first W4 kernel covers the synthetic `[0, peak, 0]` history used by `StructuralMonteCarlo`. General
`simulate_structural_response` curves, including time-varying thrust and lane-specific geometry, remain CPU work, as do
the post-thermal CFD, ignition and flight proxies and detailed-ballistics post-processing. The vectorized static
surrogate API now accepts `xp=`; its JAX numerical kernel is separately JIT-tested when the optional dependency exists.

### 14.8 CPU post-processing for W2 and W3 ensembles (2026-10-03)

`run_advanced_physics_ensemble(..., workers=N)` and `run_robustness_ensemble(..., workers=N)` now use an ordered,
bounded `ProcessPoolExecutor` for the CPU work after the batched thermal or burn solve. The default (`workers=None` or `1`)
remains serial. The worker count also continues to configure the `cpu-reference` solver when that backend is selected.
Work is grouped in small batches, with at most eight batches per process and 128 jobs overall in flight. The pool is
capped at 16 processes and the host's available CPU count. This bounds retained histories even when callers request a
larger worker count. The pool uses Python's `spawn` context so it cannot inherit initialized JAX or CUDA state.
Applications that call these APIs from a script must create the pool under an `if __name__ == "__main__":` guard, as
required by multiprocessing's spawn mode.

The advanced-physics path was checked against `simulate_advanced_physics`; the robustness path was checked against scalar
reports, including lane order and report assembly. Process startup is part of the measured call. On this host six workers
were slower at every tested size: for 256 W2 lanes, models took 1.333 s to 4.390 s and total time 2.238 s to 5.294 s;
for 1,024 W2 lanes, models took 5.367 s to 15.222 s and total time 7.591 s to 17.460 s. For 216 W3 lanes, detailed
ballistics took 0.557 s serially and 1.542 s with workers, with total runtime increasing from 11.194 s to 12.358 s.
Keep the default serial on this host; the process option remains available for workloads and hosts that benefit from it.
The benchmarks are recorded in `docs/gpu_backend_benchmarks.md` and `benchmarks/results/postprocess_*.json`.

This is parallel post-processing after a completed solve. The Phase 5 pipeline that overlaps accelerator solving,
post-processing and packing remains pending.

### 14.9 History output policies (2026-10-03)

`simulate_burn(..., history="decimated:N")` now returns at most N native accepted points, evenly spaced by accepted-point
index, with every canonical history channel. `history="uniform:N"` returns N uniformly spaced times, linearly interpolates
the seven ensemble channels (pressure, thrust, generated and nozzle flow, burn area, mean regression rate and generated-mass
integral), and includes diagnostics from the native grid: peak-thrust time, maximum absolute pressure derivative and the
axial mass-flux maximum and its location. Both policies are available on `cpu-reference`, `cpu-vectorized` and JAX; the
reference backend formats its scalar history, while the vectorized solvers retain accepted points up to `max_steps`.

JAX chunks history-producing launches against the existing 2 GiB history budget. It currently copies each launch's native
history to host memory for interpolation and axial diagnostics before discarding it; moving these operations onto the device
remains a memory and throughput optimization. A lane that reaches `max_steps` still follows the existing overflow and
reference-fallback behavior.

### 14.10 Heterogeneous burn scheduling (2026-10-03)

`simulate_burn` accepts an engine list such as `[("jax", "cuda:0"), ("cpu-reference", 6)]`. The new
`solidpy.executor.HeterogeneousExecutor` starts one feeder thread per backend/device, pulls compatible cost-sorted lanes
from a shared dynamic queue, returns lanes in their original order, and records backend selection and fallback reasons.
Capability misses are sent to the reference unless strict mode is requested. A backend exception, missing result or
`step_overflow` retries the affected chunk or lanes on the reference. Multiple fake devices were tested concurrently;
the result ordering and mass mapping were checked through the CPU-vectorized implementation.

The focused review suite passed (`69 passed, 1 skipped` across executor, ensemble, registry and batch-result tests).
This host has no JAX installation or GPU driver, so a real JAX multi-device run is still required before relying on
concurrent accelerator launches in production.

Remaining Phase 5 work: heterogeneous thermal services, process-pool reuse and native-thread/core reservation controls,
continuous refill of active device batches, and a chunk pipeline that overlaps W2/W3 CPU post-processing with device
solves. The scheduler dynamically shares queued chunks based on completion time; it does not perform a calibration pass.

## Appendix A. State vector and padded batch schema

State per lane (size `G + 7`):

| Index | Meaning |
|---|---|
| 0 | gas mass |
| 1 | thermal inventory (`gas_mass * temperature`) |
| 2 .. 2+G-1 | regression depth of each grain |
| 2+G | integral of generated mass flow |
| 3+G | integral of igniter mass flow |
| 4+G | integral of nozzle mass flow |
| 5+G | integral of unscaled thrust |
| 6+G | integral of `chamber_pressure * throat_area` |

`ProblemBatch` fields (leading axis = lane; `G` = padded grain axis; `K` = padded table axis):

| Group | Fields |
|---|---|
| Grains | `outer_radius[G]`, `inner_radius[G]`, `height[G]`, `ends_burn[G]`, `is_star[G]`, `n_points[G]`, `epsilon[G]`, `slot_fraction[G]`, `grain_valid[G]`, `burnout_depth[G]` (pack-time) |
| Motor | `chamber_volume`, `free_volume`, `throat_area`, `exit_area`, `expansion_ratio`, `nozzle_half_angle` (divergence factor), `propellant_volume` |
| Propellant | `density`, `gas_constant`, `burn_rate_model`, `a`, `n`, table `(x[K], coeffs[K,4], len)`, `erosive_k`, `erosive_alpha`, thermo mode scalar/table with spline coefficients for `c*`, `k`, `Tc`, `eta_c`, `eta_Cf`, `discharge_coefficient` |
| Environment | `ambient_pressure` |
| Sources | igniter mode and `(t[K], mdot[K])`, activation mode and `(t[K], a[K])`, `igniter_temperature`, `igniter_burn_time`, `ignition_ramp_time`, breakpoint list `(t[K], len)` |
| Solver | `rtol`, `atol`, `max_step`, `burn_timeout`, `tail_off_timeout`, `tail_off_method`, `max_steps` |
| Bookkeeping | `lane_id`, capability flags, original-object references for fallback |

## Appendix B. Parity test matrix (per backend)

| Test family | Reference | Compared quantities | Limit |
|---|---|---|---|
| Kernel unit | scalar methods | every kernel output | 1e-12 (NumPy) / 1e-10 (GPU) relative |
| Integrator | `solve_ivp` on the same right-hand side | initial step, first accepted steps, event times | initial step identical; event times 1e-6 relative; step-for-step identity not required (Appendix D) |
| Golden corpus, integrals | stored reference results | impulse, generated and nozzle mass, mass-balance residual | 1e-5 relative (measured max 4.8e-6), times 1e-6 relative |
| Golden corpus, grid-sampled maxima | stored reference results | peak pressure, peak thrust, peak mass flows | 2e-3 relative (measured max 8e-4), or tighter once peaks are defined on the continuous solution |
| Grid-sensitive diagnostics | `BurnSimulation` result | time of the thrust maximum, maximum pressure-rise rate on the native grid, axial-flux maximum, time and grain | 2e-3 relative on the value; the time of the maximum within one accepted step |
| Uniform-grid curves | linear interpolation of the reference accepted-step grid | each resampled channel | 1e-9 relative where the grid is identical |
| Acceptance policy | `evaluate_numerical_acceptance` | policy metrics and mass balance | policy limits (outer contract) |
| Random ensemble | `BurnSimulation` | distribution of deltas, failure set equality | max delta within limit; failure sets equal |
| Existing test files | parametrized by backend | each test's own assertions | unchanged |
| Cross-device | same backend, different device | golden corpus | envelope recorded in `_tolerances.py` |

## Appendix C. Glossary

* **Lane**: one motor in a batch.
* **Reference backend**: today's scalar code; the numerical source of truth.
* **Bucket**: a group of lanes sharing the padded grain count and a similar expected step count.
* **Capability**: a feature a backend declares it can execute for a lane.
* **Parity certificate**: a recorded, versioned statement that a backend passed the parity suite
  against the reference.

## Appendix D. Spike results (first implementation, 2026-10-02)

**What exists.** A separate git worktree and branch (`spike/gpu-batch-backend`, based on the revision
the reference path is pinned at) with `solidpy/batch/{kernels,dop853,engine,problem}.py`: closed-form
tubular and star geometry, power-law burn rate with the erosive term, scalar thermochemistry, nozzle
flow and thrust, the conservative right-hand side, a batched DOP853 with scipy's controller constants,
the 7th-order dense output, per-grain terminal burnout events (snap and restart as the scalar code
does) and the blowdown stage to the 1% cutoff, all as two `lax.while_loop`s in JAX (float64), with a
"metrics" history policy (running reductions only). Not covered yet: igniter and activation profiles,
tabulated burn rate and thermochemistry, uniform-grid curves, the axial-flux diagnostic.

**Corpus.** 154 motors drawn from a real ensemble (six size and geometry blocks, 1-24 grains), each
with the CPU reference result (`BurnSimulation`), 1.28 s per motor on average on one core.

**Kernel parity.** The batched right-hand side agrees with the scalar `_conservative_rhs` to <= 1.3e-13
relative at states sampled along four real trajectories (most derived quantities are bit-identical).
Initial step selection is identical to scipy's. A first version of this check ran in single precision
by accident (JAX defaults to float32) and showed 1e-7-1e-3 differences; x64 must be enabled by the
package itself, which it now is.

**Whole-simulation parity (float64, consumer GPU).** 152 of 154 lanes reproduce the CPU outcome
(`completed`, cutoff reached). The other two ran past the step budget (the CPU itself needed 11,176 and
4,187 accepted steps; the budget was 3,000-4,000) and were flagged by the overflow check, by design.
Relative differences over the 152 lanes:

| Quantity | max | 99th percentile | median |
|---|---|---|---|
| total impulse | 2.6e-6 | 2.3e-6 | 1.8e-7 |
| generated mass integral | 2.4e-6 | 2.0e-6 | 9e-10 |
| nozzle mass integral | 2.4e-6 | 2.0e-6 | 1.5e-9 |
| peak chamber pressure (grid-sampled) | 5.9e-4 | 3.2e-4 | 1.8e-6 |
| peak thrust (grid-sampled) | 6.3e-4 | 4.3e-4 | 2.0e-6 |
| peak generated / nozzle flow (grid-sampled) | 7.3e-4 / 5.9e-4 | 5.4e-4 / 3.2e-4 | 2e-6 |
| final time | 1.8e-6 | 1.6e-6 | 1.9e-9 |

Accepted-step counts coincide on only 12% of lanes (median difference 0.55%, maximum 6.8%). The
mechanism: summation order in the stage combinations differs from NumPy's, which perturbs the error
estimate by about 1e-8 relative (a near-cancelling quantity); the first steps of the ignition
transient amplify that to 1e-7 in time within a few steps; the step sequences then differ slightly for
the rest of the run. Consequence for the design: integrals agree to ~1e-6, grid-sampled maxima to
~1e-3, and that floor is a property of grid sampling (the reference path itself varies by the same
order between two CPU environments), which is why sections 8.1 and Appendix B were revised.

**Throughput (consumer 8 GB GPU, float64, whole loop compiled, metrics mode; the 154-motor corpus
tiled to the batch size, heavy-tail lanes included).**

| lanes | warm run | lanes/s | speedup vs 1 core | vs 6 cores |
|---|---|---|---|---|
| 154 | 3.1 s | 50 | 64x | 10.6x |
| 616 | 5.4 s | 113 | 145x | 24x |
| 1,232 | 8.4 s | 147 | 188x | 31x |
| 2,464 | 13.5 s | 183 | 234x | 39x |
| 4,928 | 21.6 s | 229 | 293x | 49x |
| 9,856 | 47.0 s | 210 | 268x | 45x |

The first call compiles for 10-60 s (growing with the batch). Throughput saturates around 4-10 k lanes.
In unsorted batches the loop runs 3,195 iterations (the heaviest lane) while the median lane needs about
500, so most of the device idles; **sorting lanes by expected cost gave 2.5x at equal chunk size**
(chunks of 308 lanes: 72 lanes/s unsorted, 184 lanes/s sorted, whose iteration counts were 374, 593,
771 and 3,195). A cost-based scheduler (and sending the heaviest lanes elsewhere) is therefore a
first-class component, not an optimization to postpone. In history-keeping mode a 2,464-lane batch
exhausted the 8 GB device (the buffer alone is 1.8 GB), while metrics mode fits 9,856 lanes: the history
policies of section 5.8 are necessary, not optional.

**Reading against the plan.** The measured saturation figure (about 45-49x against 6 physical cores,
before sorting) lies inside the 15-60x planning range of section 10.1. The Phase 3 criterion (>= 5x
against all CPU cores at 4,000 lanes or more) is met by a wide margin on this consumer GPU in double
precision. Results on a data-center GPU are pending.

**Data-center GPU, through PyTorch (the machine above, 24 GB Ampere-class card, float64).** Same
right-hand-side micro-benchmark as in 10.1 (no integrator): eager mode 2.2-2.5 ms per evaluation at any
batch size (launch-bound, host core of that machine); CUDA-graph replay 0.43 ms at 2,048 lanes, 1.16 ms
at 8,192 and 1.14 ms at 32,768 lanes, i.e. 0.035 microseconds per lane-evaluation at 32 k lanes versus
0.064 on the consumer card (about 1.8x), although the card's double-precision peak is ~20x higher: this
kernel family is latency- and memory-bound, not arithmetic-bound, so a larger GPU buys roughly 2x until the
kernels are fused. Below ~8 k lanes the consumer card was faster (0.62 ms vs 1.16 ms per evaluation at
8,192 lanes) because per-kernel latency is lower at its higher clock.
