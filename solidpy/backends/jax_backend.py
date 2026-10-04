# -*- coding: utf-8 -*-
"""The batched burn model on JAX: the whole solve is one jit-compiled program.

The kernels and the integrator are the ones the NumPy backend runs; only the loop (``lax.while_loop``), the
conditional (``lax.cond``) and the scatter into the history buffers differ. Float64 is enabled for the duration
of each solve with JAX's scoped switch, never as global state. JAX is imported when a ``JaxBackend`` is created,
so ``import solidpy`` and ``solidpy.backends.available()`` never load it.

Compiling takes 10 to 60 s the first time a batch shape is seen. Lanes are therefore padded to a power-of-two
count, and the grain axis and every table axis to a bucket, so a sweep reuses one program; a persistent cache can
be enabled with ``JAX_COMPILATION_CACHE_DIR``.
"""

from __future__ import annotations

import functools
import time
from typing import Any, Dict, List, Optional

import numpy as np

from ._protocol import (
    BACKEND_API_VERSION, HISTORY_POLICY_TEMPLATES, SUPPORTED, BackendUnavailable, Capabilities, SolveOptions,
    parse_history_policy, refused_lanes, unsupported_lane_error,
)
from .numpy_vectorized import DEFAULT_MAX_STEPS, NumpyBackend

INSTALL_HINT = 'pip install "solidpy[jax-cuda12]"   (or "solidpy[jax]" for CPU only)'
#: Grain-axis buckets (a batch is padded up to the next one) and the smallest lane bucket.
GRAIN_BUCKETS = (4, 8, 16, 24, 32)
MIN_LANE_BUCKET = 64
#: Bytes the transient accepted-step history of one launch may take on the device.
HISTORY_BUDGET_BYTES = 2 * 1024**3
DEFAULT_MAX_LANES = 8192


def _bucket_lanes(count: int, minimum: int = MIN_LANE_BUCKET) -> int:
    """The compiled lane count a launch of ``count`` lanes is padded to: a power of two, at least ``minimum``."""
    return max(minimum, 1 << (count - 1).bit_length())


def _floor_lanes(count: int) -> int:
    """The largest power of two not above ``count`` (a launch is padded up to a power of two, so it must not start above)."""
    return 1 << (max(int(count), 1).bit_length() - 1)


def _bucket_nodes(count: int) -> int:
    """The wall-cell axis a thermal launch is padded to: a multiple of 4 (the cost is linear in it), at least 4."""
    return max(4, -(-count // 4) * 4)


def _bucket_grains(count: int) -> int:
    for size in GRAIN_BUCKETS:
        if count <= size:
            return size
    return count


def _driver():
    """The loop, conditional and scatter of the compiled programs."""
    import jax.numpy as jnp
    from jax import lax

    from ..batch.integrators import solver

    def store(buffer, rows, cols, values):
        return buffer.at[rows, cols].set(values)

    return solver.Driver(
        jnp,
        lambda cond, body, carry: lax.while_loop(cond, body, carry),
        lambda any_lane, if_true, if_false, operand: lax.cond(any_lane, if_true, if_false, operand),
        lambda count, body, state: lax.fori_loop(0, count, body, state),
        store,
        lambda body, init, xs, reverse=False: lax.scan(body, init, xs, reverse=reverse),
    )


@functools.lru_cache(maxsize=None)
def _compiled_thermal():
    """The jitted wall integration of a ``ThermalBatch`` (jax caches per input shape)."""
    import jax

    from ..batch.integrators import thermal_solver

    driver = _driver()
    return jax.jit(lambda P: thermal_solver.solve_thermal(driver, P))


@functools.lru_cache(maxsize=16)
def _compiled_advanced_physics_proxies():
    """Jitted transient structural, CFD and ignition kernels, cached by JAX for each array shape."""
    import jax

    from ..batch.kernels.advanced_physics import advanced_physics_proxies

    return jax.jit(lambda arrays: advanced_physics_proxies(arrays, jax.numpy))


@functools.lru_cache(maxsize=16)
def _compiled_detailed_ballistics():
    """Jitted detailed-ballistics kernel, cached for each padded history and grain shape."""
    import jax

    from ..batch.kernels.detailed_ballistics import detailed_ballistics

    return jax.jit(lambda arrays: detailed_ballistics(arrays, jax.numpy))


@functools.lru_cache(maxsize=None)
def _compiled(keep_history: bool, max_steps: int):
    """The jitted solve for one history policy and step budget (jax caches per input shape)."""
    import jax

    from ..batch.integrators import solver

    driver = _driver()
    config = solver.SolveConfig(keep_history=keep_history, max_steps=max_steps)
    # the iteration cap is an argument, not a static value, so the tiers of a batch share one compiled program
    return jax.jit(lambda P, y0, cap: solver.solve_burn_and_blowdown(driver, P, y0, config, cap))


class JaxBackend:
    name = "jax"
    api_version = BACKEND_API_VERSION

    def __init__(self, device: Optional[str] = None, max_lanes: int = DEFAULT_MAX_LANES):
        try:
            import jax
        except Exception as exc:  # a jaxlib that cannot load (e.g. no AVX) raises more than ImportError
            raise BackendUnavailable(f"the jax backend could not import JAX ({exc}). Install it with: {INSTALL_HINT}") from exc
        self._jax = jax
        self.max_lanes = int(max_lanes)
        self._device = self._select_device(device)
        self.device = self._label(self._device)
        self._structural_compiled_cache = {}
        #: Seconds of the last ``solve_burn``: ``device_s`` (padding, transfer, compute, copy back) and
        #: ``assemble_s`` (result mappings on the host). The first call of a shape includes compilation.
        self.last_timings: Dict[str, float] = {}
        self._device_s = 0.0

    # -- devices ---------------------------------------------------------------------------------------------
    @staticmethod
    def _label(device) -> str:
        return "cpu" if device.platform == "cpu" else f"cuda:{device.id}"

    def _select_device(self, device: Optional[str]):
        jax = self._jax
        if device is None:
            return jax.devices()[0]
        kind, _, index = device.partition(":")
        platform = {"cpu": "cpu", "cuda": "gpu", "gpu": "gpu"}.get(kind)
        if platform is None:
            raise ValueError(f"unknown device {device!r}; use 'cpu', 'cuda:N' or 'gpu:N'")
        try:
            candidates = jax.devices(platform)
        except RuntimeError as exc:
            raise BackendUnavailable(f"JAX has no {platform} device ({exc}). Install it with: {INSTALL_HINT}") from exc
        position = int(index) if index else 0
        if position >= len(candidates):
            raise BackendUnavailable(f"JAX sees {len(candidates)} {platform} device(s), not {position + 1}")
        return candidates[position]

    def devices(self) -> List[str]:
        out = []
        for platform in ("gpu", "cpu"):
            try:
                out.extend(self._label(d) for d in self._jax.devices(platform))
            except RuntimeError:
                continue
        return out

    def capabilities(self) -> Capabilities:
        return Capabilities({f: SUPPORTED for f in NumpyBackend.SUPPORTED_FEATURES},
                            history_policies=HISTORY_POLICY_TEMPLATES,
                            services=("thermal_ablation", "structural_response", "advanced_physics_proxies",
                                      "detailed_ballistics"))

    def _x64(self):
        jax = self._jax
        if hasattr(jax, "enable_x64"):
            return jax.enable_x64(True)
        from jax.experimental import enable_x64  # older JAX

        return enable_x64()

    # -- solving ---------------------------------------------------------------------------------------------
    def _lanes_per_launch(self, grains: int, full: bool, max_steps: int) -> int:
        """Lanes per launch, a power of two so that padding to the bucket never exceeds the budget.

        The stored history takes ``(max_steps + 1) x (grains + 9)`` doubles per lane (the times and the state),
        twice over because the loop carry is held in and out.
        """
        if not full:
            return _floor_lanes(self.max_lanes)
        per_lane = 2 * (max_steps + 1) * (grains + 9) * 8
        return _floor_lanes(min(self.max_lanes, max(HISTORY_BUDGET_BYTES // per_lane, 1)))  # may be below 64

    def _run(self, sub, full: bool, max_steps: int, cap: Optional[int], floor: int = MIN_LANE_BUCKET):
        """Solve one padded launch and return the outputs as NumPy arrays for the real lanes only.

        ``sub`` already has its grain axis padded to a bucket; the lane axis is padded here.
        """
        jax = self._jax
        jnp = jax.numpy
        lanes = len(sub)
        padded = sub.select(np.concatenate([np.arange(lanes), np.zeros(_bucket_lanes(lanes, floor) - lanes, dtype=int)]))
        started = time.perf_counter()
        with self._x64(), jax.default_device(self._device):
            if jnp.zeros(1).dtype != np.float64:
                raise RuntimeError("float64 is not available in this JAX build; the jax backend needs it")
            P = {name: jax.device_put(jnp.asarray(array), self._device) for name, array in padded.arrays.items()}
            y0 = jax.device_put(jnp.asarray(padded.initial_state()), self._device)
            limit = jnp.asarray(np.iinfo(np.int64).max if cap is None else cap)
            out = jax.device_get(_compiled(full, max_steps)(P, y0, limit))
        self._device_s += time.perf_counter() - started
        return {name: np.asarray(v)[:lanes] for name, v in out.items() if np.ndim(v)}

    def solve_burn(self, batch, options=None):
        """Solve ``batch`` on the device and return the canonical results; every lane must be supported."""
        from ..batch.assemble import assemble
        from ..batch.result import BatchResult
        from ..batch.tiers import DEFAULT_TIERS, solve_in_tiers

        options = SolveOptions() if options is None else options
        if not isinstance(options, SolveOptions):
            raise TypeError(f"options must be a SolveOptions, got {type(options).__name__}")
        refused = refused_lanes(batch, self.capabilities())
        if refused:
            raise unsupported_lane_error(self.name, refused)
        batch = batch.with_table_buckets()  # table widths are part of the compiled shapes
        history_kind, _ = parse_history_policy(options.history)
        stores_history = history_kind != "metrics"
        max_steps = options.max_steps or DEFAULT_MAX_STEPS[history_kind]
        # Output policies need accepted points; capped tiers would allocate and discard those buffers.
        tiers = options.tiers if options.tiers is not None else (() if stores_history else DEFAULT_TIERS)
        grains = _bucket_grains(batch.g_max)
        per_launch = self._lanes_per_launch(grains, stores_history, max_steps)
        provenance = self.provenance()
        results: List[Dict[str, Any]] = []
        launches: List[Any] = []
        self._device_s, assemble_s = 0.0, 0.0
        for start in range(0, len(batch), per_launch):
            chunk = batch.select(np.arange(start, min(start + per_launch, len(batch)))).with_g_max(grains)
            floor = min(MIN_LANE_BUCKET, per_launch)  # a launch the budget keeps small is not padded back up
            out, info = solve_in_tiers(
                chunk, lambda sub, cap: self._run(sub, stores_history, max_steps, cap, floor), tiers
            )
            begin = time.perf_counter()
            results.extend(assemble(chunk, out, options.history, provenance))
            assemble_s += time.perf_counter() - begin
            launches.append(info)
        self.last_timings = {"device_s": self._device_s, "assemble_s": assemble_s}
        return BatchResult(results, self.name, {**provenance, "tiers": launches})

    # -- thermal ablation ------------------------------------------------------------------------------------
    def _thermal_lanes_per_launch(self, nodes: int, steps: int) -> int:
        """Lanes per thermal launch: the per-step series dominate the memory (four copies of two doubles per step), the
        solver state is about 200 doubles per wall cell and lane."""
        per_lane = 4 * steps * 2 * 8 + 200 * nodes * 8
        return _floor_lanes(min(self.max_lanes, max(HISTORY_BUDGET_BYTES // per_lane, 1)))

    def _run_thermal(self, padded):
        jax = self._jax
        jnp = jax.numpy
        started = time.perf_counter()
        with self._x64(), jax.default_device(self._device):
            if jnp.zeros(1).dtype != np.float64:
                raise RuntimeError("float64 is not available in this JAX build; the jax backend needs it")
            P = {name: jax.device_put(jnp.asarray(array), self._device) for name, array in padded.arrays.items()}
            out = jax.device_get(_compiled_thermal()(P))
        self._device_s += time.perf_counter() - started
        return {name: np.asarray(v) for name, v in out.items() if np.ndim(v)}

    def thermal_ablation(self, batch, options=None):
        """Integrate the walls of ``batch`` (a ``ThermalBatch``) on the device; every lane must be supported.

        Lanes are padded to a power-of-two count, the wall cells to a multiple of 4 and the time steps to a bucket, each
        launch to its own shape, so that a sweep reuses compiled programs and a batch sorted by length (as
        ``simulate_thermal`` sorts it) does not pay the longest lane in every launch. The result of a lane whose
        integration did not finish is ``None`` (listed in ``execution["failed_lanes"]``). ``options`` only has to be valid.
        """
        from ..batch.integrators import radau, thermal_solver
        from ..batch.problem import table_bucket
        from ..batch.result import BatchResult
        from ..batch.thermal import assemble

        options = SolveOptions() if options is None else options
        if not isinstance(options, SolveOptions):
            raise TypeError(f"options must be a SolveOptions, got {type(options).__name__}")
        refused = refused_lanes(batch, self.capabilities())
        if refused:
            raise unsupported_lane_error(self.name, refused)
        # the budget is set by the whole batch's shape (the most a launch can need); each launch is then padded to its own
        per_launch = self._thermal_lanes_per_launch(_bucket_nodes(batch.n_max), table_bucket(batch.t_max))
        results: List[Any] = []
        steps_taken = attempts = 0
        self._device_s, assemble_s = 0.0, 0.0
        for start in range(0, len(batch), per_launch):
            chunk = batch.select(np.arange(start, min(start + per_launch, len(batch))))  # trimmed to this launch
            chunk = chunk.with_padding(_bucket_nodes(chunk.n_max), table_bucket(chunk.t_max))
            lanes = len(chunk)
            floor = min(MIN_LANE_BUCKET, per_launch)  # a launch the budget keeps small is not padded back up
            padded = chunk.select(np.concatenate([np.arange(lanes), np.zeros(_bucket_lanes(lanes, floor) - lanes, dtype=int)]),
                                  trim=False)
            out = self._run_thermal(padded)
            out = {name: value[:lanes] for name, value in out.items()}
            begin = time.perf_counter()
            results.extend(assemble(chunk, out))
            assemble_s += time.perf_counter() - begin
            steps_taken += int(out["steps"].sum())
            attempts += int(out["attempts"].sum())
        self.last_timings = {"device_s": self._device_s, "assemble_s": assemble_s}
        return BatchResult(results, self.name, {
            **self.provenance(), "service": "thermal_ablation", "integrator": "radau-iia5", "rtol": thermal_solver.RTOL,
            "atol": thermal_solver.ATOL, "max_attempts": radau.MAX_ATTEMPTS,
            "failed_lanes": [i for i, r in enumerate(results) if r is None],
            "radau_steps": steps_taken, "radau_attempts": attempts,
        })

    def structural_response(
        self, geometry, chamber_pressure_pa, casing_material, casing_strength_factor=1.0, *,
        bolt_count=0, bolt_diameter_m=0.0, bolt_strength_mpa=0.0,
        closure_bolts_applicable=True, thermal=None,
    ):
        """Evaluate structural peak-pressure lanes on the selected JAX device."""
        from ..batch.kernels.structural_response import structural_response_vectorized
        from ..Multiphysics import _closure_bolt_configuration

        jax = self._jax
        jnp = jax.numpy
        start = time.perf_counter()
        wall_temperature = (thermal or {}).get(
            "simulation.advanced.thermal.casing_inner_wall_temp_c"
        )
        geometry_key = tuple(getattr(geometry, name) for name in (
            "motor_length_m", "motor_inner_diameter_m", "casing_wall_thickness_m", "dry_mass_kg",
        ))
        material_key = tuple(getattr(casing_material, name) for name in (
            "density_kg_m3", "modulus_gpa", "yield_strength_mpa", "resolved_allowable_stress_mpa",
            "resolved_ultimate_strength_mpa", "poisson_ratio", "max_service_temp_c", "material_family",
        ))
        cache_key = (
            geometry_key, material_key, float(casing_strength_factor), int(bolt_count),
            float(bolt_diameter_m), float(bolt_strength_mpa), bool(closure_bolts_applicable),
            wall_temperature, tuple(np.shape(chamber_pressure_pa)),
        )
        with self._x64(), jax.default_device(self._device):
            pressures = jax.device_put(jnp.asarray(chamber_pressure_pa, dtype=jnp.float64), self._device)
            compiled = self._structural_compiled_cache.get(cache_key)
            if compiled is None:
                def numerical(values):
                    output = structural_response_vectorized(
                        geometry, values, casing_material, casing_strength_factor,
                        bolt_count=bolt_count, bolt_diameter_m=bolt_diameter_m,
                        bolt_strength_mpa=bolt_strength_mpa,
                        closure_bolts_applicable=closure_bolts_applicable,
                        thermal=(None if wall_temperature is None else {
                            "simulation.advanced.thermal.casing_inner_wall_temp_c": wall_temperature,
                        }), xp=jnp,
                    )
                    return {name: value for name, value in output.items()
                            if not isinstance(value, (str, type(None)))}

                compiled = jax.jit(numerical)
                self._structural_compiled_cache[cache_key] = compiled
                if len(self._structural_compiled_cache) > 32:
                    self._structural_compiled_cache.pop(next(iter(self._structural_compiled_cache)))
            output = jax.device_get(compiled(pressures))
        self._device_s = time.perf_counter() - start
        status, applicability, reason = _closure_bolt_configuration(
            bolt_count, bolt_diameter_m, bolt_strength_mpa, closure_bolts_applicable,
        )
        output = {name: np.asarray(value) for name, value in output.items()}
        output.update({
            "simulation.advanced.structural.closure_bolt_status": status,
            "simulation.advanced.structural.closure_bolt_applicability": applicability,
            "simulation.advanced.structural.closure_bolt_reason": reason,
            "simulation.advanced.structural.thermal_service_status":
                "computed" if wall_temperature is not None
                else "not_modeled",
        })
        if status != "configured":
            output.update({
                "simulation.advanced.structural.closure_bolt_shear_safety_factor": None,
                "simulation.advanced.structural.closure_bolt_bearing_safety_factor": None,
                "simulation.advanced.structural.closure_bolt_shear_stress_mpa": None,
                "simulation.advanced.structural.closure_bolt_bearing_stress_mpa": None,
            })
        if wall_temperature is None:
            output.update({
                "simulation.advanced.structural.thermal_service_margin": None,
                "simulation.advanced.structural.thermoelastic_margin": None,
            })
        return output

    def advanced_physics_proxies(self, batch, options=None):
        """Evaluate transient advanced-physics proxy lanes on the selected JAX device."""
        from ..batch.advanced_physics import assemble_advanced_physics_proxies
        from ..batch.result import BatchResult

        jax = self._jax
        jnp = jax.numpy
        with self._x64(), jax.default_device(self._device):
            arrays = batch.namespace(jnp)
            placed = jax.tree_util.tree_map(lambda value: jax.device_put(value, self._device), arrays)
            output = jax.device_get(_compiled_advanced_physics_proxies()(placed))
        return BatchResult(
            assemble_advanced_physics_proxies(batch, output), self.name,
            {**self.provenance(), "service": "advanced_physics_proxies"},
        )

    def detailed_ballistics(self, batch, options=None):
        """Build detailed histories on the selected JAX device; unsupported lanes return ``None``."""
        from ..batch.detailed_ballistics import assemble_detailed_ballistics
        from ..batch.result import BatchResult

        jax = self._jax
        with self._x64(), jax.default_device(self._device):
            arrays = batch.namespace(jax.numpy)
            placed = jax.tree_util.tree_map(lambda value: jax.device_put(value, self._device), arrays)
            output = jax.device_get(_compiled_detailed_ballistics()(placed))
        return BatchResult(
            assemble_detailed_ballistics(batch, output), self.name,
            {**self.provenance(), "service": "detailed_ballistics"},
        )

    def provenance(self) -> Dict[str, Any]:
        from ..batch.assemble import library_versions

        jax = self._jax
        versions = library_versions()
        versions["jax"] = jax.__version__
        try:
            import jaxlib

            versions["jaxlib"] = jaxlib.__version__
        except Exception:
            pass
        device = self._device
        return {
            "backend": self.name,
            "backend_api_version": self.api_version,
            "device": self.device,
            "device_name": getattr(device, "device_kind", str(device)),
            "dtype": "float64",
            "library_versions": versions,
        }
