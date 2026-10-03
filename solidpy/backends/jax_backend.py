# -*- coding: utf-8 -*-
"""The batched burn model on JAX: the whole solve is one jit-compiled program.

The kernels and the integrator are the ones the NumPy backend runs; only the loop (``lax.while_loop``), the
conditional (``lax.cond``) and the scatter into the history buffers differ. Float64 is enabled for the duration
of each solve with JAX's scoped switch, never as global state. JAX is imported when a ``JaxBackend`` is created,
so ``import solidpy`` and ``solidpy.backends.available()`` never load it.

Compiling takes 10 to 60 s the first time a batch shape is seen. Lanes are therefore padded to a power-of-two
count and the grain axis to a bucket, so a sweep reuses one program; a persistent cache can be enabled with
``JAX_COMPILATION_CACHE_DIR``.
"""

from __future__ import annotations

import functools
import time
from typing import Any, Dict, List, Optional

import numpy as np

from ._protocol import (
    BACKEND_API_VERSION, SUPPORTED, BackendUnavailable, Capabilities, SolveOptions, refused_lanes, unsupported_lane_error,
)
from .numpy_vectorized import DEFAULT_MAX_STEPS, NumpyBackend

INSTALL_HINT = 'pip install "solidpy[jax-cuda12]"   (or "solidpy[jax]" for CPU only)'
#: Grain-axis buckets (a batch is padded up to the next one) and the smallest lane bucket.
GRAIN_BUCKETS = (4, 8, 16, 24, 32)
MIN_LANE_BUCKET = 64
#: Bytes the stored history of one launch may take on the device (``history="full"``).
HISTORY_BUDGET_BYTES = 2 * 1024**3
DEFAULT_MAX_LANES = 8192


def _bucket_lanes(count: int) -> int:
    return max(MIN_LANE_BUCKET, 1 << (count - 1).bit_length())


def _floor_lanes(count: int) -> int:
    """The largest lane bucket not above ``count`` (a launch is padded up to a bucket, so it must not start above)."""
    return max(MIN_LANE_BUCKET, 1 << (max(int(count), 1).bit_length() - 1))


def _bucket_grains(count: int) -> int:
    for size in GRAIN_BUCKETS:
        if count <= size:
            return size
    return count


@functools.lru_cache(maxsize=None)
def _compiled(keep_history: bool, max_steps: int):
    """The jitted solve for one history policy and step budget (jax caches per input shape)."""
    import jax
    import jax.numpy as jnp
    from jax import lax

    from ..batch.integrators import solver

    def store(buffer, rows, cols, values):
        return buffer.at[rows, cols].set(values)

    driver = solver.Driver(
        jnp,
        lambda cond, body, carry: lax.while_loop(cond, body, carry),
        lambda any_lane, if_true, if_false, operand: lax.cond(any_lane, if_true, if_false, operand),
        lambda count, body, state: lax.fori_loop(0, count, body, state),
        store,
    )
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
        return Capabilities({f: SUPPORTED for f in NumpyBackend.SUPPORTED_FEATURES}, history_policies=("metrics", "full"))

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
        return _floor_lanes(min(self.max_lanes, max(HISTORY_BUDGET_BYTES // per_lane, 1)))

    def _run(self, sub, full: bool, max_steps: int, cap: Optional[int]):
        """Solve one padded launch and return the outputs as NumPy arrays for the real lanes only.

        ``sub`` already has its grain axis padded to a bucket; the lane axis is padded here.
        """
        jax = self._jax
        jnp = jax.numpy
        lanes = len(sub)
        padded = sub.select(np.concatenate([np.arange(lanes), np.zeros(_bucket_lanes(lanes) - lanes, dtype=int)]))
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
        full = options.history == "full"
        max_steps = options.max_steps or DEFAULT_MAX_STEPS[options.history]
        # a full history is for inspection, not throughput: capped tiers would allocate and discard its buffers
        tiers = options.tiers if options.tiers is not None else (() if full else DEFAULT_TIERS)
        grains = _bucket_grains(batch.g_max)
        per_launch = self._lanes_per_launch(grains, full, max_steps)
        provenance = self.provenance()
        results: List[Dict[str, Any]] = []
        launches: List[Any] = []
        self._device_s, assemble_s = 0.0, 0.0
        for start in range(0, len(batch), per_launch):
            chunk = batch.select(np.arange(start, min(start + per_launch, len(batch)))).with_g_max(grains)
            out, info = solve_in_tiers(chunk, lambda sub, cap: self._run(sub, full, max_steps, cap), tiers)
            begin = time.perf_counter()
            results.extend(assemble(chunk, out, options.history, provenance))
            assemble_s += time.perf_counter() - begin
            launches.append(info)
        self.last_timings = {"device_s": self._device_s, "assemble_s": assemble_s}
        return BatchResult(results, self.name, {**provenance, "tiers": launches})

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
