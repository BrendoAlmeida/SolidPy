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
from typing import Any, Dict, List, Optional

import numpy as np

from ._protocol import BACKEND_API_VERSION, SUPPORTED, BackendUnavailable, Capabilities, SolveOptions, UnsupportedLane
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
    return jax.jit(lambda P, y0: solver.solve_burn_and_blowdown(driver, P, y0, config))


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
        if not full:
            return self.max_lanes
        per_lane = (max_steps + 1) * (grains + 8) * 8
        return max(1, min(self.max_lanes, HISTORY_BUDGET_BYTES // per_lane))

    def _solve_chunk(self, chunk, full: bool, max_steps: int):
        """Solve one padded launch and return the outputs as NumPy arrays for the real lanes only."""
        jax = self._jax
        jnp = jax.numpy
        lanes = len(chunk)
        padded = chunk.select(np.concatenate([np.arange(lanes), np.zeros(_bucket_lanes(lanes) - lanes, dtype=int)]))
        padded = padded.with_g_max(_bucket_grains(padded.g_max))
        with self._x64(), jax.default_device(self._device):
            if jnp.zeros(1).dtype != np.float64:
                raise RuntimeError("float64 is not available in this JAX build; the jax backend needs it")
            P = {name: jax.device_put(jnp.asarray(array), self._device) for name, array in padded.arrays.items()}
            y0 = jax.device_put(jnp.asarray(padded.initial_state()), self._device)
            out = _compiled(full, max_steps)(P, y0)
            out = jax.device_get(out)
        return padded.select(np.arange(lanes)), {name: np.asarray(v)[:lanes] if np.ndim(v) else np.asarray(v) for name, v in out.items()}

    def solve_burn(self, batch, options=None):
        """Solve ``batch`` on the device and return the canonical results; every lane must be supported."""
        from ..batch.assemble import assemble
        from ..batch.result import BatchResult

        options = SolveOptions() if options is None else options
        if not isinstance(options, SolveOptions):
            raise TypeError(f"options must be a SolveOptions, got {type(options).__name__}")
        problems = {lane: missing for lane, missing in enumerate(batch.unsupported(self.capabilities())) if missing}
        if problems:
            raise UnsupportedLane(
                "the jax backend cannot run lane(s) " + "; ".join(f"{lane}: {', '.join(m)}" for lane, m in problems.items())
            )
        full = options.history == "full"
        max_steps = options.max_steps or DEFAULT_MAX_STEPS[options.history]
        per_launch = self._lanes_per_launch(_bucket_grains(batch.g_max), full, max_steps)
        provenance = self.provenance()
        results: List[Dict[str, Any]] = []
        for start in range(0, len(batch), per_launch):
            chunk = batch.select(np.arange(start, min(start + per_launch, len(batch))))
            solved, out = self._solve_chunk(chunk, full, max_steps)
            results.extend(assemble(solved, out, options.history, provenance))
        return BatchResult(results, self.name, provenance)

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
