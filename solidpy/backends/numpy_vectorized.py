# -*- coding: utf-8 -*-
"""The batched burn model on NumPy: the same kernels and integrator an accelerator backend runs.

It needs nothing beyond the core install, so it is also the portable fallback where an accelerator library
cannot be imported, and the oracle for the kernel logic of the accelerator backends.
"""

from __future__ import annotations

import time
from typing import Any, Dict, List, Optional

from ._protocol import BACKEND_API_VERSION, SUPPORTED, Capabilities, SolveOptions, UnsupportedLane

#: Accepted points per lane when ``SolveOptions.max_steps`` is not given. The history buffer of ``"full"``
#: reserves them for every lane, so that policy defaults to far fewer.
DEFAULT_MAX_STEPS = {"metrics": 100000, "full": 5000}


class NumpyBackend:
    name = "cpu-vectorized"
    api_version = BACKEND_API_VERSION

    #: Lane features the kernels and the integrator reproduce. Everything else is routed elsewhere.
    SUPPORTED_FEATURES = (
        "tubular_grain", "star_grain", "ends_burn", "burn_rate_power_law", "erosive_burning", "thermo_scalar",
        "tail_off_numerical", "tail_off_omitted",
    )

    def __init__(self, device: Optional[str] = None):
        if device not in (None, "cpu"):
            raise ValueError(f"the cpu-vectorized backend only runs on 'cpu', not {device!r}")
        self.device = "cpu"
        #: Seconds of the last ``solve_burn``: ``solve_s`` (the integration) and ``assemble_s`` (result mappings).
        self.last_timings: Dict[str, float] = {}

    def capabilities(self) -> Capabilities:
        return Capabilities({f: SUPPORTED for f in self.SUPPORTED_FEATURES}, history_policies=("metrics", "full"))

    def devices(self) -> List[str]:
        return ["cpu"]

    def solve_burn(self, batch, options=None):
        """Solve ``batch`` and return the canonical results; every lane must be supported (``UnsupportedLane``)."""
        import numpy as np

        from ..batch.assemble import assemble
        from ..batch.integrators import solver
        from ..batch.result import BatchResult
        from ..batch.tiers import DEFAULT_TIERS, solve_in_tiers

        options = SolveOptions() if options is None else options
        if not isinstance(options, SolveOptions):
            raise TypeError(f"options must be a SolveOptions, got {type(options).__name__}")
        problems = {lane: missing for lane, missing in enumerate(batch.unsupported(self.capabilities())) if missing}
        if problems:
            raise UnsupportedLane(
                "the cpu-vectorized backend cannot run lane(s) "
                + "; ".join(f"{lane}: {', '.join(missing)}" for lane, missing in problems.items())
            )
        full = options.history == "full"
        config = solver.SolveConfig(keep_history=full, max_steps=options.max_steps or DEFAULT_MAX_STEPS[options.history])
        tiers = DEFAULT_TIERS if options.tiers is None else options.tiers

        def run(sub, cap):
            return solver.solve_burn_and_blowdown(solver.numpy_driver(), sub.namespace(np), sub.initial_state(), config, cap)

        start = time.perf_counter()
        out, info = solve_in_tiers(batch, run, tiers)
        solved = time.perf_counter()
        results = assemble(batch, out, options.history, self.provenance())
        self.last_timings = {"solve_s": solved - start, "assemble_s": time.perf_counter() - solved}
        return BatchResult(results, self.name, {**self.provenance(), "tiers": info})

    def provenance(self) -> Dict[str, Any]:
        from ..batch.assemble import library_versions

        return {
            "backend": self.name,
            "backend_api_version": self.api_version,
            "device": "cpu",
            "dtype": "float64",
            "library_versions": library_versions(),
        }
