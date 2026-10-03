# -*- coding: utf-8 -*-
"""The batched burn model on NumPy: the same kernels and integrator an accelerator backend runs.

It needs nothing beyond the core install, so it is also the portable fallback where an accelerator library
cannot be imported, and the oracle for the kernel logic of the accelerator backends.
"""

from __future__ import annotations

import time
from typing import Any, Dict, List, Optional

from ._protocol import BACKEND_API_VERSION, SUPPORTED, Capabilities, SolveOptions, refused_lanes, unsupported_lane_error

#: Accepted points per lane when ``SolveOptions.max_steps`` is not given. The history buffer of ``"full"``
#: reserves them for every lane, so that policy defaults to far fewer.
DEFAULT_MAX_STEPS = {"metrics": 100000, "full": 5000}


class NumpyBackend:
    name = "cpu-vectorized"
    api_version = BACKEND_API_VERSION

    #: Lane features the kernels and the integrator reproduce. Everything else is routed elsewhere.
    SUPPORTED_FEATURES = (
        "tubular_grain", "star_grain", "ends_burn", "burn_rate_power_law", "burn_rate_table", "erosive_burning",
        "thermo_scalar", "thermo_table",
        "tail_off_numerical", "tail_off_omitted", "igniter_scalar", "igniter_table", "activation_scalar",
        "activation_table", "ignition_ramp",
    )

    def __init__(self, device: Optional[str] = None):
        if device not in (None, "cpu"):
            raise ValueError(f"the cpu-vectorized backend only runs on 'cpu', not {device!r}")
        self.device = "cpu"
        #: Seconds of the last ``solve_burn``: ``solve_s`` (the integration) and ``assemble_s`` (result mappings).
        self.last_timings: Dict[str, float] = {}

    def capabilities(self) -> Capabilities:
        return Capabilities({f: SUPPORTED for f in self.SUPPORTED_FEATURES}, history_policies=("metrics", "full"),
                            services=("thermal_ablation",))

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
        refused = refused_lanes(batch, self.capabilities())
        if refused:
            raise unsupported_lane_error(self.name, refused)
        full = options.history == "full"
        config = solver.SolveConfig(keep_history=full, max_steps=options.max_steps or DEFAULT_MAX_STEPS[options.history])
        # a full history is for inspection, not throughput: capped tiers would allocate and discard its buffers
        tiers = options.tiers if options.tiers is not None else (() if full else DEFAULT_TIERS)

        def run(sub, cap):
            return solver.solve_burn_and_blowdown(solver.numpy_driver(), sub.namespace(np), sub.initial_state(), config, cap)

        start = time.perf_counter()
        out, info = solve_in_tiers(batch, run, tiers)
        solved = time.perf_counter()
        provenance = self.provenance()
        results = assemble(batch, out, options.history, provenance)
        self.last_timings = {"solve_s": solved - start, "assemble_s": time.perf_counter() - solved}
        return BatchResult(results, self.name, {**provenance, "tiers": info})

    def thermal_ablation(self, batch, options=None):
        """Integrate the walls of ``batch`` (a ``ThermalBatch``); every lane must be supported (``UnsupportedLane``).

        The result of a lane whose integration did not finish is ``None`` (listed in ``execution["failed_lanes"]``);
        ``solidpy.ensemble.simulate_thermal`` reruns those on the scalar reference. ``options`` only has to be valid.
        """
        import numpy as np

        from ..batch.integrators import radau, solver, thermal_solver
        from ..batch.result import BatchResult
        from ..batch.thermal import assemble

        options = SolveOptions() if options is None else options
        if not isinstance(options, SolveOptions):
            raise TypeError(f"options must be a SolveOptions, got {type(options).__name__}")
        refused = refused_lanes(batch, self.capabilities())
        if refused:
            raise unsupported_lane_error(self.name, refused)
        start = time.perf_counter()
        out = thermal_solver.solve_thermal(solver.numpy_driver(), batch.namespace(np))
        solved = time.perf_counter()
        results = assemble(batch, out)
        self.last_timings = {"solve_s": solved - start, "assemble_s": time.perf_counter() - solved}
        return BatchResult(results, self.name, {
            **self.provenance(), "service": "thermal_ablation", "integrator": "radau-iia5",
            "rtol": thermal_solver.RTOL, "atol": thermal_solver.ATOL, "max_attempts": radau.MAX_ATTEMPTS,
            "failed_lanes": [i for i, r in enumerate(results) if r is None],
            "radau_steps": int(np.sum(out["steps"])), "radau_attempts": int(np.sum(out["attempts"])),
        })

    def provenance(self) -> Dict[str, Any]:
        from ..batch.assemble import library_versions

        return {
            "backend": self.name,
            "backend_api_version": self.api_version,
            "device": "cpu",
            "dtype": "float64",
            "library_versions": library_versions(),
        }
