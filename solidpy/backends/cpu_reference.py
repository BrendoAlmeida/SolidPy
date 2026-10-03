# -*- coding: utf-8 -*-
"""The scalar reference as a backend: ``BurnSimulation`` run once per lane.

Results are exactly what ``BurnSimulation.result`` holds, with nothing added, so this backend is the
numerical source of truth every other backend is compared with.
"""

from __future__ import annotations

import platform
from concurrent.futures import ProcessPoolExecutor
from typing import Any, Dict, List, Optional

from ._protocol import BACKEND_API_VERSION, SUPPORTED, Capabilities


def _run_lane(arguments):
    """Solve one lane. Module level so a process pool can pickle it."""
    from ..Burn import BurnSimulation

    motor, propellant, environment, settings = arguments
    return BurnSimulation(motor.grains[0], motor, propellant, environment, **settings).result


class ReferenceBackend:
    name = "cpu-reference"
    api_version = BACKEND_API_VERSION

    def __init__(self, device: Optional[str] = None):
        if device not in (None, "cpu"):
            raise ValueError(f"the cpu-reference backend only runs on 'cpu', not {device!r}")
        self.device = "cpu"

    def capabilities(self) -> Capabilities:
        from ..batch.problem import FEATURES

        # the scalar code runs every lane, including custom classes and instance-level overrides
        return Capabilities({feature: SUPPORTED for feature in FEATURES}, history_policies=("full",))

    def devices(self) -> List[str]:
        return ["cpu"]

    def solve_burn(self, batch, options=None):
        """Solve every lane of ``batch`` with the scalar solver; ``options.workers`` > 1 uses a process pool.

        The result always carries the full adaptive history: the ``history`` option does not apply.
        """
        from ..batch.result import BatchResult

        workers = getattr(options, "workers", None)
        jobs = list(zip(batch.motors, batch.propellants, batch.environments, batch.settings))
        if workers and workers > 1 and len(jobs) > 1:
            with ProcessPoolExecutor(max_workers=min(workers, len(jobs))) as pool:
                results = list(pool.map(_run_lane, jobs, chunksize=1))
        else:
            results = [_run_lane(job) for job in jobs]
        return BatchResult(results, self.name, self.provenance())

    def provenance(self) -> Dict[str, Any]:
        import numpy
        import scipy

        return {
            "backend": self.name,
            "backend_api_version": self.api_version,
            "device": "cpu",
            "dtype": "float64",
            "library_versions": {"python": platform.python_version(), "numpy": numpy.__version__,
                                 "scipy": scipy.__version__},
        }
