# -*- coding: utf-8 -*-
"""The scalar reference as a backend: ``BurnSimulation`` run once per lane.

Results are exactly what ``BurnSimulation.result`` holds, with nothing added, so this backend is the
numerical source of truth every other backend is compared with.
"""

from __future__ import annotations

import copy
import pickle
import platform
from concurrent.futures import ProcessPoolExecutor
from typing import Any, Dict, List, Optional

from ._protocol import BACKEND_API_VERSION, SUPPORTED, Capabilities, SolveOptions


class _ScaledBurnRate:
    """``propellant.evaluate_burn_rate`` times a factor, the way ``Robustness`` applies a scenario's burn rate factor."""

    def __init__(self, original, factor):
        self.original = original
        self.factor = factor

    def __call__(self, chamber_pressure, port_mass_flux=0.0):
        return self.factor * self.original(chamber_pressure, port_mass_flux)


def _run_lane(arguments):
    """Solve one lane. Module level so a process pool can pickle it."""
    from ..Burn import BurnSimulation

    motor, propellant, environment, settings, burn_rate_factor = arguments
    if burn_rate_factor != 1.0:
        propellant = copy.deepcopy(propellant)  # the lane's propellant may be shared with lanes of another factor
        propellant.evaluate_burn_rate = _ScaledBurnRate(propellant.evaluate_burn_rate, burn_rate_factor)
    return BurnSimulation(motor.grains[0], motor, propellant, environment, **settings).result


def _picklable(job):
    try:
        pickle.dumps(job)
    except Exception:
        return False
    return True


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
        # the result always carries the full adaptive history, which contains everything "metrics" asks for
        return Capabilities({feature: SUPPORTED for feature in FEATURES}, history_policies=("metrics", "full"))

    def devices(self) -> List[str]:
        return ["cpu"]

    def solve_burn(self, batch, options=None):
        """Solve every lane of ``batch`` with the scalar solver; ``options.workers`` > 1 uses a process pool.

        The result always carries the full adaptive history, so ``options.history`` only has to be valid.
        Lanes that cannot be sent to a worker (a lambda igniter, an instance-level method override) are solved
        in this process, so a pool never loses the batch to a pickling error. An exception raised by the
        scalar solver for a lane propagates, as it does when calling ``BurnSimulation`` directly.
        """
        from ..batch.result import BatchResult

        options = SolveOptions() if options is None else options
        if not isinstance(options, SolveOptions):
            raise TypeError(f"options must be a SolveOptions, got {type(options).__name__}")
        factors = [float(f) for f in batch.arrays["burn_rate_factor"]]
        jobs = list(zip(batch.motors, batch.propellants, batch.environments, batch.settings, factors))
        results = [None] * len(jobs)
        pooled = []
        if options.workers and options.workers > 1 and len(jobs) > 1:
            pooled = [i for i, job in enumerate(jobs) if _picklable(job)]
        if pooled:
            with ProcessPoolExecutor(max_workers=min(options.workers, len(pooled))) as pool:
                for i, result in zip(pooled, pool.map(_run_lane, [jobs[i] for i in pooled], chunksize=1)):
                    results[i] = result
        for i, job in enumerate(jobs):
            if results[i] is None:
                results[i] = _run_lane(job)
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
