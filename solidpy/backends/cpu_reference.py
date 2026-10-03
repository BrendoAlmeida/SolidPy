# -*- coding: utf-8 -*-
"""The scalar reference as a backend: ``BurnSimulation`` run once per lane.

Results are exactly what ``BurnSimulation.result`` holds, with nothing added, so this backend is the
numerical source of truth every other backend is compared with.
"""

from __future__ import annotations

import copy
import platform
from contextlib import contextmanager
from concurrent.futures import ProcessPoolExecutor
from typing import Any, Dict, List, Optional

from ._protocol import (
    BACKEND_API_VERSION, HISTORY_POLICY_TEMPLATES, SUPPORTED, Capabilities, SolveOptions, parse_history_policy,
)
from .._parallel import process_worker_count, process_worker_initializer, safe_process_context, spawn_pickle_safe


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


def _run_thermal_lane(lane):
    """Solve the wall of one thermal lane with the scalar model. Module level so a process pool can pickle it."""
    from ..Multiphysics import simulate_thermal_ablation

    return simulate_thermal_ablation(
        lane["geometry"], lane["curve"], casing_material=lane["casing_material"], nozzle_material=lane["nozzle_material"],
        flame_temp_k=lane["flame_temp_k"], r_specific=lane["r_specific"], gamma=lane["gamma"],
        initial_temperature_k=lane["initial_temperature_k"], liner_thickness_factor=lane["liner_thickness_factor"],
    )


def _picklable(job):
    return spawn_pickle_safe(job)


class ReferenceBackend:
    name = "cpu-reference"
    api_version = BACKEND_API_VERSION

    def __init__(self, device: Optional[str] = None):
        if device not in (None, "cpu"):
            raise ValueError(f"the cpu-reference backend only runs on 'cpu', not {device!r}")
        self.device = "cpu"

    def capabilities(self) -> Capabilities:
        from ..batch.problem import FEATURES
        from ..batch.thermal import THERMAL_FEATURES

        # the scalar code runs every lane, including custom classes and instance-level overrides
        # the result always carries the full adaptive history, which contains everything "metrics" asks for
        return Capabilities({feature: SUPPORTED for feature in FEATURES + THERMAL_FEATURES},
                            history_policies=HISTORY_POLICY_TEMPLATES,
                            services=("thermal_ablation", "structural_response"))

    def devices(self) -> List[str]:
        return ["cpu"]

    @contextmanager
    def process_pool(self, options, task_count):
        """Open one reusable spawn pool for a sequence of reference chunks, if requested."""
        if options.workers and options.workers > 1 and task_count > 1:
            with ProcessPoolExecutor(
                max_workers=process_worker_count(options.workers, task_count),
                mp_context=safe_process_context(), initializer=process_worker_initializer,
            ) as pool:
                yield pool
        else:
            yield None

    @staticmethod
    def _burn_results(jobs, process_pool):
        results = [None] * len(jobs)
        pooled = [i for i, job in enumerate(jobs) if process_pool is not None and _picklable(job)]
        if pooled:
            for i, result in zip(pooled, process_pool.map(_run_lane, [jobs[i] for i in pooled], chunksize=1)):
                results[i] = result
        for i, job in enumerate(jobs):
            if results[i] is None:
                results[i] = _run_lane(job)
        return results

    @staticmethod
    def _thermal_results(lanes, process_pool):
        results = [None] * len(lanes)
        pooled = [i for i, lane in enumerate(lanes) if process_pool is not None and _picklable(lane)]
        if pooled:
            for i, result in zip(
                pooled, process_pool.map(_run_thermal_lane, [lanes[i] for i in pooled], chunksize=1)
            ):
                results[i] = result
        for i, lane in enumerate(lanes):
            if results[i] is None:
                results[i] = _run_thermal_lane(lane)
        return results

    def solve_burn(self, batch, options=None, *, process_pool=None):
        """Solve every lane of ``batch`` with the scalar solver; ``options.workers`` > 1 uses a process pool.

        The scalar solver creates its full adaptive history. ``decimated:N`` and ``uniform:N`` are formatted from it
        after the solve; ``metrics`` and ``full`` retain their existing reference results.
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
        if process_pool is None:
            with self.process_pool(options, len(jobs)) as pool:
                results = self._burn_results(jobs, pool)
        else:
            results = self._burn_results(jobs, process_pool)
        history_kind, _ = parse_history_policy(options.history)
        if history_kind in ("decimated", "uniform"):
            from ..batch.assemble import _history_for_policy

            for motor, result in zip(batch.motors, results):
                if result.get("history") is not None:
                    result["history"] = _history_for_policy(result["history"], motor, options.history)
        return BatchResult(results, self.name, self.provenance())

    def thermal_ablation(self, batch, options=None, *, process_pool=None):
        """The scalar ``simulate_thermal_ablation`` once per lane; ``options.workers`` > 1 uses a process pool.

        The results are what the scalar function returns, so this backend is the oracle of the batched ones. An
        exception the scalar model raises for a lane propagates, as when calling it directly.
        """
        from ..batch.result import BatchResult

        options = SolveOptions() if options is None else options
        if not isinstance(options, SolveOptions):
            raise TypeError(f"options must be a SolveOptions, got {type(options).__name__}")
        lanes = list(batch.lanes)
        if process_pool is None:
            with self.process_pool(options, len(lanes)) as pool:
                results = self._thermal_results(lanes, pool)
        else:
            results = self._thermal_results(lanes, process_pool)
        return BatchResult(results, self.name, {**self.provenance(), "service": "thermal_ablation"})

    def structural_response(
        self, geometry, chamber_pressure_pa, casing_material, casing_strength_factor=1.0, *,
        bolt_count=0, bolt_diameter_m=0.0, bolt_strength_mpa=0.0,
        closure_bolts_applicable=True, thermal=None,
    ):
        """Evaluate peak-pressure lanes through the scalar structural reference."""
        import numpy as np

        from ..Multiphysics import simulate_structural_response

        responses = []
        for pressure in np.asarray(chamber_pressure_pa, dtype=float):
            curve = {
                "time_s": np.asarray([0.0, 0.001, 1.0]),
                "thrust_n": np.zeros(3),
                "chamber_pressure_pa": np.asarray([0.0, float(pressure), 0.0]),
            }
            responses.append(simulate_structural_response(
                geometry, curve, thermal, casing_material=casing_material,
                casing_strength_factor=casing_strength_factor, bolt_count=bolt_count,
                bolt_diameter_m=bolt_diameter_m, bolt_strength_mpa=bolt_strength_mpa,
                closure_bolts_applicable=closure_bolts_applicable,
            ))
        if not responses:
            return {}
        result = {}
        for name in responses[0]:
            values = [response[name] for response in responses]
            result[name] = None if values[0] is None else (
                values[0] if isinstance(values[0], str) else np.asarray(values, dtype=float)
            )
        return result

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
