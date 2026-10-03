# -*- coding: utf-8 -*-
"""The part of a ``BurnSimulation`` that the post-processing reads, for a lane a batch backend solved.

``build_detailed_ballistics`` and everything built on it read five things from the simulation they are given:
``result``, ``motor``, ``propellant``, ``environment_pressure`` and ``evaluate_burn_area_activation``.
``BurnSimulation`` solves in its constructor, so a lane that a batch backend already solved cannot be wrapped in
one; this view carries the same five.
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any, Dict, Mapping

from ..Burn import BurnSimulation


class SimulationView:
    """A solved lane that looks like a ``BurnSimulation`` to the detailed-ballistics post-processing."""

    def __init__(self, result: Dict[str, Any], motor, propellant, environment_pressure: float,
                 settings: Mapping[str, Any]):
        self.result = result
        self.motor = motor
        self.propellant = propellant
        self.environment_pressure = environment_pressure
        # the activation rule needs only these two settings; it is the scalar method itself, not a copy of it
        self._activation_inputs = SimpleNamespace(
            burn_area_activation=settings["burn_area_activation"], ignition_ramp_time=settings["ignition_ramp_time"],
        )

    @classmethod
    def from_lane(cls, batch, lane: int, result: Dict[str, Any]) -> "SimulationView":
        """The view of lane ``lane`` of ``batch`` with the canonical ``result`` a backend returned for it."""
        return cls(result, batch.motors[lane], batch.propellants[lane],
                   float(batch.arrays["ambient_pressure"][lane]), batch.settings[lane])

    def evaluate_burn_area_activation(self, time, regressed_length):
        """``BurnSimulation.evaluate_burn_area_activation`` for this lane's activation setting."""
        return BurnSimulation.evaluate_burn_area_activation(self._activation_inputs, time, regressed_length)
