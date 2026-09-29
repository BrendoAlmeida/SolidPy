# -*- coding: utf-8 -*-

_author_ = ""
_copyright_ = "MIT"
_license_ = ""

import math
from numbers import Integral, Real

import numpy as np


class Motor:
    """Physical motor geometry and connected gas control volume.

    Grain positions are axial start coordinates measured from the nozzle-side
    stack reference. Positive coordinates point away from the nozzle.
    """

    def __init__(
        self,
        grains,
        grain_number=None,
        chamber_inner_radius=None,
        nozzle_throat_radius=None,
        nozzle_exit_radius=None,
        nozzle_angle=None,
        chamber_length=None,
        grain_separation=0.0,
        dry_mass_kg=None,
        dry_center_of_mass_position_m=None,
        *,
        connected_chamber_volume_m3=None,
    ):
        if not isinstance(grains, (list, tuple)):
            grains = [grains]
        if not grains:
            raise ValueError("at least one grain is required")
        chamber_inner_radius = self._finite_dimension(
            "chamber_inner_radius", chamber_inner_radius
        )
        nozzle_throat_radius = self._finite_dimension(
            "nozzle_throat_radius", nozzle_throat_radius
        )
        nozzle_exit_radius = self._finite_dimension(
            "nozzle_exit_radius", nozzle_exit_radius
        )
        if nozzle_exit_radius <= nozzle_throat_radius:
            raise ValueError("nozzle exit radius must be larger than throat radius")

        self.grain_separation = self._finite_dimension(
            "grain_separation", grain_separation, allow_zero=True
        )
        self.dry_mass_kg = dry_mass_kg
        self.dry_center_of_mass_position_m = dry_center_of_mass_position_m
        self.connected_chamber_volume_m3 = (
            None
            if connected_chamber_volume_m3 is None
            else self._finite_dimension(
                "connected_chamber_volume_m3", connected_chamber_volume_m3
            )
        )

        if grain_number is not None:
            if (
                isinstance(grain_number, bool)
                or not isinstance(grain_number, Integral)
                or grain_number < 1
            ):
                raise ValueError("grain_number must be a positive integer")
        if grain_number is not None and len(grains) == 1:
            self.grains = [grains[0]] * grain_number
        else:
            self.grains = list(grains)

        self.grain_number = len(self.grains)
        for index, grain in enumerate(self.grains):
            outer_radius = self._finite_dimension(
                f"grains[{index}].outer_radius", grain.outer_radius
            )
            inner_radius = self._finite_dimension(
                f"grains[{index}].initial_inner_radius",
                grain.initial_inner_radius,
                allow_zero=True,
            )
            if inner_radius >= outer_radius:
                raise ValueError(f"grains[{index}] must have positive web thickness")
            if outer_radius > chamber_inner_radius:
                raise ValueError(f"grains[{index}] outer radius exceeds chamber inner radius")
            height = self._finite_dimension(
                f"grains[{index}].initial_height", grain.initial_height
            )
            volume = self._finite_dimension(f"grains[{index}].volume", grain.volume)
            envelope_volume = math.pi * outer_radius * outer_radius * height
            if volume > envelope_volume * (1 + 16 * np.finfo(float).eps):
                raise ValueError(f"grains[{index}] volume exceeds its physical envelope")
        self.evaluate_chamber_length(chamber_length)
        self.chamber_area = self._finite_dimension(
            "chamber_area", math.pi * chamber_inner_radius * chamber_inner_radius
        )
        self.nozzle_throat_area = self._finite_dimension(
            "nozzle_throat_area", math.pi * nozzle_throat_radius * nozzle_throat_radius
        )
        self.nozzle_exit_area = self._finite_dimension(
            "nozzle_exit_area", math.pi * nozzle_exit_radius * nozzle_exit_radius
        )
        self.nozzle_angle = nozzle_angle
        self.expansion_ratio = self.nozzle_exit_area / self.nozzle_throat_area
        self.evaluate_free_volume()
        self.evaluate_total_burn_area()
        self.evaluate_Kn()

    @property
    def grain(self):
        """Backward-compat accessor: returns the first grain."""
        return self.grains[0]

    @staticmethod
    def _finite_dimension(name, value, allow_zero=False):
        domain = "non-negative" if allow_zero else "positive"
        if isinstance(value, bool) or not isinstance(value, Real):
            raise ValueError(f"{name} must be a finite {domain} real number")
        value = float(value)
        if not math.isfinite(value) or value < 0.0 or (not allow_zero and value == 0.0):
            raise ValueError(f"{name} must be a finite {domain} real number")
        return value

    def evaluate_chamber_length(self, chamber_length):
        heights = [
            self._finite_dimension(f"grains[{index}].initial_height", grain.initial_height)
            for index, grain in enumerate(self.grains)
        ]
        positions = []
        position = 0.0
        for height in heights:
            positions.append(position)
            position += height + self.grain_separation
        stack_length = math.fsum(heights) + self.grain_separation * (len(heights) - 1)
        stack_length = self._finite_dimension("grain stack length", stack_length)
        if chamber_length is None:
            self.chamber_length = stack_length
        else:
            self.chamber_length = self._finite_dimension("chamber_length", chamber_length)
        tolerance = 16 * np.finfo(float).eps * max(stack_length, self.chamber_length)
        if stack_length > self.chamber_length + tolerance:
            raise ValueError("grain stack, including separations, exceeds physical chamber_length")
        self.grain_axial_positions_m = tuple(positions)

    def evaluate_chamber_volume(self):
        volume = self.connected_chamber_volume_m3
        if volume is None:
            volume = self.chamber_area * self.chamber_length
        self.chamber_volume = self._finite_dimension("chamber_volume", volume)
        return self.chamber_volume

    def evaluate_propellant_volume(self):
        self.propellant_volume = self._finite_dimension(
            "propellant_volume",
            sum(
                self._finite_dimension(f"grains[{index}].volume", grain.volume)
                for index, grain in enumerate(self.grains)
            ),
        )
        return self.propellant_volume

    def evaluate_free_volume(self):
        self.free_volume = self._finite_dimension(
            "initial free volume",
            self.evaluate_chamber_volume() - self.evaluate_propellant_volume(),
        )
        return self.free_volume

    def evaluate_Kn(self):
        self.Kn = self.total_burn_area / self.nozzle_throat_area
        return self.Kn

    def evaluate_total_burn_area(self):
        self.total_burn_area = sum(g.burn_area for g in self.grains)
