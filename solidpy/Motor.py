# -*- coding: utf-8 -*-

_author_ = ""
_copyright_ = "MIT"
_license_ = ""

import numpy as np


class Motor:
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
    ):
        if not isinstance(grains, (list, tuple)):
            grains = [grains]

        self.grain_separation = grain_separation

        # Legacy mode: single grain replicated grain_number times
        if grain_number is not None and len(grains) == 1:
            self.grains = [grains[0]] * grain_number
        else:
            self.grains = list(grains)

        self.grain_number = len(self.grains)

        self.evaluate_chamber_length(chamber_length)
        self.chamber_area = np.pi * chamber_inner_radius**2
        self.nozzle_throat_area = np.pi * nozzle_throat_radius**2
        self.nozzle_exit_area = np.pi * nozzle_exit_radius**2
        self.nozzle_angle = nozzle_angle
        self.expansion_ratio = self.nozzle_exit_area / self.nozzle_throat_area
        self.evaluate_chamber_volume()
        self.evaluate_propellant_volume()
        self.evaluate_free_volume()
        self.evaluate_total_burn_area()
        self.evaluate_Kn()

    @property
    def grain(self):
        """Backward-compat accessor: returns the first grain."""
        return self.grains[0]

    def evaluate_chamber_length(self, chamber_length):
        if chamber_length is None:
            total_grain_height = sum(g.initial_height for g in self.grains)
            total_gaps = self.grain_separation * max(len(self.grains) - 1, 0)
            self.chamber_length = total_grain_height + total_gaps
        else:
            self.chamber_length = chamber_length

    def evaluate_chamber_volume(self):
        self.chamber_volume = self.chamber_area * self.chamber_length
        return self.chamber_volume

    def evaluate_propellant_volume(self):
        self.propellant_volume = sum(g.volume for g in self.grains)
        return self.propellant_volume

    def evaluate_free_volume(self):
        self.free_volume = (
            self.evaluate_chamber_volume() - self.evaluate_propellant_volume()
        )
        return self.free_volume

    def evaluate_Kn(self):
        self.Kn = self.total_burn_area / self.nozzle_throat_area
        return self.Kn

    def evaluate_total_burn_area(self):
        self.total_burn_area = sum(g.burn_area for g in self.grains)
