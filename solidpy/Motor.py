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
        dry_mass_kg=None,
        dry_center_of_mass_position_m=None,
    ):
        if not isinstance(grains, (list, tuple)):
            grains = [grains]

        # Explicit geometry validation. The previous code either crashed deep
        # inside solve_ivp with a misleading `math domain error`, fell back to
        # an unflagged degenerate motor (all-zero outputs, Isp=NaN), or
        # returned a subsonic exit Mach via fsolve without signaling the
        # error. Surface the cause at construction time with a clear message.
        if nozzle_exit_radius is not None and nozzle_throat_radius is not None:
            if nozzle_exit_radius <= nozzle_throat_radius:
                raise ValueError(
                    f"nozzle exit radius ({nozzle_exit_radius*1e3:.2f} mm) must be "
                    f"larger than throat radius ({nozzle_throat_radius*1e3:.2f} mm); "
                    "a convergent-divergent nozzle cannot have exit ≤ throat."
                )
        for grain in grains if isinstance(grains, (list, tuple)) else [grains]:
            if grain.outer_radius is not None and chamber_inner_radius is not None:
                if grain.outer_radius > chamber_inner_radius:
                    raise ValueError(
                        f"grain outer radius ({grain.outer_radius*1e3:.2f} mm) is "
                        f"larger than chamber inner radius ({chamber_inner_radius*1e3:.2f} mm); "
                        "the grain would not fit inside the chamber."
                    )

        self.grain_separation = grain_separation
        self.dry_mass_kg = dry_mass_kg
        self.dry_center_of_mass_position_m = dry_center_of_mass_position_m

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
        if self.chamber_volume < self.propellant_volume:
            raise ValueError(
                f"chamber volume ({self.chamber_volume*1e6:.1f} cm³) is smaller than "
                f"propellant volume ({self.propellant_volume*1e6:.1f} cm³); "
                "the grain would not fit inside the chamber."
            )
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
