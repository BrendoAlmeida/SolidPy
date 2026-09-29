# -*- coding: utf-8 -*-

_author_ = "Caio Eduardo dos Santos de Souza, João Lemes Gribel Soares, Thais Silva Melo, Tiago Mariotto Lucio"
_copyright_ = "MIT"
_license_ = "x"

import math

import numpy as np


class Grain:
    def __init__(
        self,
        outer_radius,
        initial_inner_radius,
        initial_height=None,
        mass=None,
        geometry="tubular",
        n_points=6,
        epsilon=0.2,
        slot_fraction=0.8,
        ends_burn=False,
    ):
        """Propellant grain with tubular or star (slotted-cylinder) geometry.

        Star grain parameters (ignored for tubular geometry):
            n_points     — number of radial slots (star arms).
            epsilon      — half-angle of each slot in radians.
            slot_fraction— fraction of the web occupied by slots;
                           1.0 means slots extend to the outer case.

        ends_burn:
            When True the two end (transversal) faces are treated as inhibited
            (e.g. by a liner/spacer) and do not regress: their area contribution
            is set to zero and the grain height stays at ``initial_height``
            throughout the burn. Only radial burn fronts keep regressing.
            Star grains use fixed-angle slots with inhibited radial sidewalls.
            Default False preserves the existing
            BATES-style behaviour where both end faces burn and the grain
            shortens axially.
        """
        self.outer_radius = outer_radius
        self.initial_inner_radius = initial_inner_radius
        # Geometry sanity: a tubular/star grain needs an outer radius strictly
        # larger than the core, otherwise the web is zero-or-negative and the
        # motor degenerates silently (burn_area=0, Isp=NaN) without ever
        # raising. Surface the cause at construction time with a clear message.
        if outer_radius <= initial_inner_radius:
            raise ValueError(
                f"grain outer radius ({outer_radius*1e3:.2f} mm) must be larger "
                f"than the core radius ({initial_inner_radius*1e3:.2f} mm); "
                "web thickness would be zero or negative."
            )
        self.ends_burn = bool(ends_burn)
        self.inner_radius = initial_inner_radius
        self.mass = mass
        self.n_points = max(int(n_points), 1)
        self.epsilon = float(epsilon)
        self.slot_fraction = min(max(float(slot_fraction), 0.0), 1.0)
        if geometry == "star" and (
            not math.isfinite(self.epsilon)
            or self.epsilon <= 0.0
            or self.n_points * self.epsilon >= np.pi
        ):
            raise ValueError("star slots require finite epsilon > 0 and n_points * epsilon < pi")
        self.evaluate_grain_initial_height(initial_height)
        self.height = self.initial_height
        self.geometry = geometry
        self.evaluate_grain_geometry()
        self.evaluate_grain_volume()
        self.density = self.evaluate_grain_density()

    def evaluate_grain_initial_height(self, initial_height):
        if initial_height is None:
            self.initial_height = 3 * self.outer_radius + self.inner_radius
        else:
            self.initial_height = initial_height

    def evaluate_grain_density(self):
        if self.mass is not None:
            density = self.mass / self.volume
            return density
        return None

    def evaluate_grain_geometry(self):
        if self.geometry == "tubular":
            self.evaluate_tubular_burn_area(0, update_state=True)
        elif self.geometry == "star":
            self.evaluate_star_burn_area(0, update_state=True)
        else:
            print("Not a valid geometry type")

    def calculate_tubular_geometry(self, regressed_length):
        regressed_length = max(float(regressed_length), 0.0)
        web_thickness = self.outer_radius - self.initial_inner_radius
        burned_through = (
            regressed_length >= web_thickness
            or regressed_length >= self.initial_height / 2
        )

        inner_radius = min(
            self.initial_inner_radius + regressed_length, self.outer_radius
        )
        if self.ends_burn:
            # Inhibited end faces do not regress axially: the grain keeps its
            # full length and only the cylindrical bore widens.
            height = self.initial_height
            burned_through = regressed_length >= web_thickness
        else:
            height = max(self.initial_height - 2 * regressed_length, 0.0)
        if burned_through:
            return height, inner_radius, 0.0

        longitudinal_area = 2 * np.pi * inner_radius * height
        if self.ends_burn:
            return height, inner_radius, longitudinal_area

        transversal_area = 2 * np.pi * (self.outer_radius**2 - inner_radius**2)
        burn_area = transversal_area + longitudinal_area
        return height, inner_radius, burn_area

    def evaluate_tubular_burn_area(self, regressed_length, update_state=False):
        height, inner_radius, burn_area = self.calculate_tubular_geometry(
            regressed_length
        )
        if update_state:
            self.height = height
            self.inner_radius = inner_radius
            self.burn_area = burn_area

        return burn_area

    def calculate_star_geometry(self, regressed_length):
        """Return the fixed-angle radial-front slotted-cylinder geometry.

        ``fixed_angle_radial_front_v1`` keeps slot angular boundaries fixed:
        radial slot walls are inhibited and only bore and slot-floor arcs
        regress. This approximation does not model isotropic star regression.
        Its burn area equals the negative derivative of remaining volume.
        """
        w = max(float(regressed_length), 0.0)
        N = self.n_points
        eps = self.epsilon
        Ri = self.initial_inner_radius
        Ro = self.outer_radius
        L0 = self.initial_height
        web = Ro - Ri
        Rs = Ri + self.slot_fraction * web  # slot floor initial radius
        w_floor = max(Ro - Rs, 0.0)         # regression when floor reaches case

        burned_through = w >= web or (not self.ends_burn and w >= L0 / 2)
        if burned_through:
            height = L0 if self.ends_burn else max(L0 - 2 * w, 0.0)
            return height, min(Ri + w, Ro), 0.0

        # Inhibited end faces preserve the initial axial length.
        h = L0 if self.ends_burn else L0 - 2 * w
        r_bore = Ri + w

        if w < w_floor:
            # Phase 1: slot floor still within the grain
            r_floor = Rs + w
            P_lat = (2 * np.pi - 2 * N * eps) * r_bore + 2 * N * eps * r_floor
            A_end = np.pi * (Ro ** 2 - r_bore ** 2) - N * eps * (r_floor ** 2 - r_bore ** 2)
        else:
            # Phase 2: slot floor merged with outer case
            P_lat = (2 * np.pi - 2 * N * eps) * r_bore
            A_end = (np.pi - N * eps) * (Ro ** 2 - r_bore ** 2)

        end_faces_area = 0.0 if self.ends_burn else 2.0 * A_end
        burn_area = max(P_lat * h + end_faces_area, 0.0)
        return h, r_bore, burn_area

    def evaluate_star_burn_area(self, regressed_length, update_state=False):
        h, r_bore, burn_area = self.calculate_star_geometry(regressed_length)
        if update_state:
            self.height = h
            self.inner_radius = r_bore
            self.burn_area = burn_area
        return burn_area

    def evaluate_burn_area(self, regressed_length, update_state=False):
        """Dispatch to the correct geometry model."""
        if self.geometry == "star":
            return self.evaluate_star_burn_area(regressed_length, update_state)
        return self.evaluate_tubular_burn_area(regressed_length, update_state)

    def evaluate_port_area(self, regressed_length):
        """Cross-sectional area of the gas port at the given regression depth.

        Used for erosive burning (Lenoir-Robillard port mass flux G = ṁ/A_port).
        For tubular grains this is π*r_bore². For star grains the N slots add
        additional area.
        """
        w = max(float(regressed_length), 0.0)
        Ri = self.initial_inner_radius
        Ro = self.outer_radius
        r_bore = min(Ri + w, Ro)

        if self.geometry != "star":
            return np.pi * r_bore ** 2

        N = self.n_points
        eps = self.epsilon
        Rs = Ri + self.slot_fraction * (Ro - Ri)
        w_floor = max(Ro - Rs, 0.0)
        r_floor = min(Rs + w, Ro) if w < w_floor else Ro
        slot_area = N * eps * max(r_floor ** 2 - r_bore ** 2, 0.0)
        return np.pi * r_bore ** 2 + slot_area

    @property
    def geometry_model(self):
        """Identify the regression approximation used by this grain."""
        return "fixed_angle_radial_front_v1" if self.geometry == "star" else "tubular_radial_axial_v1"

    @property
    def burnout_regression_m(self):
        """Regression distance at radial or axial exhaustion, in metres."""
        web = self.outer_radius - self.initial_inner_radius
        return web if self.ends_burn else min(web, self.initial_height / 2.0)

    def calculate_remaining_volume(self, regressed_length):
        """Return remaining solid volume without modifying grain state."""
        regression = max(float(regressed_length), 0.0)
        if regression >= self.burnout_regression_m:
            return 0.0
        height = self.initial_height if self.ends_burn else self.initial_height - 2.0 * regression
        solid_area = np.pi * self.outer_radius**2 - self.evaluate_port_area(regression)
        return max(float(solid_area * height), 0.0)

    def evaluate_grain_volume(self):
        regression = self.inner_radius - self.initial_inner_radius
        self.volume = self.calculate_remaining_volume(regression)
        return self.volume



# Grao_Leviata = Grain(outer_radius=71.92 / 2000, initial_inner_radius=31.92 / 2000)
# print(Grao_Leviata.burn_area)
# print(Grao_Leviata.volume)
