# -*- coding: utf-8 -*-

_author_ = "Caio Eduardo dos Santos de Souza, João Lemes Gribel Soares, Thais Silva Melo, Tiago Mariotto Lucio"
_copyright_ = "MIT"
_license_ = "x"

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
            throughout the burn. Only the cylindrical bore (and slot walls, for
            star grains) keeps regressing. Default False preserves the existing
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
        """Burn area for the slotted-cylinder (N-arm star) grain at regression w.

        The cross-section has a central bore at Ri plus N radial slots of
        half-angle epsilon extending outward to Rs = Ri + slot_fraction*(Ro-Ri).
        The outer surface (at Ro) is case-bonded and does not burn.

        Analytical model (Nakka/BurnSim convention):
          Phase 1 — slot floor not yet at case  (w < Ro - Rs):
            P_lat = (2π - 2Nε)(Ri+w) + 2N(Rs-Ri) + 2Nε(Rs+w)
            A_end = π(Ro²-(Ri+w)²) - Nε((Rs+w)²-(Ri+w)²)

          Phase 2 — slot floor at case  (w ≥ Ro - Rs):
            P_lat = (2π - 2Nε)(Ri+w) + 2N(Ro-(Ri+w))
            A_end = (π - Nε)(Ro²-(Ri+w)²)

        Returns:
            tuple: (height, inner_radius, burn_area)
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

        burned_through = (w >= web) or (w >= L0 / 2)
        if burned_through:
            return max(L0 - 2 * w, 0.0), min(Ri + w, Ro), 0.0

        # Inhibited end faces keep the grain at full length; only the bore /
        # slot walls regress. Let h be set first so the burned_through branch
        # above still uses the classical L0-2*w web-burnout criterion.
        h = L0 if self.ends_burn else L0 - 2 * w
        r_bore = Ri + w

        if w < w_floor:
            # Phase 1: slot floor still within the grain
            r_floor = Rs + w
            slot_wall = Rs - Ri                                          # constant
            P_lat = (2 * np.pi - 2 * N * eps) * r_bore + 2 * N * slot_wall + 2 * N * eps * r_floor
            A_end = np.pi * (Ro ** 2 - r_bore ** 2) - N * eps * (r_floor ** 2 - r_bore ** 2)
        else:
            # Phase 2: slot floor merged with outer case
            slot_wall = max(Ro - r_bore, 0.0)
            P_lat = (2 * np.pi - 2 * N * eps) * r_bore + 2 * N * slot_wall
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

    def evaluate_grain_volume(self):
        Ri = self.inner_radius
        Ro = self.outer_radius
        h = self.height
        if self.geometry == "star":
            Rs = self.initial_inner_radius + self.slot_fraction * (Ro - self.initial_inner_radius)
            slot_area = self.n_points * self.epsilon * (Rs ** 2 - Ri ** 2)
            self.volume = (np.pi * (Ro ** 2 - Ri ** 2) - slot_area) * h
        else:
            self.volume = np.pi * (Ro ** 2 - Ri ** 2) * h
        return self.volume


# Grao_Leviata = Grain(outer_radius=71.92 / 2000, initial_inner_radius=31.92 / 2000)
# print(Grao_Leviata.burn_area)
# print(Grao_Leviata.volume)
