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
    ):

        self.outer_radius = outer_radius
        self.initial_inner_radius = initial_inner_radius
        self.inner_radius = initial_inner_radius
        self.mass = mass
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
            pass
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
        height = max(self.initial_height - 2 * regressed_length, 0.0)
        if burned_through:
            return height, inner_radius, 0.0

        transversal_area = 2 * np.pi * (self.outer_radius**2 - inner_radius**2)
        longitudinal_area = 2 * np.pi * inner_radius * height
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

    def evaluate_grain_volume(self):
        self.volume = (
            np.pi * (self.outer_radius**2 - self.inner_radius**2) * self.height
        )
        return self.volume


# Grao_Leviata = Grain(outer_radius=71.92 / 2000, initial_inner_radius=31.92 / 2000)
# print(Grao_Leviata.burn_area)
# print(Grao_Leviata.volume)
