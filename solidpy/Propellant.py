# -*- coding: utf-8 -*-

_author_ = ""
_copyright_ = "MIT"
_license_ = "x"

import math
import csv
import scipy.interpolate as interpolate
import scipy.constants as const


class Propellant:
    def __init__(
        self,
        specific_heat_ratio,
        products_molecular_mass,
        combustion_temperature,
        density=None,
        **kwargs
    ):
        self.specific_heat_ratio = specific_heat_ratio
        self.density = density
        self.products_molecular_mass = products_molecular_mass
        self.products_constant = const.R / products_molecular_mass
        self.combustion_temperature = combustion_temperature

        self.__dict__.update(kwargs)
        self._burn_rate_interpolator = None
        if "interpolation_list" in self.__dict__:
            self._burn_rate_interpolator = self._load_burn_rate_interpolator(
                self.interpolation_list
            )
        self.calc_cstar()

        self.mean_burn_rate_index = 0
        self.mean_burn_rate_value = 0

    def _load_burn_rate_interpolator(self, interpolation_list):
        pressure_list = []
        burn_rate_list = []
        with open(interpolation_list, "r") as interpolation_data:
            reader = csv.reader(interpolation_data)
            next(reader)
            for line in reader:
                pressure_list.append(float(line[0]))
                burn_rate_list.append(float(line[1]))

        interpolation_kind = "cubic" if len(pressure_list) >= 4 else "linear"
        # Clamp to measured range rather than extrapolating: cubic splines can
        # oscillate or go negative outside the data, giving unphysical rates.
        return interpolate.interp1d(
            pressure_list,
            burn_rate_list,
            kind=interpolation_kind,
            bounds_error=False,
            fill_value=(burn_rate_list[0], burn_rate_list[-1]),
        )

    def evaluate_burn_rate(self, chamber_pressure, port_mass_flux=0.0):
        """Propellant burn rate with optional Lenoir-Robillard erosive correction.

        Args:
            chamber_pressure (float): chamber pressure [Pa]
            port_mass_flux (float): port mass flux G = ṁ/A_port [kg/(m²·s)].
                Pass 0 (default) to skip the erosive correction.

        Returns:
            float: effective burn rate [m/s]
        """
        if "interpolation_list" in self.__dict__:
            if chamber_pressure < 0:
                return 0
            r0 = float(self._burn_rate_interpolator(chamber_pressure * 1e-6)) / 1000
        elif "burn_rate_a" in self.__dict__ and "burn_rate_n" in self.__dict__:
            r0 = self.burn_rate_a * math.pow(chamber_pressure * 1e-6, self.burn_rate_n) / 1000
        else:
            raise TypeError(
                "Missing arguments. You must pass either an `interpolation_list` path or scalar ballistic "
                "coefficients `burn_rate_a` and `burn_rate_n` arguments to Propellant class "
            )

        # Lenoir-Robillard erosive burning correction.
        # r_total = r_0 + k_e * G^0.8 * exp(-alpha_e * r_0 / G)
        # Only active when port_mass_flux > 0 and erosive parameters are set.
        # Ref: Lenoir & Robillard (1957); Sutton & Biblarz §12.6.
        k_e = getattr(self, "erosive_burning_coefficient", 0.0)
        alpha_e = getattr(self, "erosive_alpha", 35.0)
        G = float(port_mass_flux)
        if k_e > 0.0 and G > 1e-3:
            erosive_correction = k_e * G ** 0.8 * math.exp(-alpha_e * r0 / max(G, 1e-9))
            return r0 + max(erosive_correction, 0.0)

        return r0

    def calc_cstar(self):
        k = self.specific_heat_ratio
        self.cstar = math.sqrt(
            (const.R * self.combustion_temperature)
            / (self.products_molecular_mass * k)
            * ((k + 1) / 2) ** ((k + 1) / (k - 1))
        )
        return self.cstar

    # Test - Média Móvel
    def mean_burn_rate(self, current_burn_rate):
        self.mean_burn_rate_value *= self.mean_burn_rate_index
        self.mean_burn_rate_value += current_burn_rate
        self.mean_burn_rate_index += 1
        self.mean_burn_rate_value /= self.mean_burn_rate_index

    def pressure_coeff(self):
        return

    def exponential_pressure_coeff(self):
        return


# knsb = Propellant(
#     1.1361,
#     1700,
#     39.86e-3, #kg/mol
#     1600, #K
#     interpolation_list=r'C:\Users\ansys\Desktop\SolidPy\data\burnrate\KNSB.csv'
# )

# knsu = Propellant(
#     1.1361,
#     1700,
#     39.86e-3, #kg/mol
#     1600, #K
#     burn_rate_a=5.8,
#     burn_rate_n=0.22
# )
