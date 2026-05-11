# -*- coding: utf-8 -*-

_author_ = ""
_copyright_ = "MIT"
_license_ = "x"

import math
import csv
import warnings
import scipy.interpolate as interpolate
import scipy.constants as const
import numpy as np


PA_PER_PSIA = 6894.757293168
FT_TO_M = 0.3048
RANKINE_TO_KELVIN = 5.0 / 9.0
DEFAULT_CEA_PRESSURE_RANGE_PA = (101325.0, 150.0 * 101325.0)


def _import_rocketcea():
    try:
        from rocketcea.cea_obj import CEA_Obj, add_new_propellant
    except ImportError as exc:
        raise ImportError(
            "RocketCEA support requires the optional 'rocketcea' package. "
            "Install it with: pip install rocketcea"
        ) from exc
    return CEA_Obj, add_new_propellant


class Propellant:
    def __init__(
        self,
        specific_heat_ratio,
        products_molecular_mass,
        combustion_temperature,
        density=None,
        *,
        cstar=None,
        cea_formulation=None,
        cea_name=None,
        cea_pressure_range_pa=None,
        cea_n_points=40,
        cea_expansion_ratio=10.0,
        **kwargs
    ):
        self.specific_heat_ratio = specific_heat_ratio
        self.density = density
        self.products_molecular_mass = products_molecular_mass
        self.products_constant = const.R / products_molecular_mass
        self.combustion_temperature = combustion_temperature
        self.cstar = cstar

        self.cea_formulation = cea_formulation
        self.cea_name = cea_name
        self.cea_pressure_range_pa = (
            DEFAULT_CEA_PRESSURE_RANGE_PA
            if cea_pressure_range_pa is None
            else cea_pressure_range_pa
        )
        self.cea_n_points = cea_n_points
        self.cea_expansion_ratio = cea_expansion_ratio
        self._cea_obj = None

        self._thermo_table = None      # set by load_thermo_table() or RocketCEA
        self._cstar_func = None
        self._gamma_func = None
        self._temperature_func = None

        self.__dict__.update(kwargs)
        self._burn_rate_interpolator = None
        if "interpolation_list" in self.__dict__:
            self._burn_rate_interpolator = self._load_burn_rate_interpolator(
                self.interpolation_list
            )
        if self.cstar is None:
            self.calc_cstar()
        else:
            self.cstar = float(self.cstar)

        if self.cea_formulation is not None:
            self._load_cea_thermo()

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

    def load_thermo_table(self, data):
        """Load pressure-dependent thermochemical properties for more accurate simulation.

        Accepts a 4-column table: [[pressure_pa, cstar_m_s, k, Tc_K], ...].
        Once loaded, cstar_at_pressure(), k_at_pressure(), and Tc_at_pressure()
        return interpolated values instead of the constant class attributes.
        Rows must be sorted in ascending pressure order.

        ``data`` can be:
          - A path string or file-like object for a CSV file (header skipped).
          - A (N, 4) numeric array or nested list.

        The pressure column is expected in Pa.  After loading, the base
        specific_heat_ratio, combustion_temperature, and cstar attributes are
        updated to the value at the median table pressure so existing code
        that reads those attributes is still reasonable.
        """
        if isinstance(data, (str, bytes)):
            rows = []
            with open(data, "r") as fh:
                reader = csv.reader(fh)
                next(reader)
                for line in reader:
                    rows.append([float(x) for x in line[:4]])
            arr = np.array(rows, dtype=float)
        else:
            arr = np.asarray(data, dtype=float)

        if arr.ndim != 2 or arr.shape[1] < 4:
            raise ValueError("Thermochemistry table must have columns: pressure_pa, cstar_m_s, k, Tc_K")
        if len(arr) < 2:
            raise ValueError("Thermochemistry table must have at least 2 rows")

        self._load_thermo_array(arr)

    def _load_thermo_array(self, arr):
        arr = np.asarray(arr, dtype=float)
        if arr.ndim != 2 or arr.shape[1] < 4:
            raise ValueError("Thermochemistry table must have columns: pressure_pa, cstar_m_s, k, Tc_K")
        if len(arr) < 2:
            raise ValueError("Thermochemistry table must have at least 2 rows")

        order = np.argsort(arr[:, 0])
        arr = arr[order]
        pressures = arr[:, 0]
        if np.any(np.diff(pressures) <= 0.0):
            raise ValueError("Thermochemistry table pressures must be unique and ascending")

        cstars = arr[:, 1]
        ks = arr[:, 2]
        tcs = arr[:, 3]

        def _make_interp(y):
            kind = "cubic" if len(y) >= 4 else "linear"
            return interpolate.interp1d(
                pressures, y, kind=kind, bounds_error=False,
                fill_value=(float(y[0]), float(y[-1]))
            )

        self._cstar_func = _make_interp(cstars)
        self._gamma_func = _make_interp(ks)
        self._temperature_func = _make_interp(tcs)

        self._thermo_table = {
            "pressure_pa": pressures,
            "cstar_interp": self._cstar_func,
            "k_interp": self._gamma_func,
            "Tc_interp": self._temperature_func,
        }

        # Update base attributes to the mid-range value so existing code stays sane.
        mid = pressures[len(pressures) // 2]
        self.specific_heat_ratio = float(self.get_gamma(mid))
        self.combustion_temperature = float(self.Tc_at_pressure(mid))
        self.cstar = float(self.get_cstar(mid))

    def _load_cea_thermo(self):
        card = str(self.cea_formulation).strip()
        if not card:
            raise ValueError("cea_formulation must be a non-empty RocketCEA propellant card")

        CEA_Obj, add_new_propellant = _import_rocketcea()
        self.cea_name = self.cea_name or f"solidpy_propellant_{id(self):x}"
        add_new_propellant(self.cea_name, card)
        self._cea_obj = CEA_Obj(propName=self.cea_name)

        pressures_pa = self._build_cea_pressure_grid()
        eps = max(float(self.cea_expansion_ratio), 1.001)
        rows = []

        for pressure_pa in pressures_pa:
            pressure_psia = pressure_pa / PA_PER_PSIA
            try:
                _, cstar_ft_s, chamber_temperature_r = self._cea_obj.get_IvacCstrTc(
                    Pc=pressure_psia,
                    eps=eps,
                    MR=0,
                )
                gamma = self._cea_obj.get_Chamber_MolWt_gamma(
                    Pc=pressure_psia,
                    eps=eps,
                    MR=0,
                )[1]
                rows.append([
                    pressure_pa,
                    float(cstar_ft_s) * FT_TO_M,
                    float(gamma),
                    float(chamber_temperature_r) * RANKINE_TO_KELVIN,
                ])
            except Exception:
                warnings.warn(
                    f"RocketCEA evaluation failed at P={pressure_pa:.0f} Pa; skipping.",
                    stacklevel=2,
                )

        if len(rows) < 2:
            raise RuntimeError(
                "RocketCEA produced fewer than two valid thermodynamic points "
                "for the requested pressure range."
            )

        self._load_thermo_array(np.array(rows, dtype=float))

    def _build_cea_pressure_grid(self):
        n_points = int(self.cea_n_points)
        if n_points < 2:
            raise ValueError("cea_n_points must be at least 2")

        try:
            p_low, p_high = self.cea_pressure_range_pa
        except (TypeError, ValueError) as exc:
            raise ValueError("cea_pressure_range_pa must be a (low_pa, high_pa) pair") from exc

        p_low = float(p_low)
        p_high = float(p_high)
        if p_low <= 0.0 or p_high <= 0.0 or p_high <= p_low:
            raise ValueError("cea_pressure_range_pa must contain positive ascending pressures")

        return np.linspace(p_low, p_high, n_points)

    def get_cstar(self, pressure_pa):
        """Return characteristic velocity c* [m/s] at the given chamber pressure."""
        if self._cstar_func is not None:
            return float(self._cstar_func(pressure_pa))
        return float(self.cstar)

    def get_gamma(self, pressure_pa):
        """Return specific heat ratio gamma/k at the given chamber pressure."""
        if self._gamma_func is not None:
            return float(self._gamma_func(pressure_pa))
        return float(self.specific_heat_ratio)

    def cstar_at_pressure(self, pressure_pa):
        """Return characteristic velocity c* [m/s] at the given chamber pressure."""
        return self.get_cstar(pressure_pa)

    def k_at_pressure(self, pressure_pa):
        """Return specific heat ratio k at the given chamber pressure."""
        return self.get_gamma(pressure_pa)

    def Tc_at_pressure(self, pressure_pa):
        """Return adiabatic flame temperature Tc [K] at the given chamber pressure."""
        if self._temperature_func is not None:
            return float(self._temperature_func(pressure_pa))
        return float(self.combustion_temperature)

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


def build_cea_properties(
    propellant_name,
    pressure_range_pa=None,
    n_points=20,
    expansion_ratio=10.0,
):
    """Build a pressure-dependent thermochemistry table via NASA CEA (rocketcea).

    Calls the ``rocketcea`` Python wrapper (pip install rocketcea) for
    a named solid propellant and returns a (N×4) NumPy array suitable for
    Propellant.load_thermo_table():

        [[pressure_pa, cstar_m_s, k, Tc_K], ...]

    The table spans ``pressure_range_pa`` at ``n_points`` logarithmically spaced
    points.  If ``rocketcea`` is not installed an ImportError is raised with
    installation instructions.

    Args:
        propellant_name (str) — rocketcea propellant name, e.g. ``'KNSB'``.
        pressure_range_pa     — (P_low, P_high) in Pa; default (1e5, 10e6).
        n_points (int)        — number of pressure points (default 20).
        expansion_ratio (float) — nozzle area ratio for Isp reference (default 10).

    Returns:
        np.ndarray of shape (n_points, 4): columns [pressure_pa, cstar_m_s, k, Tc_K].

    Example::

        from solidpy import Propellant, build_cea_properties

        table = build_cea_properties('KNSB', pressure_range_pa=(1e5, 10e6))
        knsb = Propellant(
            specific_heat_ratio=1.1361,
            density=1700,
            products_molecular_mass=39.9e-3,
            combustion_temperature=1600,
            burn_rate_a=5.13,
            burn_rate_n=0.22,
        )
        knsb.load_thermo_table(table)
    """
    CEA_Obj, _ = _import_rocketcea()

    if pressure_range_pa is None:
        pressure_range_pa = (1e5, 10e6)

    P_low, P_high = float(pressure_range_pa[0]), float(pressure_range_pa[1])
    pressures_pa = np.logspace(np.log10(max(P_low, 1.0)), np.log10(P_high), int(n_points))

    cea = CEA_Obj(propName=str(propellant_name))
    rows = []
    for p_pa in pressures_pa:
        p_psia = p_pa / PA_PER_PSIA
        try:
            eps = max(float(expansion_ratio), 1.001)
            isp_vac, cstar, tc_r = cea.get_IvacCstrTc(Pc=p_psia, eps=eps, MR=0)
            k = cea.get_Chamber_MolWt_gamma(Pc=p_psia, MR=0, eps=eps)[1]
            cstar_ms = cstar * FT_TO_M
            tc_k = tc_r * RANKINE_TO_KELVIN
            rows.append([p_pa, cstar_ms, float(k), tc_k])
        except Exception:
            warnings.warn(f"CEA evaluation failed at P={p_pa:.0f} Pa — skipping.", stacklevel=2)

    if not rows:
        raise RuntimeError("CEA produced no valid results for the given propellant/pressure range.")

    return np.array(rows, dtype=float)


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
