# -*- coding: utf-8 -*-

_author_ = ""
_copyright_ = "MIT"
_license_ = ""

import math
import hashlib
import json
from pathlib import Path
import subprocess
from types import SimpleNamespace
from numbers import Real
import numpy as np
import matplotlib.pyplot as plt

from scipy.optimize import fsolve, brentq
from scipy.integrate import solve_ivp
try:
    from scipy.integrate import cumulative_trapezoid
except ImportError:
    from scipy.integrate import cumtrapz as cumulative_trapezoid
from matplotlib.font_manager import FontProperties

try:
    from .Grain import Grain
    from .Propellant import Propellant
    from .Motor import Motor
    from .Environment import Environment
    from .Export import Export
except ImportError:
    from Grain import Grain
    from Propellant import Propellant
    from Motor import Motor
    from Environment import Environment
    from Export import Export


class Burn:
    def __init__(
        self,
        grain,
        motor,
        propellant,
        environment=None,
        *,
        eta_c: float = 1.0,
        eta_Cf: float = 1.0,
        discharge_coefficient: float = 1.0,
    ):
        """Initialise a burn simulation.

        Args:
            grain:       Grain object (or list of Grain objects via motor.grains).
            motor:       Motor object.
            propellant:  Propellant object.
            environment: Environment object (defaults to sea-level standard).
            eta_c: Combustion efficiency, applied as T_0_eff = eta_c**2 * T_0.
            eta_Cf: Efficiency applied once to total reported thrust.
            discharge_coefficient: Nozzle mass-flow and momentum coefficient.

        Efficiencies must be finite real numbers in (0, 1].
        """
        if environment is None:
            environment = Environment()
        self.motor = motor
        self.grain = grain
        self.propellant = propellant
        self.environment = environment
        self.eta_c = self._validate_efficiency("eta_c", eta_c)
        self.eta_Cf = self._validate_efficiency("eta_Cf", eta_Cf)
        self.discharge_coefficient = self._validate_efficiency(
            "discharge_coefficient", discharge_coefficient
        )

        self.gravity = environment.gravity
        self.environment_pressure = environment.atmospheric_pressure

        self.parameters = self.set_parameters()
        self._exit_mach_cache = {}

    @staticmethod
    def _validate_efficiency(name, value):
        if isinstance(value, bool) or not isinstance(value, Real):
            raise ValueError(f"{name} must be a finite real number in (0, 1]")
        value = float(value)
        if not math.isfinite(value) or not 0.0 < value <= 1.0:
            raise ValueError(f"{name} must be a finite real number in (0, 1]")
        return value

    @staticmethod
    def _validate_temperature(value):
        if isinstance(value, bool) or not isinstance(value, Real):
            raise ValueError("chamber_temperature must be a finite positive real number")
        value = float(value)
        if not math.isfinite(value) or value <= 0.0:
            raise ValueError("chamber_temperature must be a finite positive real number")
        return value

    @property
    def applied_efficiencies(self):
        """Return the resolved efficiency values used by the burn model."""
        return {
            "eta_c": self.eta_c,
            "eta_Cf": self.eta_Cf,
            "discharge_coefficient": self.discharge_coefficient,
        }

    def set_parameters(self):
        parameters = (
            self.propellant.combustion_temperature,  # T_0
            self.propellant.products_constant,  # R
            self.propellant.density,  # rho_g
            self.propellant.specific_heat_ratio,  # k
            self.motor.nozzle_throat_area,  # A_t
        )
        return parameters

    def _parameters_at_pressure(self, chamber_pressure=None):
        pressure = (
            self.environment_pressure
            if chamber_pressure is None
            else max(float(chamber_pressure), 0.0)
        )
        # While a BurnSimulation solves, the inputs below cannot change, so the last pressure's
        # answer is reused (a right-hand-side evaluation asks for it ~14 times). Outside that scope
        # ``_solve_cache`` is None and every call recomputes, as before.
        cache = getattr(self, "_solve_cache", None)
        if cache is not None:
            last = cache.get("parameters")
            if last is not None and last[0] == pressure:
                return last[1]
        # c* scales with sqrt(T_0), so eta_c enters through effective temperature.
        t0_eff = self.propellant.Tc_at_pressure(pressure) * (self.eta_c ** 2)
        result = (
            t0_eff,  # T_0 (effective)
            self.propellant.products_constant,  # R
            self.propellant.density,  # rho_g
            self.propellant.get_gamma(pressure),  # k
            self.motor.nozzle_throat_area,  # A_t
        )
        if cache is not None:
            cache["parameters"] = (pressure, result)
        return result

    def evaluate_nozzle_mass_flow(self, chamber_pressure, *, chamber_temperature=None):
        """Calculation of total nozzle mass flow.

        Source:
            https://www.grc.nasa.gov/www/k-12/rocket/rktthsum.html

        Args:
            chamber_pressure (float): current chamber pressure

        Returns:
            float: nozzle mass flow for the specified chamber pressure
        """
        if chamber_temperature is not None:
            self._validate_temperature(chamber_temperature)
        if chamber_pressure <= self.environment_pressure:
            return 0.0

        T_0, R, _, k, A_t = self._parameters_at_pressure(chamber_pressure)
        if chamber_temperature is not None:
            T_0 = self._validate_temperature(chamber_temperature)
        pressure_ratio = self.environment_pressure / chamber_pressure
        critical_pressure_ratio = math.pow(2 / (k + 1), k / (k - 1))

        if pressure_ratio <= critical_pressure_ratio:
            mass_flow = (
                chamber_pressure
                * A_t
                * np.sqrt(k / (R * T_0))
                * math.pow((2 / (k + 1)), ((k + 1) / (2 * (k - 1))))
            )
            return self.discharge_coefficient * mass_flow

        unchoked_term = (
            math.pow(pressure_ratio, 2 / k)
            - math.pow(pressure_ratio, (k + 1) / k)
        )
        mass_flow = (
            A_t
            * chamber_pressure
            * math.sqrt((2 * k / ((k - 1) * R * T_0)) * max(unchoked_term, 0.0))
        )
        return self.discharge_coefficient * mass_flow

    def is_nozzle_choked(self, chamber_pressure):
        """Return whether chamber-to-ambient pressure ratio chokes the throat."""
        if chamber_pressure <= self.environment_pressure:
            return False
        _, _, _, k, _ = self._parameters_at_pressure(chamber_pressure)
        critical_pressure_ratio = math.pow(2 / (k + 1), k / (k - 1))
        return self.environment_pressure / chamber_pressure <= critical_pressure_ratio

    def evaluate_exit_mach(self, chamber_pressure=None):
        """Calculation of mach number at nozzle exit
        (ratio of flow speed to the local sound speed).

        Source:
        https://www.grc.nasa.gov/www/k-12/rocket/rktthsum.html

        Returns:
            float: mach number
        """
        _, _, _, k, _ = self._parameters_at_pressure(chamber_pressure)
        if chamber_pressure is not None and not self.is_nozzle_choked(
            chamber_pressure
        ):
            if chamber_pressure <= self.environment_pressure:
                self.exit_mach = 0.0
                return self.exit_mach
            pressure_ratio = self.environment_pressure / chamber_pressure
            self.exit_mach = math.sqrt(
                max(
                    (2 / (k - 1))
                    * (math.pow(1 / pressure_ratio, (k - 1) / k) - 1),
                    0.0,
                )
            )
            return min(self.exit_mach, 1.0)

        cache_key = (round(k, 12), self.motor.expansion_ratio)
        if cache_key in self._exit_mach_cache:
            self.exit_mach = self._exit_mach_cache[cache_key]
            return self.exit_mach

        def func(mach_number):
            mach_number = float(np.asarray(mach_number).reshape(-1)[0])
            return (
                math.pow((k + 1) / 2, -(k + 1) / (2 * (k - 1)))
                * math.pow(
                    (1 + (k - 1) / 2 * mach_number**2),
                    (k + 1) / (2 * (k - 1)),
                )
                / mach_number
                - self.motor.expansion_ratio
            )

        # Use a bracketed root-find for the supersonic root.
        # A/A*(M) is strictly monotonically increasing for M > 1, so brentq
        # is guaranteed to converge once we bracket the root. fsolve with a
        # fixed initial guess of M=2 silently fails for high expansion ratios
        # combined with low γ (e.g. KNSB, ε ≳ 16) — it returns the initial
        # guess unchanged with a large residual.
        lo = 1.0 + 1e-9
        # Grow the upper bracket until func(M_hi) > 0 (root is bracketed).
        hi = 50.0
        for _ in range(20):
            if func(hi) > 0:
                break
            hi *= 2.0
        else:
            # Fallback: expansion ratio may be extremely large; try brentq
            # anyway with the last hi — if it still can't bracket, re-raise.
            pass

        if func(lo) * func(hi) >= 0:
            # Safety: bracket failed (should not happen for physical inputs).
            # Fall back to fsolve so we do not raise unexpectedly, but warn.
            import warnings
            warnings.warn(
                f"evaluate_exit_mach: could not bracket supersonic root for "
                f"k={k:.6g}, expansion_ratio={self.motor.expansion_ratio:.6g}. "
                "Falling back to fsolve; result may be inaccurate.",
                RuntimeWarning,
                stacklevel=2,
            )
            self.exit_mach = fsolve(func, np.array(2))[0]
        else:
            self.exit_mach = brentq(func, lo, hi, xtol=1e-10, rtol=1e-10)
            # Verify convergence to a tight tolerance.
            residual = abs(func(self.exit_mach))
            if residual > 1e-6:
                import warnings
                warnings.warn(
                    f"evaluate_exit_mach: brentq residual {residual:.3e} exceeds "
                    "tolerance after solving. Result may be inaccurate.",
                    RuntimeWarning,
                    stacklevel=2,
                )

        self._exit_mach_cache[cache_key] = self.exit_mach
        return self.exit_mach

    def evaluate_exit_pressure(self, chamber_pressure):
        """Calculation of the pressure at nozzle exit .

        Source:
        https://www.grc.nasa.gov/www/k-12/rocket/rktthsum.html

        Args:
            chamber_pressure (float): current chamber pressure

        Returns:
            float: exit pressure for the specified chamber pressure
        """
        if not self.is_nozzle_choked(chamber_pressure):
            self.exit_pressure = self.environment_pressure
            return self.exit_pressure

        _, _, _, k, _ = self._parameters_at_pressure(chamber_pressure)
        self.exit_pressure = chamber_pressure * math.pow(
            (1 + (k - 1) / 2 * self.evaluate_exit_mach(chamber_pressure) ** 2),
            -k / (k - 1),
        )
        return self.exit_pressure

    def evaluate_exit_temperature(self, chamber_pressure=None, *, chamber_temperature=None):
        """Calculation of fluid temperature at nozzle exit.

        Source:
        https://www.grc.nasa.gov/www/k-12/rocket/rktthsum.html

        Returns:
            float: exit temperature
        """
        T_0, _, _, k, _ = self._parameters_at_pressure(chamber_pressure)
        if chamber_temperature is not None:
            T_0 = self._validate_temperature(chamber_temperature)
        if chamber_pressure is not None and not self.is_nozzle_choked(
            chamber_pressure
        ):
            if chamber_pressure <= self.environment_pressure:
                self.exit_temperature = T_0
                return self.exit_temperature
            self.exit_temperature = T_0 * math.pow(
                self.environment_pressure / chamber_pressure, (k - 1) / k
            )
            return self.exit_temperature

        self.exit_temperature = T_0 / (
            1 + (k - 1) / 2 * self.evaluate_exit_mach(chamber_pressure) ** 2
        )
        return self.exit_temperature

    def evaluate_exit_velocity(self, chamber_pressure=None, *, chamber_temperature=None):
        """Calculation of fluid velocity at nozzle exit.

        Source:
        https://www.grc.nasa.gov/www/k-12/rocket/rktthsum.html

        Returns:
            float: exit velocity
        """
        T_0, R, _, k, _ = self._parameters_at_pressure(chamber_pressure)
        if chamber_temperature is not None:
            T_0 = self._validate_temperature(chamber_temperature)
        if chamber_pressure is not None and not self.is_nozzle_choked(
            chamber_pressure
        ):
            if chamber_pressure <= self.environment_pressure:
                self.exit_velocity = 0.0
                return self.exit_velocity
            self.exit_velocity = math.sqrt(
                max(
                    (2 * k / (k - 1))
                    * R
                    * T_0
                    * (
                        1
                        - math.pow(
                            self.environment_pressure / chamber_pressure,
                            (k - 1) / k,
                        )
                    ),
                    0.0,
                )
            )
            return self.exit_velocity

        self.exit_velocity = self.evaluate_exit_mach(chamber_pressure) * math.sqrt(
            k * R * self.evaluate_exit_temperature(chamber_pressure, chamber_temperature=chamber_temperature)
        )
        return self.exit_velocity

    def _nozzle_divergence_factor(self):
        """Thrust loss factor for conical nozzles: λ = (1 + cos α) / 2.

        For a perfect (bell) nozzle or when angle is unspecified, λ = 1.
        Sutton & Biblarz (2010) §3.4; Rogers/RASAero Part 4, eq.(9).
        """
        angle = self.motor.nozzle_angle
        if angle is None or angle <= 0.0:
            return 1.0
        return 0.5 * (1.0 + math.cos(float(angle)))

    def evaluate_Cf(self, chamber_pressure, *, chamber_temperature=None):
        """Return reported thrust divided by chamber pressure and throat area."""
        if chamber_temperature is not None:
            self._validate_temperature(chamber_temperature)
        if chamber_pressure <= self.environment_pressure:
            self.Cf = 0.0
        else:
            self.Cf = self.evaluate_thrust(chamber_pressure, chamber_temperature=chamber_temperature) / (
                chamber_pressure * self.motor.nozzle_throat_area
            )
        return self.Cf

    def evaluate_thrust_components(self, chamber_pressure, *, chamber_temperature=None):
        """Return momentum, pressure, and reported total thrust in newtons.

        Ideal momentum includes conical divergence once. The discharge
        coefficient scales momentum only; eta_Cf scales the summed thrust.
        Combustion efficiency enters through the effective gas temperature.
        """
        if chamber_temperature is not None:
            self._validate_temperature(chamber_temperature)
        if chamber_pressure <= self.environment_pressure:
            return {
                "momentum_ideal_n": 0.0,
                "momentum_n": 0.0,
                "pressure_n": 0.0,
                "total_n": 0.0,
            }
        ideal_mass_flow = (
            self.evaluate_nozzle_mass_flow(chamber_pressure, chamber_temperature=chamber_temperature)
            / self.discharge_coefficient
        )
        ideal_momentum = (
            self._nozzle_divergence_factor()
            * ideal_mass_flow
            * self.evaluate_exit_velocity(chamber_pressure, chamber_temperature=chamber_temperature)
        )
        momentum = self.discharge_coefficient * ideal_momentum
        pressure = (
            self.evaluate_exit_pressure(chamber_pressure) - self.environment_pressure
        ) * self.motor.nozzle_exit_area
        return {
            "momentum_ideal_n": ideal_momentum,
            "momentum_n": momentum,
            "pressure_n": pressure,
            "total_n": self.eta_Cf * (momentum + pressure),
        }

    def evaluate_thrust(self, chamber_pressure, *, chamber_temperature=None):
        """Return thrust with discharge and total-thrust efficiencies applied."""
        self.thrust = self.evaluate_thrust_components(chamber_pressure, chamber_temperature=chamber_temperature)["total_n"]
        return self.thrust

    def evaluate_total_impulse(self, thrust_list, time_list):
        """Numerical integration by trapezoids for total impulse
        approximation.

        Args:
            thrust_list (float list or float arrays): list of thrust values
            for each time step
            time_list (float list or float arrays): list of time steps

        Returns:
            float: the total impulse correspondent to the integral of
            the given values
        """
        if len(time_list) < 2:
            return 0.0
        total_impulse = cumulative_trapezoid(thrust_list, time_list)[-1]
        return total_impulse

    def evaluate_specific_impulse(self, thrust_list, time_list):
        """Calculation of motor's specific impulse.

        Args:
            thrust_list (float list or float arrays): list of thrust values
            time_list (float list or float arrays): list of time steps

        Returns:
            float: the specific impulse for the given values and propellant mass
        """
        specific_impulse = self.evaluate_total_impulse(thrust_list, time_list) / (
            self.propellant.density
            * sum(g.volume for g in self.motor.grains)
            * self.environment.standard_gravity
        )
        return specific_impulse

    def compute_total_burn_area(self, regressed_lengths):
        """Sum burn area over all active grains.

        Args:
            regressed_lengths: scalar (shared regression for all grains) or a
                sequence of per-grain regression values.
        """
        grains = self.motor.grains
        try:
            lengths = list(regressed_lengths)
        except TypeError:
            lengths = [float(regressed_lengths)] * len(grains)
        if len(lengths) != len(grains):
            lengths = [lengths[0]] * len(grains)
        total = 0.0
        for grain, r in zip(grains, lengths):
            total += grain.evaluate_burn_area(r, update_state=False)
        return total

    def evaluate_burn_rate(
        self, chamber_pressure, chamber_pressure_derivative, free_volume, burn_area
    ):
        """Calculation of propellant rate of regression, i.e. burn rate

        Args:
            chamber_pressure (float): current chamber pressure
            chamber_pressure_derivative (float): current chamber pressure derivative
            free_volume (float): current combustion chamber free volume
            burn_area (float): current total grain burn area accounting for regression

        Returns:
            float: current propellant burn rate
        """
        T_0, R, rho_g, _, _ = self._parameters_at_pressure(chamber_pressure)

        rho_0 = chamber_pressure / (R * T_0)  # product_gas_density
        nozzle_mass_flow = self.evaluate_nozzle_mass_flow(chamber_pressure)

        burn_rate = (
            free_volume / (R * T_0) * chamber_pressure_derivative + nozzle_mass_flow
        ) / (burn_area * (rho_g - rho_0))

        return burn_rate


class BurnSimulation(Burn):
    def __init__(
        self,
        grain,
        motor,
        propellant,
        environment=None,
        max_step_size=0.01,
        tail_off_evaluation=True,
        igniter_mass_flow=None,
        igniter_burn_time=0.0,
        igniter_temperature=None,
        burn_area_activation=None,
        ignition_ramp_time=0.0,
        tail_off_method="numerical",
        *,
        eta_c: float = 1.0,
        eta_Cf: float = 1.0,
        discharge_coefficient: float = 1.0,
        rtol: float = 1e-8,
        atol: float = 1e-10,
        burn_timeout_s: float = 100.0,
        tail_off_timeout_s: float = 100.0,
        solve_cache: bool = True,
    ):
        Burn.__init__(
            self,
            grain,
            motor,
            propellant,
            environment,
            eta_c=eta_c,
            eta_Cf=eta_Cf,
            discharge_coefficient=discharge_coefficient,
        )
        self.max_step_size = self._positive_setting("max_step_size", max_step_size)
        self.rtol = self._positive_setting("rtol", rtol)
        self.atol = self._positive_setting("atol", atol)
        self.burn_timeout_s = self._positive_setting("burn_timeout_s", burn_timeout_s)
        self.tail_off_timeout_s = self._positive_setting("tail_off_timeout_s", tail_off_timeout_s)
        self.igniter_mass_flow = igniter_mass_flow
        self.igniter_burn_time = self._nonnegative_setting("igniter_burn_time", igniter_burn_time)
        self.igniter_temperature = (
            self._parameters_at_pressure(self.environment_pressure)[0]
            if igniter_temperature is None
            else igniter_temperature
        )
        self.igniter_temperature = self._positive_setting("igniter_temperature", self.igniter_temperature)
        self.burn_area_activation = burn_area_activation
        self.ignition_ramp_time = self._nonnegative_setting("ignition_ramp_time", ignition_ramp_time)
        self.tail_off_method = str(tail_off_method).lower()
        if self.tail_off_method not in {"numerical", "analytical"}:
            raise ValueError("tail_off_method must be numerical or analytical")
        self._burnout_times = [None] * len(self.motor.grains)
        for name, value in (("propellant_density", propellant.density),
                            ("gas_constant", propellant.products_constant),
                            ("combustion_temperature", propellant.combustion_temperature)):
            self._positive_setting(name, value)
        self._positive_setting("effective_combustion_temperature", self._parameters_at_pressure(self.environment_pressure)[0])
        if not math.isfinite(propellant.specific_heat_ratio) or propellant.specific_heat_ratio <= 1.0:
            raise ValueError("specific_heat_ratio must be finite and greater than one")
        self._validate_source_profiles()
        self._termination_reason = "burn_timeout"

        # ``solve_cache`` only avoids repeating identical evaluations while solving; results are
        # bit-for-bit the same with it off, and the cache is released when the solve ends.
        self._solve_cache = {"quantities": {}} if solve_cache else None
        try:
            self.grain_burn_solution = self.evaluate_grain_burn_solution()
            self.tail_off_solution = (
                self.evaluate_tail_off_solution() if tail_off_evaluation else None
            )
            self.total_burn_solution = self.evaluate_complete_solution()
            self.result = self._build_result(tail_off_evaluation)
        finally:
            self._solve_cache = None

    """Solver required functions"""

    def evaluate_igniter_mass_flow(self, time):
        """Evaluate optional igniter gas mass flow at the current time.

        The value can be supplied as a callable ``m_dot(t)``, as a scalar
        paired with ``igniter_burn_time``, or as a two-column ``(time, m_dot)``
        table. Returned units are kg/s.
        """
        source = self.igniter_mass_flow

        if source is None:
            return 0.0
        if callable(source):
            if self.igniter_burn_time > 0 and time >= self.igniter_burn_time:
                return 0.0
            value = float(source(time))
            if not math.isfinite(value) or value < 0.0:
                raise ValueError("igniter_mass_flow callable must return finite non-negative values")
            return value
        if np.isscalar(source):
            if self.igniter_burn_time <= 0.0 or time >= self.igniter_burn_time:
                return 0.0
            return max(float(source), 0.0)

        profile = np.asarray(source, dtype=float)
        if profile.ndim != 2 or profile.shape[1] != 2:
            raise ValueError(
                "igniter_mass_flow must be None, a callable, a scalar, or a two-column time/mass-flow table"
            )

        profile_time = profile[:, 0]
        profile_mass_flow = profile[:, 1]
        if time < profile_time[0] or time >= profile_time[-1]:
            return 0.0
        return max(float(np.interp(time, profile_time, profile_mass_flow)), 0.0)

    def evaluate_burn_area_activation(self, time, regressed_length):
        """Evaluate the ignited fraction of the available burn area.

        The activation factor represents flame spreading over the exposed
        grain surface. It is dimensionless and clipped to the physical range
        [0, 1]. It can be a callable, a constant, a two-column time table, or
        the built-in smooth ramp controlled by ``ignition_ramp_time``.
        """
        source = self.burn_area_activation

        if source is None:
            if self.ignition_ramp_time <= 0.0:
                return 1.0
            progress = min(max(time / self.ignition_ramp_time, 0.0), 1.0)
            return progress * progress * (3.0 - 2.0 * progress)
        if callable(source):
            try:
                activation = source(time, regressed_length)
            except TypeError:
                activation = source(time)
            if not math.isfinite(float(activation)):
                raise ValueError("burn_area_activation callable must return finite values")
            return min(max(float(activation), 0.0), 1.0)
        if np.isscalar(source):
            return min(max(float(source), 0.0), 1.0)

        profile = np.asarray(source, dtype=float)
        if profile.ndim != 2 or profile.shape[1] != 2:
            raise ValueError(
                "burn_area_activation must be None, a callable, a scalar, or a two-column time/activation table"
            )

        profile_time = profile[:, 0]
        profile_activation = profile[:, 1]
        if time < profile_time[0]:
            return min(max(float(profile_activation[0]), 0.0), 1.0)
        if time > profile_time[-1]:
            return min(max(float(profile_activation[-1]), 0.0), 1.0)
        activation = np.interp(time, profile_time, profile_activation)
        return min(max(float(activation), 0.0), 1.0)

    @staticmethod
    def _positive_setting(name, value):
        if isinstance(value, bool) or not isinstance(value, Real):
            raise ValueError(f"{name} must be a finite positive real number")
        value = float(value)
        if not math.isfinite(value) or value <= 0.0:
            raise ValueError(f"{name} must be a finite positive real number")
        return value

    @staticmethod
    def _nonnegative_setting(name, value):
        if isinstance(value, bool) or not isinstance(value, Real):
            raise ValueError(f"{name} must be a finite non-negative real number")
        value = float(value)
        if not math.isfinite(value) or value < 0.0:
            raise ValueError(f"{name} must be a finite non-negative real number")
        return value

    def _validate_source_profiles(self):
        if not math.isfinite(self.igniter_burn_time):
            raise ValueError("igniter_burn_time must be finite")
        if not math.isfinite(self.ignition_ramp_time):
            raise ValueError("ignition_ramp_time must be finite")
        for name in ("igniter_mass_flow", "burn_area_activation"):
            source = getattr(self, name)
            if source is None or callable(source):
                continue
            if np.isscalar(source):
                value = self._nonnegative_setting(name, source)
                if name == "burn_area_activation" and value > 1.0:
                    raise ValueError("burn_area_activation must not exceed 1")
                continue
            profile = np.asarray(source, dtype=float)
            if (profile.ndim != 2 or profile.shape[1] != 2 or len(profile) < 2
                    or not np.all(np.isfinite(profile)) or np.any(np.diff(profile[:, 0]) <= 0)
                    or np.any(profile < 0)):
                raise ValueError(f"{name} table must have increasing non-negative time and finite non-negative values")
            if name == "burn_area_activation" and np.any(profile[:, 1] > 1.0):
                raise ValueError("burn_area_activation must not exceed 1")

    def _source_end_time(self):
        source = self.igniter_mass_flow
        if source is None:
            return 0.0
        if callable(source) or np.isscalar(source):
            return self.igniter_burn_time
        return float(np.asarray(source)[-1, 0])

    def _source_breakpoints(self, start, stop):
        points = [stop]
        for name in ("igniter_mass_flow", "burn_area_activation"):
            source = getattr(self, name)
            if source is not None and not callable(source) and not np.isscalar(source):
                points.extend(np.asarray(source, dtype=float)[:, 0])
        points.extend([self._source_end_time(), self.ignition_ramp_time])
        return sorted(set(float(t) for t in points if start < t <= stop))

    def _state_quantities(self, time, state, active=None):
        """Derived quantities at one state.

        Within a solve, evaluations with ``active=None`` (the post-processing of the integrated
        history, which revisits the same points several times) are memoised per (time, state);
        the integrator's own calls pass ``active`` and are never cached. The returned dictionary
        is shared between calls: treat it as read-only.
        """
        cache = getattr(self, "_solve_cache", None)
        if active is not None or cache is None:
            return self._state_quantities_uncached(time, state, active)
        memo = cache["quantities"]
        key = (float(time), np.asarray(state, dtype=float).tobytes())
        hit = memo.get(key)
        if hit is None:
            hit = memo[key] = self._state_quantities_uncached(time, state, None)
        return hit

    def _state_quantities_uncached(self, time, state, active=None):
        n = len(self.motor.grains)
        regressions = np.asarray(state[2:2 + n])
        remaining = sum(g.calculate_remaining_volume(r) for g, r in zip(self.motor.grains, regressions))
        volume = self.motor.chamber_volume - remaining
        gas_mass = max(float(state[0]), np.finfo(float).tiny)
        thermal_inventory = max(float(state[1]), np.finfo(float).tiny)
        temperature = thermal_inventory / gas_mass
        pressure = self.propellant.products_constant * thermal_inventory / volume
        pressure_for_rate = max(pressure, self.environment_pressure)
        if active is None:
            active = regressions < np.asarray([g.burnout_regression_m for g in self.motor.grains])
        areas = np.asarray([
            g.evaluate_burn_area(min(max(float(r), 0.0), np.nextafter(g.burnout_regression_m, 0.0)))
            if is_active else 0.0
            for g, r, is_active in zip(self.motor.grains, regressions, active)
        ])
        activation = self.evaluate_burn_area_activation(time, float(np.mean(regressions)))
        port_area = np.mean([g.evaluate_port_area(r) for g, r in zip(self.motor.grains, regressions)])
        nozzle_flow = self.evaluate_nozzle_mass_flow(pressure, chamber_temperature=temperature)
        port_mass_flux = 0.5 * nozzle_flow / max(float(port_area), 1e-9)
        burn_rate = self.propellant.evaluate_burn_rate(pressure_for_rate, port_mass_flux) if np.any(active) else 0.0
        regression_rates = np.asarray(active, dtype=float) * burn_rate * activation
        if not math.isfinite(float(burn_rate)) or burn_rate < 0:
            raise ValueError("propellant burn rate must be finite and non-negative")
        generated_grains = self.propellant.density * areas * regression_rates
        igniter_flow = self.evaluate_igniter_mass_flow(time)
        components = self.evaluate_thrust_components(pressure, chamber_temperature=temperature)
        return {
            "pressure": pressure, "volume": volume, "temperature": temperature,
            "regressions": regressions, "regression_rates": regression_rates,
            "areas": areas * activation, "generated_grains": generated_grains,
            "generated": float(np.sum(generated_grains)), "igniter": igniter_flow,
            "nozzle": nozzle_flow, "components": components,
        }

    def _conservative_rhs(self, time, state, active):
        q = self._state_quantities(time, state, active)
        source_temperature = self._parameters_at_pressure(q["pressure"])[0]
        mass_rate = q["generated"] + q["igniter"] - q["nozzle"]
        thermal_rate = (q["generated"] * source_temperature
                        + q["igniter"] * self.igniter_temperature
                        - q["nozzle"] * q["temperature"])
        unscaled_thrust = q["components"]["momentum_n"] + q["components"]["pressure_n"]
        return [mass_rate, thermal_rate, *q["regression_rates"],
                q["generated"], q["igniter"], q["nozzle"], unscaled_thrust,
                q["pressure"] * self.motor.nozzle_throat_area]

    def vector_field(self, time, state_variables):
        """Return derivatives for the legacy pressure, volume, regression state.

        The canonical solver separately integrates gas mass and a prescribed
        source-temperature mixing inventory. This compatibility entry point
        assumes the effective combustion temperature at the supplied state.
        """
        n = len(self.motor.grains)
        pressure = max(float(state_variables[0]), self.environment_pressure)
        volume = float(state_variables[1])
        temperature, gas_constant, density, _, _ = self._parameters_at_pressure(pressure)
        regressions = state_variables[2:2 + n]
        geometric_volume = self.motor.chamber_volume - sum(g.calculate_remaining_volume(r) for g, r in zip(self.motor.grains, regressions))
        mass = pressure * geometric_volume / (gas_constant * temperature)
        state = [mass, mass * temperature, *state_variables[2:2 + n], *([0.0] * 5)]
        q = self._state_quantities(time, state)
        dv_dt = q["generated"] / density
        thermal_rate = (q["generated"] * temperature + q["igniter"] * self.igniter_temperature
                        - q["nozzle"] * temperature)
        dp_dt = (gas_constant * thermal_rate - pressure * dv_dt) / volume
        return [dp_dt, dv_dt, *q["regression_rates"]]

    @staticmethod
    def _join_segments(segments):
        times, states = [], []
        for segment in segments:
            start = 1 if times else 0
            times.extend(segment.t[start:])
            states.extend(segment.y[:, start:].T)
        return np.asarray(times), np.asarray(states).T

    def _integrate_stage(self, start, stop, state, active, cutoff=None, *, stop_after_burnout=True):
        segments = []
        success = True
        reached_cutoff = False
        for boundary in self._source_breakpoints(start, stop):
            while start < boundary:
                events = []
                grain_event_indices = []
                for index in np.flatnonzero(active):
                    def burnout(time, y, index=index):
                        return y[2 + index] - self.motor.grains[index].burnout_regression_m
                    burnout.terminal = True
                    burnout.direction = 1
                    events.append(burnout)
                    grain_event_indices.append(index)
                if cutoff is not None:
                    def end_blowdown(time, y):
                        return self._state_quantities(time, y, active)["pressure"] - cutoff
                    end_blowdown.terminal = True
                    end_blowdown.direction = -1
                    events.append(end_blowdown)
                def rhs(time, y):
                    source_time = np.nextafter(boundary, -np.inf) if time >= boundary else time
                    return self._conservative_rhs(source_time, y, active)
                raw = solve_ivp(
                    rhs, (start, boundary), state, method="DOP853", events=events or None,
                    max_step=self.max_step_size, rtol=self.rtol, atol=self.atol,
                )
                segments.append(raw)
                state = raw.y[:, -1].copy()
                previous = start
                start = float(raw.t[-1])
                if not raw.success:
                    success = False
                    break
                if raw.status == 1:
                    if cutoff is not None and len(raw.t_events[-1]):
                        reached_cutoff = True
                        break
                    tolerance = 64 * np.finfo(float).eps
                    reported_events = {
                        index
                        for event_position, index in enumerate(grain_event_indices)
                        if len(raw.t_events[event_position])
                    }
                    # Trust the localized terminal event even when its returned state undershoots the depth.
                    # Coalesce exact same-depth, same-state grains; those events are simultaneous by construction.
                    simultaneous = set(reported_events)
                    for event_index in reported_events:
                        event_depth = self.motor.grains[event_index].burnout_regression_m
                        event_regression = state[2 + event_index]
                        simultaneous.update(
                            index for index in grain_event_indices
                            if self.motor.grains[index].burnout_regression_m == event_depth
                            and state[2 + index] == event_regression
                        )
                    for index in grain_event_indices:
                        depth = self.motor.grains[index].burnout_regression_m
                        if index in simultaneous or state[2 + index] >= depth * (1 - tolerance):
                            state[2 + index] = depth
                            active[index] = False
                            self._burnout_times[index] = start
                    raw.y[:, -1] = state
                    if stop_after_burnout and not np.any(active):
                        break
                if start <= previous and np.any(active):
                    success = False
                    break
            if not success or reached_cutoff or (stop_after_burnout and not np.any(active)):
                break
        if not segments:
            segments = [SimpleNamespace(t=np.asarray([start]), y=np.asarray(state)[:, None])]
        times, states = self._join_segments(segments)
        return SimpleNamespace(t=times, y=states, success=success, reached_cutoff=reached_cutoff)

    def _public_solution(self, raw):
        quantities = [self._state_quantities(t, y) for t, y in zip(raw.t, raw.y.T)]
        regression = raw.y[2:2 + len(self.motor.grains)]
        temperatures = np.asarray([q["temperature"] for q in quantities])
        pressure = np.asarray([q["pressure"] for q in quantities])
        volume = np.asarray([q["volume"] for q in quantities])
        return [raw.t, pressure, volume, np.mean(regression, axis=0),
                *self.process_solution(pressure, temperatures)]

    def solve_burn(self):
        """Integrate source burning with independent terminal grain events."""
        self._burnout_times = [None] * len(self.motor.grains)
        temperature, gas_constant, _, _, _ = self._parameters_at_pressure(self.environment_pressure)
        mass = self.environment_pressure * self.motor.free_volume / (gas_constant * temperature)
        n = len(self.motor.grains)
        initial = [mass, mass * temperature, *([0.0] * n), *([0.0] * 5)]
        self.initial_gas_temperature_k = temperature
        self._gas_mass_initial = mass
        self._burn_raw = self._integrate_stage(0.0, self.burn_timeout_s, initial, np.ones(n, dtype=bool))
        if not self._burn_raw.success:
            self._termination_reason = "solver_failure"
        elif all(t is not None for t in self._burnout_times):
            self._termination_reason = "propellant_burnout"
        else:
            self._termination_reason = "burn_timeout"
        public = self._public_solution(self._burn_raw)
        return SimpleNamespace(t=self._burn_raw.t, y=np.vstack(public[1:3] + list(self._burn_raw.y[2:2 + n])),
                               success=self._burn_raw.success, conservative_state=self._burn_raw.y)

    def solve_tail_off_regime(self):
        if self.tail_off_method == "analytical":
            return self.solve_analytical_tail_off_regime()
        return self.solve_numerical_tail_off_regime()

    def solve_analytical_tail_off_regime(self):
        """Retain a sampled isothermal exponential approximation for legacy use."""
        start = float(self._burn_raw.t[-1])
        state = self._burn_raw.y[:, -1]
        q = self._state_quantities(start, state)
        coefficient = (self.propellant.products_constant * q["temperature"] * self.motor.nozzle_throat_area
                       * self.discharge_coefficient / (self.propellant.get_cstar(q["pressure"]) * self.eta_c))
        peak = max(self.grain_burn_solution[1])
        cutoff = self.environment_pressure + 0.01 * max(peak - self.environment_pressure, 0.0)
        duration = min(self.tail_off_timeout_s, max(q["volume"] / coefficient * math.log(max(q["pressure"] / max(cutoff, 1.0), 1.0)), 0.0))
        time = (np.linspace(start, start + duration, max(2, int(math.ceil(duration / self.max_step_size)) + 1))
                if duration > 0 else np.asarray([start]))
        pressure = q["pressure"] * np.exp(-coefficient * (time - start) / q["volume"])
        mass = pressure * q["volume"] / (self.propellant.products_constant * q["temperature"])
        states = np.repeat(state[:, None], len(time), axis=1)
        states[0] = mass
        states[1] = mass * q["temperature"]
        offset = 2 + len(self.motor.grains)
        states[offset + 2] += state[0] - mass
        components = [self.evaluate_thrust_components(p, chamber_temperature=q["temperature"]) for p in pressure]
        thrust = [c["momentum_n"] + c["pressure_n"] for c in components]
        states[offset + 3] += cumulative_trapezoid(thrust, time, initial=0.0)
        states[offset + 4] += cumulative_trapezoid(pressure * self.motor.nozzle_throat_area, time, initial=0.0)
        self._tail_raw = SimpleNamespace(t=time, y=states, success=True, reached_cutoff=False)
        self._termination_reason = "analytical_approximation"
        return self._public_solution(self._tail_raw)[:4]

    def solve_numerical_tail_off_regime(self):
        """Continue all gas inventories until the 1% peak gauge-pressure event."""
        start = float(self._burn_raw.t[-1])
        state = self._burn_raw.y[:, -1].copy()
        if self._termination_reason != "propellant_burnout":
            self._tail_raw = SimpleNamespace(t=np.asarray([start]), y=state[:, None], success=False)
            return self._public_solution(self._tail_raw)[:4]
        stop = start + self.tail_off_timeout_s
        active = np.zeros(len(self.motor.grains), dtype=bool)
        source_end = self._source_end_time()
        segments = [self._burn_raw]
        if source_end > start:
            before = self._integrate_stage(start, min(source_end, stop), state, active, stop_after_burnout=False)
            segments.append(before)
            start, state = float(before.t[-1]), before.y[:, -1]
            if not before.success:
                self._termination_reason = "solver_failure"
                self._tail_raw = before
                return self._public_solution(before)[:4]
        reference_peak = max(self._state_quantities(t, y)["pressure"] for raw in segments for t, y in zip(raw.t, raw.y.T))
        self._blowdown_reference_peak = reference_peak
        cutoff = self.environment_pressure + 0.01 * max(reference_peak - self.environment_pressure, 0.0)
        self._blowdown_cutoff_pressure = cutoff
        if start >= stop:
            after = SimpleNamespace(t=np.asarray([start]), y=state[:, None], success=True, reached_cutoff=False)
        elif self._state_quantities(start, state)["pressure"] <= cutoff:
            after = SimpleNamespace(t=np.asarray([start]), y=state[:, None], success=True, reached_cutoff=True)
        else:
            after = self._integrate_stage(start, stop, state, active, cutoff=cutoff)
        tails = segments[1:] + [after]
        time, states = self._join_segments(tails)
        self._tail_raw = SimpleNamespace(t=time, y=states, success=after.success, reached_cutoff=after.reached_cutoff)
        self._termination_reason = ("completed" if after.reached_cutoff else "blowdown_timeout") if after.success else "solver_failure"
        if callable(self.igniter_mass_flow) and self.igniter_burn_time <= 0:
            self._termination_reason = "unknown_igniter_duration"
        return self._public_solution(self._tail_raw)[:4]

    def process_solution(self, chamber_pressure_list, temperatures=None):
        if temperatures is None:
            temperatures = [None] * len(chamber_pressure_list)
        thrust, exit_pressure, exit_velocity = [], [], []
        for pressure, temperature in zip(chamber_pressure_list, temperatures):
            thrust.append(self.evaluate_thrust(pressure, chamber_temperature=temperature))
            exit_pressure.append(self.evaluate_exit_pressure(pressure))
            exit_velocity.append(self.evaluate_exit_velocity(pressure, chamber_temperature=temperature))
        return thrust, exit_pressure, exit_velocity

    def evaluate_grain_burn_solution(self):
        self.solve_burn()
        self.per_grain_regression_burn = list(self._burn_raw.y[2:2 + len(self.motor.grains)])
        return self._public_solution(self._burn_raw)

    def evaluate_tail_off_solution(self):
        self.solve_tail_off_regime()
        return self._public_solution(self._tail_raw)

    def evaluate_complete_solution(self):
        if self.tail_off_solution is None:
            return list(self.grain_burn_solution)
        return [np.concatenate((burn, tail[1:])) for burn, tail in zip(self.grain_burn_solution, self.tail_off_solution)]

    @staticmethod
    def _flow_interval(time, flow):
        indices = np.flatnonzero(flow > 0.0)
        if not len(indices):
            return 0.0, 0.0
        start = float(time[max(int(indices[0]) - 1, 0)])
        end = float(time[min(int(indices[-1]) + 1, len(time) - 1)])
        return start, end

    def _build_result(self, tail_off_evaluation):
        raws = [self._burn_raw] + ([self._tail_raw] if self.tail_off_solution is not None else [])
        time, states = self._join_segments(raws)
        quantities = [self._state_quantities(t, y) for t, y in zip(time, states.T)]
        def series(key):
            return np.asarray([q[key] for q in quantities], dtype=float)
        components = [q["components"] for q in quantities]
        n = len(self.motor.grains)
        offset = 2 + n
        history = {
            "time_s": time, "chamber_pressure_pa": series("pressure"),
            "free_volume_m3": series("volume"), "regression_m": states[2:offset].T,
            "regression_rate_m_s": series("regression_rates"),
            "burn_area_m2": np.sum(series("areas"), axis=1),
            "burn_area_grains_m2": series("areas"),
            "mdot_generated_kg_s": series("generated"),
            "mdot_generated_grains_kg_s": series("generated_grains"),
            "mdot_igniter_kg_s": series("igniter"), "mdot_nozzle_kg_s": series("nozzle"),
            "gas_mass_kg": states[0], "gas_temperature_k": series("temperature"),
            "thrust_n": np.asarray([c["total_n"] for c in components]),
            "momentum_ideal_n": np.asarray([c["momentum_ideal_n"] for c in components]),
            "momentum_n": np.asarray([c["momentum_n"] for c in components]),
            "pressure_n": np.asarray([c["pressure_n"] for c in components]),
            "exit_pressure_pa": np.asarray([self.evaluate_exit_pressure(p) for p in series("pressure")]),
            "exit_velocity_m_s": np.asarray([self.evaluate_exit_velocity(p, chamber_temperature=t) for p, t in zip(series("pressure"), series("temperature"))]),
            "generated_mass_integral_kg": states[offset],
            "igniter_mass_integral_kg": states[offset + 1],
            "nozzle_mass_integral_kg": states[offset + 2],
            "impulse_integral_ns": states[offset + 3] * self.eta_Cf,
            "pressure_throat_integral_ns": states[offset + 4],
        }
        solid_remaining = sum(g.calculate_remaining_volume(r) for g, r in zip(self.motor.grains, history["regression_m"][-1]))
        consumed = self.propellant.density * (self.motor.propellant_volume - solid_remaining)
        generated, igniter, nozzle = (float(states[offset + i, -1]) for i in range(3))
        residual = consumed + igniter - nozzle - (states[0, -1] - states[0, 0])
        burn_start, burn_end = self._flow_interval(time, history["mdot_generated_kg_s"])
        if all(t is not None for t in self._burnout_times):
            burn_end = max(self._burnout_times)
        nozzle_start, nozzle_end = self._flow_interval(time, history["mdot_nozzle_kg_s"])
        burn_duration, nozzle_duration = burn_end - burn_start, nozzle_end - nozzle_start
        metrics = {
            "propellant_mass_consumed_kg": float(consumed), "integrated_generated_mass_kg": generated,
            "generated_mass_integral_kg": generated,
            "propellant_mass_initial_kg": float(self.motor.propellant_volume * self.propellant.density),
            "propellant_mass_remaining_kg": float(solid_remaining * self.propellant.density),
            "propellant_burn_start_s": burn_start, "propellant_burn_end_s": burn_end,
            "propellant_burn_duration_s": burn_duration,
            "nozzle_flow_start_s": nozzle_start, "nozzle_flow_end_s": nozzle_end,
            "nozzle_flow_duration_s": nozzle_duration,
            "mass_flow_avg_generated_kg_s": generated / burn_duration if burn_duration > 0 else 0.0,
            "max_generated_mass_flow_kg_s": float(np.max(history["mdot_generated_kg_s"])),
            "mass_flow_avg_nozzle_kg_s": nozzle / nozzle_duration if nozzle_duration > 0 else 0.0,
            "max_nozzle_mass_flow_kg_s": float(np.max(history["mdot_nozzle_kg_s"])),
            "nozzle_mass_integral_kg": nozzle, "igniter_mass_injected_kg": igniter,
            "gas_mass_initial_kg": float(states[0, 0]), "gas_mass_cutoff_kg": float(states[0, -1]),
            "other_declared_outflows_kg": 0.0, "mass_balance_residual_kg": float(residual),
            "mass_flow_balance_error_pct": float(100 * abs(residual) / max(consumed + igniter, 1e-15)),
            "total_impulse_ns": float(history["impulse_integral_ns"][-1]),
            "peak_chamber_pressure_pa": float(np.max(history["chamber_pressure_pa"])),
            "peak_thrust_n": float(np.max(history["thrust_n"])),
            "pressure_throat_integral_ns": float(history["pressure_throat_integral_ns"][-1]),
            "grain_burnout_times_s": tuple(self._burnout_times),
        }
        cea_used = self.propellant.cea_formulation is not None or self.propellant._cea_obj is not None
        scalar = not cea_used and self.propellant._thermo_table is None
        reason = self._termination_reason
        if not tail_off_evaluation and reason == "propellant_burnout":
            reason = "tail_off_omitted"
        numerical_completed = reason == "completed"
        if not scalar:
            reason = "unsupported_thermochemistry"
        resolved = {
            "combustion_temperature_k": float(self.propellant.combustion_temperature),
            "gas_constant_j_kg_k": float(self.propellant.products_constant),
            "propellant_density_kg_m3": float(self.propellant.density),
            "specific_heat_ratio": float(self.propellant.specific_heat_ratio),
            "initial_gas_temperature_k": self.initial_gas_temperature_k,
            "igniter_temperature_k": self.igniter_temperature,
            "connected_chamber_volume_m3": self.motor.chamber_volume,
            "physical_chamber_length_m": self.motor.chamber_length,
            "grain_geometry_models": [g.geometry_model for g in self.motor.grains],
            "grain_geometry": [
                {"outer_radius_m": g.outer_radius, "inner_radius_m": g.initial_inner_radius,
                 "height_m": g.initial_height, "ends_inhibited": g.ends_burn,
                 "geometry": g.geometry, "n_points": g.n_points, "epsilon_rad": g.epsilon,
                 "slot_fraction": g.slot_fraction}
                for g in self.motor.grains
            ],
            "nozzle_throat_area_m2": self.motor.nozzle_throat_area,
            "nozzle_exit_area_m2": self.motor.nozzle_exit_area,
            "nozzle_angle_rad": self.motor.nozzle_angle,
            "ambient_pressure_pa": self.environment_pressure,
            "burn_rate_a": getattr(self.propellant, "burn_rate_a", None),
            "burn_rate_n": getattr(self.propellant, "burn_rate_n", None),
            "erosive_burning_coefficient": getattr(self.propellant, "erosive_burning_coefficient", 0.0),
            "erosive_alpha": getattr(self.propellant, "erosive_alpha", 35.0),
            "efficiencies": self.applied_efficiencies,
            "igniter_burn_time_s": self.igniter_burn_time,
            "ignition_ramp_time_s": self.ignition_ramp_time,

        }
        interpolator = self.propellant._burn_rate_interpolator
        if interpolator is not None:
            resolved["burn_rate_pressure_table_mpa"] = interpolator.x.tolist()
            resolved["burn_rate_table_mm_s"] = interpolator.y.tolist()
        for name in ("igniter_mass_flow", "burn_area_activation"):
            source = getattr(self, name)
            if callable(source):
                resolved[name] = {"type": "callable", "identity": getattr(source, "__qualname__", type(source).__qualname__),
                                  "module": getattr(source, "__module__", None), "support_s": self.igniter_burn_time if name == "igniter_mass_flow" else None}
            elif source is None:
                resolved[name] = None
            elif np.isscalar(source):
                resolved[name] = float(source)
            else:
                resolved[name] = np.asarray(source).tolist()
        source_hash = hashlib.sha256()
        source_hash.update(json.dumps(resolved, sort_keys=True, default=float).encode())
        for filename in ("Burn.py", "Grain.py", "Propellant.py"):
            source_hash.update((Path(__file__).parent / filename).read_bytes())
        try:
            checkout_root = Path(__file__).resolve().parent.parent
            git_root = subprocess.run(["git", "rev-parse", "--show-toplevel"], cwd=Path(__file__).parent,
                                      capture_output=True, text=True, timeout=1, check=True).stdout.strip()
            git_sha = (subprocess.run(["git", "rev-parse", "HEAD"], cwd=checkout_root,
                                      capture_output=True, text=True, timeout=1, check=True).stdout.strip()
                       if Path(git_root).resolve() == checkout_root else None)
        except (OSError, subprocess.SubprocessError):
            git_sha = None
        return {
            "history": history, "metrics": metrics,
            "status": {"completed": numerical_completed and scalar, "termination_reason": reason,
                       "numerical_blowdown_completed": numerical_completed,
                       "scalar_contract_supported": scalar,
                       "burnout_completed": all(t is not None for t in self._burnout_times),
                       "blowdown_cutoff_pressure_pa": getattr(self, "_blowdown_cutoff_pressure", None),
                       "blowdown_reference_peak_pressure_pa": getattr(self, "_blowdown_reference_peak", None)},
            "efficiencies": {**self.applied_efficiencies, "efficiency_semantics": "native_split"},
            "provenance": {
                "eta_c_applied": self.eta_c, "eta_cf_applied": self.eta_Cf,
                "discharge_coefficient_applied": self.discharge_coefficient,
                "efficiency_semantics": "native_split", "cea_used": cea_used,
                "thermochemistry_source": "cea_legacy" if cea_used else ("scalar" if scalar else "pressure_table_legacy"),
                "physics_provider_hash": source_hash.hexdigest(), "solidpy_git_sha": git_sha,
                "solidpy_git_sha_status": "available" if git_sha else "unavailable",
                "resolved_inputs": resolved,
                "gas_temperature_model": "prescribed_source_temperature_mixing_v1",
                "activation_model": "uniform_front_rate_scaling_v1",
                "flow_interval_method": "adaptive_positive_source_bracket",
                "integration_method": "adaptive_ode_quadrature",
                "solver_settings": {"method": "DOP853", "rtol": self.rtol, "atol": self.atol,
                                    "max_step_size_s": self.max_step_size,
                                    "burn_timeout_s": self.burn_timeout_s, "tail_off_timeout_s": self.tail_off_timeout_s,
                                    "tail_off_method": self.tail_off_method},
            },
        }


class BurnExport(Export):
    def __init__(self, BurnSimulation):
        self.BurnSimulation = BurnSimulation
        (
            self.time,
            self.chamber_pressure,
            self.free_volume,
            self.regressed_length,
            self.thrust,
            self.exit_pressure,
            self.exit_velocity,
        ) = self.BurnSimulation.total_burn_solution

        self.burn_exporting()
        self.post_processing()

    def burn_exporting(self):
        """Method that calls Export class for solution exporting in a csv.

        Returns:
            None
        """
        try:
            Export.raw_simulation_data_export(
                self.BurnSimulation.total_burn_solution,
                "data/burn_simulation/burn_data.csv",
                [
                    "Time",
                    "Chamber Pressure",
                    "Free Volume",
                    "Regressed Length",
                    "Thrust",
                    "Exit Pressure",
                    "Exit Velocity",
                ],
            )
        except OSError as err:
            print("OS error: {0}".format(err))
        return None

    def post_processing(self):
        """Method for post solution values processing, allowing for final
        notable burn evaluations and notable solution points, such as extrema.

        Returns:
            None
        """
        (
            self.max_chamber_pressure,
            self.end_free_volume,
            self.end_regressed_length,
            self.max_thrust,
            self.max_exit_pressure,
            self.max_exit_velocity,
        ) = Export.evaluate_max_variables_list(
            self.BurnSimulation.total_burn_solution[0],
            self.BurnSimulation.total_burn_solution[1:],
        )

        # Parabolic refinement of the raw argmax pressure peak: important
        # for structural sizing when the adaptive mesh brackets the true
        # pressure maximum between samples. Falls back silently to the
        # raw argmax when the bracket would not yield a concave-down vertex.
        self.max_chamber_pressure_refined = Export.refine_peak_parabolic(
            self.BurnSimulation.total_burn_solution[0],
            self.BurnSimulation.total_burn_solution[1],
        )

        self.total_impulse = self.BurnSimulation.evaluate_total_impulse(
            self.thrust, self.time
        )

        self.specific_impulse = self.BurnSimulation.evaluate_specific_impulse(
            self.thrust, self.time
        )

        self.propellant_mass = (
            sum(g.volume for g in self.BurnSimulation.motor.grains)
            * self.BurnSimulation.propellant.density
        )

        return None

    def all_info(self):
        """Console logging of notable burn characteristics.

        Returns:
            None
        """
        print("Total Impulse: {:.2f} Ns".format(self.total_impulse))
        print("Max Thrust: {:.2f} N at {:.2f} s".format(*self.max_thrust))
        print("Mean Thrust: {:.2f} N".format(Export.evaluate_mean(self.thrust)))
        print(
            "Max Chamber Pressure: {:.2f} bar at {:.2f} s".format(
                self.max_chamber_pressure[0] / 1e5, self.max_chamber_pressure[1]
            )
        )
        print(
            "Max Chamber Pressure (refined): {:.2f} bar at {:.2f} s".format(
                self.max_chamber_pressure_refined[0] / 1e5,
                self.max_chamber_pressure_refined[1],
            )
        )
        print(
            "Mean Chamber Pressure: {:.2f} bar".format(
                Export.evaluate_mean(self.chamber_pressure) / 1e5
            )
        )
        print("Propellant mass: {:.2f} g".format(1000 * self.propellant_mass))
        print("Specific Impulse: {:.2f} s".format(self.specific_impulse))
        print("Burnout Time: {:.2f} s".format(self.time[-1]))

        return None

    def plotting(self):
        """Plot graphs of notable burn list values.

        Returns:
            None
        """
        plt.figure(1, figsize=(16, 9))
        plt.plot(self.time, self.thrust, color="b", linewidth=0.75, label=r"$F_T$")
        plt.grid(True)
        plt.xlabel("time (s)")
        plt.ylabel("thrust (N)")
        plt.legend(prop=FontProperties(size=16))
        plt.title("Thrust as function of time")
        plt.savefig("data/burn_simulation/graphs/thrust.png", dpi=200)

        plt.figure(2, figsize=(16, 9))
        plt.plot(
            self.time, self.chamber_pressure, color="b", linewidth=0.75, label=r"$p_c$"
        )
        plt.grid(True)
        plt.xlabel("time (s)")
        plt.ylabel("chamber pressure (pa)")
        plt.legend(prop=FontProperties(size=16))
        plt.title("Chamber Pressure as function of time")
        plt.savefig("data/burn_simulation/graphs/chamber_pressure.png", dpi=200)

        plt.figure(3, figsize=(16, 9))
        plt.plot(
            self.time, self.exit_pressure, color="b", linewidth=0.75, label=r"$p_e$"
        )
        plt.grid(True)
        plt.xlabel("time (s)")
        plt.ylabel("exit pressure (pa)")
        plt.legend(prop=FontProperties(size=16))
        plt.title("Exit Pressure as function of time")
        plt.savefig("data/burn_simulation/graphs/exit_pressure.png", dpi=200)

        plt.figure(4, figsize=(16, 9))
        plt.plot(
            self.time, self.free_volume, color="b", linewidth=0.75, label=r"$\forall_c$"
        )
        plt.grid(True)
        plt.xlabel("time (s)")
        plt.ylabel("free volume (m³)")
        plt.legend(prop=FontProperties(size=16))
        plt.title("Free Volume as function of time")
        plt.savefig("data/burn_simulation/graphs/free_volume.png", dpi=200)

        plt.figure(5, figsize=(16, 9))
        plt.plot(
            self.time,
            self.regressed_length,
            color="b",
            linewidth=0.75,
            label=r"$\ell_{regr}$",
        )
        plt.grid(True)
        plt.xlabel("time (s)")
        plt.ylabel("regressed length (m)")
        plt.legend(prop=FontProperties(size=16))
        plt.title("Regressed Grain Length as function of time")
        plt.savefig("data/burn_simulation/graphs/regressed_length.png", dpi=200)

        if self.BurnSimulation.tail_off_solution:
            plt.figure(6, figsize=(16, 9))
            plt.plot(
                self.BurnSimulation.tail_off_solution[0],
                self.BurnSimulation.tail_off_solution[1],
                color="b",
                linewidth=0.75,
                label=r"$p^{toff}_c$",
            )
            plt.grid(True)
            plt.xlabel("time (s)")
            plt.ylabel("tail of chamber pressure (pa)")
            plt.legend(prop=FontProperties(size=16))
            plt.title("Tail Off Chamber Pressure as function of time")
            plt.savefig(
                "data/burn_simulation/graphs/tail_off_chamber_pressure.png", dpi=200
            )

        return None


if __name__ == "__main__":
    """Burn definitions"""
    Grao_Leviata = Grain(
        outer_radius=71.92 / 2000,
        initial_inner_radius=31.92 / 2000,
    )
    Leviata = Motor(
        Grao_Leviata,
        grain_number=4,
        chamber_inner_radius=77.92 / 2000,
        nozzle_throat_radius=17.5 / 2000,
        nozzle_exit_radius=44.44 / 2000,
        nozzle_angle=15 * np.pi / 180,
        chamber_length=600 / 1000,
    )
    KNSB = Propellant(
        specific_heat_ratio=1.1361,
        density=1700,
        products_molecular_mass=39.9e-3,
        combustion_temperature=1600,
        # burn_rate_a=5.13,
        # burn_rate_n=0.22,
        interpolation_list="data/burnrate/KNSB3.csv",
        # interpolation_list="data/burnrate/simulated/KNSB_Leviata_sim.csv",
    )

    Ambient = Environment(latitude=-0.38390456, altitude=627, ellipsoidal_model=True)

    """Class instances"""
    Simulation = BurnSimulation(
        Grao_Leviata, Leviata, KNSB, Ambient, tail_off_evaluation=True
    )
    ExportPlot = BurnExport(Simulation)

    """Desired outputs"""
    ExportPlot.all_info()
    ExportPlot.plotting()
