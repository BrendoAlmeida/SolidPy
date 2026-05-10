# -*- coding: utf-8 -*-

_author_ = ""
_copyright_ = "MIT"
_license_ = ""

import math
import numpy as np
import matplotlib.pyplot as plt

from scipy.optimize import fsolve
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
    def __init__(self, grain, motor, propellant, environment=None):
        if environment is None:
            environment = Environment()
        self.motor = motor
        self.grain = grain
        self.propellant = propellant
        self.environment = environment

        self.gravity = environment.gravity
        self.environment_pressure = environment.atmospheric_pressure

        self.parameters = self.set_parameters()
        self._exit_mach_cache = {}

    def set_parameters(self):
        parameters = (
            self.propellant.combustion_temperature,  # T_0
            self.propellant.products_constant,  # R
            self.propellant.density,  # rho_g
            self.propellant.specific_heat_ratio,  # k
            self.motor.nozzle_throat_area,  # A_t
        )
        return parameters

    def evaluate_nozzle_mass_flow(self, chamber_pressure):
        """Calculation of total nozzle mass flow.

        Source:
            https://www.grc.nasa.gov/www/k-12/rocket/rktthsum.html

        Args:
            chamber_pressure (float): current chamber pressure

        Returns:
            float: nozzle mass flow for the specified chamber pressure
        """
        T_0, R, _, k, A_t = self.parameters
        if chamber_pressure <= self.environment_pressure:
            return 0.0

        pressure_ratio = self.environment_pressure / chamber_pressure
        critical_pressure_ratio = math.pow(2 / (k + 1), k / (k - 1))

        if pressure_ratio <= critical_pressure_ratio:
            return (
                chamber_pressure
                * A_t
                * np.sqrt(k / (R * T_0))
                * math.pow((2 / (k + 1)), ((k + 1) / (2 * (k - 1))))
            )

        unchoked_term = (
            math.pow(pressure_ratio, 2 / k)
            - math.pow(pressure_ratio, (k + 1) / k)
        )
        return (
            A_t
            * chamber_pressure
            * math.sqrt((2 * k / ((k - 1) * R * T_0)) * max(unchoked_term, 0.0))
        )

    def is_nozzle_choked(self, chamber_pressure):
        """Return whether chamber-to-ambient pressure ratio chokes the throat."""
        _, _, _, k, _ = self.parameters
        if chamber_pressure <= self.environment_pressure:
            return False
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
        _, _, _, k, _ = self.parameters
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

        cache_key = (k, self.motor.expansion_ratio)
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

        self.exit_mach = fsolve(func, np.array(2))[0]
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
        _, _, _, k, _ = self.parameters
        if not self.is_nozzle_choked(chamber_pressure):
            self.exit_pressure = self.environment_pressure
            return self.exit_pressure

        self.exit_pressure = chamber_pressure * math.pow(
            (1 + (k - 1) / 2 * self.evaluate_exit_mach(chamber_pressure) ** 2),
            -k / (k - 1),
        )
        return self.exit_pressure

    def evaluate_exit_temperature(self, chamber_pressure=None):
        """Calculation of fluid temperature at nozzle exit.

        Source:
        https://www.grc.nasa.gov/www/k-12/rocket/rktthsum.html

        Returns:
            float: exit temperature
        """
        T_0, _, _, k, _ = self.parameters
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

    def evaluate_exit_velocity(self, chamber_pressure=None):
        """Calculation of fluid velocity at nozzle exit.

        Source:
        https://www.grc.nasa.gov/www/k-12/rocket/rktthsum.html

        Returns:
            float: exit velocity
        """
        _, R, _, k, _ = self.parameters
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
                    * self.propellant.combustion_temperature
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
            k * R * self.evaluate_exit_temperature(chamber_pressure)
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

    def evaluate_Cf(self, chamber_pressure):
        """Calculation of the engine's thrust coefficient.

        Source:
        Rogers, RasAero: The Solid Rocket Motor - Part 4 - Departures
        from Ideal Performance for Conical Nozzles and Bell Nozzles,
        page 28, eq.(9). High Power Rocketry.

        Args:
            chamber_pressure (float): current chamber pressure

        Returns:
            (float): the motor's thrust coefficient for
            a given chamber pressure
        """
        _, _, _, k, _ = self.parameters
        lambda_div = self._nozzle_divergence_factor()

        if chamber_pressure <= self.environment_pressure:
            self.Cf = 0.0
            return self.Cf
        if not self.is_nozzle_choked(chamber_pressure):
            thrust = self.evaluate_nozzle_mass_flow(
                chamber_pressure
            ) * self.evaluate_exit_velocity(chamber_pressure)
            self.Cf = lambda_div * thrust / (chamber_pressure * self.motor.nozzle_throat_area)
            return self.Cf

        self.Cf = lambda_div * (
            math.sqrt(
                (2 * k**2 / (k - 1))
                * math.pow(2 / (k + 1), (k + 1) / (k - 1))
                * (
                    1
                    - math.pow(
                        self.evaluate_exit_pressure(chamber_pressure)
                        / chamber_pressure,
                        (k - 1) / k,
                    )
                )
            )
            + (
                (
                    self.evaluate_exit_pressure(chamber_pressure)
                    - self.environment_pressure
                )
                / chamber_pressure
            )
            * self.motor.expansion_ratio
        )
        return self.Cf

    def evaluate_thrust(self, chamber_pressure):
        """Calculation of engine's thrust

        Args:
            chamber_pressure (float): current chamber

        Returns:
            float: motor's thrust for a given chamber pressure
        """
        self.thrust = (
            self.evaluate_Cf(chamber_pressure)
            * chamber_pressure
            * self.motor.nozzle_throat_area
        )
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
        T_0, R, rho_g, _, _ = self.parameters

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
    ):
        Burn.__init__(self, grain, motor, propellant, environment)
        self.max_step_size = max_step_size
        self.igniter_mass_flow = igniter_mass_flow
        self.igniter_burn_time = max(float(igniter_burn_time), 0.0)
        self.igniter_temperature = (
            propellant.combustion_temperature
            if igniter_temperature is None
            else float(igniter_temperature)
        )
        self.burn_area_activation = burn_area_activation
        self.ignition_ramp_time = max(float(ignition_ramp_time), 0.0)
        self.tail_off_method = str(tail_off_method).lower()

        self.grain_burn_solution = self.evaluate_grain_burn_solution()
        self.tail_off_solution = (
            self.evaluate_tail_off_solution() if tail_off_evaluation else None
        )
        self.total_burn_solution = self.evaluate_complete_solution()

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
            return max(float(source(time)), 0.0)
        if np.isscalar(source):
            if self.igniter_burn_time <= 0.0 or time > self.igniter_burn_time:
                return 0.0
            return max(float(source), 0.0)

        profile = np.asarray(source, dtype=float)
        if profile.ndim != 2 or profile.shape[1] != 2:
            raise ValueError(
                "igniter_mass_flow must be None, a callable, a scalar, or a two-column time/mass-flow table"
            )

        profile_time = profile[:, 0]
        profile_mass_flow = profile[:, 1]
        if time < profile_time[0] or time > profile_time[-1]:
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

    def vector_field(self, time, state_variables):
        """Vector field for the ODE state [P, V, r_0, r_1, ..., r_{n-1}].

        Each grain carries its own regression distance r_i so grains with
        different geometries can burn out independently. All grains share
        the same pressure-dependent burn rate (same propellant).

        Args:
            time (float): current time
            state_variables (sequence): [chamber_pressure, free_volume,
                r_grain_0, r_grain_1, ..., r_grain_{n-1}]

        Returns:
            list: time-derivatives of each state variable
        """
        n_grains = len(self.motor.grains)
        chamber_pressure = state_variables[0]
        free_volume = state_variables[1]
        per_grain_regression = list(state_variables[2:2 + n_grains])

        T_0, R, rho_g, _, _ = self.parameters
        rho_0 = chamber_pressure / (R * T_0)
        nozzle_mass_flow = self.evaluate_nozzle_mass_flow(chamber_pressure)
        igniter_mass_flow = self.evaluate_igniter_mass_flow(time)

        # Representative regression for activation/area calls that expect scalar.
        mean_regression = sum(per_grain_regression) / max(n_grains, 1)
        geometric_burn_area = self.compute_total_burn_area(per_grain_regression)
        burn_area = geometric_burn_area * self.evaluate_burn_area_activation(
            time, mean_regression
        )

        # Port mass flux G = ṁ_nozzle / A_port for Lenoir-Robillard erosive model.
        # Use the mean port area across all grains; star grains include slot area.
        total_port_area = sum(
            g.evaluate_port_area(r)
            for g, r in zip(self.motor.grains, per_grain_regression)
        )
        port_area = total_port_area / max(n_grains, 1)
        port_mass_flux = nozzle_mass_flow / max(port_area, 1e-9)

        burn_rate = self.propellant.evaluate_burn_rate(chamber_pressure, port_mass_flux)

        dp_dt = (
            (
                burn_area * burn_rate * (rho_g - rho_0)
                + igniter_mass_flow * self.igniter_temperature / T_0
                - nozzle_mass_flow
            )
            * R
            * T_0
            / free_volume
        )
        dv_dt = burn_area * burn_rate
        # Each grain regresses at the same burn rate (same propellant, same P).
        dr_dt = [burn_rate] * n_grains

        return [dp_dt, dv_dt] + dr_dt

    def solve_burn(self):
        """Initial conditions setting and solver instantiation.

        State vector: [chamber_pressure, free_volume, r_0, r_1, ..., r_{n-1}]
        where r_i is the regression distance for each grain.

        Returns:
            object: solution object from solve_ivp
        """
        n_grains = len(self.motor.grains)
        state_variables = (
            [self.environment_pressure, self.motor.free_volume]
            + [0.0] * n_grains
        )

        def end_burn_propellant(time, state_variables):
            free_volume = state_variables[1]
            per_grain_regression = list(state_variables[2:2 + n_grains])
            if (self.motor.chamber_volume - free_volume < 1e-6) or (
                self.compute_total_burn_area(per_grain_regression) <= 0.0
            ):
                return 0
            return 1

        end_burn_propellant.terminal = True

        solution = solve_ivp(
            self.vector_field,
            (0.0, 100.0),
            state_variables,
            method="DOP853",
            events=end_burn_propellant,
            max_step=max(float(self.max_step_size), 1e-6),
            atol=1e-8,
            rtol=1e-10,
        )

        return solution

    def solve_tail_off_regime(self):
        """Evaluate the chamber gas depressurization after total grain burn.

        Returns:
            list: solution of the tail off regime, grouping time steps
            and chamber pressure
        """
        if self.tail_off_method == "analytical":
            return self.solve_analytical_tail_off_regime()
        if self.tail_off_method != "numerical":
            raise ValueError("tail_off_method must be 'numerical' or 'analytical'")
        return self.solve_numerical_tail_off_regime()

    def solve_analytical_tail_off_regime(self):
        """Evaluate the original choked-flow analytical tail-off model."""

        T_0, R, _, _, A_t = self.parameters

        # Set initial values at the end of grain burn simulation
        (
            self.initial_tail_off_time,
            self.initial_tail_off_chamber_pressure,
            self.initial_tail_off_free_volume,
        ) = (
            self.grain_burn_solution[0][-1],
            self.grain_burn_solution[1][-1],
            self.grain_burn_solution[2][-1],
        )

        # Analytical solution to the fluid behavior after grain burn
        self.evaluate_tail_off_chamber_pressure = (
            lambda time: self.initial_tail_off_chamber_pressure
            * math.exp(
                -R
                * T_0
                * A_t
                / (self.initial_tail_off_free_volume * self.propellant.calc_cstar())
                * (time - self.initial_tail_off_time)
            )
        )

        # Keep the same time pacing for uniform union with burn solution
        time_steps = np.linspace(
            self.initial_tail_off_time,
            100.0,
            int((100.0 - self.initial_tail_off_time) / self.max_step_size),
        )

        tail_off_time = []
        tail_off_chamber_pressure = []
        tail_off_free_volume = []
        tail_off_regressed_length = []

        for time in time_steps:
            chamber_pressure = self.evaluate_tail_off_chamber_pressure(time)
            if chamber_pressure / self.environment_pressure > 1.0001:
                tail_off_time.append(time)
                tail_off_chamber_pressure.append(chamber_pressure)
                tail_off_free_volume.append(self.motor.chamber_volume)
                tail_off_regressed_length.append(
                    self.grain_burn_solution[3][-1]
                )
            else:
                break

        self.tail_off_solution = [
            tail_off_time,
            tail_off_chamber_pressure,
            tail_off_free_volume,
            tail_off_regressed_length,
        ]

        return self.tail_off_solution

    def solve_numerical_tail_off_regime(self):
        """Numerically integrate post-burn chamber blowdown."""
        T_0, R, _, _, _ = self.parameters

        initial_time = self.grain_burn_solution[0][-1]
        initial_chamber_pressure = self.grain_burn_solution[1][-1]
        free_volume = self.motor.chamber_volume
        regressed_length = self.grain_burn_solution[3][-1]

        if initial_chamber_pressure <= self.environment_pressure * 1.0001:
            self.tail_off_solution = [
                [initial_time],
                [initial_chamber_pressure],
                [free_volume],
                [regressed_length],
            ]
            return self.tail_off_solution

        def tail_off_vector(time, state_variables):
            chamber_pressure = max(float(state_variables[0]), self.environment_pressure)
            nozzle_mass_flow = self.evaluate_nozzle_mass_flow(chamber_pressure)
            return [-nozzle_mass_flow * R * T_0 / free_volume]

        def end_tail_off(time, state_variables):
            return state_variables[0] - self.environment_pressure * 1.0001

        end_tail_off.terminal = True
        end_tail_off.direction = -1

        solution = solve_ivp(
            tail_off_vector,
            (initial_time, 100.0),
            [initial_chamber_pressure],
            method="DOP853",
            events=end_tail_off,
            max_step=max(float(self.max_step_size), 1e-6),
            atol=1e-8,
            rtol=1e-10,
        )

        tail_off_time = solution.t
        tail_off_chamber_pressure = solution.y[0]
        tail_off_free_volume = np.full_like(tail_off_time, free_volume, dtype=float)
        tail_off_regressed_length = np.full_like(
            tail_off_time, regressed_length, dtype=float
        )

        self.tail_off_solution = [
            tail_off_time,
            tail_off_chamber_pressure,
            tail_off_free_volume,
            tail_off_regressed_length,
        ]

        return self.tail_off_solution

    def process_solution(self, chamber_pressure_list):
        """Iteration through the solve_ivp solution in order to compute
        notable burn characteristics besides the state variables, such as thrust,
        exit pressure and exit velocity.

        Args:
            chamber_pressure_list (list): chamber pressure solution
            evaluated by solve_ivp

        Returns:
            tuple: thrust, exit pressure and exit velocity lists
        """
        thrust_list = []
        exit_velocity_list = []
        exit_pressure_list = []

        for chamber_pressure in chamber_pressure_list:
            thrust_list.append(self.evaluate_thrust(chamber_pressure))
            exit_velocity_list.append(self.evaluate_exit_velocity(chamber_pressure))
            exit_pressure_list.append(self.evaluate_exit_pressure(chamber_pressure))

        return thrust_list, exit_pressure_list, exit_velocity_list

    def evaluate_grain_burn_solution(self):
        """Adapts solve_ivp results to a simple matrix containing each
        burn characteristic.

        The returned list layout is:
          [time, chamber_pressure, free_volume, regressed_length,
           thrust, exit_pressure, exit_velocity]

        regressed_length is the mean across all grains for backward
        compatibility. Per-grain regressions are stored in
        self.per_grain_regression_burn.

        Returns:
            list: list containing the solution and added burn computations
        """
        n_grains = len(self.motor.grains)
        raw = self.solve_burn()

        # Per-grain regressions: rows 2..2+n_grains of ODE solution.
        per_grain = [raw.y[2 + i] for i in range(n_grains)]
        self.per_grain_regression_burn = per_grain

        # Mean regression for downstream backward-compatible output.
        mean_regression = np.mean(np.vstack(per_grain), axis=0)

        grain_burn_solution = [
            raw.t,
            raw.y[0],
            raw.y[1],
            mean_regression,
            *self.process_solution(raw.y[0]),
        ]

        return grain_burn_solution

    def evaluate_tail_off_solution(self):
        """Adapts solve_ivp results to a simple matrix containing each
        burn characteristic.

        Returns:
            list: list containing the solution and added burn computations
        """
        tail_off_solution = self.solve_tail_off_regime()
        tail_off_solution = [
            *tail_off_solution,
            *self.process_solution(tail_off_solution[1]),
        ]

        return tail_off_solution

    def evaluate_complete_solution(self):
        """Groups both grain burn and tail-off solutions after processing.

        Returns:
            matrix: contains tail-off and grain burn regime
        """
        grain_solution = self.grain_burn_solution
        tail_off_solution = self.tail_off_solution
        total_burn_solution = []

        if tail_off_solution:
            tail_off_start = 0
            if (
                len(grain_solution[0]) > 0
                and len(tail_off_solution[0]) > 0
                and tail_off_solution[0][0] <= grain_solution[0][-1]
            ):
                tail_off_start = 1

            for grain_parameter, tail_off_parameter in zip(
                grain_solution, tail_off_solution
            ):
                total_burn_solution.append(
                    np.append(grain_parameter, tail_off_parameter[tail_off_start:])
                )
        else:
            for grain_parameter in grain_solution:
                total_burn_solution.append(grain_parameter)

        return total_burn_solution


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
