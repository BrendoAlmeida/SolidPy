import math

import numpy as np
import pytest

from solidpy import Burn, BurnSimulation, Environment, Grain, Motor, Propellant


@pytest.fixture
def motor_stack():
    grain = Grain(outer_radius=0.035, initial_inner_radius=0.015, initial_height=0.12)
    motor = Motor(
        grains=[grain],
        chamber_inner_radius=0.037,
        chamber_length=0.14,
        nozzle_throat_radius=0.008,
        nozzle_exit_radius=0.018,
        nozzle_angle=math.radians(15),
    )
    propellant = Propellant(
        specific_heat_ratio=1.1308,
        products_molecular_mass=0.04197,
        combustion_temperature=1720.0,
        density=1879.0,
        burn_rate_a=7.36,
        burn_rate_n=0.32,
    )
    return grain, motor, propellant, Environment()


@pytest.mark.parametrize("burn_class", [Burn, BurnSimulation])
@pytest.mark.parametrize("parameter", ["eta_c", "eta_Cf", "discharge_coefficient"])
@pytest.mark.parametrize(
    "value", [0.0, -0.1, 1.01, np.nan, np.inf, -np.inf, "0.9", "invalid", True, False, np.bool_(True), None]
)
def test_invalid_efficiencies_fail_before_solving(motor_stack, burn_class, parameter, value):
    with pytest.raises(ValueError, match=parameter):
        burn_class(*motor_stack, **{parameter: value})


def test_numeric_efficiency_boundaries_and_resolved_values(motor_stack):
    burn = Burn(*motor_stack, eta_c=1, eta_Cf=np.float64(1.0), discharge_coefficient=np.float32(0.75))
    assert burn.applied_efficiencies == {
        "eta_c": 1.0,
        "eta_Cf": 1.0,
        "discharge_coefficient": 0.75,
    }
    assert all(type(value) is float for value in burn.applied_efficiencies.values())


def test_thrust_decomposition_matches_isentropic_formula(motor_stack):
    burn = Burn(*motor_stack, eta_c=0.87, eta_Cf=0.91, discharge_coefficient=0.82)
    _, motor, propellant, environment = motor_stack
    pressure = 3.5e6
    gamma = propellant.specific_heat_ratio
    temperature = propellant.combustion_temperature * burn.eta_c**2
    gas_constant = propellant.products_constant
    exit_mach = burn.evaluate_exit_mach(pressure)
    exit_pressure = pressure * (1 + (gamma - 1) * exit_mach**2 / 2) ** (-gamma / (gamma - 1))
    ideal_flow = pressure * motor.nozzle_throat_area * math.sqrt(gamma / (gas_constant * temperature))
    ideal_flow *= (2 / (gamma + 1)) ** ((gamma + 1) / (2 * (gamma - 1)))
    velocity = math.sqrt(
        2 * gamma * gas_constant * temperature / (gamma - 1)
        * (1 - (exit_pressure / pressure) ** ((gamma - 1) / gamma))
    )
    momentum = (1 + math.cos(motor.nozzle_angle)) / 2 * ideal_flow * velocity
    pressure_thrust = (exit_pressure - environment.atmospheric_pressure) * motor.nozzle_exit_area
    components = burn.evaluate_thrust_components(pressure)
    assert components["momentum_ideal_n"] == pytest.approx(momentum)
    assert components["momentum_n"] == pytest.approx(0.82 * momentum)
    assert components["pressure_n"] == pytest.approx(pressure_thrust)
    assert components["total_n"] == pytest.approx(0.91 * (0.82 * momentum + pressure_thrust))
    assert burn.evaluate_thrust(pressure) == components["total_n"]
    assert burn.evaluate_Cf(pressure) == pytest.approx(components["total_n"] / (pressure * motor.nozzle_throat_area))


@pytest.mark.parametrize("pressure", [1.5e5, 3.5e6])
def test_discharge_scales_flow_and_momentum_only(motor_stack, pressure):
    ideal = Burn(*motor_stack)
    reduced = Burn(*motor_stack, discharge_coefficient=0.8)
    ideal_components = ideal.evaluate_thrust_components(pressure)
    reduced_components = reduced.evaluate_thrust_components(pressure)
    assert reduced.evaluate_nozzle_mass_flow(pressure) == pytest.approx(0.8 * ideal.evaluate_nozzle_mass_flow(pressure))
    assert reduced_components["momentum_ideal_n"] == pytest.approx(ideal_components["momentum_ideal_n"])
    assert reduced_components["momentum_n"] == pytest.approx(0.8 * ideal_components["momentum_n"])
    assert reduced_components["pressure_n"] == ideal_components["pressure_n"]
    assert reduced.evaluate_Cf(pressure) == pytest.approx(reduced.evaluate_thrust(pressure) / (pressure * reduced.motor.nozzle_throat_area))


@pytest.mark.parametrize("pressure", [-1.0, 0.0, 101325.0])
def test_low_pressure_has_zero_thrust_and_flow(motor_stack, pressure):
    burn = Burn(*motor_stack, eta_Cf=0.8, discharge_coefficient=0.7)
    assert burn.evaluate_nozzle_mass_flow(pressure) == 0.0
    assert burn.evaluate_thrust(pressure) == 0.0
    assert burn.evaluate_Cf(pressure) == 0.0
    assert all(value == 0.0 for value in burn.evaluate_thrust_components(pressure).values())


def test_thrust_efficiency_changes_impulse_without_changing_solver_states(motor_stack):
    baseline = BurnSimulation(*motor_stack, max_step_size=0.05, tail_off_evaluation=False)
    reduced = BurnSimulation(*motor_stack, max_step_size=0.05, tail_off_evaluation=False, eta_Cf=0.83)
    for index in [0, 1, 2, 3, 5, 6]:
        np.testing.assert_array_equal(baseline.total_burn_solution[index], reduced.total_burn_solution[index])
    time, pressure, _, _, thrust, *_ = baseline.total_burn_solution
    np.testing.assert_allclose(reduced.total_burn_solution[4], 0.83 * np.asarray(thrust))
    baseline_flow = [baseline.evaluate_nozzle_mass_flow(p) for p in pressure]
    reduced_flow = [reduced.evaluate_nozzle_mass_flow(p) for p in pressure]
    np.testing.assert_array_equal(baseline_flow, reduced_flow)
    assert reduced.propellant.cstar_at_pressure(3.5e6) == baseline.propellant.cstar_at_pressure(3.5e6)
    assert reduced.evaluate_total_impulse(reduced.total_burn_solution[4], time) == pytest.approx(
        0.83 * baseline.evaluate_total_impulse(thrust, time)
    )
    assert reduced.applied_efficiencies["eta_Cf"] == 0.83


def test_combustion_efficiency_changes_pressure_solution(motor_stack):
    baseline = BurnSimulation(*motor_stack, max_step_size=0.05, tail_off_evaluation=False)
    reduced = BurnSimulation(*motor_stack, max_step_size=0.05, tail_off_evaluation=False, eta_c=0.9)
    assert max(reduced.total_burn_solution[1]) < max(baseline.total_burn_solution[1])
    assert reduced._parameters_at_pressure(3.5e6)[0] == pytest.approx(0.81 * baseline._parameters_at_pressure(3.5e6)[0])
