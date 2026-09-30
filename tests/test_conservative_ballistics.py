import math

import numpy as np
import pytest

from solidpy import (
    Burn,
    BurnSimulation,
    Environment,
    Grain,
    Motor,
    Propellant,
    evaluate_numerical_acceptance,
)


def make_stack(geometry="tubular", grains=None):
    if grains is None:
        grains = [Grain(0.035, 0.015, initial_height=0.12, geometry=geometry)]
    motor = Motor(
        grains, chamber_inner_radius=0.037,
        chamber_length=sum(g.initial_height for g in grains) + 0.02,
        nozzle_throat_radius=0.008, nozzle_exit_radius=0.018,
    )
    propellant = Propellant(
        1.1308, 0.04197, 1720.0, density=1879.0,
        burn_rate_a=7.36, burn_rate_n=0.32,
    )
    return grains[0], motor, propellant, Environment()


def simulate(**kwargs):
    return BurnSimulation(*make_stack(), max_step_size=0.03, **kwargs)


@pytest.mark.parametrize("geometry", ["tubular", "star"])
def test_canonical_inventory_and_numerical_blowdown_conserve_mass(geometry):
    simulation = BurnSimulation(*make_stack(geometry), max_step_size=0.03)
    result = simulation.result
    history, metrics = result["history"], result["metrics"]
    assert result["status"]["completed"]
    assert np.all(np.diff(history["time_s"]) > 0)
    assert np.ptp(np.diff(history["time_s"])) > 0.0
    assert history["regression_m"].shape == (len(history["time_s"]), 1)
    np.testing.assert_allclose(
        history["gas_mass_kg"] - history["gas_mass_kg"][0],
        history["generated_mass_integral_kg"] + history["igniter_mass_integral_kg"]
        - history["nozzle_mass_integral_kg"], atol=1e-12,
    )
    gas_constant = simulation.propellant.products_constant
    np.testing.assert_allclose(history["chamber_pressure_pa"] * history["free_volume_m3"],
                               history["gas_mass_kg"] * gas_constant * history["gas_temperature_k"])
    cutoff = simulation.environment_pressure + 0.01 * (
        metrics["peak_chamber_pressure_pa"] - simulation.environment_pressure
    )
    assert history["chamber_pressure_pa"][-1] == pytest.approx(cutoff, rel=1e-8)
    assert metrics["gas_mass_cutoff_kg"] > 0.0
    assert metrics["mass_flow_balance_error_pct"] < 0.001
    assert metrics["generated_mass_integral_kg"] == pytest.approx(metrics["propellant_mass_initial_kg"], rel=1e-6)
    assert metrics["propellant_mass_consumed_kg"] == metrics["propellant_mass_initial_kg"]
    assert metrics["mass_flow_avg_generated_kg_s"] == pytest.approx(
        metrics["generated_mass_integral_kg"] / metrics["propellant_burn_duration_s"]
    )
    assert metrics["mass_flow_avg_nozzle_kg_s"] == pytest.approx(
        metrics["nozzle_mass_integral_kg"] / metrics["nozzle_flow_duration_s"]
    )
    assert result["provenance"]["integration_method"] == "adaptive_ode_quadrature"
    assert len(simulation.total_burn_solution) == 7
    np.testing.assert_array_equal(simulation.total_burn_solution[0], history["time_s"])
    np.testing.assert_allclose(simulation.total_burn_solution[4], history["thrust_n"])


def test_different_grains_stop_regression_and_sources_independently():
    grains = [Grain(0.025, 0.019, initial_height=0.08), Grain(0.035, 0.015, initial_height=0.12)]
    simulation = BurnSimulation(*make_stack(grains=grains), max_step_size=0.03)
    history = simulation.result["history"]
    burnout = simulation.result["metrics"]["grain_burnout_times_s"]
    assert burnout[0] < burnout[1]
    stopped = history["time_s"] >= burnout[0]
    np.testing.assert_allclose(history["regression_m"][stopped, 0], grains[0].burnout_regression_m, atol=1e-15)
    assert np.all(history["mdot_generated_grains_kg_s"][stopped, 0] == 0.0)
    assert np.any(history["mdot_generated_grains_kg_s"][stopped, 1] > 0.0)
    assert simulation.result["status"]["completed"]


@pytest.mark.parametrize("source", [0.003, [[0.0, 0.0], [2.0, 0.006], [4.0, 0.0]]])
def test_hot_igniter_injection_continues_after_burnout_with_its_actual_mass(source):
    simulation = simulate(igniter_mass_flow=source, igniter_burn_time=4.0, igniter_temperature=3500.0)
    history, metrics = simulation.result["history"], simulation.result["metrics"]
    assert simulation.result["status"]["completed"]
    assert metrics["igniter_mass_injected_kg"] == pytest.approx(0.012, rel=1e-6)
    assert history["time_s"][-1] >= 4.0
    assert np.any(history["mdot_igniter_kg_s"][history["time_s"] > metrics["propellant_burn_end_s"]] > 0.0)
    assert np.max(history["gas_temperature_k"]) > simulation.initial_gas_temperature_k
    assert metrics["mass_flow_balance_error_pct"] < 0.001
    np.testing.assert_allclose(
        history["mdot_nozzle_kg_s"],
        [simulation.evaluate_nozzle_mass_flow(p, chamber_temperature=t)
         for p, t in zip(history["chamber_pressure_pa"], history["gas_temperature_k"])],
    )


def test_partial_activation_scales_front_regression_and_geometric_consumption_consistently():
    simulation = simulate(burn_area_activation=0.4)
    metrics = simulation.result["metrics"]
    assert simulation.result["status"]["completed"]
    assert metrics["generated_mass_integral_kg"] == pytest.approx(metrics["propellant_mass_consumed_kg"], rel=1e-6)
    assert metrics["mass_flow_balance_error_pct"] < 0.001


def test_nonuniform_time_integral_is_not_a_sample_mean():
    burn = Burn(*make_stack())
    time = np.asarray([0.0, 0.05, 0.5, 2.0])
    thrust = 3.0 + 2.0 * time
    integral = burn.evaluate_total_impulse(thrust, time)
    assert integral == pytest.approx(10.0)
    assert integral / 2.0 != pytest.approx(np.mean(thrust))
    assert burn.evaluate_total_impulse([3.0], [0.0]) == 0.0


def test_numerical_acceptance_passes_for_coarse_and_refined_solver_runs():
    coarse = BurnSimulation(*make_stack(), max_step_size=0.03).result
    refined = BurnSimulation(*make_stack(), max_step_size=0.015).result

    report = evaluate_numerical_acceptance(coarse, refined)

    assert report["status"] == "passed"
    assert report["passed"] is True
    assert all(item["passed"] for item in report["convergence"].values())


@pytest.mark.parametrize("parameters,reason", [
    ({"burn_timeout_s": 0.01}, "burn_timeout"),
    ({"tail_off_timeout_s": 1e-6}, "blowdown_timeout"),
    ({"tail_off_evaluation": False}, "tail_off_omitted"),
    ({"tail_off_method": "analytical"}, "analytical_approximation"),
    ({"igniter_mass_flow": lambda time: 0.0}, "unknown_igniter_duration"),
])
def test_incomplete_termination_is_explicit(parameters, reason):
    result = simulate(**parameters).result
    assert not result["status"]["completed"]
    assert result["status"]["termination_reason"] == reason
    assert len(result["history"]["time_s"]) > 1
    assert result["metrics"]["gas_mass_cutoff_kg"] > 0.0


def test_zero_sources_report_zero_flow_means_without_consuming_geometry():
    simulation = simulate(burn_area_activation=0.0, burn_timeout_s=0.03)
    metrics = simulation.result["metrics"]
    assert metrics["propellant_mass_consumed_kg"] == 0.0
    assert metrics["mass_flow_avg_generated_kg_s"] == 0.0
    assert metrics["mass_flow_avg_nozzle_kg_s"] == 0.0
    assert metrics["propellant_burn_duration_s"] == 0.0
    assert metrics["nozzle_flow_duration_s"] == 0.0
    assert simulation.result["status"]["termination_reason"] == "burn_timeout"


def test_tiny_grain_does_not_trigger_absolute_volume_burnout_threshold():
    grain = Grain(0.002, 0.001, initial_height=0.005)
    motor = Motor(grain, chamber_inner_radius=0.0025, chamber_length=0.007,
                  nozzle_throat_radius=0.0006, nozzle_exit_radius=0.001)
    propellant = make_stack()[2]
    simulation = BurnSimulation(grain, motor, propellant, max_step_size=0.002)
    assert grain.volume < 1e-6
    assert simulation.result["status"]["completed"]
    assert simulation.result["metrics"]["propellant_mass_consumed_kg"] == pytest.approx(grain.volume * propellant.density)
    assert simulation.result["metrics"]["propellant_burn_duration_s"] > 0.0


def test_coarse_refined_canonical_quadratures_and_peaks_converge():
    coarse = BurnSimulation(*make_stack(), max_step_size=0.04, rtol=1e-6, atol=1e-9).result
    refined = BurnSimulation(*make_stack(), max_step_size=0.01, rtol=1e-9, atol=1e-12).result
    assert coarse["status"]["completed"] and refined["status"]["completed"]
    for key in ["peak_chamber_pressure_pa", "peak_thrust_n", "max_generated_mass_flow_kg_s", "max_nozzle_mass_flow_kg_s"]:
        assert coarse["metrics"][key] == pytest.approx(refined["metrics"][key], rel=0.02)
    for key in ["total_impulse_ns", "generated_mass_integral_kg", "nozzle_mass_integral_kg"]:
        assert coarse["metrics"][key] == pytest.approx(refined["metrics"][key], rel=0.01)


@pytest.mark.parametrize("parameter", ["rtol", "atol", "max_step_size", "burn_timeout_s", "tail_off_timeout_s", "igniter_temperature"])
@pytest.mark.parametrize("value", [0.0, -1.0, math.nan, math.inf, True, "0.01"])
def test_invalid_positive_solver_settings_are_rejected(parameter, value):
    with pytest.raises(ValueError, match=parameter):
        BurnSimulation(*make_stack(), **{parameter: value})


@pytest.mark.parametrize("parameter", ["igniter_burn_time", "ignition_ramp_time"])
@pytest.mark.parametrize("value", [-1.0, math.nan, math.inf, True, "0.01"])
def test_invalid_source_durations_are_rejected(parameter, value):
    with pytest.raises(ValueError, match=parameter):
        BurnSimulation(*make_stack(), **{parameter: value})


@pytest.mark.parametrize("source", [-0.1, math.nan, [[0.0, 1.0], [0.0, 2.0]], [[0.0, 1.0], [1.0, -1.0]]])
def test_invalid_igniter_profiles_are_rejected(source):
    with pytest.raises(ValueError, match="igniter_mass_flow"):
        BurnSimulation(*make_stack(), igniter_mass_flow=source)


def test_legacy_pressure_dependent_thermochemistry_is_explicitly_outside_contract():
    stack = make_stack()
    stack[2].load_thermo_table([[101325, 1100, 1.13, 1720], [3e6, 1100, 1.13, 1720]])
    result = BurnSimulation(*stack, max_step_size=0.05).result
    assert not result["status"]["completed"]
    assert result["status"]["termination_reason"] == "unsupported_thermochemistry"
    assert result["provenance"]["thermochemistry_source"] == "pressure_table_legacy"
    assert not result["provenance"]["cea_used"]


@pytest.mark.parametrize("method", ["evaluate_nozzle_mass_flow", "evaluate_exit_temperature", "evaluate_exit_velocity", "evaluate_thrust_components", "evaluate_thrust", "evaluate_Cf"])
@pytest.mark.parametrize("temperature", [0.0, -1.0, math.nan, math.inf, True, "1720"])
def test_optional_nozzle_temperature_rejects_invalid_values(method, temperature):
    burn = Burn(*make_stack())
    with pytest.raises(ValueError, match="chamber_temperature"):
        getattr(burn, method)(2e6, chamber_temperature=temperature)


def test_provider_hash_identifies_resolved_burn_law_and_geometry():
    baseline = simulate().result
    stack = make_stack()
    stack[2].burn_rate_a *= 1.1
    changed = BurnSimulation(*stack, max_step_size=0.03).result
    assert baseline["provenance"]["physics_provider_hash"] != changed["provenance"]["physics_provider_hash"]
    assert baseline["provenance"]["solidpy_git_sha"]
    assert baseline["provenance"]["resolved_inputs"]["initial_gas_temperature_k"] == 1720.0
