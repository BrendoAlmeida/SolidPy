"""Characterisation of how ``Robustness`` applies a scenario, so the batch path can rely on the same objects.

``LEGACY_APPLY`` is ``_apply_scenario`` as it was before ``_scenario_objects`` was extracted from it.
"""

import copy

import numpy as np
import pytest

from solidpy import Environment, RobustnessScenario, run_robustness_analysis
from solidpy import Robustness
from solidpy.Robustness import (
    BURN_RATE_TEMP_REFERENCE_K,
    BURN_RATE_TEMP_SENSITIVITY_PER_K,
    _apply_scenario,
    _scenario_objects,
    build_latin_hypercube_scenarios,
    default_robustness_scenarios,
)
from test_robustness import make_motor_stack


def legacy_apply(grain, motor, propellant, environment, scenario):
    grain, motor, propellant, environment = copy.deepcopy((grain, motor, propellant, environment))
    if environment is None:
        environment = Environment()

    temperature_factor = 1.0 + BURN_RATE_TEMP_SENSITIVITY_PER_K * (
        float(scenario.initial_temperature_k) - BURN_RATE_TEMP_REFERENCE_K
    )
    burn_rate_factor = max(float(scenario.burn_rate_factor) * temperature_factor, 0.1)
    original_burn_rate = propellant.evaluate_burn_rate
    propellant.evaluate_burn_rate = (
        lambda chamber_pressure, port_mass_flux=0.0,
        _orig=original_burn_rate, _factor=burn_rate_factor:
        _factor * _orig(chamber_pressure, port_mass_flux)
    )

    density_factor = max(float(scenario.density_factor), 0.01)
    propellant.density *= density_factor

    throat_factor = max(float(scenario.throat_factor), 0.01)
    motor.nozzle_throat_area *= throat_factor**2
    motor.expansion_ratio = motor.nozzle_exit_area / max(motor.nozzle_throat_area, 1e-12)

    if scenario.ambient_pressure_pa is not None:
        environment.atmospheric_pressure = max(float(scenario.ambient_pressure_pa), 0.0)

    return grain, motor, propellant, environment


SCENARIOS = default_robustness_scenarios() + build_latin_hypercube_scenarios(sample_count=6) + [
    RobustnessScenario("extreme_low", burn_rate_factor=0.01, initial_temperature_k=150.0, throat_factor=0.001,
                       density_factor=0.0, ambient_pressure_pa=-5.0),
]


def state(grain, motor, propellant, environment):
    pressures = np.geomspace(1e5, 9e6, 6)
    return {
        "density": propellant.density,
        "throat_area": motor.nozzle_throat_area,
        "expansion_ratio": motor.expansion_ratio,
        "ambient": environment.atmospheric_pressure,
        "rates": [propellant.evaluate_burn_rate(p, g) for p in pressures for g in (0.0, 5.0, 300.0)],
    }


@pytest.mark.parametrize("scenario", SCENARIOS, ids=[s.scenario_id for s in SCENARIOS])
def test_apply_scenario_is_unchanged_by_the_extraction(scenario):
    stack = make_motor_stack()

    assert state(*_apply_scenario(*stack, scenario)) == state(*legacy_apply(*stack, scenario))


@pytest.mark.parametrize("scenario", SCENARIOS, ids=[s.scenario_id for s in SCENARIOS])
def test_scenario_objects_carry_the_factor_instead_of_an_override(scenario):
    grain, motor, propellant, environment = make_motor_stack()
    before = state(grain, motor, propellant, environment)

    _, scenario_motor, scenario_propellant, scenario_environment, factor = _scenario_objects(
        grain, motor, propellant, environment, scenario
    )

    assert "evaluate_burn_rate" not in vars(scenario_propellant)
    temperature = 1.0 + BURN_RATE_TEMP_SENSITIVITY_PER_K * (scenario.initial_temperature_k - BURN_RATE_TEMP_REFERENCE_K)
    assert factor == max(scenario.burn_rate_factor * temperature, 0.1)
    legacy = legacy_apply(grain, motor, propellant, environment, scenario)[2]
    for pressure in (3e5, 2e6, 7e6):  # the factor times the unscaled rate is what the override gives
        assert factor * scenario_propellant.evaluate_burn_rate(pressure, 40.0) == legacy.evaluate_burn_rate(pressure, 40.0)
    assert state(grain, motor, propellant, environment) == before  # the inputs were copied, not changed


def test_a_robustness_run_gives_the_same_result_through_the_extracted_helper(monkeypatch):
    grain, motor, propellant, environment = make_motor_stack()
    scenarios = [s for s in default_robustness_scenarios() if s.scenario_id in ("low_burn_rate", "cold_start", "wide_throat")]
    kwargs = dict(scenarios=scenarios, max_step_size=0.03, max_time_points=250)

    now = run_robustness_analysis(grain, motor, propellant, environment, **kwargs)
    monkeypatch.setattr(Robustness, "_apply_scenario", legacy_apply)
    before = run_robustness_analysis(grain, motor, propellant, environment, **kwargs)

    assert now["summary"] == before["summary"]
    for a, b in zip(now["scenarios"], before["scenarios"]):
        assert a["summary"] == b["summary"] and a["scenario_id"] == b["scenario_id"]
        np.testing.assert_array_equal(a["thrust_n"], b["thrust_n"])
