# -*- coding: utf-8 -*-

import numpy as np

from solidpy import (
    Environment,
    Grain,
    Motor,
    Propellant,
    RobustnessScenario,
    run_robustness_analysis,
)


def make_motor_stack():
    grain = Grain(
        outer_radius=71.92 / 2000,
        initial_inner_radius=31.92 / 2000,
        mass=700 / 1000,
    )
    motor = Motor(
        grain,
        grain_number=4,
        chamber_inner_radius=77.92 / 2000,
        nozzle_throat_radius=17.5 / 2000,
        nozzle_exit_radius=44.44 / 2000,
        nozzle_angle=15 * np.pi / 180,
        chamber_length=600 / 1000,
        dry_mass_kg=3.0,
        dry_center_of_mass_position_m=0.0,
    )
    propellant = Propellant(
        specific_heat_ratio=1.1361,
        density=1700,
        products_molecular_mass=39.9e-3,
        combustion_temperature=1600,
        interpolation_list="data/burnrate/KNSB3.csv",
    )
    environment = Environment(101325, 1.25, -0.38390456)
    return grain, motor, propellant, environment


def test_robustness_analysis_aggregates_scenarios():
    result = run_robustness_analysis(
        *make_motor_stack(),
        scenarios=[
            RobustnessScenario("slow", burn_rate_factor=0.98),
            RobustnessScenario("fast", burn_rate_factor=1.02),
        ],
        max_step_size=0.03,
        max_time_points=250,
    )

    assert result["nominal"]["scenario_id"] == "nominal"
    assert len(result["scenarios"]) == 2
    summary = result["summary"]
    assert summary["simulation.robustness.scenario_count"] == 2.0
    assert 0.0 <= summary["simulation.robustness.valid_ratio"] <= 1.0
    assert summary["simulation.robustness.peak_thrust_mean_n"] > 0
    assert summary["simulation.robustness.total_impulse_p95_ns"] >= summary[
        "simulation.robustness.total_impulse_p05_ns"
    ]
