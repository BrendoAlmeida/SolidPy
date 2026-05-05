# -*- coding: utf-8 -*-

import numpy as np

from solidpy import Environment, Grain, Motor, Propellant, run_detailed_ballistics


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


def test_detailed_ballistics_returns_internal_series():
    result = run_detailed_ballistics(
        *make_motor_stack(),
        max_step_size=0.02,
        max_time_points=300,
    )

    time_s = result["time_s"]
    assert len(time_s) > 10
    assert np.all(np.diff(time_s) > 0)

    expected_series = [
        "thrust_n",
        "chamber_pressure_pa",
        "mass_generated_kg_s",
        "mass_nozzle_kg_s",
        "propellant_mass_kg",
        "burn_area_m2",
        "throat_diameter_m",
        "throat_ablation_m",
        "cf",
        "ignition_active_fraction",
    ]
    for key in expected_series:
        assert len(result[key]) == len(time_s)

    summary = result["summary"]
    assert summary["simulation.nominal.total_impulse_ns"] > 0
    assert summary["simulation.nominal.peak_thrust_n"] > 0
    assert summary["simulation.nominal.chamber_pressure_max_mpa"] > 0
