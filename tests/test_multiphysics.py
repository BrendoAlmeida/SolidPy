# -*- coding: utf-8 -*-

import numpy as np

from solidpy import (
    CasingMaterial,
    Environment,
    Grain,
    Motor,
    NozzleMaterial,
    Propellant,
    geometry_from_components,
    run_detailed_ballistics,
    simulate_advanced_components,
    simulate_advanced_physics,
    simulate_thermal_ablation,
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


def test_advanced_physics_returns_all_component_metrics():
    grain, motor, propellant, environment = make_motor_stack()
    curve = run_detailed_ballistics(
        grain,
        motor,
        propellant,
        environment,
        max_step_size=0.03,
        max_time_points=250,
    )
    geometry = geometry_from_components(
        grain,
        motor,
        propellant,
        casing_wall_thickness_m=0.004,
        dry_mass_kg=3.0,
    )
    advanced = simulate_advanced_physics(
        geometry,
        curve,
        casing_material=CasingMaterial(),
        nozzle_material=NozzleMaterial(),
        flame_temp_k=propellant.combustion_temperature,
        r_specific=propellant.products_constant,
    )

    expected_keys = [
        "simulation.advanced.thermal.throat_ablation_mm",
        "simulation.advanced.structural.safety_factor",
        "simulation.advanced.cfd.reynolds_proxy",
        "simulation.advanced.ignition.delay_s",
        "simulation.advanced.flight.max_altitude_m",
    ]
    for key in expected_keys:
        assert key in advanced
        assert np.isfinite(advanced[key])

    assert advanced["simulation.advanced.structural.max_stress_mpa"] > 0
    assert advanced["simulation.advanced.flight.delta_v_m_s"] >= 0


def test_advanced_components_groups_results():
    grain, motor, propellant, environment = make_motor_stack()
    curve = run_detailed_ballistics(
        grain,
        motor,
        propellant,
        environment,
        max_step_size=0.04,
        max_time_points=200,
    )
    geometry = geometry_from_components(
        grain,
        motor,
        propellant,
        casing_wall_thickness_m=0.004,
        dry_mass_kg=3.0,
    )
    grouped = simulate_advanced_components(geometry, curve)

    for key in ["thermal", "structural", "cfd", "ignition", "flight", "nominal_advanced"]:
        assert key in grouped


def test_thermal_ablation_implicit_solver_keeps_liner_temperatures():
    grain, motor, propellant, _environment = make_motor_stack()
    geometry = geometry_from_components(
        grain,
        motor,
        propellant,
        casing_wall_thickness_m=0.004,
        dry_mass_kg=3.0,
    )
    time_s = np.linspace(0.0, 1.0, 6)
    curve = {
        "time_s": time_s,
        "thrust_n": np.full_like(time_s, 450.0),
        "mass_flow_kg_s": np.full_like(time_s, 0.42),
        "chamber_pressure_pa": np.full_like(time_s, 2.4e6),
        "gamma": propellant.specific_heat_ratio,
    }

    thermal = simulate_thermal_ablation(
        geometry,
        curve,
        casing_material=CasingMaterial(liner_thickness_m=0.002),
        nozzle_material=NozzleMaterial(),
        flame_temp_k=propellant.combustion_temperature,
        r_specific=propellant.products_constant,
    )

    expected_keys = [
        "simulation.advanced.thermal.liner_hot_face_temp_c",
        "simulation.advanced.thermal.liner_casing_interface_temp_c",
        "simulation.advanced.thermal.casing_inner_wall_temp_c",
        "simulation.advanced.thermal.casing_outer_wall_temp_c",
    ]
    for key in expected_keys:
        assert key in thermal
        assert np.isfinite(thermal[key])
    assert thermal["simulation.advanced.metadata.thermal_node_count"] > 4.0
