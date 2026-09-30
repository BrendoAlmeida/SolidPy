import json
from dataclasses import replace

import numpy as np
import pytest

from solidpy import (
    CasingMaterial,
    MotorGeometry,
    StructuralMonteCarlo,
    casing_burst_pressure_pa,
)
from solidpy.Multiphysics import (
    simulate_advanced_components,
    simulate_advanced_physics,
    simulate_structural_response,
)
from solidpy import Grain, Motor, NozzleMaterial
from solidpy.surrogate_physics import (
    compute_structural_features,
    compute_structural_features_vectorized,
)


def make_geometry():
    return MotorGeometry(
        motor_length_m=0.24,
        motor_inner_diameter_m=0.08,
        casing_wall_thickness_m=0.004,
        grain_outer_diameter_m=0.07,
        grain_core_diameter_m=0.03,
        grain_gap_m=0.0,
        grain_length_each_m=0.10,
        grain_number=2,
        fill_length_m=0.20,
        throat_diameter_m=0.012,
        exit_diameter_m=0.028,
        free_volume_m3=0.001,
        propellant_mass_kg=1.0,
        dry_mass_kg=3.0,
        motor_initial_mass_kg=4.0,
        motor_final_mass_kg=3.0,
    )


def make_curve(time=None, pressure=None):
    time = np.asarray([0.0, 1.0] if time is None else time, dtype=float)
    pressure = np.asarray([0.0, 3.0e6] if pressure is None else pressure, dtype=float)
    return {"time_s": time, "thrust_n": np.zeros_like(time), "chamber_pressure_pa": pressure}


def test_public_scalar_and_vector_burst_match_structural_kernel():
    material = CasingMaterial(ultimate_strength_mpa=820.0)
    radius, wall, factor = 0.04, 0.004, 0.73
    expected = (2.0 / np.sqrt(3.0)) * 820.0e6 * factor * np.log1p(wall / radius)
    public = casing_burst_pressure_pa(radius, wall, 820.0, casing_strength_factor=factor)
    vector = casing_burst_pressure_pa(
        np.asarray([radius, radius]), np.asarray([wall, wall]), np.asarray([820.0, 820.0]),
        casing_strength_factor=np.asarray([factor, 1.0]),
    )
    structural = simulate_structural_response(
        make_geometry(), make_curve(), None, casing_material=material,
        casing_strength_factor=factor,
    )
    assert public == pytest.approx(expected)
    np.testing.assert_allclose(vector, [expected, expected / factor])
    assert structural["simulation.advanced.structural.burst_pressure_mpa"] * 1e6 == pytest.approx(expected)
    assert structural["simulation.advanced.structural.casing_burst_pressure_mpa"] == pytest.approx(expected / 1e6)


def test_static_scalar_and_vector_burst_paths_match_transient_kernel():
    grain = Grain(0.035, 0.015, initial_height=0.10)
    motor = Motor(
        grain, chamber_inner_radius=0.037, chamber_length=0.12,
        nozzle_throat_radius=0.008, nozzle_exit_radius=0.018,
        nozzle_angle=np.deg2rad(15.0),
    )
    casing, nozzle = CasingMaterial(ultimate_strength_mpa=820.0), NozzleMaterial()
    pressure = np.asarray([2.0e6, 4.0e6])
    static = compute_structural_features(
        motor, casing, nozzle, chamber_pressure_pa=float(pressure[-1]),
        casing_wall_thickness_m=0.004, grain=grain, propellant_mass_kg=0.5,
    )
    vector = compute_structural_features_vectorized(
        chamber_radius_m=0.037, throat_radius_m=0.008, exit_radius_m=0.018,
        chamber_length_m=0.12, casing_wall_thickness_m=0.004,
        casing_density_kg_m3=casing.density_kg_m3, bulkhead_fraction=casing.bulkhead_fraction,
        liner_thickness_m=0.0, liner_density_kg_m3=casing.liner_density_kg_m3,
        nozzle_density_kg_m3=nozzle.density_kg_m3,
        nozzle_wall_thickness_factor=nozzle.wall_thickness_factor,
        nozzle_min_wall_thickness_m=nozzle.min_wall_thickness_m,
        divergent_half_angle_rad=motor.nozzle_angle,
        chamber_pressure_pa=pressure, port_area_m2=grain.evaluate_port_area(0.0),
        propellant_mass_kg=0.5, ultimate_strength_mpa=820.0,
    )
    curve = {"time_s": np.asarray([0.0, 1.0]), "thrust_n": np.zeros(2),
             "chamber_pressure_pa": np.full(2, pressure[-1])}
    transient = simulate_structural_response(
        replace(make_geometry(), motor_inner_diameter_m=0.074), curve, None,
        casing_material=casing,
    )
    expected = casing_burst_pressure_pa(0.037, 0.004, 820.0)
    assert static.burst_pressure_pa == pytest.approx(expected)
    np.testing.assert_allclose(vector.burst_pressure_pa, [
        casing_burst_pressure_pa(0.037, 0.004, 820.0),
        casing_burst_pressure_pa(0.037, 0.004, 820.0),
    ])
    assert transient["simulation.advanced.structural.burst_pressure_mpa"] * 1e6 == pytest.approx(expected)


@pytest.mark.parametrize("value", [0.0, -1.0, np.nan, np.inf, True, "0.04"])
def test_burst_helper_rejects_invalid_physical_inputs(value):
    with pytest.raises(ValueError):
        casing_burst_pressure_pa(value, 0.004, 620.0)


@pytest.mark.parametrize("factor", [0.0, -1.0, np.nan, np.inf])
def test_burst_helper_rejects_invalid_strength_factor(factor):
    with pytest.raises(ValueError):
        casing_burst_pressure_pa(0.04, 0.004, 620.0, casing_strength_factor=factor)


@pytest.mark.parametrize(
    "call",
    [
        lambda: simulate_structural_response(make_geometry(), make_curve(), None),
        lambda: simulate_advanced_physics(make_geometry(), make_curve()),
        lambda: simulate_advanced_components(make_geometry(), make_curve()),
        lambda: StructuralMonteCarlo(make_geometry(), None, peak_pressure_distribution=lambda: 3e6),
    ],
)
def test_dynamic_structural_kernels_require_casing_material(call):
    with pytest.raises(ValueError, match="casing_material"):
        call()


@pytest.mark.parametrize(
    "options,status,applicability,reason",
    [
        ({}, "not_configured", "not_modeled", "closure_bolts_not_configured"),
        ({"closure_bolts_applicable": False}, "not_configured", "not_applicable", "closure_bolts_not_applicable"),
        ({"bolt_count": 4}, "model_not_available", "not_modeled", "incomplete_closure_bolt_properties"),
    ],
)
def test_unavailable_fastener_responses_are_nullable_and_json_finite(options, status, applicability, reason):
    structural = simulate_structural_response(
        make_geometry(), make_curve(), None, casing_material=CasingMaterial(), **options,
    )
    assert structural["simulation.advanced.structural.closure_bolt_status"] == status
    assert structural["simulation.advanced.structural.closure_bolt_applicability"] == applicability
    assert structural["simulation.advanced.structural.closure_bolt_reason"] == reason
    for key in (
        "closure_bolt_shear_safety_factor",
        "closure_bolt_bearing_safety_factor",
        "closure_bolt_shear_stress_mpa",
        "closure_bolt_bearing_stress_mpa",
    ):
        assert structural[f"simulation.advanced.structural.{key}"] is None
    json.dumps(structural, allow_nan=False)


def test_configured_fastener_responses_are_computed_and_finite():
    structural = simulate_structural_response(
        make_geometry(), make_curve(), None, casing_material=CasingMaterial(),
        bolt_count=4, bolt_diameter_m=0.006, bolt_strength_mpa=800.0,
    )
    assert structural["simulation.advanced.structural.closure_bolt_status"] == "configured"
    assert structural["simulation.advanced.structural.closure_bolt_applicability"] == "applicable"
    assert structural["simulation.advanced.structural.closure_bolt_shear_safety_factor"] > 0.0
    json.dumps(structural, allow_nan=False)


def test_thermal_service_margin_is_distinct_and_legacy_name_is_an_alias():
    geometry, curve, material = make_geometry(), make_curve(), CasingMaterial()
    missing = simulate_structural_response(geometry, curve, None, casing_material=material)
    assert missing["simulation.advanced.structural.thermal_service_margin"] is None
    assert missing["simulation.advanced.structural.thermoelastic_margin"] is None
    assert missing["simulation.advanced.structural.thermal_service_status"] == "not_modeled"

    thermal = {"simulation.advanced.thermal.casing_inner_wall_temp_c": 120.0}
    computed = simulate_structural_response(geometry, curve, thermal, casing_material=material)
    assert computed["simulation.advanced.structural.thermal_service_status"] == "computed"
    assert computed["simulation.advanced.structural.thermal_service_margin"] == pytest.approx(
        computed["simulation.advanced.structural.thermoelastic_margin"]
    )


def test_pressure_integral_uses_nonuniform_time_trapezoid():
    structural = simulate_structural_response(
        make_geometry(), make_curve([0.0, 0.1, 1.0], [0.0, 1e6, 0.0]),
        None, casing_material=CasingMaterial(),
    )
    assert structural["simulation.advanced.structural.pressurization_impulse_mpa_s"] == pytest.approx(0.5)
    inner_radius, outer_radius = 0.04, 0.044
    expected_limit = (620.0 * 1e6 * (outer_radius**2 - inner_radius**2)
                      / (np.sqrt(3.0) * outer_radius**2))
    assert structural["simulation.advanced.structural.yield_pressure_mpa"] == pytest.approx(expected_limit / 1e6)
    assert structural["simulation.advanced.structural.ultimate_elastic_limit_pressure_mpa"] == pytest.approx(expected_limit / 1e6)


def test_structural_ensemble_records_failures_and_missing_bolt_probability():
    model = StructuralMonteCarlo(
        make_geometry(), CasingMaterial(),
        peak_pressure_distribution=lambda: 3.0e6, random_seed=7,
    )
    result = model.run(2)
    assert result["robustness_policy_id"] == "structural_monte_carlo_v1"
    assert result["result_role"] == "ensemble"
    assert result["nominal"] is None
    assert len(result["scenario_ids"]) == len(result["scenarios"]) == 2
    assert result["status"] == "completed"
    assert result["failure_probability_bolts"] is None
    assert result["provenance"]["random_seed"] == 7
    assert result["provenance"]["thermal_source"] == "not_provided"
    assert len(result["provenance"]["physics_provider_hash"]) == 64
    repeated = StructuralMonteCarlo(
        make_geometry(), CasingMaterial(),
        peak_pressure_distribution=lambda: 3.0e6, random_seed=7,
    ).run(2)
    assert repeated["provenance"]["physics_provider_hash"] == result["provenance"]["physics_provider_hash"]
    assert all(record["structural"]["simulation.advanced.structural.thermal_service_margin"] is None
               for record in result["scenarios"])
    json.dumps(result, allow_nan=False)


def test_structural_ensemble_with_no_evaluated_samples_is_incomplete():
    model = StructuralMonteCarlo(
        make_geometry(), CasingMaterial(),
        peak_pressure_distribution=lambda: np.nan, random_seed=7,
    )
    result = model.run(2)
    assert result["status"] == "incomplete"
    assert result["n_evaluated"] == 0
    assert result["failure_probability"] is None
    assert len(result["scenario_ids"]) == 2
    assert all(item["status"] == "failed" for item in result["scenarios"])
    json.dumps(result, allow_nan=False)


def test_structural_ensemble_sampling_is_reproducible_for_a_fixed_seed():
    first = StructuralMonteCarlo(
        make_geometry(), CasingMaterial(), peak_pressure_distribution=lambda: 3.0e6,
        parameter_sigmas={"perturb_peak_pressure": {"mean": 1.0e5, "sigma": 2.0e4}},
        random_seed=91,
    ).run(4)
    second = StructuralMonteCarlo(
        make_geometry(), CasingMaterial(), peak_pressure_distribution=lambda: 3.0e6,
        parameter_sigmas={"perturb_peak_pressure": {"mean": 1.0e5, "sigma": 2.0e4}},
        random_seed=91,
    ).run(4)
    assert first["scenario_ids"] == second["scenario_ids"]
    assert first["samples"] == second["samples"]
    assert first["peak_pressure_pa"] == second["peak_pressure_pa"]
