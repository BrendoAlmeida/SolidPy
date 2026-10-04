"""Parity tests for the array kernels used by advanced-physics ensembles."""

from dataclasses import replace

import numpy as np
import pytest

from solidpy.Multiphysics import (
    CasingMaterial,
    simulate_cfd_proxies,
    simulate_ignition_proxy,
    simulate_structural_response,
)
from solidpy.batch.advanced_physics import AdvancedPhysicsBatch
from solidpy.batch.kernels.advanced_physics import advanced_physics_proxies
from test_robustness import make_motor_stack
from solidpy import geometry_from_components
from solidpy.backends import get_backend


def _numeric_reference(geometry, curve, thermal, casing, flame, specific):
    structural = simulate_structural_response(geometry, curve, thermal, casing_material=casing)
    return {
        **structural,
        **simulate_cfd_proxies(geometry, curve, thermal, r_specific=specific, flame_temp_k=flame),
        **simulate_ignition_proxy(geometry, curve, thermal, structural),
    }


@pytest.fixture
def advanced_inputs():
    grain, motor, propellant, _ = make_motor_stack()
    geometry = geometry_from_components(
        grain, motor, propellant, casing_wall_thickness_m=0.004, dry_mass_kg=3.0,
    )
    second_geometry = replace(
        geometry, casing_wall_thickness_m=0.006, grain_gap_m=0.0025, throat_diameter_m=0.018,
    )
    curves = [
        {
            "time_s": np.asarray([0.0, 0.04, 0.11]),
            "thrust_n": np.asarray([0.0, 2300.0, 800.0]),
            "chamber_pressure_pa": np.asarray([0.0, 3.2e6, 1.8e6]),
            "mass_flow_kg_s": np.asarray([0.0, 0.42, 0.14]),
        },
        {
            "time_s": np.asarray([0.0]),
            "thrust_n": np.asarray([120.0]),
            "chamber_pressure_pa": np.asarray([0.0]),
            "mass_flow_kg_s": np.asarray([0.03]),
        },
    ]
    thermals = [
        {
            "simulation.advanced.thermal.heat_load_kj_m2": 310.0,
            "simulation.advanced.thermal.throat_ablation_mm": 0.4,
            "simulation.advanced.thermal.casing_inner_wall_temp_c": 165.0,
        },
        {
            "simulation.advanced.thermal.heat_load_kj_m2": 125.0,
            "simulation.advanced.thermal.throat_ablation_mm": 0.1,
        },
    ]
    casings = [
        CasingMaterial(),
        CasingMaterial(material_family="composite", modulus_gpa=70.0, allowable_stress_mpa=180.0,
                       ultimate_strength_mpa=310.0, poisson_ratio=0.25),
    ]
    flames = [2400.0, 1650.0]
    specifics = [290.0, 312.0]
    return [geometry, second_geometry], curves, thermals, casings, flames, specifics


def test_advanced_proxy_kernels_match_scalar_models_and_preserve_single_point_fallback(advanced_inputs):
    geometries, curves, thermals, casings, flames, specifics = advanced_inputs
    batch = AdvancedPhysicsBatch.from_objects(
        geometries, curves, thermals, casings, flame_temp_k=flames, r_specific=specifics,
    )

    output = advanced_physics_proxies(batch.namespace(np), np)

    assert batch.t_max == 3
    assert batch.select([1]).t_max == 2
    for lane in range(len(batch)):
        expected = _numeric_reference(
            geometries[lane], curves[lane], thermals[lane], casings[lane], flames[lane], specifics[lane],
        )
        for key, expected_value in expected.items():
            if isinstance(expected_value, (int, float, np.number)):
                assert output[key][lane] == pytest.approx(expected_value, rel=2e-12, abs=1e-12), key


def test_numpy_and_reference_services_preserve_the_scalar_result_schema(advanced_inputs):
    geometries, curves, thermals, casings, flames, specifics = advanced_inputs
    batch = AdvancedPhysicsBatch.from_objects(
        geometries, curves, thermals, casings, flame_temp_k=flames, r_specific=specifics,
    )

    numpy_results = get_backend("cpu-vectorized").advanced_physics_proxies(batch).to_results()
    reference_results = get_backend("cpu-reference").advanced_physics_proxies(batch).to_results()

    assert len(numpy_results) == len(reference_results) == len(batch)
    for lane, (vectorized, reference) in enumerate(zip(numpy_results, reference_results)):
        assert vectorized.keys() == reference.keys()
        expected = _numeric_reference(
            geometries[lane], curves[lane], thermals[lane], casings[lane], flames[lane], specifics[lane],
        )
        assert vectorized["simulation.advanced.structural.thermal_service_margin"] == expected[
            "simulation.advanced.structural.thermal_service_margin"
        ]
        assert vectorized["simulation.advanced.structural.closure_bolt_status"] == expected[
            "simulation.advanced.structural.closure_bolt_status"
        ]
        for key, expected_value in reference.items():
            if isinstance(expected_value, (int, float, np.number)):
                assert vectorized[key] == pytest.approx(expected_value, rel=2e-12, abs=1e-12), key


def test_jax_service_matches_the_reference_when_jax_is_installed(advanced_inputs):
    pytest.importorskip("jax")
    geometries, curves, thermals, casings, flames, specifics = advanced_inputs
    batch = AdvancedPhysicsBatch.from_objects(
        geometries, curves, thermals, casings, flame_temp_k=flames, r_specific=specifics,
    )

    actual = get_backend("jax", device="cpu").advanced_physics_proxies(batch).to_results()
    expected = get_backend("cpu-reference").advanced_physics_proxies(batch).to_results()

    for actual_lane, expected_lane in zip(actual, expected):
        assert actual_lane.keys() == expected_lane.keys()
        for key, value in expected_lane.items():
            if isinstance(value, (int, float, np.number)):
                assert actual_lane[key] == pytest.approx(value, rel=2e-12, abs=1e-12), key


@pytest.mark.gpu
def test_jax_gpu_service_matches_the_reference(advanced_inputs):
    pytest.importorskip("jax")
    geometries, curves, thermals, casings, flames, specifics = advanced_inputs
    batch = AdvancedPhysicsBatch.from_objects(
        geometries, curves, thermals, casings, flame_temp_k=flames, r_specific=specifics,
    )

    actual = get_backend("jax", device="cuda:0").advanced_physics_proxies(batch).to_results()
    expected = get_backend("cpu-reference").advanced_physics_proxies(batch).to_results()

    for actual_lane, expected_lane in zip(actual, expected):
        assert actual_lane.keys() == expected_lane.keys()
        for key, value in expected_lane.items():
            if isinstance(value, (int, float, np.number)):
                assert actual_lane[key] == pytest.approx(value, rel=2e-12, abs=1e-12), key
