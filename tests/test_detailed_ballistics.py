# -*- coding: utf-8 -*-

import numpy as np
import pytest

from solidpy import Environment, Grain, Motor, Propellant, run_detailed_ballistics
from solidpy.DetailedBallistics import (
    _evaluate_dynamic_mass_and_cg,
    _linear_endpoint_clamped,
    _remaining_geometry,
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
        dry_center_of_mass_position_m=0.3,
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


@pytest.mark.parametrize("field,value", [
    ("dry_mass_kg", None),
    ("dry_center_of_mass_position_m", None),
    ("dry_mass_kg", np.nan),
    ("dry_center_of_mass_position_m", np.nan),
    ("dry_mass_kg", -1.0),
    ("dry_center_of_mass_position_m", -1.0),
])
def test_detailed_ballistics_rejects_invalid_dry_hardware(field, value):
    grain, motor, propellant, environment = make_motor_stack()
    setattr(motor, field, value)
    with pytest.raises(ValueError):
        run_detailed_ballistics(grain, motor, propellant, environment, max_step_size=0.03)


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

    for key in [
        "propellant_center_of_mass_position_m",
        "motor_mass_kg",
        "motor_center_of_mass_position_m",
    ]:
        assert len(result[key]) == len(time_s)
        assert np.all(np.isfinite(result[key]))
    assert result["schema_version"] == 4.0
    assert result["summary"]["simulation.schema_version"] == 4.0
    assert np.all(result["propellant_mass_kg"] >= 0.0)
    assert np.all(result["motor_mass_kg"] >= result["propellant_mass_kg"])

    summary = result["summary"]
    assert summary["simulation.nominal.total_impulse_ns"] > 0
    assert summary["simulation.nominal.peak_thrust_n"] > 0
    assert summary["simulation.nominal.chamber_pressure_max_mpa"] > 0
    assert result["interpolation"] == {
        "method": "linear",
        "outside_domain": "endpoint_clamping",
        "extrapolation": False,
        "domain": "time_s",
    }
    motor = make_motor_stack()[1]
    dry_cg = motor.dry_center_of_mass_position_m
    np.testing.assert_allclose(result["motor_mass_kg"], result["propellant_mass_kg"] + motor.dry_mass_kg)
    np.testing.assert_allclose(
        result["motor_mass_kg"] * result["motor_center_of_mass_position_m"],
        result["propellant_mass_kg"] * result["propellant_center_of_mass_position_m"]
        + motor.dry_mass_kg * dry_cg,
    )
    grain, _, propellant, _ = make_motor_stack()
    initial_height, initial_area = _remaining_geometry(grain, 0.0)
    expected_initial_mass = 4.0 * initial_height * initial_area * propellant.density
    np.testing.assert_allclose(result["propellant_mass_kg"][0], expected_initial_mass, rtol=1e-6)
    assert np.all(np.isfinite(result["motor_mass_kg"][-1:]))
    assert result["motor_mass_kg"][-1] == pytest.approx(motor.dry_mass_kg, abs=0.05)
    np.testing.assert_allclose(result["motor_center_of_mass_position_m"][-1], dry_cg, atol=0.05)


def test_interpolation_is_linear_and_endpoint_clamped():
    values = _linear_endpoint_clamped(np.array([-1.0, 0.5, 3.0]), [0.0, 1.0, 2.0], [2.0, 4.0, 8.0])
    np.testing.assert_allclose(values, [2.0, 3.0, 8.0])


def test_dynamic_geometry_and_moments_use_independent_centroids():
    tubular = Grain(outer_radius=0.05, initial_inner_radius=0.02, initial_height=0.2, ends_burn=False)
    inhibited = Grain(outer_radius=0.05, initial_inner_radius=0.02, initial_height=0.2, ends_burn=True)
    motor = Motor(
        [tubular, inhibited], chamber_inner_radius=0.06, nozzle_throat_radius=0.01,
        nozzle_exit_radius=0.02, grain_separation=0.03, dry_mass_kg=2.0,
        dry_center_of_mass_position_m=0.4,
    )
    propellant = type("PropellantStub", (), {"density": 1000.0})()
    masses, cgs, motor_masses, motor_cgs = _evaluate_dynamic_mass_and_cg(
        motor, propellant, [0.01, 0.2]
    )
    first_height, first_area = _remaining_geometry(tubular, 0.01)
    second_height, second_area = _remaining_geometry(inhibited, 0.01)
    first_mass = first_area * first_height * propellant.density
    second_mass = second_area * second_height * propellant.density
    expected_mass = first_mass + second_mass
    expected_moment = first_mass * (0.01 + first_height / 2.0)
    expected_moment += second_mass * (tubular.initial_height + motor.grain_separation + second_height / 2.0)
    np.testing.assert_allclose(masses[0], expected_mass)
    np.testing.assert_allclose(cgs[0], expected_moment / expected_mass)
    np.testing.assert_allclose(motor_masses[0], masses[0] + motor.dry_mass_kg)
    np.testing.assert_allclose(motor_cgs[0] * motor_masses[0], cgs[0] * masses[0] + motor.dry_center_of_mass_position_m * motor.dry_mass_kg)
    assert masses[1] == 0.0
    assert cgs[1] == motor.dry_center_of_mass_position_m
    assert motor_cgs[1] == motor.dry_center_of_mass_position_m


def _star_phase_area(grain, regression):
    """Independent star-burn cross-sectional area (Nakka phase rules)."""
    w = max(float(regression), 0.0)
    n, eps = grain.n_points, grain.epsilon
    ri, ro = grain.initial_inner_radius, grain.outer_radius
    rs = ri + grain.slot_fraction * (ro - ri)
    r_bore = min(ri + w, ro)
    if w < ro - rs:
        # Phase 1: slot floor still inside the grain, its own walls burn.
        r_floor = min(rs + w, ro)
        return np.pi * (ro**2 - r_bore**2) - n * eps * (r_floor**2 - r_bore**2)
    # Phase 2: slot floor has merged with the outer case.
    return (np.pi - n * eps) * (ro**2 - r_bore**2)


def test_star_geometry_remaining_volume_and_end_modes():
    for ends_burn in (False, True):
        grain = Grain(
            outer_radius=0.05, initial_inner_radius=0.01, initial_height=0.2,
            geometry="star", n_points=5, epsilon=0.1, slot_fraction=0.5,
            ends_burn=ends_burn,
        )
        web = grain.outer_radius - grain.initial_inner_radius
        w_floor = grain.outer_radius - (
            grain.initial_inner_radius + grain.slot_fraction * web
        )
        regressions = (0.0, 0.005, 0.02, w_floor, w_floor + 1e-9, 0.04)
        for regression in regressions:
            height, area = _remaining_geometry(grain, regression)
            expected_height = (
                grain.initial_height
                if ends_burn
                else max(grain.initial_height - 2 * max(regression, 0.0), 0.0)
            )
            assert np.isclose(height, expected_height, rtol=1e-9, atol=1e-12)
            expected_area = max(_star_phase_area(grain, regression), 0.0)
            assert np.isclose(area, expected_area, rtol=1e-9, atol=1e-12)
            if ends_burn:
                assert np.isclose(height, grain.initial_height)
        # Zero propellant cross-section at burnout (bore reaches the case).
        assert _remaining_geometry(grain, web)[1] == 0.0
        # Phase-1 → phase-2 cross-section is continuous at the slot-floor/case boundary.
        area_before = _remaining_geometry(grain, max(w_floor - 1e-9, 0.0))[1]
        area_after = _remaining_geometry(grain, w_floor)[1]
        assert np.isclose(area_before, area_after, rtol=1e-6, atol=1e-9)


def test_tubular_geometry_remaining_volume_matches_independent_formula():
    for ends_burn in (False, True):
        grain = Grain(
            outer_radius=0.05, initial_inner_radius=0.02, initial_height=0.2,
            ends_burn=ends_burn,
        )
        web = grain.outer_radius - grain.initial_inner_radius
        for regression in (0.0, 0.005, 0.02, web, web + 0.005, 0.15):
            height, area = _remaining_geometry(grain, regression)
            inner_radius = min(
                grain.initial_inner_radius + max(regression, 0.0), grain.outer_radius
            )
            expected_height = (
                grain.initial_height
                if ends_burn
                else max(grain.initial_height - 2 * max(regression, 0.0), 0.0)
            )
            expected_area = max(
                np.pi * (grain.outer_radius**2 - inner_radius**2), 0.0
            )
            assert np.isclose(height, expected_height, rtol=1e-9, atol=1e-12)
            assert np.isclose(area, expected_area, rtol=1e-9, atol=1e-12)


def test_star_grain_dynamic_centroid_includes_regression_offset():
    grain = Grain(
        outer_radius=0.05, initial_inner_radius=0.01, initial_height=0.2,
        geometry="star", n_points=5, epsilon=0.1, slot_fraction=0.5,
        ends_burn=False,
    )
    motor = Motor(
        [grain], chamber_inner_radius=0.06, nozzle_throat_radius=0.01,
        nozzle_exit_radius=0.02, dry_mass_kg=0.0,
        dry_center_of_mass_position_m=0.0,
    )
    propellant = type("PropellantStub", (), {"density": 1000.0})()
    regression = 0.01
    masses, cgs, _, _ = _evaluate_dynamic_mass_and_cg(
        motor, propellant, [regression]
    )
    height, area = _remaining_geometry(grain, regression)
    expected_mass = area * height * propellant.density
    expected_cg = regression + height / 2.0
    np.testing.assert_allclose(masses[0], expected_mass, rtol=1e-9, atol=1e-12)
    np.testing.assert_allclose(cgs[0], expected_cg, rtol=1e-9, atol=1e-12)


def test_detailed_ballistics_converges_with_explicit_precision_tolerance():
    coarse = run_detailed_ballistics(*make_motor_stack(), max_step_size=0.03, max_time_points=500)
    fine = run_detailed_ballistics(*make_motor_stack(), max_step_size=0.01, max_time_points=500)
    coarse_mass = np.interp(fine["time_s"], coarse["time_s"], coarse["propellant_mass_kg"])
    coarse_cg = np.interp(fine["time_s"], coarse["time_s"], coarse["motor_center_of_mass_position_m"])
    np.testing.assert_allclose(coarse_mass, fine["propellant_mass_kg"], rtol=1e-2, atol=3e-3)
    np.testing.assert_allclose(coarse_cg, fine["motor_center_of_mass_position_m"], rtol=1e-2, atol=3e-3)
