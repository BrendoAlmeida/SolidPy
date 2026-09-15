# -*- coding: utf-8 -*-

import math

import numpy as np
import pytest
from scipy import sparse

from solidpy import (
    CasingMaterial,
    DEFAULT_BULKHEAD_FRACTION,
    DEFAULT_NOZZLE_CONVERGENT_HALF_ANGLE_DEG,
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
from solidpy.Multiphysics import (
    _casing_mass_with_bulkheads_kg,
    _casing_mass_with_bulkheads_kg_vectorized,
    _liner_mass_kg,
    _liner_mass_kg_vectorized,
    _nozzle_mass_kg,
    _nozzle_mass_kg_vectorized,
)


def test_bulkhead_fraction_default_is_public_and_canonical():
    assert DEFAULT_BULKHEAD_FRACTION == 1.35
    assert CasingMaterial().bulkhead_fraction == DEFAULT_BULKHEAD_FRACTION


def test_vectorized_mass_helpers_match_scalar_helpers():
    radius = np.array([0.037, 0.041, 0.045])
    wall = np.array([0.004, 0.005, 0.006])
    length = np.array([0.14, 0.16, 0.18])
    density = np.array([7850.0, 7800.0, 8050.0])
    fraction = np.array([1.35, 1.2, 1.5])
    casing_vector = _casing_mass_with_bulkheads_kg_vectorized(
        radius, wall, length, density, fraction
    )
    casing_scalar = np.array([
        _casing_mass_with_bulkheads_kg(
            r, w, l, CasingMaterial(density_kg_m3=d, bulkhead_fraction=f)
        )
        for r, w, l, d, f in zip(radius, wall, length, density, fraction)
    ])
    np.testing.assert_allclose(casing_vector, casing_scalar)

    liner_thickness = np.array([0.001, 0.0, -0.001])
    liner_density = np.array([1100.0, 0.0, 1200.0])
    liner_vector = _liner_mass_kg_vectorized(
        radius, length, liner_thickness, liner_density
    )
    liner_scalar = np.array([
        _liner_mass_kg(
            r,
            l,
            CasingMaterial(
                liner_thickness_m=t,
                liner_density_kg_m3=d,
            ),
        )
        for r, l, t, d in zip(radius, length, liner_thickness, liner_density)
    ])
    np.testing.assert_allclose(liner_vector, liner_scalar)

    throat = np.array([0.008, 0.009, 0.010])
    exit_radius = np.array([0.018, 0.020, 0.023])
    angles = np.array([math.radians(12.0), math.radians(15.0), math.radians(18.0)])
    nozzle_density = np.array([1800.0, 1750.0, 1900.0])
    factors = np.array([1.15, 1.1, 1.3])
    minimum_walls = np.array([0.004, 0.003, 0.005])
    nozzle_vector = _nozzle_mass_kg_vectorized(
        radius,
        throat,
        exit_radius,
        angles,
        wall,
        nozzle_density,
        factors,
        minimum_walls,
    )
    nozzle_scalar = np.array([
        _nozzle_mass_kg(
            r,
            t,
            e,
            a,
            w,
            NozzleMaterial(
                density_kg_m3=d,
                wall_thickness_factor=f,
                min_wall_thickness_m=m,
            ),
        )
        for r, t, e, a, w, d, f, m in zip(
            radius,
            throat,
            exit_radius,
            angles,
            wall,
            nozzle_density,
            factors,
            minimum_walls,
        )
    ])
    np.testing.assert_allclose(nozzle_vector, nozzle_scalar)


def test_vectorized_mass_helpers_broadcast_scalars_and_return_zero_dim_arrays():
    result = _casing_mass_with_bulkheads_kg_vectorized(
        np.array([0.037, 0.040, 0.043]),
        0.004,
        0.14,
        7850.0,
        DEFAULT_BULKHEAD_FRACTION,
    )
    assert result.shape == (3,)
    scalar_result = _liner_mass_kg_vectorized(0.037, 0.14, 0.001, 1100.0)
    assert scalar_result.shape == ()
    assert isinstance(scalar_result, np.ndarray)


def test_vectorized_liner_ignores_invalid_density_when_inactive():
    result = _liner_mass_kg_vectorized(
        np.array([0.037, 0.040]),
        np.array([0.14, 0.16]),
        np.array([0.0, -1e308]),
        np.array([math.nan, math.inf]),
    )

    assert result.shape == (2,)
    np.testing.assert_array_equal(result, np.zeros(2))


def test_vectorized_nozzle_mass_accepts_degenerate_nozzle():
    result = _nozzle_mass_kg_vectorized(
        0.037,
        0.008,
        0.008,
        math.radians(15.0),
        0.004,
        1800.0,
        1.15,
        0.004,
    )
    expected = _nozzle_mass_kg(
        0.037,
        0.008,
        0.008,
        math.radians(15.0),
        0.004,
        NozzleMaterial(
            density_kg_m3=1800.0,
            wall_thickness_factor=1.15,
            min_wall_thickness_m=0.004,
        ),
    )

    assert result.shape == ()
    assert float(result) == pytest.approx(expected)
    with pytest.raises(ValueError, match="exit_radius_m"):
        _nozzle_mass_kg_vectorized(
            0.037,
            0.008,
            0.007,
            math.radians(15.0),
            0.004,
            1800.0,
            1.15,
            0.004,
        )


@pytest.mark.parametrize(
    "angle",
    [0.0, -1.0, math.pi / 2.0, math.nan, math.inf],
)
def test_vectorized_nozzle_mass_rejects_invalid_angles(angle):
    with pytest.raises(ValueError, match="divergent_half_angle_rad"):
        _nozzle_mass_kg_vectorized(
            0.037,
            0.008,
            0.018,
            angle,
            0.004,
            1800.0,
            1.15,
            0.004,
        )


def test_vectorized_mass_helpers_reject_shape_mismatch_and_overflow():
    with pytest.raises(ValueError, match="broadcastable"):
        _casing_mass_with_bulkheads_kg_vectorized(
            np.ones(2),
            np.ones(3),
            0.14,
            7850.0,
            1.35,
        )
    with pytest.raises(ValueError):
        _casing_mass_with_bulkheads_kg_vectorized(
            0.037,
            1e308,
            1e308,
            7850.0,
            1.35,
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


def _geometry_with_materials(
    *,
    casing_wall_thickness_m=0.004,
    casing_material=None,
    nozzle_material=None,
):
    grain, motor, propellant, _environment = make_motor_stack()
    return geometry_from_components(
        grain,
        motor,
        propellant,
        casing_wall_thickness_m=casing_wall_thickness_m,
        casing_material=casing_material or CasingMaterial(),
        nozzle_material=nozzle_material or NozzleMaterial(),
    ), motor


def test_geometry_legacy_signature_preserves_casing_only_dry_mass():
    grain, motor, propellant, _environment = make_motor_stack()
    geometry = geometry_from_components(
        grain,
        motor,
        propellant,
        0.004,
        None,
        7850.0,
    )
    inner_radius = math.sqrt(motor.chamber_area / math.pi)
    wall = max(0.004, 1e-5)
    outer_radius = inner_radius + wall
    expected = (
        math.pi
        * max(outer_radius**2 - inner_radius**2, 0.0)
        * motor.chamber_length
        * max(7850.0, 1.0)
    )

    assert geometry.dry_mass_kg == expected
    assert geometry.casing_mass_kg == expected
    assert geometry.liner_mass_kg == 0.0
    assert geometry.nozzle_mass_kg == 0.0


def test_geometry_material_model_adds_bulkheads_and_exposes_components():
    casing = CasingMaterial(
        density_kg_m3=7800.0,
        bulkhead_fraction=1.35,
        liner_thickness_m=0.002,
        liner_density_kg_m3=1100.0,
    )
    nozzle = NozzleMaterial(
        density_kg_m3=1800.0,
        wall_thickness_factor=1.15,
        min_wall_thickness_m=0.004,
    )
    geometry, motor = _geometry_with_materials(
        casing_material=casing,
        nozzle_material=nozzle,
    )
    inner_diameter = 2.0 * math.sqrt(motor.chamber_area / math.pi)
    wall = 0.004
    shell_volume = math.pi * ((inner_diameter / 2.0 + wall) ** 2 - (inner_diameter / 2.0) ** 2) * motor.chamber_length
    bulkhead_volume = (
        2.0
        * (math.pi / 4.0)
        * inner_diameter**2
        * wall
        * casing.bulkhead_fraction
    )
    liner_inner_diameter = inner_diameter - 2.0 * casing.liner_thickness_m
    liner_volume = (
        math.pi
        / 4.0
        * (inner_diameter**2 - liner_inner_diameter**2)
        * motor.chamber_length
    )

    assert geometry.casing_mass_kg == pytest.approx(
        (shell_volume + bulkhead_volume) * casing.density_kg_m3
    )
    assert geometry.liner_mass_kg == pytest.approx(
        liner_volume * casing.liner_density_kg_m3
    )
    assert geometry.dry_mass_kg == pytest.approx(
        geometry.casing_mass_kg + geometry.liner_mass_kg + geometry.nozzle_mass_kg
    )


@pytest.mark.parametrize("legacy_density", [math.nan, math.inf])
def test_geometry_material_model_ignores_legacy_casing_density(legacy_density):
    grain, motor, propellant, _environment = make_motor_stack()
    casing = CasingMaterial(density_kg_m3=7800.0, bulkhead_fraction=1.35)
    geometry = geometry_from_components(
        grain,
        motor,
        propellant,
        casing_wall_thickness_m=0.004,
        casing_density_kg_m3=legacy_density,
        casing_material=casing,
        nozzle_material=NozzleMaterial(density_kg_m3=0.0),
    )

    inner_radius = math.sqrt(motor.chamber_area / math.pi)
    wall = 0.004
    shell_volume = math.pi * ((inner_radius + wall) ** 2 - inner_radius**2) * motor.chamber_length
    bulkhead_volume = (
        2.0
        * (math.pi / 4.0)
        * (2.0 * inner_radius) ** 2
        * wall
        * casing.bulkhead_fraction
    )
    expected_casing_mass = (shell_volume + bulkhead_volume) * casing.density_kg_m3

    assert geometry.casing_mass_kg == pytest.approx(expected_casing_mass)
    assert geometry.dry_mass_kg == pytest.approx(expected_casing_mass)


def test_geometry_bulkhead_correction_is_strictly_above_legacy_shell():
    grain, motor, propellant, _environment = make_motor_stack()
    casing = CasingMaterial(density_kg_m3=7800.0, bulkhead_fraction=1.35)
    geometry = geometry_from_components(
        grain,
        motor,
        propellant,
        0.004,
        casing_material=casing,
        nozzle_material=NozzleMaterial(density_kg_m3=0.0),
    )
    inner_radius = math.sqrt(motor.chamber_area / math.pi)
    shell_volume = math.pi * ((inner_radius + 0.004) ** 2 - inner_radius**2) * motor.chamber_length
    assert geometry.casing_mass_kg > shell_volume * casing.density_kg_m3


def test_geometry_liner_mass_is_exactly_zero_when_liner_is_disabled():
    geometry, _motor = _geometry_with_materials(
        casing_material=CasingMaterial(liner_thickness_m=0.0),
        nozzle_material=NozzleMaterial(density_kg_m3=0.0),
    )
    assert geometry.liner_mass_kg == 0.0


def test_geometry_nozzle_density_factor_and_floor_control_mass():
    casing = CasingMaterial()
    density_zero, _motor = _geometry_with_materials(
        casing_material=casing,
        nozzle_material=NozzleMaterial(density_kg_m3=0.0),
    )
    low_factor, _motor = _geometry_with_materials(
        casing_material=casing,
        nozzle_material=NozzleMaterial(
            density_kg_m3=1800.0,
            wall_thickness_factor=1.15,
            min_wall_thickness_m=0.004,
        ),
    )
    high_factor, _motor = _geometry_with_materials(
        casing_material=casing,
        nozzle_material=NozzleMaterial(
            density_kg_m3=1800.0,
            wall_thickness_factor=2.0,
            min_wall_thickness_m=0.004,
        ),
    )
    high_density, _motor = _geometry_with_materials(
        casing_material=casing,
        nozzle_material=NozzleMaterial(
            density_kg_m3=3600.0,
            wall_thickness_factor=1.15,
            min_wall_thickness_m=0.004,
        ),
    )

    assert density_zero.nozzle_mass_kg == 0.0
    assert high_factor.nozzle_mass_kg > low_factor.nozzle_mass_kg
    assert high_density.nozzle_mass_kg > low_factor.nozzle_mass_kg

    # With a 1 mm casing wall, factor 1.15 is below the 4 mm floor.
    floor_geometry, _motor = _geometry_with_materials(
        casing_wall_thickness_m=0.001,
        casing_material=casing,
        nozzle_material=NozzleMaterial(
            density_kg_m3=1800.0,
            wall_thickness_factor=1.15,
            min_wall_thickness_m=0.004,
        ),
    )
    factor_geometry, _motor = _geometry_with_materials(
        casing_wall_thickness_m=0.001,
        casing_material=casing,
        nozzle_material=NozzleMaterial(
            density_kg_m3=1800.0,
            wall_thickness_factor=1.15,
            min_wall_thickness_m=0.0,
        ),
    )
    assert floor_geometry.nozzle_mass_kg == pytest.approx(
        factor_geometry.nozzle_mass_kg * (0.004 / (0.001 * 1.15))
    )


def test_geometry_degenerate_divergent_nozzle_has_finite_zero_length_cylinder_limit():
    grain, motor, propellant, _environment = make_motor_stack()
    motor.nozzle_exit_area = motor.nozzle_throat_area
    casing = CasingMaterial()
    nozzle = NozzleMaterial(density_kg_m3=1800.0)
    geometry = geometry_from_components(
        grain,
        motor,
        propellant,
        0.004,
        casing_material=casing,
        nozzle_material=nozzle,
    )

    throat_radius = math.sqrt(motor.nozzle_throat_area / math.pi)
    chamber_radius = math.sqrt(motor.chamber_area / math.pi)
    wall = max(0.004 * nozzle.wall_thickness_factor, nozzle.min_wall_thickness_m)
    conv_delta = chamber_radius - throat_radius
    conv_angle = math.radians(DEFAULT_NOZZLE_CONVERGENT_HALF_ANGLE_DEG)
    conv_slant = math.hypot(
        conv_delta / math.tan(conv_angle),
        conv_delta,
    )
    # Equal throat/exit radii define a zero-length divergent cone.  The
    # native convention documented by geometry_from_components gives that
    # degenerate cone zero divergent lateral area; only the convergent shell
    # contributes to the expected mass.
    divergent_area = 0.0
    convergent_area = math.pi * (chamber_radius + throat_radius) * conv_slant
    expected = (
        (divergent_area + convergent_area) * wall
        * nozzle.density_kg_m3
    )

    assert divergent_area == 0.0
    assert np.isfinite(geometry.nozzle_mass_kg)
    assert geometry.nozzle_mass_kg == pytest.approx(expected)


@pytest.mark.parametrize(
    "invalid_angle",
    [None, 0.0, -0.1, math.nan, math.inf, math.pi / 2.0],
)
def test_geometry_rejects_invalid_divergent_nozzle_angle(invalid_angle):
    grain, motor, propellant, _environment = make_motor_stack()
    motor.nozzle_angle = invalid_angle

    with pytest.raises(ValueError, match=r"motor\.nozzle_angle"):
        geometry_from_components(
            grain,
            motor,
            propellant,
            0.004,
            casing_material=CasingMaterial(),
            nozzle_material=NozzleMaterial(),
        )


@pytest.mark.parametrize(
    ("attribute", "value"),
    [
        ("chamber_area", math.nan),
        ("nozzle_throat_area", math.inf),
        ("nozzle_exit_area", math.nan),
    ],
)
def test_geometry_rejects_nonfinite_motor_areas(attribute, value):
    grain, motor, propellant, _environment = make_motor_stack()
    setattr(motor, attribute, value)

    with pytest.raises(ValueError, match=rf"motor\.{attribute}"):
        geometry_from_components(
            grain,
            motor,
            propellant,
            0.004,
            casing_material=CasingMaterial(),
            nozzle_material=NozzleMaterial(),
        )


@pytest.mark.parametrize(
    "material",
    [
        CasingMaterial(density_kg_m3=math.nan),
        CasingMaterial(bulkhead_fraction=math.inf),
        CasingMaterial(liner_thickness_m=0.001, liner_density_kg_m3=math.nan),
    ],
)
def test_geometry_rejects_nonfinite_casing_material_values(material):
    grain, motor, propellant, _environment = make_motor_stack()

    with pytest.raises(ValueError):
        geometry_from_components(
            grain,
            motor,
            propellant,
            0.004,
            casing_material=material,
            nozzle_material=NozzleMaterial(density_kg_m3=0.0),
        )


@pytest.mark.parametrize(
    "material",
    [
        NozzleMaterial(density_kg_m3=math.nan),
        NozzleMaterial(wall_thickness_factor=math.inf),
        NozzleMaterial(min_wall_thickness_m=math.nan),
    ],
)
def test_geometry_rejects_nonfinite_nozzle_material_values(material):
    grain, motor, propellant, _environment = make_motor_stack()

    with pytest.raises(ValueError):
        geometry_from_components(
            grain,
            motor,
            propellant,
            0.004,
            casing_material=CasingMaterial(),
            nozzle_material=material,
        )


def test_geometry_rejects_nonfinite_wall_thickness_and_overflow():
    grain, motor, propellant, _environment = make_motor_stack()

    with pytest.raises(ValueError, match="casing_wall_thickness_m"):
        geometry_from_components(
            grain,
            motor,
            propellant,
            math.nan,
            casing_material=CasingMaterial(),
            nozzle_material=NozzleMaterial(),
        )

    with pytest.raises(ValueError):
        geometry_from_components(
            grain,
            motor,
            propellant,
            1.0e308,
            casing_material=CasingMaterial(),
            nozzle_material=NozzleMaterial(),
        )


def test_geometry_clamps_negative_wall_thickness_in_material_mode():
    geometry, _motor = _geometry_with_materials(
        casing_wall_thickness_m=-1.0,
        casing_material=CasingMaterial(),
        nozzle_material=NozzleMaterial(density_kg_m3=0.0),
    )

    assert geometry.casing_wall_thickness_m == 1e-5
    assert geometry.casing_mass_kg >= 0.0


def test_geometry_material_model_honors_explicit_dry_mass_override():
    geometry, _motor = _geometry_with_materials(
        casing_material=CasingMaterial(liner_thickness_m=0.002),
        nozzle_material=NozzleMaterial(),
    )

    grain, motor, propellant, _environment = make_motor_stack()
    overridden_zero = geometry_from_components(
        grain,
        motor,
        propellant,
        casing_wall_thickness_m=0.004,
        dry_mass_kg=-3.0,
        casing_material=CasingMaterial(liner_thickness_m=0.002),
        nozzle_material=NozzleMaterial(),
    )

    assert overridden_zero.dry_mass_kg == 0.0
    assert overridden_zero.casing_mass_kg == pytest.approx(geometry.casing_mass_kg)
    assert overridden_zero.liner_mass_kg == pytest.approx(geometry.liner_mass_kg)
    assert overridden_zero.nozzle_mass_kg == pytest.approx(geometry.nozzle_mass_kg)

    overridden_positive = geometry_from_components(
        grain,
        motor,
        propellant,
        casing_wall_thickness_m=0.004,
        dry_mass_kg=3.25,
        casing_material=CasingMaterial(liner_thickness_m=0.002),
        nozzle_material=NozzleMaterial(),
    )
    assert overridden_positive.dry_mass_kg == 3.25


def test_geometry_negative_liner_thickness_is_inactive():
    geometry, _motor = _geometry_with_materials(
        casing_material=CasingMaterial(
            liner_thickness_m=-0.001,
            liner_density_kg_m3=math.nan,
        ),
        nozzle_material=NozzleMaterial(density_kg_m3=0.0),
    )

    assert geometry.liner_mass_kg == 0.0


def test_default_nozzle_convergent_angle_is_publicly_exported():
    assert DEFAULT_NOZZLE_CONVERGENT_HALF_ANGLE_DEG == 45.0


def test_geometry_nozzle_zero_density_is_valid_and_zero_mass():
    geometry, _motor = _geometry_with_materials(
        casing_material=CasingMaterial(),
        nozzle_material=NozzleMaterial(density_kg_m3=0.0),
    )
    assert geometry.nozzle_mass_kg == 0.0


def test_geometry_material_properties_are_valid_for_native_model():
    geometry, _motor = _geometry_with_materials(
        casing_material=CasingMaterial(
            density_kg_m3=7800.0,
            bulkhead_fraction=1.35,
            liner_thickness_m=0.002,
            liner_density_kg_m3=1100.0,
        ),
        nozzle_material=NozzleMaterial(
            density_kg_m3=1800.0,
            wall_thickness_factor=1.15,
            min_wall_thickness_m=0.004,
        ),
    )

    for value in (
        geometry.casing_mass_kg,
        geometry.liner_mass_kg,
        geometry.nozzle_mass_kg,
        geometry.dry_mass_kg,
    ):
        assert np.isfinite(value)


def test_geometry_legacy_mode_keeps_valid_nozzle_angle_irrelevant():
    grain, motor, propellant, _environment = make_motor_stack()
    motor.nozzle_angle = None
    geometry = geometry_from_components(
        grain,
        motor,
        propellant,
        0.004,
        dry_mass_kg=3.0,
    )
    assert geometry.dry_mass_kg == 3.0


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


def test_thermal_ablation_thin_liner_stays_below_recovery_temperature():
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
        "thrust_n": np.full_like(time_s, 4500.0),
        "mass_flow_kg_s": np.full_like(time_s, 0.6),
        "chamber_pressure_pa": np.full_like(time_s, 8.0e6),
        "gamma": propellant.specific_heat_ratio,
    }

    thermal = simulate_thermal_ablation(
        geometry,
        curve,
        casing_material=CasingMaterial(liner_thickness_m=1.5e-4),
        nozzle_material=NozzleMaterial(),
        flame_temp_k=2800.0,
        r_specific=propellant.products_constant,
    )

    recovery_temp_c = thermal["simulation.advanced.metadata.max_chamber_temperature_k"] - 273.15
    for key in [
        "simulation.advanced.thermal.liner_hot_face_temp_c",
        "simulation.advanced.thermal.liner_casing_interface_temp_c",
        "simulation.advanced.thermal.casing_inner_wall_temp_c",
        "simulation.advanced.thermal.casing_outer_wall_temp_c",
    ]:
        assert np.isfinite(thermal[key])
        assert thermal[key] <= recovery_temp_c + 1e-3


def test_thermal_ablation_thin_liner_is_stable_across_time_grids():
    grain, motor, propellant, _environment = make_motor_stack()
    geometry = geometry_from_components(
        grain,
        motor,
        propellant,
        casing_wall_thickness_m=0.004,
        dry_mass_kg=3.0,
    )

    def simulate_for_time_grid(time_s):
        curve = {
            "time_s": time_s,
            "thrust_n": np.full_like(time_s, 4500.0),
            "mass_flow_kg_s": np.full_like(time_s, 0.6),
            "chamber_pressure_pa": np.full_like(time_s, 8.0e6),
            "gamma": propellant.specific_heat_ratio,
        }
        return simulate_thermal_ablation(
            geometry,
            curve,
            casing_material=CasingMaterial(liner_thickness_m=1.5e-4),
            nozzle_material=NozzleMaterial(),
            flame_temp_k=2800.0,
            r_specific=propellant.products_constant,
        )

    coarse = simulate_for_time_grid(np.linspace(0.0, 1.0, 6))
    fine = simulate_for_time_grid(np.linspace(0.0, 1.0, 101))

    for key in [
        "simulation.advanced.thermal.liner_hot_face_temp_c",
        "simulation.advanced.thermal.liner_casing_interface_temp_c",
        "simulation.advanced.thermal.casing_inner_wall_temp_c",
        "simulation.advanced.thermal.casing_outer_wall_temp_c",
        "simulation.advanced.thermal.heat_load_kj_m2",
    ]:
        assert np.isclose(coarse[key], fine[key], rtol=3e-3, atol=1e-3)


def test_thermal_ablation_uses_sparse_analytic_jacobian(monkeypatch):
    import solidpy.Multiphysics as multiphysics

    grain, motor, propellant, _environment = make_motor_stack()
    geometry = geometry_from_components(
        grain,
        motor,
        propellant,
        casing_wall_thickness_m=0.004,
        dry_mass_kg=3.0,
    )
    time_s = np.linspace(0.0, 0.2, 3)
    curve = {
        "time_s": time_s,
        "thrust_n": np.full_like(time_s, 4500.0),
        "mass_flow_kg_s": np.full_like(time_s, 0.6),
        "chamber_pressure_pa": np.full_like(time_s, 8.0e6),
        "gamma": propellant.specific_heat_ratio,
    }
    captured = {}
    solve_ivp = multiphysics.solve_ivp

    def capture_solver(*args, **kwargs):
        captured["rhs"] = args[0]
        captured["jacobian"] = kwargs["jac"]
        return solve_ivp(*args, **kwargs)

    monkeypatch.setattr(multiphysics, "solve_ivp", capture_solver)
    thermal = simulate_thermal_ablation(
        geometry,
        curve,
        casing_material=CasingMaterial(liner_thickness_m=1.5e-4),
        nozzle_material=NozzleMaterial(),
        flame_temp_k=2800.0,
        r_specific=propellant.products_constant,
    )

    state = np.full(
        int(thermal["simulation.advanced.metadata.thermal_node_count"]),
        600.0,
    )
    direction = np.zeros_like(state)
    direction[0] = 1.0
    perturbation = 1e-4
    finite_difference = (
        captured["rhs"](0.0, state + perturbation * direction)
        - captured["rhs"](0.0, state - perturbation * direction)
    ) / (2.0 * perturbation)
    jacobian = captured["jacobian"](0.0, state)

    assert sparse.isspmatrix_csc(jacobian)
    assert np.allclose(jacobian.dot(direction), finite_difference, rtol=1e-5, atol=1e-5)
