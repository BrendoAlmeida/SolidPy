import math
from types import SimpleNamespace

import numpy as np
import pytest

from solidpy import CasingMaterial, Grain, Motor, NozzleMaterial
from solidpy.Multiphysics import geometry_from_components
from solidpy.surrogate_physics import compute_structural_features


def make_motor(grains=None, **kwargs):
    if grains is None:
        grains = Grain(outer_radius=0.035, initial_inner_radius=0.015, initial_height=0.12)
    parameters = {
        "chamber_inner_radius": 0.037,
        "nozzle_throat_radius": 0.008,
        "nozzle_exit_radius": 0.018,
        "nozzle_angle": math.radians(15),
        "chamber_length": 0.14,
    }
    parameters.update(kwargs)
    return Motor(grains, **parameters)


def test_connected_cylinder_and_frustum_subtract_propellant_once():
    grains = [
        Grain(outer_radius=0.035, initial_inner_radius=0.015, initial_height=0.12),
        Grain(outer_radius=0.033, initial_inner_radius=0.012, initial_height=0.10),
    ]
    cylinder_volume = math.pi * 0.037**2 * 0.24
    frustum_volume = math.pi * 0.029 / 3 * (0.037**2 + 0.037 * 0.008 + 0.008**2)
    connected_volume = cylinder_volume + frustum_volume
    motor = make_motor(
        grains, chamber_length=0.24, grain_separation=0.003,
        connected_chamber_volume_m3=connected_volume,
    )
    assert motor.chamber_volume == pytest.approx(connected_volume)
    assert motor.propellant_volume == pytest.approx(sum(grain.volume for grain in grains))
    assert motor.free_volume == pytest.approx(connected_volume - motor.propellant_volume)
    assert motor.evaluate_free_volume() == pytest.approx(motor.free_volume)
    assert motor.evaluate_chamber_volume() == pytest.approx(connected_volume)
    assert motor.chamber_length == 0.24
    assert motor.grain_axial_positions_m == pytest.approx((0.0, 0.123))


def test_connected_volume_preserves_physical_structure_and_mass():
    baseline = make_motor()
    connected = make_motor(connected_chamber_volume_m3=baseline.chamber_volume * 1.7)
    parameters = {
        "casing_material": CasingMaterial(),
        "nozzle_material": NozzleMaterial(),
        "chamber_pressure_pa": 3.5e6,
        "casing_wall_thickness_m": 0.004,
        "propellant_mass_kg": 1.0,
    }
    baseline_features = compute_structural_features(baseline, **parameters)
    connected_features = compute_structural_features(connected, **parameters)
    assert connected.chamber_length == baseline.chamber_length
    assert connected_features == baseline_features

    baseline_geometry = geometry_from_components(
        baseline.grain, baseline, SimpleNamespace(density=1700.0),
        casing_wall_thickness_m=0.004,
        casing_material=parameters["casing_material"],
        nozzle_material=parameters["nozzle_material"],
    )
    connected_geometry = geometry_from_components(
        connected.grain, connected, SimpleNamespace(density=1700.0),
        casing_wall_thickness_m=0.004,
        casing_material=parameters["casing_material"],
        nozzle_material=parameters["nozzle_material"],
    )
    assert connected_geometry.connected_chamber_volume_m3 > baseline_geometry.connected_chamber_volume_m3
    assert connected_geometry.motor_length_m == baseline_geometry.motor_length_m
    assert connected_geometry.casing_body_length_m == baseline_geometry.casing_body_length_m
    assert connected_geometry.motor_initial_mass_kg == pytest.approx(
        baseline_geometry.motor_initial_mass_kg
    )


def test_legacy_replicated_grains_and_inferred_physical_length():
    grain = Grain(outer_radius=0.035, initial_inner_radius=0.015, initial_height=0.12)
    motor = make_motor(grain, grain_number=3, chamber_length=None, grain_separation=0.002)
    assert motor.grains == [grain] * 3
    assert motor.grain is grain
    assert motor.chamber_length == pytest.approx(0.364)
    assert motor.chamber_volume == pytest.approx(motor.chamber_area * 0.364)
    assert motor.propellant_volume == pytest.approx(3 * grain.volume)
    assert motor.grain_axial_positions_m == pytest.approx((0.0, 0.122, 0.244))


@pytest.mark.parametrize("value", [0.0, -1.0, math.nan, math.inf, -math.inf, "0.01", True, np.bool_(True)])
def test_connected_volume_rejects_invalid_numbers(value):
    with pytest.raises(ValueError, match="connected_chamber_volume_m3"):
        make_motor(connected_chamber_volume_m3=value)


def test_connected_volume_accepts_numpy_real_scalar():
    motor = make_motor(connected_chamber_volume_m3=np.float64(0.002))
    assert motor.chamber_volume == 0.002


@pytest.mark.parametrize("factor", [1.0, 0.9])
def test_nonpositive_initial_free_volume_is_rejected(factor):
    grain = Grain(outer_radius=0.035, initial_inner_radius=0.015, initial_height=0.12)
    with pytest.raises(ValueError, match="initial free volume"):
        make_motor(grain, connected_chamber_volume_m3=grain.volume * factor)


def test_large_connected_volume_cannot_hide_axial_stack_overflow():
    grain = Grain(outer_radius=0.035, initial_inner_radius=0.015, initial_height=0.12)
    with pytest.raises(ValueError, match="physical chamber_length"):
        make_motor(
            grain, grain_number=2, chamber_length=0.24, grain_separation=0.001,
            connected_chamber_volume_m3=0.1,
        )


def test_stack_fit_accepts_roundoff_at_physical_boundary():
    grains = [
        Grain(outer_radius=0.035, initial_inner_radius=0.015, initial_height=height)
        for height in [0.1, 0.2]
    ]
    motor = make_motor(grains, chamber_length=0.3)
    assert motor.grain_axial_positions_m == (0.0, 0.1)


def test_connected_volume_cannot_hide_radial_overflow():
    with pytest.raises(ValueError, match="outer radius"):
        make_motor(chamber_inner_radius=0.034, connected_chamber_volume_m3=0.1)


@pytest.mark.parametrize("field", ["volume", "initial_height", "outer_radius"])
@pytest.mark.parametrize("value", [0.0, -1.0, math.nan, math.inf, "0.1", True])
def test_invalid_individual_grain_geometry_is_rejected(field, value):
    grain = Grain(outer_radius=0.035, initial_inner_radius=0.015, initial_height=0.12)
    setattr(grain, field, value)
    with pytest.raises(ValueError, match=field):
        make_motor(grain, connected_chamber_volume_m3=0.1)


def test_individual_grain_volume_must_fit_its_physical_envelope():
    grain = Grain(outer_radius=0.035, initial_inner_radius=0.015, initial_height=0.12)
    grain.volume = math.pi * grain.outer_radius**2 * grain.initial_height * 1.01
    with pytest.raises(ValueError, match="physical envelope"):
        make_motor(grain, connected_chamber_volume_m3=0.1)


@pytest.mark.parametrize("field", ["chamber_length", "chamber_inner_radius", "nozzle_throat_radius", "nozzle_exit_radius"])
@pytest.mark.parametrize("value", [0.0, -1.0, math.nan, math.inf, "0.1", True])
def test_motor_dimensions_reject_invalid_numbers(field, value):
    with pytest.raises(ValueError, match=field):
        make_motor(**{field: value})


@pytest.mark.parametrize("separation", [-0.001, math.nan, math.inf, "0.002", True])
def test_invalid_grain_separation_is_rejected(separation):
    with pytest.raises(ValueError, match="grain_separation"):
        make_motor(grain_separation=separation)


def test_empty_grain_stack_is_rejected():
    with pytest.raises(ValueError, match="at least one grain"):
        make_motor([])


@pytest.mark.parametrize("count", [0, -1, 1.5, True, np.bool_(True)])
@pytest.mark.parametrize("explicit_list", [False, True])
def test_invalid_grain_count_is_rejected(count, explicit_list):
    grain = Grain(outer_radius=0.035, initial_inner_radius=0.015, initial_height=0.05)
    grains = [grain, grain] if explicit_list else grain
    with pytest.raises(ValueError, match="grain_number"):
        make_motor(grains, grain_number=count)
