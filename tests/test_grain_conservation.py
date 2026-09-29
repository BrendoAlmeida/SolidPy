import math

import numpy as np
import pytest
from scipy.integrate import quad

from solidpy import Grain


@pytest.mark.parametrize("geometry", ["tubular", "star"])
@pytest.mark.parametrize("ends_inhibited", [False, True])
@pytest.mark.parametrize("height", [0.04, 0.2])
def test_burn_area_is_remaining_volume_derivative(geometry, ends_inhibited, height):
    grain = Grain(
        0.05, 0.01, initial_height=height, geometry=geometry,
        n_points=5, epsilon=0.1, slot_fraction=0.5, ends_burn=ends_inhibited,
    )
    assert grain.calculate_remaining_volume(0.0) == pytest.approx(grain.volume)
    assert grain.calculate_remaining_volume(grain.burnout_regression_m) == 0.0
    for fraction in [0.1, 0.49, 0.51, 0.9]:
        regression = fraction * grain.burnout_regression_m
        delta = 1e-7 * grain.burnout_regression_m
        derivative = (
            grain.calculate_remaining_volume(regression - delta)
            - grain.calculate_remaining_volume(regression + delta)
        ) / (2 * delta)
        assert derivative == pytest.approx(grain.evaluate_burn_area(regression), rel=1e-7)
    integral = quad(
        grain.evaluate_burn_area, 0.0, grain.burnout_regression_m,
        points=[0.02] if grain.burnout_regression_m > 0.02 else None,
        epsabs=1e-13,
    )[0]
    assert integral == pytest.approx(grain.volume, rel=1e-10)


@pytest.mark.parametrize("ends_inhibited", [False, True])
def test_star_ports_solid_volume_and_phase_transition_are_consistent(ends_inhibited):
    grain = Grain(
        0.05, 0.01, initial_height=0.2, geometry="star",
        n_points=5, epsilon=0.1, slot_fraction=0.5, ends_burn=ends_inhibited,
    )
    assert grain.geometry_model == "fixed_angle_radial_front_v1"
    for regression in [0.0, 0.01, 0.02, 0.03, 0.04]:
        height = grain.initial_height if ends_inhibited else grain.initial_height - 2 * regression
        solid_area = grain.calculate_remaining_volume(regression) / height
        assert solid_area + grain.evaluate_port_area(regression) == pytest.approx(math.pi * grain.outer_radius**2)
    for evaluator in [grain.evaluate_port_area, grain.calculate_remaining_volume]:
        assert evaluator(0.02 - 1e-10) == pytest.approx(evaluator(0.02 + 1e-10), rel=1e-7)
    height_at_floor = grain.initial_height if ends_inhibited else grain.initial_height - 0.04
    area_drop = grain.evaluate_burn_area(0.02 - 1e-10) - grain.evaluate_burn_area(0.02 + 1e-10)
    assert area_drop == pytest.approx(2 * grain.n_points * grain.epsilon * grain.outer_radius * height_at_floor)
    assert grain.evaluate_burn_area(grain.burnout_regression_m) == 0.0


@pytest.mark.parametrize("epsilon", [0.0, -0.1, math.nan, math.inf, math.pi / 5, math.pi])
def test_star_slots_reject_invalid_or_overlapping_angles(epsilon):
    with pytest.raises(ValueError, match="star slots"):
        Grain(0.05, 0.01, geometry="star", n_points=5, epsilon=epsilon)


def test_remaining_volume_does_not_mutate_geometry():
    grain = Grain(0.05, 0.01, initial_height=0.2, geometry="star")
    initial = (grain.height, grain.inner_radius, grain.volume, grain.burn_area)
    assert grain.calculate_remaining_volume(-0.01) == pytest.approx(grain.volume)
    grain.calculate_remaining_volume(0.02)
    assert (grain.height, grain.inner_radius, grain.volume, grain.burn_area) == initial
    assert np.isfinite(grain.calculate_remaining_volume(0.02))


@pytest.mark.parametrize("geometry", ["tubular", "star"])
@pytest.mark.parametrize("ends_inhibited", [False, True])
def test_updated_grain_volume_uses_canonical_regression_geometry(geometry, ends_inhibited):
    grain = Grain(
        0.05, 0.01, initial_height=0.2, geometry=geometry,
        n_points=5, epsilon=0.1, slot_fraction=0.5, ends_burn=ends_inhibited,
    )
    for regression in [0.0, 0.01, 0.02, 0.03, 0.04]:
        grain.evaluate_burn_area(regression, update_state=True)
        assert grain.evaluate_grain_volume() == pytest.approx(grain.calculate_remaining_volume(regression))
