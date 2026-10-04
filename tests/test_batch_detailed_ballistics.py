"""Parity checks for lane-batched detailed-ballistics post-processing."""

import numpy as np
import pytest

from solidpy import Grain, Motor, run_detailed_ballistics
from solidpy.DetailedBallistics import build_detailed_ballistics
from solidpy.batch.detailed_ballistics import DetailedBallisticsBatch, assemble_detailed_ballistics
from solidpy.batch.kernels.detailed_ballistics import detailed_ballistics
from solidpy.batch.simulation_view import SimulationView
from solidpy.backends import get_backend
from test_detailed_ballistics import make_motor_stack


def _solved_view(step, *, star=False, activation=None):
    if star:
        _, _, propellant, environment = make_motor_stack()
        grain = Grain(
            outer_radius=0.05, initial_inner_radius=0.012, initial_height=0.22,
            geometry="star", n_points=5, epsilon=0.1, slot_fraction=0.55,
        )
        motor = Motor(
            grain, chamber_inner_radius=0.06, nozzle_throat_radius=0.009, nozzle_exit_radius=0.02,
            dry_mass_kg=2.1, dry_center_of_mass_position_m=0.31,
        )
    else:
        grain, motor, propellant, environment = make_motor_stack()
    detailed = run_detailed_ballistics(
        grain, motor, propellant, environment, max_step_size=step, burn_area_activation=activation,
    )
    simulation = detailed["simulation"]
    return SimulationView(
        simulation.result, simulation.motor, simulation.propellant, simulation.environment_pressure,
        {
            "burn_area_activation": simulation.burn_area_activation,
            "ignition_ramp_time": simulation.ignition_ramp_time,
        },
    )


def _assert_detailed_equal(actual, expected):
    assert actual.keys() == expected.keys()
    for key, expected_value in expected.items():
        value = actual[key]
        if isinstance(expected_value, np.ndarray):
            np.testing.assert_allclose(value, expected_value, rtol=2e-9, atol=2e-11, err_msg=key)
        elif isinstance(expected_value, dict):
            assert value.keys() == expected_value.keys()
            for child, target in expected_value.items():
                if isinstance(target, (int, float)):
                    assert value[child] == pytest.approx(target, rel=2e-12, abs=1e-12), f"{key}.{child}"
                else:
                    assert value[child] == target, f"{key}.{child}"
        elif key in ("canonical_result", "status", "provenance"):
            assert value is expected_value
        else:
            assert value == expected_value


def test_numpy_kernel_matches_scalar_histories_for_ragged_tubular_and_star_lanes():
    views = [
        _solved_view(0.03, activation=[[0.0, 0.2], [0.04, 0.9], [0.2, 1.0]]),
        _solved_view(0.02, star=True, activation=0.63),
    ]
    options = [
        {"resample_step": 0.021, "max_time_points": 55, "nozzle_ablation_scale": 0.8},
        {"resample_step": 0.034, "max_time_points": 37, "nozzle_ablation_scale": 1.2},
    ]
    batch = DetailedBallisticsBatch.from_views(views, options)
    actual = assemble_detailed_ballistics(batch, detailed_ballistics(batch.namespace(np), np))
    assert batch.arrays["raw_regression_m"].shape[2] == 4
    assert batch.arrays["query_time_s"].shape[1] == 55

    for lane, (view, lane_options) in enumerate(zip(views, options)):
        expected = build_detailed_ballistics(view, **lane_options)
        _assert_detailed_equal(actual[lane], expected)

    serviced = get_backend("cpu-vectorized").detailed_ballistics(batch).to_results()
    for lane, result in enumerate(serviced):
        _assert_detailed_equal(result, actual[lane])


def test_scalar_backend_service_keeps_custom_activation_callbacks():
    view = _solved_view(0.03, activation=lambda time, regression: 0.5)
    options = {"resample_step": 0.03, "max_time_points": 32}
    batch = DetailedBallisticsBatch.from_views([view], options)
    result = get_backend("cpu-reference").detailed_ballistics(batch).to_results()[0]
    expected = build_detailed_ballistics(view, **options)
    _assert_detailed_equal(result, expected)


def test_batch_selection_trims_each_ragged_axis_and_scalar_callback_is_marked():
    views = [
        _solved_view(0.03),
        _solved_view(0.02, star=True, activation=lambda time, regression: 0.5),
    ]
    batch = DetailedBallisticsBatch.from_views(
        views, [{"resample_step": 0.02}, {"resample_step": 0.04}]
    )
    selected = batch.select([1])
    assert selected.arrays["raw_regression_m"].shape[2] == 1
    assert selected.arrays["query_time_s"].shape[1] == len(batch.lanes[1]["query_time"])
    assert selected.unsupported_lanes == [0]
    results = assemble_detailed_ballistics(batch, detailed_ballistics(batch.namespace(np), np))
    assert results[0] is not None
    assert results[1] is None


def test_jax_backend_service_matches_numpy_when_jax_is_installed():
    pytest.importorskip("jax")
    views = [_solved_view(0.03), _solved_view(0.02, star=True)]
    batch = DetailedBallisticsBatch.from_views(views, {"resample_step": 0.03, "max_time_points": 40})
    expected = get_backend("cpu-vectorized").detailed_ballistics(batch).to_results()
    actual = get_backend("jax", device="cpu").detailed_ballistics(batch).to_results()
    for got, want in zip(actual, expected):
        _assert_detailed_equal(got, want)
