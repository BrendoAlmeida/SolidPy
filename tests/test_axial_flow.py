import math

import numpy as np
import pytest

from solidpy import (
    BurnSimulation, Environment, Grain, Motor, Propellant, backends,
    compare_axial_diagnostics, evaluate_axial_mass_flux,
)
from solidpy.backends import SolveOptions
from solidpy.batch import ProblemBatch


def make_motor(grains):
    return Motor(
        grains, chamber_inner_radius=0.06, nozzle_throat_radius=0.008,
        nozzle_exit_radius=0.018, grain_separation=0.005,
    )


def make_history(regression, generated, igniter=None, time=None):
    regression = np.atleast_2d(regression)
    generated = np.atleast_2d(generated)
    count = len(regression)
    return {
        "time_s": np.arange(count, dtype=float) if time is None else np.asarray(time),
        "regression_m": regression, "mdot_generated_grains_kg_s": generated,
        "mdot_igniter_kg_s": np.zeros(count) if igniter is None else np.asarray(igniter),
    }


def make_uniform_history_batch():
    propellant = Propellant(
        1.1308, 0.04197, 1720.0, density=1879.0, burn_rate_a=7.36, burn_rate_n=0.32,
    )
    motors = [
        make_motor([Grain(0.035, 0.015, initial_height=0.12)]),
        make_motor([
            Grain(0.035, 0.015, initial_height=0.12, geometry="star", n_points=5, epsilon=0.1),
            Grain(0.034, 0.014, initial_height=0.10),
        ]),
        make_motor([
            Grain(0.035, 0.015, initial_height=0.12),
            Grain(0.034, 0.014, initial_height=0.10, geometry="star", n_points=5, epsilon=0.1),
            Grain(0.033, 0.013, initial_height=0.09),
        ]),
    ]
    settings = {"max_step_size": 0.04, "rtol": 1e-6, "atol": 1e-9}
    batch = ProblemBatch.from_objects(
        motors, propellant, [Environment() for _ in motors], settings=settings,
    )
    return batch, motors


def assert_uniform_axial_matches_scalar_history(backend, device=None, *, rtol):
    batch, motors = make_uniform_history_batch()
    runner = backends.get_backend(backend, device=device)
    full = runner.solve_burn(batch, SolveOptions(history="full", max_steps=900, tiers=())).to_results()
    uniform = runner.solve_burn(batch, SolveOptions(history="uniform:24", max_steps=900, tiers=())).to_results()

    for motor, native, result in zip(motors, full, uniform):
        expected = evaluate_axial_mass_flux(motor, native["history"])["metrics"]
        actual = result["history"]["diagnostics"]["axial_mass_flux"]
        assert tuple(actual) == tuple(expected)
        for key in (
            "max_axial_mass_flux_grain_index", "max_axial_mass_flux_station_id",
            "max_axial_mass_flux_time_index", "max_axial_mass_flux_station_index",
        ):
            assert actual[key] == expected[key], key
        for key in (
            "max_axial_mass_flux_kg_m2_s", "max_axial_mass_flux_time_s", "max_axial_mass_flux_position_m",
        ):
            np.testing.assert_allclose(actual[key], expected[key], rtol=rtol, atol=0.0, err_msg=key)


@pytest.mark.parametrize("direction", ["negative", "positive"])
@pytest.mark.parametrize("geometry", ["tubular", "star"])
@pytest.mark.parametrize("inhibited", [False, True])
def test_local_sources_ports_and_shrinking_coordinates_match_analytical_reference(direction, geometry, inhibited):
    grains = [
        Grain(0.04, 0.01, initial_height=0.14, geometry=geometry,
              n_points=5, epsilon=0.1, slot_fraction=0.5, ends_burn=inhibited),
        Grain(0.035, 0.012, initial_height=0.10, ends_burn=inhibited),
    ]
    motor = make_motor(grains)
    regression = np.array([[0.004, 0.002], [0.01, 0.006]])
    sources = np.array([[0.12, 0.05], [0.14, 0.04]])
    history = make_history(regression, sources, igniter=[0.01, 0.02], time=[0.0, 0.3])
    diagnostic = evaluate_axial_mass_flux(motor, history, stations_per_grain=5, nozzle_direction=direction)
    expected_order = [1, 0] if direction == "negative" else [0, 1]
    assert diagnostic["grain_order"] == expected_order
    upstream = history["mdot_igniter_kg_s"].copy()
    fractions = np.linspace(0.0, 1.0, 5)
    for ordered, i in enumerate(expected_order):
        grain = grains[i]
        r = grain.initial_inner_radius + regression[:, i]
        height = np.full(2, grain.initial_height) if inhibited else grain.initial_height - 2 * regression[:, i]
        if grain.geometry == "star":
            floor = np.minimum(grain.outer_radius, grain.initial_inner_radius + 0.5 * (
                grain.outer_radius - grain.initial_inner_radius) + regression[:, i])
            port = math.pi * r**2 + 0.5 * (floor**2 - r**2)
            perimeter = (2 * math.pi - 1.0) * r + 1.0 * floor
        else:
            port = math.pi * r**2
            perimeter = 2 * math.pi * r
        end_area = np.zeros(2) if inhibited else math.pi * grain.outer_radius**2 - port
        area = perimeter * height + 2 * end_area
        face = sources[:, i] * end_area / area
        lateral = sources[:, i] * perimeter * height / area
        expected_flow = upstream[:, None] + face[:, None] + lateral[:, None] * fractions
        expected_flow[:, -1] += face
        block = slice(ordered * 5, (ordered + 1) * 5)
        np.testing.assert_allclose(diagnostic["mdot_axial_kg_s"][:, block], expected_flow)
        np.testing.assert_allclose(diagnostic["mass_flux_kg_m2_s"][:, block], expected_flow / port[:, None])
        start = motor.grain_axial_positions_m[i] + (np.zeros(2) if inhibited else regression[:, i])
        coordinate_fraction = 1 - fractions if direction == "negative" else fractions
        np.testing.assert_allclose(diagnostic["positions_m"][:, block], start[:, None] + height[:, None] * coordinate_fraction)
        upstream += sources[:, i]
    assert diagnostic["status"] == "uncalibrated_diagnostic"
    assert diagnostic["convergence"]["status"] == "not_evaluated"
    maximum = diagnostic["metrics"]
    t, s = maximum["max_axial_mass_flux_time_index"], maximum["max_axial_mass_flux_station_index"]
    assert maximum["max_axial_mass_flux_kg_m2_s"] == diagnostic["mass_flux_kg_m2_s"][t, s]
    assert maximum["max_axial_mass_flux_station_id"] == diagnostic["station_ids"][s]
    assert maximum["max_axial_mass_flux_grain_index"] == diagnostic["grain_indices"][s]
    assert maximum["max_axial_mass_flux_time_s"] == diagnostic["time_s"][t]


@pytest.mark.parametrize("geometry", ["tubular", "star"])
def test_slot_floor_phase_transition_has_correct_local_port_and_source_split(geometry):
    grain = Grain(0.04, 0.01, initial_height=0.14, geometry=geometry,
                  n_points=5, epsilon=0.1, slot_fraction=0.5)
    history = make_history([[0.018]], [[0.1]])
    diagnostic = evaluate_axial_mass_flux(make_motor([grain]), history)
    bore = 0.028
    if geometry == "star":
        port = math.pi * bore**2 + 0.5 * (0.04**2 - bore**2)
        perimeter = (2 * math.pi - 1) * bore
    else:
        port = math.pi * bore**2
        perimeter = 2 * math.pi * bore
    end_area = math.pi * 0.04**2 - port
    expected_face = 0.1 * end_area / (perimeter * 0.104 + 2 * end_area)
    assert diagnostic["mdot_end_face_grains_kg_s"][0, 0] == pytest.approx(expected_face)
    assert diagnostic["port_area_m2"][0, 0] == pytest.approx(port)


def test_empty_axially_exhausted_grain_uses_chamber_area_and_does_not_imply_nozzle_flow():
    grains = [Grain(0.04, 0.01, initial_height=0.02), Grain(0.035, 0.015, initial_height=0.1)]
    motor = make_motor(grains)
    history = make_history([[0.01, 0.001], [0.01, 0.02]], [[0, 0.1], [0, 0]], igniter=[0.01, 0.0])
    history["mdot_nozzle_kg_s"] = np.array([0.08, 0.05])
    result = evaluate_axial_mass_flux(motor, history, stations_per_grain=3)
    np.testing.assert_allclose(result["port_area_m2"][:, 3:], motor.chamber_area)
    np.testing.assert_allclose(result["mdot_axial_kg_s"][0, 3:], 0.11)
    np.testing.assert_allclose(result["mdot_axial_kg_s"][1], 0.0)
    assert result["assumptions"]["gas_storage_distribution"] == "not_modeled"
    assert result["assumptions"]["gas_drainage_distribution"] == "not_modeled"
    assert not result["assumptions"]["erosion_feedback"]
    assert not result["assumptions"]["physical_validity_gate"]


@pytest.mark.parametrize("stations", [0, 1, -1, 3.5, True, np.bool_(True), "5"])
def test_invalid_station_count_is_rejected(stations):
    motor = make_motor([Grain(0.04, 0.01, initial_height=0.1)])
    with pytest.raises(ValueError, match="stations_per_grain"):
        evaluate_axial_mass_flux(motor, make_history([[0]], [[0.1]]), stations_per_grain=stations)


@pytest.mark.parametrize("kwargs,match", [
    ({"nozzle_direction": "left"}, "nozzle_direction"),
    ({"flow_arrangement": "split"}, "single_outlet"),
    ({"flow_arrangement": "dual_outlet"}, "single_outlet"),
])
def test_unsupported_flow_configuration_is_rejected(kwargs, match):
    motor = make_motor([Grain(0.04, 0.01, initial_height=0.1)])
    with pytest.raises(ValueError, match=match):
        evaluate_axial_mass_flux(motor, make_history([[0]], [[0.1]]), **kwargs)


@pytest.mark.parametrize("name,value", [
    ("time_s", [0, 0]), ("time_s", [math.nan, 1]), ("time_s", [-1, 1]),
    ("regression_m", [[0], [math.inf]]), ("regression_m", [[0], [-1]]),
    ("regression_m", [[0], [0.04]]), ("regression_m", [0, 0]),
    ("mdot_generated_grains_kg_s", [[0.1], [-0.1]]),
    ("mdot_generated_grains_kg_s", [[0.1, 0.2], [0.1, 0.2]]),
    ("mdot_igniter_kg_s", [0]), ("mdot_igniter_kg_s", [0, math.nan]),
])
def test_invalid_history_is_rejected(name, value):
    motor = make_motor([Grain(0.04, 0.01, initial_height=0.1)])
    history = make_history([[0], [0.01]], [[0.1], [0.1]])
    history[name] = value
    with pytest.raises(ValueError, match=name):
        evaluate_axial_mass_flux(motor, history)


def test_burned_out_sources_are_rejected():
    grain = Grain(0.04, 0.01, initial_height=0.02)
    with pytest.raises(ValueError, match="burned-out"):
        evaluate_axial_mass_flux(make_motor([grain]), make_history([[0.01]], [[0.1]]))


def test_missing_required_history_is_rejected():
    grain = Grain(0.04, 0.01, initial_height=0.02)
    with pytest.raises(ValueError, match="requires history"):
        evaluate_axial_mass_flux(make_motor([grain]), {})


def test_spatial_refinement_and_linear_temporal_reference_are_exact():
    motor = make_motor([Grain(0.04, 0.01, initial_height=0.1)])
    coarse = evaluate_axial_mass_flux(motor, make_history([[0], [0]], [[0.1], [0.2]], time=[0, 1]), stations_per_grain=2)
    refined = evaluate_axial_mass_flux(motor, make_history([[0], [0], [0]], [[0.1], [0.15], [0.2]], time=[0, 0.5, 1]), stations_per_grain=17)
    comparison = compare_axial_diagnostics(coarse, refined)
    assert comparison["status"] == "evaluated_diagnostic_only"
    assert comparison["temporal"]["peak_relative_delta"] == 0.0
    assert comparison["temporal"]["profile_relative_max_error"] < 1e-15
    assert comparison["spatial"]["peak_relative_errors"] == {"coarse": 0.0, "refined": 0.0}
    assert not comparison["physical_validity_gate"]


@pytest.mark.parametrize("geometry", ["tubular", "star"])
def test_real_adaptive_core_temporal_and_spatial_refinement(geometry):
    grain = Grain(0.035, 0.015, initial_height=0.12, geometry=geometry)
    motor = Motor([grain], chamber_inner_radius=0.037, chamber_length=0.14,
                  nozzle_throat_radius=0.008, nozzle_exit_radius=0.018)
    propellant = Propellant(1.1308, 0.04197, 1720.0, density=1879.0, burn_rate_a=7.36, burn_rate_n=0.32)
    coarse_core = BurnSimulation(grain, motor, propellant, Environment(), max_step_size=0.04, rtol=1e-6, atol=1e-9)
    refined_core = BurnSimulation(grain, motor, propellant, Environment(), max_step_size=0.01, rtol=1e-9, atol=1e-12)
    assert coarse_core.result["status"]["completed"] and refined_core.result["status"]["completed"]
    coarse = evaluate_axial_mass_flux(motor, coarse_core.result["history"], stations_per_grain=3)
    refined = evaluate_axial_mass_flux(motor, refined_core.result["history"], stations_per_grain=11)
    comparison = compare_axial_diagnostics(coarse, refined)
    assert comparison["temporal"]["peak_relative_delta"] < 0.02
    assert np.isfinite(comparison["temporal"]["profile_relative_max_error"])
    assert comparison["spatial"]["peak_relative_errors"] == {"coarse": 0.0, "refined": 0.0}
    np.testing.assert_allclose(refined["mdot_axial_kg_s"][-1], 0.0)
    assert refined_core.result["history"]["mdot_nozzle_kg_s"][-1] > 0.0


@pytest.mark.parametrize("floor", [0, -1, math.nan, math.inf, True, [1], "1"])
def test_invalid_comparison_scale_is_rejected(floor):
    motor = make_motor([Grain(0.04, 0.01, initial_height=0.1)])
    diagnostic = evaluate_axial_mass_flux(motor, make_history([[0]], [[0.1]]))
    with pytest.raises(ValueError, match="scale_floor"):
        compare_axial_diagnostics(diagnostic, diagnostic, scale_floor=floor)


def test_comparison_rejects_different_geometry_orientation_and_nonoverlap():
    motor = make_motor([Grain(0.04, 0.01, initial_height=0.1)])
    history = make_history([[0], [0]], [[0.1], [0.1]])
    reference = evaluate_axial_mass_flux(motor, history)
    reversed_result = evaluate_axial_mass_flux(motor, history, nozzle_direction="positive")
    with pytest.raises(ValueError, match="nozzle_direction"):
        compare_axial_diagnostics(reference, reversed_result)
    different = evaluate_axial_mass_flux(make_motor([Grain(0.04, 0.02, initial_height=0.1)]), history)
    with pytest.raises(ValueError, match="geometry"):
        compare_axial_diagnostics(reference, different)
    history["time_s"] = np.array([2, 3])
    with pytest.raises(ValueError, match="common time"):
        compare_axial_diagnostics(reference, evaluate_axial_mass_flux(motor, history))


def test_cpu_vectorized_uniform_axial_metrics_match_scalar_with_mixed_grain_counts():
    assert_uniform_axial_matches_scalar_history("cpu-vectorized", rtol=1e-12)


def test_jax_cpu_uniform_axial_metrics_match_scalar_with_mixed_grain_counts():
    pytest.importorskip("jax")
    from solidpy.backends import _tolerances

    assert_uniform_axial_matches_scalar_history("jax", device="cpu", rtol=_tolerances.KERNEL_RTOL_GPU)


@pytest.mark.gpu
def test_jax_gpu_uniform_axial_metrics_match_scalar_with_mixed_grain_counts():
    pytest.importorskip("jax")
    from solidpy.backends import _tolerances

    assert_uniform_axial_matches_scalar_history("jax", device="cuda:0", rtol=_tolerances.KERNEL_RTOL_GPU)
