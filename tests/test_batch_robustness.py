"""Robustness scenarios as lanes of a batch backend, against the scalar ``run_robustness_analysis``."""

import numpy as np
import pytest

from solidpy import Environment, Grain, Motor, Propellant, default_robustness_scenarios, run_robustness_analysis
from solidpy import backends, ensemble
from solidpy.backends import _tolerances as tol
from solidpy.ensemble import run_robustness_ensemble
from test_robustness import make_motor_stack

SCENARIO_IDS = ("low_burn_rate", "wide_throat", "low_density", "cold_start", "high_altitude")
KWARGS = dict(monte_carlo_sample_count=3, max_step_size=0.02, max_time_points=400)


def scenarios():
    return [s for s in default_robustness_scenarios() if s.scenario_id in SCENARIO_IDS]


def second_design():
    """A two-grain design with a power-law propellant: another grain count, another burn model."""
    grain = Grain(outer_radius=0.0305, initial_inner_radius=0.0115, initial_height=0.09)
    motor = Motor([grain, grain], chamber_inner_radius=0.032, chamber_length=0.2, nozzle_throat_radius=0.0085,
                  nozzle_exit_radius=0.021, nozzle_angle=0.2618, dry_mass_kg=1.5, dry_center_of_mass_position_m=0.0)
    propellant = Propellant(1.1308, 0.04197, 1720.0, density=1879.0, burn_rate_a=7.36, burn_rate_n=0.32)
    return grain, motor, propellant, Environment()


@pytest.fixture(scope="module")
def design():
    return make_motor_stack()


@pytest.fixture(scope="module")
def scalar_report(design):
    return run_robustness_analysis(*design, scenarios=scenarios(), **KWARGS)


# ---- the comparison -------------------------------------------------------------------------------------------
SUMMARY_RTOL = {
    "total_impulse_ns": tol.INTEGRAL_RTOL, "generated_mass_integral_kg": tol.INTEGRAL_RTOL,
    "nozzle_mass_integral_kg": tol.INTEGRAL_RTOL, "igniter_mass_injected_kg": tol.INTEGRAL_RTOL,
    "gas_mass_initial_kg": tol.INTEGRAL_RTOL, "isp_effective_s": tol.INTEGRAL_RTOL, "avg_thrust_n": tol.INTEGRAL_RTOL,
    "mass_flow_avg_kg_s": tol.INTEGRAL_RTOL, "mass_flow_avg_nozzle_kg_s": tol.INTEGRAL_RTOL,
    "cstar_effective_m_s": tol.INTEGRAL_RTOL, "ignition_active_fraction_final": tol.INTEGRAL_RTOL,
    "burn_time_s": tol.TIME_RTOL, "nozzle_flow_duration_s": tol.TIME_RTOL,
    "peak_thrust_n": tol.GRID_SAMPLED_RTOL, "chamber_pressure_max_mpa": tol.GRID_SAMPLED_RTOL,
    "max_nozzle_mass_flow_kg_s": tol.GRID_SAMPLED_RTOL, "gas_mass_cutoff_kg": tol.GRID_SAMPLED_RTOL,
    "max_mass_flow_kg_s": tol.GENERATED_FLOW_PEAK_COARSE_RTOL,
    "throat_ablation_mm": tol.DETAILED_ACCUMULATED_RTOL, "final_throat_diameter_mm": tol.DETAILED_ACCUMULATED_RTOL,
    "pressure_rise_rate_max_mpa_s": tol.FINITE_DIFFERENCE_RTOL,
}
ROBUSTNESS_RTOL = {"burn_time": tol.TIME_RTOL, "peak_thrust": tol.GRID_SAMPLED_RTOL, "total_impulse": tol.INTEGRAL_RTOL}
SERIES = ("time_s", "thrust_n", "chamber_pressure_pa", "free_volume_m3", "regressed_length_m", "burn_area_m2",
          "mass_flow_kg_s", "mass_nozzle_kg_s", "gas_mass_kg", "impulse_integral_ns", "propellant_mass_kg",
          "motor_mass_kg", "motor_center_of_mass_position_m", "exit_pressure_pa", "throat_diameter_m",
          "throat_ablation_m", "cf", "ignition_active_fraction")
#: Series that drop to zero when a grain burns out: samples inside the last accepted step before a burnout read a
#: fraction of the jump that depends on where the solver put that step (see the version 4 note of the tolerances).
JUMPING_SERIES = ("burn_area_m2", "mass_flow_kg_s")


def before_a_burnout(lane, step):
    """Mask of the samples of ``lane`` that lie within one maximum step before one of its grain burnouts."""
    time = np.asarray(lane["time_s"])
    mask = np.zeros(len(time), dtype=bool)
    for burnout in lane["canonical_result"]["metrics"]["grain_burnout_times_s"]:
        if burnout is not None:
            mask |= (time >= burnout - step) & (time < burnout)
    return mask


def assert_summaries_close(got, want, where):
    assert got.keys() == want.keys(), where
    for key, expected in want.items():
        name = key.split(".")[-1]
        if not isinstance(expected, float):
            assert got[key] == expected, (where, key)
        elif name == "mass_conservation_error_pct":
            assert abs(got[key] - expected) <= tol.MASS_BALANCE_ATOL_PCT, (where, key, got[key], expected)
        else:
            assert name in SUMMARY_RTOL or key in ("simulation.schema_version",), f"uncategorised summary key {key}"
            rtol = SUMMARY_RTOL.get(name, 0.0)
            assert got[key] == pytest.approx(expected, rel=rtol, abs=0.0), (where, key)


def assert_reports_close(got, want, step=KWARGS["max_step_size"]):
    assert set(got) == set(want)
    assert got["status"] == want["status"] and got["scenario_ids"] == want["scenario_ids"]
    for a, b in zip([got["nominal"]] + got["scenarios"], [want["nominal"]] + want["scenarios"]):
        where = a["scenario_id"]
        assert set(a) == set(b) and a["scenario_id"] == b["scenario_id"] and a["scenario_kind"] == b["scenario_kind"]
        assert a.get("scenario_factors") == b.get("scenario_factors"), where
        assert a["status"]["termination_reason"] == b["status"]["termination_reason"], where
        assert a["status"]["completed"] == b["status"]["completed"], where
        assert a["provenance"]["physics_provider_hash"] == b["provenance"]["physics_provider_hash"], where
        assert_summaries_close(a["summary"], b["summary"], where)
        skipped = before_a_burnout(b, step)
        for name in SERIES:
            x, y = np.asarray(a[name]), np.asarray(b[name])
            assert x.shape == y.shape, (where, name)
            rtol = 1e-6 if name == "time_s" else tol.DETAILED_SERIES_RTOL
            if name in JUMPING_SERIES:
                x, y = x[~skipped], y[~skipped]
            np.testing.assert_allclose(x, y, rtol=0.0, atol=rtol * max(float(np.max(np.abs(y))), 1e-300),
                                       err_msg=f"{where} {name}")
    assert got["summary"].keys() == want["summary"].keys()
    for key, expected in want["summary"].items():
        quantity = next((q for q in ROBUSTNESS_RTOL if f".{q}_" in key), None)
        if quantity is None:
            assert got["summary"][key] == expected, key  # counts and ratios
        else:
            rtol = ROBUSTNESS_RTOL[quantity]
            mean = want["summary"][f"simulation.robustness.{quantity}_mean_{key.rsplit('_', 1)[-1]}"]
            assert got["summary"][key] == pytest.approx(expected, rel=rtol, abs=rtol * abs(mean)), key


def same_exactly(a, b, path=""):
    """Deep equality for report structures (arrays, nested mappings), except the simulation objects."""
    if isinstance(a, dict):
        assert a.keys() == b.keys(), path
        for key in a:
            if key != "simulation":
                same_exactly(a[key], b[key], f"{path}.{key}")
    elif isinstance(a, (list, tuple)):
        assert len(a) == len(b), path
        for i, (x, y) in enumerate(zip(a, b)):
            same_exactly(x, y, f"{path}[{i}]")
    elif isinstance(a, np.ndarray):
        np.testing.assert_array_equal(a, b, err_msg=path)
    else:
        assert a == b or (a != a and b != b), path


# ---- tests ----------------------------------------------------------------------------------------------------
def test_the_default_path_does_not_touch_the_batch_machinery(design, monkeypatch):
    monkeypatch.setattr(ensemble, "run_robustness_ensemble", lambda *a, **k: pytest.fail("batch path used"))

    report = run_robustness_analysis(*design, scenarios=scenarios()[:1], max_step_size=0.03, max_time_points=100)

    assert report["status"] == "completed"


def test_the_reference_backend_gives_the_scalar_report_bit_for_bit(design, scalar_report):
    report = run_robustness_analysis(*design, scenarios=scenarios(), backend="cpu-reference", **KWARGS)

    same_exactly(report, scalar_report)


def test_the_reference_backend_in_a_process_pool_gives_the_same_report(design, scalar_report):
    report = run_robustness_analysis(*design, scenarios=scenarios(), backend="cpu-reference", workers=2, **KWARGS)

    same_exactly(report, scalar_report)


def test_the_numpy_backend_matches_the_scalar_report_within_the_tolerances(design, scalar_report):
    report = run_robustness_analysis(*design, scenarios=scenarios(), backend="cpu-vectorized", **KWARGS)

    assert_reports_close(report, scalar_report)
    execution = report["nominal"]["canonical_result"]["provenance"]["execution"]
    assert execution["backend"] == "cpu-vectorized" and execution["scenario_inputs"] == {"burn_rate_factor": 1.0}
    factors = [s["canonical_result"]["provenance"]["execution"]["scenario_inputs"]["burn_rate_factor"]
               for s in report["scenarios"]]
    assert factors[0] == pytest.approx(0.94 * (1.0 + 0.005 * (298.15 - 298.15)))  # low_burn_rate
    assert len(set(factors)) > 3  # the temperature and the sampled scenarios changed it per lane


def test_the_numpy_backend_matches_the_scalar_report_on_a_power_law_design():
    design = second_design()
    options = dict(scenarios=scenarios()[:3], monte_carlo_sample_count=2, max_step_size=0.02, max_time_points=300)

    report = run_robustness_analysis(*design, backend="cpu-vectorized", **options)
    expected = run_robustness_analysis(*design, **options)

    assert_reports_close(report, expected, step=0.02)
    assert report["status"] == "completed"


def test_several_designs_are_one_batch_and_each_report_matches_its_own_run():
    designs = [make_motor_stack(), second_design()]
    options = dict(scenarios=scenarios()[:3], max_step_size=0.02, max_time_points=300, backend="cpu-vectorized")

    together = run_robustness_ensemble(designs, **options)
    alone = [run_robustness_ensemble([d], **options)[0] for d in designs]

    assert len(together) == 2
    for a, b in zip(together, alone):
        assert_reports_close(a, b)
        assert a["nominal"]["summary"] == b["nominal"]["summary"]  # lanes do not see each other: no change at all
    assert together[0]["nominal"]["propellant_mass_kg"].shape != together[1]["nominal"]["propellant_mass_kg"].shape


def test_dropping_the_series_keeps_every_scalar_output_and_the_statistics():
    designs = [make_motor_stack()]
    options = dict(scenarios=scenarios()[:3], max_step_size=0.03, max_time_points=200, backend="cpu-vectorized",
                   validator=lambda result: result["summary"]["simulation.nominal.peak_thrust_n"] > 0.0)

    full = run_robustness_ensemble(designs, **options)[0]
    slim = run_robustness_ensemble(designs, keep_series=False, **options)[0]

    assert slim["summary"] == full["summary"] and slim["status"] == full["status"]
    assert slim["provenance"] == full["provenance"] and slim["scenario_ids"] == full["scenario_ids"]
    for a, b in zip([slim["nominal"]] + slim["scenarios"], [full["nominal"]] + full["scenarios"]):
        assert not any(isinstance(v, np.ndarray) for v in a.values()) and "canonical_result" not in a
        assert "simulation" not in a and a["interpolation"] == b["interpolation"]  # small mappings are kept
        assert a["summary"] == b["summary"] and a["status"] == b["status"] and a["scenario_id"] == b["scenario_id"]
        assert a.get("valid") == b.get("valid") and a.get("scenario_factors") == b.get("scenario_factors")
        assert a["provenance"]["physics_provider_hash"] == b["provenance"]["physics_provider_hash"]
    assert all("valid" in r for r in slim["scenarios"]) and "valid" not in slim["nominal"]


def test_chunking_does_not_change_the_reports():
    designs = [make_motor_stack(), second_design()]
    options = dict(scenarios=scenarios()[:2], max_step_size=0.03, max_time_points=200, backend="cpu-vectorized")

    one = run_robustness_ensemble(designs, **options)
    split = run_robustness_ensemble(designs, chunk_lanes=1, **options)  # one design per solve

    for a, b in zip(one, split):
        assert_reports_close(a, b)


def test_a_per_design_setting_overrides_the_common_one_and_the_validator_sees_every_scenario():
    seen = []
    designs = [make_motor_stack()[:3] + (None,), make_motor_stack()[:3] + (None, {"max_step_size": 0.05})]

    reports = run_robustness_ensemble(
        designs, scenarios=scenarios()[:2], max_step_size=0.02, max_time_points=300, backend="cpu-vectorized",
        validator=lambda result: seen.append(result["scenario_id"]) or True,
    )

    assert seen == [s.scenario_id for s in scenarios()[:2]] * 2
    assert reports[0]["summary"]["simulation.robustness.valid_ratio"] == 1.0
    steps = [len(r["nominal"]["time_s"]) for r in reports]
    assert steps[0] > steps[1]  # the coarser per-design step gives fewer resampled points


def test_lanes_the_backend_cannot_run_fall_back_to_the_reference_and_strict_refuses_them(design):
    sources = dict(igniter_mass_flow=lambda t: 0.002 if t < 0.1 else 0.0, igniter_burn_time=0.1)
    options = dict(scenarios=scenarios()[:2], max_step_size=0.03, max_time_points=200)

    report = run_robustness_analysis(*design, backend="cpu-vectorized", **options, **sources)
    expected = run_robustness_analysis(*design, **options, **sources)

    for got, want in zip([report["nominal"]] + report["scenarios"], [expected["nominal"]] + expected["scenarios"]):
        assert got["canonical_result"]["provenance"]["execution"]["fallback"]["ran_on"] == "cpu-reference"
        assert got["summary"] == want["summary"]
    factors = [r["canonical_result"]["provenance"]["execution"]["scenario_inputs"]["burn_rate_factor"]
               for r in [report["nominal"]] + report["scenarios"]]
    assert factors == [1.0, pytest.approx(0.94), 1.0]  # the reference lanes record the factor they ran with too
    with pytest.raises(backends.UnsupportedLane, match="igniter_callable"):
        run_robustness_ensemble([design], backend="cpu-vectorized", strict=True, **options, **sources)


def test_the_igniter_energy_factor_scales_the_igniter_time_of_each_scenario(design):
    from solidpy import RobustnessScenario

    options = dict(max_step_size=0.03, max_time_points=200, igniter_mass_flow=0.01, igniter_burn_time=0.2)
    scenario_list = [RobustnessScenario("weak", igniter_energy_factor=0.5), RobustnessScenario("strong", igniter_energy_factor=1.5)]

    report = run_robustness_analysis(*design, scenarios=scenario_list, backend="cpu-vectorized", **options)
    expected = run_robustness_analysis(*design, scenarios=scenario_list, **options)

    injected = [s["summary"]["simulation.nominal.igniter_mass_injected_kg"] for s in report["scenarios"]]
    assert injected == pytest.approx([0.01 * 0.1, 0.01 * 0.3], rel=1e-6)
    assert_reports_close(report, expected)


def test_a_missing_dry_hardware_is_refused_before_anything_is_solved(design, monkeypatch):
    grain, motor, propellant, environment = design
    bare = Motor(grain, grain_number=4, chamber_inner_radius=77.92 / 2000, nozzle_throat_radius=17.5 / 2000,
                 nozzle_exit_radius=44.44 / 2000, nozzle_angle=0.26, chamber_length=0.6)
    monkeypatch.setattr(ensemble, "simulate_burn", lambda *a, **k: pytest.fail("solved before validating"))

    with pytest.raises(ValueError, match="explicit dry hardware mass"):
        run_robustness_ensemble([(grain, bare, propellant, environment)], backend="cpu-vectorized")


@pytest.mark.parametrize("designs, kwargs, error", [
    ([(1, 2)], {}, "a design is"),
    ([(1, 2, 3, 4, {}, 6)], {}, "a design is"),
    ([make_motor_stack()], {"chunk_lanes": 0}, "chunk_lanes must be a positive integer"),
    ([make_motor_stack()], {"chunk_lanes": True}, "chunk_lanes must be a positive integer"),
])
def test_invalid_arguments_are_refused(designs, kwargs, error):
    with pytest.raises(ValueError, match=error):
        run_robustness_ensemble(designs, backend="cpu-vectorized", **kwargs)


def test_device_and_workers_need_a_backend(design):
    for argument in ({"device": "cuda:0"}, {"workers": 4}):
        with pytest.raises(ValueError, match="pass backend= as well"):
            run_robustness_analysis(*design, scenarios=[], **argument)


def test_a_design_is_never_split_across_chunks():
    one = run_robustness_ensemble([make_motor_stack()], scenarios=scenarios()[:3], max_step_size=0.03, max_time_points=200,
                                  backend="cpu-vectorized", chunk_lanes=2)  # a design is 4 lanes, more than the limit

    assert len(one) == 1 and len(one[0]["scenarios"]) == 3


def test_no_designs_give_no_reports():
    assert run_robustness_ensemble([], backend="cpu-vectorized") == []


def test_auto_keeps_a_single_design_on_the_reference_and_reproduces_the_scalar_report(design, scalar_report):
    report = run_robustness_analysis(*design, scenarios=scenarios(), backend="auto", **KWARGS)

    same_exactly(report, scalar_report)  # 'auto' needs thousands of lanes to leave the reference
