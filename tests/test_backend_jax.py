import importlib.util

import numpy as np
import pytest

import golden_corpus as gc
from solidpy import backends
from solidpy.backends import BackendUnavailable, SolveOptions, UnsupportedLane
from solidpy.backends import _tolerances as tol
from solidpy.batch import ProblemBatch

PLAIN_TAGS = {"scalar_thermo", "power_law", "igniter_none", "activation_none", "tail_off_numerical"}
ROUNDING_DECIDED = ("solver-failure",)  # see tests/test_batch_solver.py


def test_a_missing_jax_is_reported_with_the_install_command_and_never_imported(monkeypatch):
    real = importlib.util.find_spec
    monkeypatch.setattr(importlib.util, "find_spec", lambda name, *a, **k: None if name == "jax" else real(name, *a, **k))

    assert backends.available()["jax"].startswith("missing: pip install")
    assert "solidpy[jax-cuda12]" in backends.available()["jax"]
    with pytest.raises(BackendUnavailable, match=r"needs the missing package\(s\): jax.*solidpy\[jax-cuda12\]"):
        backends.get_backend("jax")


jax = pytest.importorskip("jax")

from solidpy.backends import jax_backend  # noqa: E402  (imports numpy only; jax is imported when a backend is created)


def pack(cases):
    built = [gc.build_objects(c) for c in cases]
    return ProblemBatch.from_objects(
        [b[1] for b in built], [b[2] for b in built], [b[3] for b in built], [b[4] for b in built]
    )


@pytest.fixture(scope="module")
def subset():
    corpus = gc.load_corpus()["cases"]
    reference = gc.load_reference()["records"]
    cases = [
        c for c in corpus
        if PLAIN_TAGS <= set(c["tags"]) and not {"tail_off_omitted", "tail_off_analytical"} & set(c["tags"])
        and reference[c["id"]]["history_points"] <= 150 and c["family"] not in ROUNDING_DECIDED
    ]
    return cases, reference, pack(cases)


def compare_with_the_reference(cases, reference, results):
    bad = []
    for case, got in zip(cases, results):
        stored = reference[case["id"]]
        if got["status"]["termination_reason"] != stored["status"]["termination_reason"]:
            bad.append((case["id"], "reason", got["status"]["termination_reason"]))
            continue
        for key in ("total_impulse_ns", "generated_mass_integral_kg", "nozzle_mass_integral_kg"):
            if not np.isclose(got["metrics"][key], stored["metrics"][key], rtol=tol.INTEGRAL_RTOL, atol=0):
                bad.append((case["id"], key))
        for key in ("peak_chamber_pressure_pa", "peak_thrust_n", "max_nozzle_mass_flow_kg_s"):
            if not np.isclose(got["metrics"][key], stored["metrics"][key], rtol=tol.GRID_SAMPLED_RTOL, atol=0):
                bad.append((case["id"], key))
        key = "max_generated_mass_flow_kg_s"
        if not np.isclose(got["metrics"][key], stored["metrics"][key], rtol=tol.GENERATED_FLOW_PEAK_RTOL, atol=0):
            bad.append((case["id"], key))
        for got_t, stored_t in zip(got["metrics"]["grain_burnout_times_s"], stored["metrics"]["grain_burnout_times_s"]):
            if (got_t is None) != (stored_t is None):  # a lane that timed out leaves some grains unburnt
                bad.append((case["id"], "burnout set"))
            elif stored_t is not None and not np.isclose(got_t, stored_t, rtol=tol.TIME_RTOL, atol=0):
                bad.append((case["id"], "burnout time"))
    return bad


def test_the_pure_helpers_bucket_lanes_and_grains():
    assert [jax_backend._bucket_lanes(n) for n in (1, 64, 65, 1000, 4096, 4097)] == [64, 64, 128, 1024, 4096, 8192]
    assert [jax_backend._bucket_grains(n) for n in (1, 4, 5, 8, 9, 24, 25, 33)] == [4, 4, 8, 8, 16, 24, 32, 33]


def test_a_launch_never_pads_above_its_budget_and_a_full_history_launch_fits_the_memory_budget():
    assert [jax_backend._floor_lanes(n) for n in (1, 3, 63, 64, 100, 8192, 9000)] == [1, 2, 32, 64, 64, 8192, 8192]
    assert jax_backend._bucket_lanes(5) == 64 and jax_backend._bucket_lanes(5, 4) == 8 and jax_backend._bucket_lanes(3, 1) == 4
    backend = backends.get_backend("jax", device="cpu")

    metrics_lanes = backend._lanes_per_launch(8, False, 100000)
    full_lanes = backend._lanes_per_launch(8, True, 5000)

    assert metrics_lanes == jax_backend._floor_lanes(backend.max_lanes)
    assert full_lanes & (full_lanes - 1) == 0 and full_lanes >= 1
    assert jax_backend._bucket_lanes(full_lanes, min(jax_backend.MIN_LANE_BUCKET, full_lanes)) * 2 * 5001 * 17 * 8 <= (
        jax_backend.HISTORY_BUDGET_BYTES)
    # a very long history can allow fewer than 64 lanes: the budget wins over the compile bucket
    assert backend._lanes_per_launch(8, True, 2_000_000) < jax_backend.MIN_LANE_BUCKET


def test_the_backend_is_listed_described_and_selects_devices():
    assert backends.available()["jax"] == "ok"

    description = backends.describe("jax", device="cpu")

    assert description["name"] == "jax" and "cpu" in description["devices"]
    assert description["provenance"]["dtype"] == "float64" and description["provenance"]["device"] == "cpu"
    assert {"jax", "numpy", "scipy"} <= set(description["provenance"]["library_versions"])
    assert description["capabilities"] == backends.describe("cpu-vectorized")["capabilities"]
    with pytest.raises(ValueError, match="unknown device 'tpu:0'"):
        backends.get_backend("jax", device="tpu:0")
    with pytest.raises(BackendUnavailable):
        backends.get_backend("jax", device="cuda:99")


def test_jax_on_the_cpu_device_matches_the_reference_on_the_corpus_subset(subset):
    cases, reference, batch = subset

    results = backends.get_backend("jax", device="cpu").solve_burn(batch).to_results()

    assert len(cases) > 40
    assert compare_with_the_reference(cases, reference, results) == []
    execution = results[0]["provenance"]["execution"]
    assert execution["backend"] == "jax" and execution["device"] == "cpu" and execution["dtype"] == "float64"


def test_jax_agrees_with_the_numpy_backend_lane_by_lane(subset):
    cases, reference, batch = subset
    few = batch.select(np.arange(0, len(cases), 7))

    from_jax = backends.get_backend("jax", device="cpu").solve_burn(few).to_results()
    from_numpy = backends.get_backend("cpu-vectorized").solve_burn(few).to_results()

    for a, b in zip(from_jax, from_numpy):
        assert a["status"]["termination_reason"] == b["status"]["termination_reason"]
        assert a["provenance"]["physics_provider_hash"] == b["provenance"]["physics_provider_hash"]
        for key in ("total_impulse_ns", "generated_mass_integral_kg", "propellant_mass_consumed_kg"):
            assert a["metrics"][key] == pytest.approx(b["metrics"][key], rel=tol.INTEGRAL_RTOL), key
        for key in ("peak_chamber_pressure_pa", "peak_thrust_n"):
            assert a["metrics"][key] == pytest.approx(b["metrics"][key], rel=tol.GRID_SAMPLED_RTOL), key


def test_float64_is_scoped_to_the_solve_and_does_not_leak_into_the_users_jax(subset):
    cases, reference, batch = subset
    before = jax.config.jax_enable_x64

    backends.get_backend("jax", device="cpu").solve_burn(batch.select([0, 1]))

    assert jax.config.jax_enable_x64 == before
    assert jax.numpy.ones(2).dtype == (np.float64 if before else np.float32)


def test_a_launch_with_a_different_lane_count_pads_to_a_bucket_and_returns_only_real_lanes(subset):
    cases, reference, batch = subset
    five = batch.select(np.arange(5))

    results = backends.get_backend("jax", device="cpu").solve_burn(five).to_results()

    assert len(results) == 5
    assert compare_with_the_reference(cases[:5], reference, results) == []


def test_history_full_returns_the_canonical_channels_and_the_step_budget_flags_failures(subset):
    cases, reference, batch = subset
    one = batch.select([0])
    backend = backends.get_backend("jax", device="cpu")

    full = backend.solve_burn(one, SolveOptions(history="full", max_steps=600)).to_results()[0]
    short = backend.solve_burn(one, SolveOptions(max_steps=12)).to_results()[0]
    numpy_full = backends.get_backend("cpu-vectorized").solve_burn(one, SolveOptions(history="full", max_steps=600)).to_results()[0]

    assert list(full["history"]) == list(numpy_full["history"])
    assert full["history"]["time_s"][0] == 0.0 and (np.diff(full["history"]["time_s"]) > 0).all()
    assert short["provenance"]["execution"]["step_overflow"] is True and not short["status"]["completed"]


def test_tiers_give_the_same_results_as_one_uncapped_launch_and_are_reported(subset):
    cases, reference, batch = subset
    few = batch.select(np.arange(0, len(cases), 4))
    backend = backends.get_backend("jax", device="cpu")

    plain = backend.solve_burn(few, SolveOptions(tiers=()))
    tiered = backend.solve_burn(few, SolveOptions(tiers=(150,)))  # about half of these lanes finish in 150

    launches = tiered.execution["tiers"][0]
    assert [tier[0] for tier in launches] == [150, None] and launches[0][1] == len(few)
    assert 0 < launches[1][1] < len(few) and launches[1][1] == launches[0][2]  # the rest were rerun in a smaller batch
    assert plain.execution["tiers"][0] == [(None, len(few), 0)]
    for a, b in zip(plain.to_results(), tiered.to_results()):
        assert a["status"]["termination_reason"] == b["status"]["termination_reason"]
        for key in ("total_impulse_ns", "generated_mass_integral_kg"):
            assert b["metrics"][key] == pytest.approx(a["metrics"][key], rel=tol.INTEGRAL_RTOL), key
        for t_a, t_b in zip(a["metrics"]["grain_burnout_times_s"], b["metrics"]["grain_burnout_times_s"]):
            assert t_b == pytest.approx(t_a, rel=tol.TIME_RTOL)


def test_jax_runs_source_lanes_the_source_only_stage_and_the_blowdown_quirk_like_the_reference():
    corpus = gc.load_corpus()["cases"]
    reference = gc.load_reference()["records"]
    cases = []
    for family in ("igniter-scalar", "igniter-table", "igniter-after", "activation-scalar", "activation-table",
                   "ramp", "combo-table", "quirk"):
        members = [c for c in corpus if c["family"] == family]
        cases.append(min(members, key=lambda c: reference[c["id"]]["history_points"]))

    results = backends.get_backend("jax", device="cpu").solve_burn(pack(cases)).to_results()

    assert compare_with_the_reference(cases, reference, results) == []
    quirk = results[-1]
    assert quirk["status"]["termination_reason"] == "blowdown_timeout"
    for case, got in zip(cases, results):
        stored = reference[case["id"]]["metrics"]["igniter_mass_injected_kg"]
        assert got["metrics"]["igniter_mass_injected_kg"] == pytest.approx(stored, rel=tol.INTEGRAL_RTOL, abs=1e-12)


def test_jax_runs_tabulated_burn_rates_and_pressure_dependent_thermochemistry_like_the_reference():
    corpus = gc.load_corpus()["cases"]
    reference = gc.load_reference()["records"]
    cases = []
    for family in ("ratetable", "ratetable", "thermotable", "thermotable"):
        members = [c for c in corpus if c["family"] == family and c not in cases]
        cases.append(min(members, key=lambda c: reference[c["id"]]["history_points"]))
    longest = max((c for c in corpus if c["family"] == "ratetable"
                   and c["propellant"]["interpolation_list"].endswith("KNSB2.csv")),
                  key=lambda c: -reference[c["id"]]["history_points"])
    cases.append(longest)  # the 1,002-row table

    results = backends.get_backend("jax", device="cpu").solve_burn(pack(cases)).to_results()

    assert compare_with_the_reference(cases, reference, results) == []
    assert [r["status"]["termination_reason"] for r in results[2:4]] == ["unsupported_thermochemistry"] * 2


def test_unsupported_lanes_are_refused_by_name():
    by_id = {c["id"]: c for c in gc.load_corpus()["cases"]}
    batch = pack([by_id["tubular-000"], by_id["igniter-callable-000"]])

    with pytest.raises(UnsupportedLane, match=r"lane\(s\) 1: igniter_callable"):
        backends.get_backend("jax", device="cpu").solve_burn(batch)


@pytest.mark.gpu
def test_on_the_accelerator_the_corpus_subset_matches_the_reference(subset):
    cases, reference, batch = subset
    backend = backends.get_backend("jax", device="cuda:0")

    results = backend.solve_burn(batch).to_results()

    assert backend.device == "cuda:0"
    assert compare_with_the_reference(cases, reference, results) == []
    assert results[0]["provenance"]["execution"]["device"] == "cuda:0"
    assert results[0]["provenance"]["execution"]["device_name"]


@pytest.mark.gpu
def test_the_accelerator_and_the_cpu_device_agree_within_the_parity_limits(subset):
    cases, reference, batch = subset
    few = batch.select(np.arange(0, len(cases), 5))

    on_gpu = backends.get_backend("jax", device="cuda:0").solve_burn(few).to_results()
    on_cpu = backends.get_backend("jax", device="cpu").solve_burn(few).to_results()

    for a, b in zip(on_gpu, on_cpu):
        assert a["status"]["termination_reason"] == b["status"]["termination_reason"]
        assert a["metrics"]["total_impulse_ns"] == pytest.approx(b["metrics"]["total_impulse_ns"], rel=tol.INTEGRAL_RTOL)


# ---- robustness scenarios as lanes ------------------------------------------------------------------------------
@pytest.fixture(scope="module")
def robustness_inputs():
    from solidpy import run_robustness_analysis
    from test_batch_robustness import KWARGS, scenarios
    from test_robustness import make_motor_stack

    design = make_motor_stack()
    options = dict(scenarios=scenarios(), **KWARGS)
    return design, options, run_robustness_analysis(*design, **options)


def test_jax_on_the_cpu_device_gives_the_scalar_robustness_report_within_the_tolerances(robustness_inputs):
    from solidpy import run_robustness_analysis
    from test_batch_robustness import assert_reports_close

    design, options, scalar = robustness_inputs

    report = run_robustness_analysis(*design, backend="jax", device="cpu", **options)

    assert_reports_close(report, scalar)
    execution = report["scenarios"][0]["canonical_result"]["provenance"]["execution"]
    assert execution["backend"] == "jax" and execution["device"] == "cpu"
    assert execution["scenario_inputs"]["burn_rate_factor"] == pytest.approx(0.94)


@pytest.mark.gpu
def test_on_the_accelerator_the_robustness_report_matches_the_scalar_one(robustness_inputs):
    from solidpy import run_robustness_analysis
    from test_batch_robustness import assert_reports_close

    design, options, scalar = robustness_inputs

    report = run_robustness_analysis(*design, backend="jax", device="cuda:0", **options)

    assert_reports_close(report, scalar)
    assert report["nominal"]["canonical_result"]["provenance"]["execution"]["device"] == "cuda:0"
