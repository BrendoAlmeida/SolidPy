import threading

import numpy as np
import pytest

import golden_corpus as gc
from solidpy import backends
from solidpy.backends import BackendUnavailable, SolveOptions, UnsupportedLane
from solidpy.backends import _tolerances as tol
from solidpy.batch import ProblemBatch
from solidpy.provenance import REFERENCE_PHYSICS_EQUIVALENCE_CLASS

PLAIN_TAGS = {"scalar_thermo", "power_law", "igniter_none", "activation_none", "tail_off_numerical"}

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
        and reference[c["id"]]["history_points"] <= 150
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


def test_device_result_stream_materializes_the_same_canonical_lanes(subset):
    cases, reference, batch = subset
    one = batch.select([0])
    backend = backends.get_backend("jax", device="cpu")

    expected = backend.solve_burn(one, SolveOptions(history="full", max_steps=600)).to_results()
    streamed = backend.solve_burn_device(one, SolveOptions(history="full", max_steps=600))
    got = streamed.to_results()

    assert len(streamed) == len(got) == 1
    assert list(got[0]["history"]) == list(expected[0]["history"])
    assert got[0]["status"] == expected[0]["status"]
    for key in ("total_impulse_ns", "peak_chamber_pressure_pa", "generated_mass_integral_kg"):
        assert got[0]["metrics"][key] == pytest.approx(expected[0]["metrics"][key])
    assert streamed.execution["chunks"] == 1


def test_device_result_can_materialize_metrics_without_returning_host_history(subset):
    cases, reference, batch = subset
    streamed = backends.get_backend("jax", device="cpu").solve_burn_device(
        batch.select([0]), SolveOptions(history="full", max_steps=600)
    )
    rows = list(streamed.iter_results(history="metrics"))

    assert len(rows) == 1 and rows[0]["history"] is None
    assert rows[0]["metrics"]["total_impulse_ns"] > 0.0


def test_history_policies_retain_then_format_jax_accepted_points(subset):
    cases, reference, batch = subset
    one = batch.select([0])
    backend = backends.get_backend("jax", device="cpu")

    decimated = backend.solve_burn(one, SolveOptions(history="decimated:16", max_steps=600)).to_results()[0]
    uniform = backend.solve_burn(one, SolveOptions(history="uniform:24", max_steps=600)).to_results()[0]

    assert 2 <= len(decimated["history"]["time_s"]) <= 16
    assert len(uniform["history"]["time_s"]) == 24
    assert np.isfinite(uniform["history"]["diagnostics"]["max_abs_pressure_derivative_pa_s"])
    assert uniform["provenance"]["execution"]["history"] == "uniform:24"


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
    execution = results[0]["provenance"]["execution"]
    assert execution["device"] == "cuda:0"
    assert execution["device_name"]
    assert execution["physics_equivalence_class"] == REFERENCE_PHYSICS_EQUIVALENCE_CLASS
    assert execution["parity_certificate"]["kernel_source_hash"] == execution["kernel_source_hash"]
    assert execution["parity_certificate"]["tolerances_version"] == execution["tolerances_version"]


@pytest.mark.gpu
def test_the_accelerator_preserves_source_endpoints_blowdown_events_and_full_history():
    corpus = gc.load_corpus()["cases"]
    reference = gc.load_reference()["records"]
    families = ("igniter-scalar", "activation-table", "ramp", "quirk")
    cases = [
        min((case for case in corpus if case["family"] == family),
            key=lambda case: reference[case["id"]]["history_points"])
        for family in families
    ]
    batch = pack(cases)
    options = SolveOptions(history="full", max_steps=1200)

    cpu = backends.get_backend("cpu-vectorized").solve_burn(batch, options).to_results()
    gpu = backends.get_backend("jax", device="cuda:0").solve_burn(batch, options).to_results()

    assert compare_with_the_reference(cases, reference, cpu) == []
    assert compare_with_the_reference(cases, reference, gpu) == []
    for case, expected, actual in zip(cases, cpu, gpu):
        for key in (
            "completed", "termination_reason", "numerical_blowdown_completed", "scalar_contract_supported",
            "burnout_completed",
        ):
            assert actual["status"][key] == expected["status"][key]
        for key in ("blowdown_cutoff_pressure_pa", "blowdown_reference_peak_pressure_pa"):
            assert actual["status"][key] == pytest.approx(
                expected["status"][key], rel=tol.GRID_SAMPLED_RTOL, abs=1e-12,
            )
        assert list(actual["history"]) == list(expected["history"])
        for result in (expected, actual):
            times = result["history"]["time_s"]
            assert times[0] == 0.0 and np.all(np.diff(times) > 0.0)
            assert times[-1] == pytest.approx(
                reference[case["id"]]["end_time_s"], rel=tol.TIME_RTOL, abs=1e-12,
            )

    igniter_case, igniter_result = cases[0], gpu[0]
    igniter_end = igniter_case["simulation"]["igniter_burn_time"]
    at_igniter_end = np.flatnonzero(np.isclose(igniter_result["history"]["time_s"], igniter_end, rtol=0, atol=1e-12))
    assert at_igniter_end.size == 1
    assert igniter_result["history"]["mdot_igniter_kg_s"][at_igniter_end[0]] == 0.0

    quirk = gpu[-1]
    knot = cases[-1]["simulation"]["burn_area_activation"][1][0]
    assert quirk["status"]["termination_reason"] == "blowdown_timeout"
    assert quirk["history"]["time_s"][-1] == pytest.approx(knot, rel=tol.TIME_RTOL)


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


# ---- thermal ablation -------------------------------------------------------------------------------------------
@pytest.fixture(scope="module")
def thermal_lanes():
    from thermal_cases import CASES, case_lane, pack_lanes, random_lanes, scalar_thermal

    lanes = [case_lane(name) for name in CASES] + random_lanes(10, seed=9)
    return pack_lanes(lanes), [scalar_thermal(lane) for lane in lanes]


def assert_thermal_close(results, scalar):
    for got, want in zip(results, scalar):
        assert set(got) == set(want)
        for key, value in want.items():
            assert got[key] == pytest.approx(value, rel=tol.THERMAL_RTOL, abs=1e-12), key


def test_the_thermal_axes_are_bucketed():
    assert [jax_backend._bucket_nodes(n) for n in (1, 4, 5, 8, 11, 12, 13, 18, 33)] == [4, 4, 8, 8, 12, 12, 16, 20, 36]


def test_a_thermal_launch_stays_inside_the_memory_budget():
    backend = backends.get_backend("jax", device="cpu")

    small = backend._thermal_lanes_per_launch(8, 200)
    huge = backend._thermal_lanes_per_launch(64, 200000)

    # 4 series of 200,000 steps of two doubles take 12.8 MB a lane: 2 GB hold 167, a power of two at most
    assert small == backend.max_lanes and huge == 128
    assert backend._thermal_lanes_per_launch(200, 200) >= 4096  # linear in the wall cells: 200 cells still fit thousands of lanes


def test_jax_on_the_cpu_device_gives_the_scalar_thermal_ablation(thermal_lanes):
    batch, scalar = thermal_lanes
    backend = backends.get_backend("jax", device="cpu")

    result = backend.thermal_ablation(batch)

    assert backend.capabilities().provides("thermal_ablation")
    assert result.execution["failed_lanes"] == [] and result.execution["device"] == "cpu"
    assert_thermal_close(result.results, scalar)
    assert set(backend.last_timings) == {"device_s", "assemble_s"}


def test_jax_gives_a_lane_the_same_result_whatever_the_padding_around_it(thermal_lanes):
    batch, scalar = thermal_lanes
    backend = backends.get_backend("jax", device="cpu")

    alone = backend.thermal_ablation(batch.select([5])).results[0]
    trio = backend.thermal_ablation(batch.select([20, 5, 0])).results[1]

    assert_thermal_close([alone, trio], [scalar[5], scalar[5]])


def test_jax_refuses_thermal_lanes_it_cannot_reproduce():
    from thermal_cases import case_lane, pack_lanes

    batch = pack_lanes([case_lane("steel"), dict(case_lane("steel"), flame_temp_k=float("nan"))])

    with pytest.raises(UnsupportedLane, match="1: non_finite_thermal_input"):
        backends.get_backend("jax", device="cpu").thermal_ablation(batch)


def test_each_thermal_launch_is_padded_to_its_own_shape(thermal_lanes, monkeypatch):
    batch, scalar = thermal_lanes
    order = np.argsort(batch.arrays["n_intervals"], kind="stable")  # what simulate_thermal hands a backend that must chunk
    backend = jax_backend.JaxBackend(device="cpu", max_lanes=8)
    shapes = []
    original = backend._run_thermal
    monkeypatch.setattr(backend, "_run_thermal", lambda padded: shapes.append((len(padded), padded.n_max, padded.t_max)) or original(padded))

    result = backend.thermal_ablation(batch.select(order))

    assert len(shapes) == 4 and len(set(shapes)) > 1  # the shortest launch is not padded to the longest
    assert shapes[0][2] < shapes[-1][2]
    assert_thermal_close(result.results, [scalar[i] for i in order])


def test_jax_chunks_a_thermal_batch_that_does_not_fit_one_launch(thermal_lanes):
    batch, scalar = thermal_lanes
    backend = jax_backend.JaxBackend(device="cpu", max_lanes=8)  # eight lanes per launch: the 25 lanes take four

    result = backend.thermal_ablation(batch)

    assert_thermal_close(result.results, scalar)


def test_the_ensemble_runs_the_advanced_physics_on_jax(thermal_lanes):
    from solidpy import CasingMaterial
    from solidpy.ensemble import run_advanced_physics_ensemble
    from test_advanced_ensemble import scalar as advanced_scalar
    from test_robustness import make_motor_stack

    from solidpy import geometry_from_components, run_detailed_ballistics

    grain, motor, propellant, environment = make_motor_stack()
    geometry = geometry_from_components(grain, motor, propellant, casing_wall_thickness_m=0.004, dry_mass_kg=3.0)
    curve = run_detailed_ballistics(grain, motor, propellant, environment, max_step_size=0.03, max_time_points=1000)
    casing = CasingMaterial(liner_thickness_m=0.002)

    got = run_advanced_physics_ensemble(
        geometry, [curve, curve], casing_material=casing, flame_temp_k=propellant.combustion_temperature,
        r_specific=propellant.products_constant, backend="jax", device="cpu",
    )

    want = advanced_scalar(geometry, curve, propellant, casing)
    for result in got:
        for key, value in want.items():
            assert result[key] == pytest.approx(value, rel=1e-8, abs=1e-9), key


@pytest.mark.gpu
def test_on_the_accelerator_the_thermal_ablation_matches_the_scalar_model(thermal_lanes):
    batch, scalar = thermal_lanes

    result = backends.get_backend("jax", device="cuda:0").thermal_ablation(batch)

    assert result.execution["failed_lanes"] == [] and result.execution["device"] == "cuda:0"
    assert_thermal_close(result.results, scalar)


@pytest.mark.gpu
def test_on_the_accelerator_auto_sends_a_large_thermal_batch_to_the_device(thermal_lanes):
    from solidpy.ensemble import AUTO_MIN_THERMAL_LANES, simulate_thermal

    batch, scalar = thermal_lanes
    lanes = np.arange(AUTO_MIN_THERMAL_LANES) % len(batch)

    result = simulate_thermal(batch.select(lanes), backend="auto")

    assert result.execution["effective_backend"] == "jax" and result.execution["fallback_lanes"] == {}
    assert_thermal_close(result.to_results()[:len(scalar)], scalar)


@pytest.mark.gpu
def test_the_heterogeneous_executor_runs_concurrent_jax_work_on_two_real_gpus(subset):
    from solidpy.ensemble import simulate_burn

    cases, reference, batch = subset
    available = backends.get_backend("jax", device="cuda:0").devices()
    devices = [device for device in available if device.startswith("cuda:")]
    if len(devices) < 2:
        pytest.skip(f"requires two JAX GPU devices; found {len(devices)}")

    ready = threading.Barrier(2)

    class SynchronizedJaxBackend:
        name = "test-synchronized-jax"
        api_version = backends.BACKEND_API_VERSION

        def __init__(self, device=None):
            self.device = device
            self.delegate = backends.get_backend("jax", device=device)
            self._first_call = True

        def capabilities(self):
            return self.delegate.capabilities()

        def devices(self):
            return self.delegate.devices()

        def provenance(self):
            return self.delegate.provenance()

        def solve_burn(self, sub_batch, options):
            if self._first_call:
                self._first_call = False
                ready.wait(timeout=15)
            return self.delegate.solve_burn(sub_batch, options)

    name = SynchronizedJaxBackend.name
    backends.register_backend(name, SynchronizedJaxBackend)
    try:
        selected = batch.select(np.arange(8))
        outcome = simulate_burn(
            selected,
            backend=[(name, devices[0]), (name, devices[1])],
            chunk_size=4,
        )
    finally:
        backends.unregister_backend(name)

    engines = outcome.execution["engines"]
    assert {engine["device"] for engine in engines} == set(devices[:2])
    assert sum(engine["lanes"] for engine in engines) == len(selected)
    assert outcome.execution["fallback_lanes"] == []
    results = outcome.to_results()
    assert compare_with_the_reference(cases[:8], reference, results) == []
    assert {result["provenance"]["execution"]["device"] for result in results} == set(devices[:2])
