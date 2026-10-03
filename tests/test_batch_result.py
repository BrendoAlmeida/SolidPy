import hashlib
import json
import re
from pathlib import Path

import numpy as np
import pytest

import golden_corpus as gc
from solidpy import backends, evaluate_numerical_acceptance
from solidpy.backends import SolveOptions, UnsupportedLane
from solidpy.backends import _tolerances as tol
from solidpy.batch import ProblemBatch, assemble
from solidpy.batch.integrators import solver

PLAIN_TAGS = {"scalar_thermo", "power_law", "igniter_none", "activation_none", "tail_off_numerical"}
LIVE_FAMILIES = ("tubular", "star", "ends-star", "mixed", "erosive", "efficiency", "lowkn", "replicated")
ROUNDING_DECIDED = ("solver-failure",)  # see tests/test_batch_solver.py


def pack(cases):
    built = [gc.build_objects(c) for c in cases]
    return ProblemBatch.from_objects(
        [b[1] for b in built], [b[2] for b in built], [b[3] for b in built], [b[4] for b in built]
    )


def solve(batch, **options):
    return backends.get_backend("cpu-vectorized").solve_burn(batch, SolveOptions(**options)).to_results()


@pytest.fixture(scope="module")
def corpus_cases():
    corpus = gc.load_corpus()["cases"]
    return {c["id"]: c for c in corpus}, gc.load_reference()["records"]


@pytest.fixture(scope="module")
def live(corpus_cases):
    """The cheapest design of several families, solved by the scalar solver and by the batched backend."""
    by_id, reference = corpus_cases
    cases = []
    for family in LIVE_FAMILIES:
        members = [c for c in by_id.values() if c["family"] == family and PLAIN_TAGS <= set(c["tags"])]
        cases.append(min(members, key=lambda c: reference[c["id"]]["history_points"]))
    scalar = [gc.simulate(c).result for c in cases]
    batched = solve(pack(cases), history="full", max_steps=900)
    return cases, scalar, batched


def test_every_backend_returns_the_canonical_top_level_structure(backend, corpus_cases):
    by_id, _ = corpus_cases
    batch = pack([by_id["tubular-000"]])

    result = backends.get_backend(backend).solve_burn(batch, SolveOptions(history="full", max_steps=900)).to_results()[0]

    assert set(result) == {"history", "metrics", "status", "efficiencies", "provenance"}
    assert result["history"] is not None
    assert result["status"]["completed"] is True


def test_the_mappings_have_the_scalar_keys_in_the_scalar_order(live):
    _, scalar, batched = live
    for expected, got in zip(scalar, batched):
        assert list(got) == list(expected)
        assert list(got["metrics"]) == list(expected["metrics"])
        assert list(got["status"]) == list(expected["status"])
        assert got["efficiencies"] == expected["efficiencies"]
        assert list(got["history"]) == list(expected["history"])
        assert [k for k in got["provenance"] if k != "execution"] == list(expected["provenance"])


def test_the_physics_provider_hash_is_the_scalar_hash_so_acceptance_can_compare_backends(live):
    _, scalar, batched = live
    for expected, got in zip(scalar, batched):
        assert got["provenance"]["physics_provider_hash"] == expected["provenance"]["physics_provider_hash"]
        assert json.dumps(got["provenance"]["resolved_inputs"], sort_keys=True, default=float) == json.dumps(
            expected["provenance"]["resolved_inputs"], sort_keys=True, default=float
        )
        for key in ("eta_c_applied", "eta_cf_applied", "discharge_coefficient_applied", "efficiency_semantics",
                    "cea_used", "thermochemistry_source", "solver_settings", "gas_temperature_model",
                    "activation_model", "flow_interval_method", "integration_method", "solidpy_git_sha"):
            assert got["provenance"][key] == expected["provenance"][key], key


def test_the_execution_block_records_how_the_result_was_produced(live):
    _, _, batched = live
    execution = batched[0]["provenance"]["execution"]

    assert execution["backend"] == "cpu-vectorized" and execution["device"] == "cpu"
    assert execution["dtype"] == "float64" and execution["history"] == "full"
    assert execution["integrator"]["name"] == "dop853_batched" and execution["fallback"] is None
    assert execution["tolerances_version"] == tol.TOLERANCES_VERSION
    assert re.fullmatch(r"[0-9a-f]{64}", execution["kernel_source_hash"]) and execution["step_overflow"] is False
    assert {"python", "numpy", "scipy"} <= set(execution["library_versions"])


def test_the_kernel_source_hash_covers_the_kernel_and_integrator_files():
    root = Path(assemble.__file__).parent
    digest = hashlib.sha256()
    for path in sorted([*(root / "kernels").glob("*.py"), *(root / "integrators").glob("*.py")]):
        digest.update(path.name.encode())
        digest.update(path.read_bytes().replace(b"\r\n", b"\n"))

    assert assemble.kernel_source_hash() == digest.hexdigest()
    assert any(p.name == "rhs.py" for p in (root / "kernels").glob("*.py"))


def test_status_and_metrics_agree_with_the_live_scalar_results(live):
    _, scalar, batched = live
    for expected, got in zip(scalar, batched):
        for key, value in expected["status"].items():
            if key.startswith("blowdown_"):
                assert got["status"][key] == pytest.approx(value, rel=tol.GRID_SAMPLED_RTOL), key
            else:
                assert got["status"][key] == value, key
        e, g = expected["metrics"], got["metrics"]
        for key in ("total_impulse_ns", "generated_mass_integral_kg", "nozzle_mass_integral_kg",
                    "propellant_mass_consumed_kg", "propellant_mass_initial_kg", "propellant_mass_remaining_kg",
                    "gas_mass_initial_kg", "pressure_throat_integral_ns"):
            assert g[key] == pytest.approx(e[key], rel=tol.INTEGRAL_RTOL), key
        # the gas mass at the cutoff is read at the event, whose pressure level is 1 % of the sampled peak
        for key in ("peak_chamber_pressure_pa", "peak_thrust_n", "max_nozzle_mass_flow_kg_s", "gas_mass_cutoff_kg"):
            assert g[key] == pytest.approx(e[key], rel=tol.GRID_SAMPLED_RTOL), key
        assert g["max_generated_mass_flow_kg_s"] == pytest.approx(e["max_generated_mass_flow_kg_s"],
                                                                  rel=tol.GENERATED_FLOW_PEAK_RTOL)
        for key in ("propellant_burn_start_s", "propellant_burn_end_s", "nozzle_flow_start_s", "nozzle_flow_end_s"):
            assert g[key] == pytest.approx(e[key], rel=tol.TIME_RTOL, abs=1e-12), key
        assert [t is None for t in g["grain_burnout_times_s"]] == [t is None for t in e["grain_burnout_times_s"]]
        np.testing.assert_allclose(
            [t for t in g["grain_burnout_times_s"] if t is not None],
            [t for t in e["grain_burnout_times_s"] if t is not None], rtol=tol.TIME_RTOL,
        )
        assert g["mass_flow_balance_error_pct"] < 1e-3 and e["mass_flow_balance_error_pct"] < 1e-3


def test_numerical_acceptance_passes_for_the_batched_result_against_the_reference(live):
    _, scalar, batched = live
    for expected, got in zip(scalar, batched):
        report = evaluate_numerical_acceptance(got, expected)

        assert report["status"] == "passed", report["incomplete_reasons"] or report["convergence"]
        assert report["incomplete_reasons"] == []
        assert report["provenance"]["coarse"]["physics_provider_hash"] == report["provenance"]["refined"][
            "physics_provider_hash"]


def test_history_channels_are_the_scalar_functions_of_the_stored_points(live):
    cases, scalar, batched = live
    for case, got in zip(cases, batched):
        simulation = gc.simulate(case)
        history = got["history"]
        count = len(history["time_s"])
        n = len(simulation.motor.grains)
        offset = 2 + n
        for k in np.unique(np.linspace(0, count - 1, 12).astype(int)):
            t = history["time_s"][k]
            state = np.concatenate([[history["gas_mass_kg"][k], history["gas_mass_kg"][k] * history["gas_temperature_k"][k]],
                                    history["regression_m"][k],
                                    [history["generated_mass_integral_kg"][k], history["igniter_mass_integral_kg"][k],
                                     history["nozzle_mass_integral_kg"][k],
                                     history["impulse_integral_ns"][k] / simulation.eta_Cf,
                                     history["pressure_throat_integral_ns"][k]]])
            q = simulation._state_quantities_uncached(t, state, None)
            where = f"{case['id']} point {k}"
            for channel, expected, scale in (
                ("chamber_pressure_pa", q["pressure"], 0.0), ("mdot_generated_kg_s", q["generated"], 0.0),
                ("mdot_nozzle_kg_s", q["nozzle"], 0.0), ("thrust_n", q["components"]["total_n"], 0.0),
                ("momentum_n", q["components"]["momentum_n"], 0.0),
                ("exit_pressure_pa", simulation.evaluate_exit_pressure(q["pressure"]), 0.0),
                ("exit_velocity_m_s", simulation.evaluate_exit_velocity(q["pressure"], chamber_temperature=q["temperature"]), 0.0),
            ):
                np.testing.assert_allclose(history[channel][k], expected, rtol=1e-9, atol=1e-9 * max(abs(expected), 1.0) * 1e-3,
                                           err_msg=f"{channel} {where}")
            np.testing.assert_allclose(history["burn_area_grains_m2"][k], q["areas"], rtol=1e-11, atol=1e-18, err_msg=where)
        assert history["regression_m"].shape == (count, n) and history["burn_area_grains_m2"].shape == (count, n)
        assert history["impulse_integral_ns"][-1] == got["metrics"]["total_impulse_ns"]
        assert history["generated_mass_integral_kg"][-1] == got["metrics"]["generated_mass_integral_kg"]
        assert (np.diff(history["time_s"]) > 0).all() and history["mdot_igniter_kg_s"].shape == (count,)


def test_the_metrics_policy_returns_no_history_and_the_same_metrics(corpus_cases):
    by_id, _ = corpus_cases
    batch = pack([by_id["star-001"]])

    light = solve(batch, history="metrics")[0]
    full = solve(batch, history="full", max_steps=900)[0]

    assert light["history"] is None and light["provenance"]["execution"]["history"] == "metrics"
    assert light["metrics"] == full["metrics"] and light["status"] == full["status"]


def test_the_corpus_subset_matches_the_stored_reference(corpus_cases):
    by_id, reference = corpus_cases
    cases = [c for c in by_id.values() if PLAIN_TAGS <= set(c["tags"]) and "tail_off_omitted" not in c["tags"]
             and not {"tail_off_analytical"} & set(c["tags"]) and reference[c["id"]]["history_points"] <= 300]
    results = solve(pack(cases))

    bad = []
    for case, got in zip(cases, results):
        if case["family"] in ROUNDING_DECIDED:
            continue
        stored = reference[case["id"]]
        if got["status"]["termination_reason"] != stored["status"]["termination_reason"]:
            bad.append((case["id"], "reason"))
            continue
        for key in ("completed", "numerical_blowdown_completed", "burnout_completed"):
            if got["status"][key] != stored["status"][key]:
                bad.append((case["id"], key))
        for key in ("total_impulse_ns", "generated_mass_integral_kg", "nozzle_mass_integral_kg"):
            if not np.isclose(got["metrics"][key], stored["metrics"][key], rtol=tol.INTEGRAL_RTOL, atol=0):
                bad.append((case["id"], key))
        for key in ("peak_chamber_pressure_pa", "peak_thrust_n", "max_nozzle_mass_flow_kg_s"):
            if not np.isclose(got["metrics"][key], stored["metrics"][key], rtol=tol.GRID_SAMPLED_RTOL, atol=0):
                bad.append((case["id"], key))
        key = "max_generated_mass_flow_kg_s"
        if not np.isclose(got["metrics"][key], stored["metrics"][key], rtol=tol.GENERATED_FLOW_PEAK_RTOL, atol=0):
            bad.append((case["id"], key))
        if not stored["status"]["completed"]:
            continue
        if got["metrics"]["mass_flow_balance_error_pct"] > max(2 * stored["metrics"]["mass_flow_balance_error_pct"], 1e-4):
            bad.append((case["id"], "mass balance"))
    assert bad == []


def test_the_tail_off_can_be_omitted_per_lane(corpus_cases):
    by_id, reference = corpus_cases
    cases = [by_id[i] for i in sorted(by_id) if i.startswith("notail-")] + [by_id["tubular-000"]]
    results = solve(pack(cases))

    for case, got in zip(cases[:-1], results[:-1]):
        stored = reference[case["id"]]
        assert got["status"]["termination_reason"] == "tail_off_omitted" == stored["status"]["termination_reason"]
        assert got["status"]["completed"] is False
        assert got["status"]["blowdown_cutoff_pressure_pa"] is None
        assert got["metrics"]["total_impulse_ns"] == pytest.approx(stored["metrics"]["total_impulse_ns"], rel=tol.INTEGRAL_RTOL)
    assert results[-1]["status"]["termination_reason"] == "completed"  # a lane that does run the blowdown


def test_the_cutoff_is_reported_when_the_blowdown_fails_but_not_when_the_source_only_stage_does(corpus_cases):
    by_id, _ = corpus_cases
    batch = pack([by_id["tubular-000"]])
    out = solver.solve_burn_and_blowdown(solver.numpy_driver(), batch.namespace(np), batch.initial_state())
    blowdown_failed = dict(out, tail_ok=np.array([False]))
    sources_failed = dict(out, tail_ok=np.array([False]), source_ok=np.array([False]))

    ok, late, early = (assemble.assemble(batch, o)[0]["status"] for o in (out, blowdown_failed, sources_failed))

    assert ok["blowdown_cutoff_pressure_pa"] == pytest.approx(float(out["cutoff"][0])) and ok["termination_reason"] == "completed"
    assert late["termination_reason"] == early["termination_reason"] == "solver_failure"
    assert late["blowdown_cutoff_pressure_pa"] == ok["blowdown_cutoff_pressure_pa"]  # as BurnSimulation sets it
    assert late["blowdown_reference_peak_pressure_pa"] == ok["blowdown_reference_peak_pressure_pa"]
    assert early["blowdown_cutoff_pressure_pa"] is None and early["blowdown_reference_peak_pressure_pa"] is None


def test_a_lane_that_runs_out_of_points_is_a_solver_failure_flagged_in_the_provenance(corpus_cases):
    by_id, _ = corpus_cases
    result = solve(pack([by_id["tubular-000"], by_id["tubular-001"]]), max_steps=30)

    assert [r["provenance"]["execution"]["step_overflow"] for r in result] == [True, True]
    assert {r["status"]["termination_reason"] for r in result} <= {"solver_failure"}
    assert not any(r["status"]["completed"] for r in result)


def test_lanes_the_kernels_cannot_run_are_refused_by_name(corpus_cases):
    by_id, _ = corpus_cases
    batch = pack([by_id["tubular-000"], by_id["igniter-callable-000"], by_id["activation-callable-000"]])

    with pytest.raises(UnsupportedLane, match=r"lane\(s\) 1: .*igniter_callable; 2: .*activation_callable"):
        backends.get_backend("cpu-vectorized").solve_burn(batch)
    description = backends.describe("cpu-vectorized")
    assert description["devices"] == ["cpu"] and set(description["history_policies"]) == {"metrics", "full"}
    assert "igniter_callable" not in description["capabilities"]  # Python callables stay on the reference
    assert description["capabilities"]["igniter_table"] == "supported"


def test_solve_options_validate_max_steps_and_the_backend_the_device():
    for bad in (0, 1, -5, 2.5, True):
        with pytest.raises(ValueError, match="max_steps must be an integer of at least 2"):
            SolveOptions(max_steps=bad)
    with pytest.raises(ValueError, match="only runs on 'cpu'"):
        backends.get_backend("cpu-vectorized", device="cuda:0")


def test_the_unsupported_lane_helpers_name_lanes_and_features(corpus_cases):
    from solidpy.backends._protocol import refused_lanes, unsupported_lane_error

    by_id, _ = corpus_cases
    batch = pack([by_id["tubular-000"], by_id["igniter-callable-000"], by_id["activation-callable-000"]])

    refused = refused_lanes(batch, backends.get_backend("cpu-vectorized").capabilities())

    assert refused == {1: ["igniter_callable"], 2: ["activation_callable"]}
    assert "backend 'x' cannot run lane(s) 1: igniter_callable; 2: activation_callable" in str(
        unsupported_lane_error("x", refused))
    assert refused_lanes(batch, backends.get_backend("cpu-reference").capabilities()) == {}


def test_the_git_sha_is_read_again_only_when_head_moves(monkeypatch):
    calls = []
    monkeypatch.setattr(assemble, "_read_git_sha", lambda: calls.append(1) or "sha")
    monkeypatch.setattr(assemble, "_GIT_TTL_S", 1e9)
    monkeypatch.setitem(assemble._git_cache, "stamp", None)
    stamps = iter([("a",), ("a",), ("a",), ("b",)])  # the fourth call sees a new commit
    monkeypatch.setattr(assemble, "_git_stamp", lambda: next(stamps))

    shas = [assemble._git_sha() for _ in range(4)]

    assert shas == ["sha"] * 4 and len(calls) == 2


def test_the_source_hashes_are_not_recomputed_for_every_batch(corpus_cases):
    by_id, _ = corpus_cases
    batch = pack([by_id["tubular-000"]])
    hits = assemble._kernel_hash.cache_info().hits, assemble._source_bytes.cache_info().hits

    solve(batch)
    solve(batch)
    solve(batch)

    assert assemble._kernel_hash.cache_info().hits >= hits[0] + 2
    assert assemble._source_bytes.cache_info().hits >= hits[1] + 2
