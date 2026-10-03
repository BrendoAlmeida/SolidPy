import numpy as np
import pytest

import golden_corpus as gc
from solidpy import backends
from solidpy.backends import _tolerances as tol
from solidpy.batch import ProblemBatch

SOURCE_FAMILIES = ("igniter-scalar", "igniter-table", "igniter-after", "activation-scalar", "activation-table",
                   "ramp", "combo-ramp", "combo-table", "quirk")
MAX_POINTS = 400


@pytest.fixture(scope="module")
def solved():
    corpus = gc.load_corpus()["cases"]
    reference = gc.load_reference()["records"]
    cases = [c for c in corpus if c["family"] in SOURCE_FAMILIES and reference[c["id"]]["history_points"] <= MAX_POINTS]
    built = [gc.build_objects(c) for c in cases]
    batch = ProblemBatch.from_objects([b[1] for b in built], [b[2] for b in built], [b[3] for b in built],
                                      [b[4] for b in built])
    results = backends.get_backend("cpu-vectorized").solve_burn(batch).to_results()
    return cases, reference, batch, results


def test_the_subset_covers_every_source_kind_the_quirk_and_a_source_after_burnout(solved):
    cases, reference, batch, results = solved
    families = {c["family"] for c in cases}
    outcomes = {reference[c["id"]]["status"]["termination_reason"] for c in cases}

    assert families == set(SOURCE_FAMILIES) and len(cases) >= 25
    assert "blowdown_timeout" in outcomes and "completed" in outcomes
    after = [c for c in cases if "igniter_after_burnout" in c["tags"]]
    assert after and all(reference[c["id"]]["metrics"]["igniter_mass_injected_kg"] > 0 for c in after)


def test_every_source_lane_ends_the_way_the_reference_ends(solved):
    cases, reference, batch, results = solved

    wrong = {
        c["id"]: (got["status"]["termination_reason"], reference[c["id"]]["status"]["termination_reason"])
        for c, got in zip(cases, results)
        if got["status"]["termination_reason"] != reference[c["id"]]["status"]["termination_reason"]
    }

    assert wrong == {}


def test_metrics_match_the_reference_including_the_igniter_mass(solved):
    cases, reference, batch, results = solved
    bad = []
    for case, got in zip(cases, results):
        stored = reference[case["id"]]
        m, s = got["metrics"], stored["metrics"]
        for key in ("total_impulse_ns", "generated_mass_integral_kg", "nozzle_mass_integral_kg",
                    "igniter_mass_injected_kg", "propellant_mass_consumed_kg"):
            if not np.isclose(m[key], s[key], rtol=tol.INTEGRAL_RTOL, atol=1e-12):
                bad.append((case["id"], key, m[key], s[key]))
        for key in ("peak_chamber_pressure_pa", "peak_thrust_n", "max_generated_mass_flow_kg_s",
                    "max_nozzle_mass_flow_kg_s", "gas_mass_cutoff_kg"):
            if not np.isclose(m[key], s[key], rtol=tol.GRID_SAMPLED_RTOL, atol=0):
                bad.append((case["id"], key, m[key], s[key]))
        for got_t, stored_t in zip(m["grain_burnout_times_s"], s["grain_burnout_times_s"]):
            if (got_t is None) != (stored_t is None) or (
                    stored_t is not None and not np.isclose(got_t, stored_t, rtol=tol.TIME_RTOL, atol=0)):
                bad.append((case["id"], "burnout", got_t, stored_t))
        end = got["metrics"]["nozzle_flow_end_s"]
        if not np.isclose(end, s["nozzle_flow_end_s"], rtol=tol.TIME_RTOL, atol=1e-12):
            bad.append((case["id"], "nozzle end", end, s["nozzle_flow_end_s"]))
        for key in ("completed", "numerical_blowdown_completed", "burnout_completed"):
            if got["status"][key] != stored["status"][key]:
                bad.append((case["id"], key))
    assert bad == []


def test_the_blowdown_truncated_by_a_breakpoint_is_reproduced(solved):
    cases, reference, batch, results = solved
    quirk = [(c, r) for c, r in zip(cases, results) if c["family"] == "quirk"]

    assert len(quirk) >= 3
    for case, got in quirk:
        assert got["status"]["termination_reason"] == "blowdown_timeout"
        assert got["status"]["numerical_blowdown_completed"] is False
        knot = case["simulation"]["burn_area_activation"][1][0]
        assert reference[case["id"]]["end_time_s"] == pytest.approx(knot, rel=1e-9)  # it stops at the breakpoint
        assert got["metrics"]["nozzle_flow_end_s"] == pytest.approx(knot, rel=1e-9)


def test_an_igniter_that_outlasts_the_burn_keeps_feeding_the_chamber_until_it_stops(solved):
    cases, reference, batch, results = solved
    for case, got in zip(cases, results):
        if "igniter_after_burnout" not in case["tags"]:
            continue
        end = case["simulation"]["igniter_burn_time"]
        assert got["metrics"]["nozzle_flow_end_s"] >= end * (1 - 1e-6)
        assert got["metrics"]["igniter_mass_injected_kg"] == pytest.approx(
            case["simulation"]["igniter_mass_flow"] * end, rel=1e-6)
        assert got["status"]["termination_reason"] == "completed"


def test_stored_histories_carry_the_igniter_flow_and_the_activated_areas(solved):
    cases, reference, batch, results = solved
    lanes = [i for i, c in enumerate(cases) if c["family"] in ("igniter-table", "ramp")][:3]
    full = backends.get_backend("cpu-vectorized").solve_burn(
        batch.select(lanes), backends.SolveOptions(history="full", max_steps=900)).to_results()

    for lane, result in zip(lanes, full):
        history = result["history"]
        assert (history["mdot_igniter_kg_s"] >= 0).all()
        igniter = cases[lane]["family"] == "igniter-table"
        assert bool(history["mdot_igniter_kg_s"].any()) == igniter
        assert history["igniter_mass_integral_kg"][-1] == pytest.approx(result["metrics"]["igniter_mass_injected_kg"])
        if cases[lane]["family"] == "ramp":  # the ramp scales the rate and the area from zero at ignition
            assert history["burn_area_m2"][0] == 0.0 and history["regression_rate_m_s"][0].max() == 0.0


def test_sources_are_evaluated_just_below_the_end_of_a_segment(solved):
    """At ``t == end of segment`` the scalar solver uses the left limit, so an igniter that stops there still flows."""
    from solidpy.batch.integrators import solver
    from solidpy.batch.kernels import rhs

    cases, reference, batch, results = solved
    lane = next(i for i, c in enumerate(cases) if c["family"] == "igniter-scalar")
    sub = batch.select([lane])
    P = sub.namespace(np)
    end = float(sub.arrays["igniter_burn_time"][0])
    y = sub.initial_state()
    offset = 2 + sub.g_max

    inside_segment = solver._fun(np, P, P["grain_valid"], np.array([end]))(np.array([end]), y)
    raw = rhs.conservative_rhs(np, y, P["grain_valid"], P, time=np.array([end]))

    assert inside_segment[0, offset + 1] > 0.0 and raw[0, offset + 1] == 0.0
    assert inside_segment[0, offset + 1] == sub.arrays["igniter_value"][0]
