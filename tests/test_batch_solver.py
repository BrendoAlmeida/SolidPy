import numpy as np
import pytest

import golden_corpus as gc
from solidpy.backends import _tolerances as tol
from solidpy.batch import ProblemBatch
from solidpy.batch.integrators import solver

PLAIN_TAGS = {"scalar_thermo", "power_law", "igniter_none", "activation_none", "tail_off_numerical"}
MAX_POINTS = 300  # reference accepted steps; keeps the lockstep loop short enough for a unit test


def reason(out, lane):
    """The scalar ``termination_reason`` the solver outputs imply (no tail-off omission here)."""
    if not out["burn_ok"][lane]:
        return "solver_failure"
    if not out["burned_out"][lane]:
        return "burn_timeout"
    if not out["tail_ok"][lane]:
        return "solver_failure"
    return "completed" if out["reached_cutoff"][lane] else "blowdown_timeout"


@pytest.fixture(scope="module")
def solved():
    corpus = gc.load_corpus()["cases"]
    reference = gc.load_reference()["records"]
    cases = [
        c for c in corpus
        if PLAIN_TAGS <= set(c["tags"]) and not {"tail_off_omitted", "tail_off_analytical"} & set(c["tags"])
        and reference[c["id"]]["history_points"] <= MAX_POINTS
    ]
    built = [gc.build_objects(c) for c in cases]
    batch = ProblemBatch.from_objects(
        [b[1] for b in built], [b[2] for b in built], [b[3] for b in built], [b[4] for b in built]
    )
    out = solver.solve_burn_and_blowdown(
        solver.numpy_driver(), batch.namespace(np), batch.initial_state(), solver.SolveConfig(keep_history=True, max_steps=1200)
    )
    return cases, reference, batch, out


def rel(actual, desired, what, rtol, floor=0.0):
    np.testing.assert_allclose(actual, desired, rtol=rtol, atol=floor, err_msg=what)


def test_the_corpus_subset_covers_the_stage_outcomes(solved):
    cases, reference, batch, out = solved
    outcomes = {reference[c["id"]]["status"]["termination_reason"] for c in cases}

    assert len(cases) > 100
    assert outcomes == {"completed", "burn_timeout", "blowdown_timeout", "solver_failure"}
    assert (batch.n_grains > 4).any() and (batch.n_grains == 1).any()


#: Designs whose reference outcome is decided by rounding, not by physics. An absurd erosive coefficient makes
#: identical grains burn out together; scipy then snaps only one of them (the root of the second event lands a
#: few ulps away), the other event fires at ``t_old`` and ``BurnSimulation`` declares a solver failure. The
#: batched solver snaps every grain within ``64 eps`` of its burnout depth at the first event and completes.
ROUNDING_DECIDED = ("solver-failure",)


def test_every_lane_ends_the_way_the_reference_ends(solved):
    cases, reference, batch, out = solved

    wrong = {
        c["id"]: (reason(out, i), reference[c["id"]]["status"]["termination_reason"])
        for i, c in enumerate(cases)
        if reason(out, i) != reference[c["id"]]["status"]["termination_reason"]
        and c["family"] not in ROUNDING_DECIDED
    }

    assert wrong == {}
    assert not out["overflow"].any()


def test_rounding_decided_failures_end_with_a_defined_outcome_and_a_finite_state(solved):
    cases, reference, batch, out = solved
    lanes = [i for i, c in enumerate(cases) if c["family"] in ROUNDING_DECIDED]

    assert len(lanes) == 3
    for i in lanes:
        assert reference[cases[i]["id"]]["status"]["termination_reason"] == "solver_failure"
        assert reason(out, i) in ("completed", "solver_failure")
        assert np.isfinite(out["y"][i]).all() and not out["overflow"][i]


def test_final_time_and_grain_burnout_times_match_the_reference(solved):
    cases, reference, batch, out = solved
    bad = []
    for i, case in enumerate(cases):
        stored = reference[case["id"]]
        if not stored["status"]["completed"]:
            continue
        if not np.isclose(out["t"][i], stored["end_time_s"], rtol=tol.TIME_RTOL, atol=0):
            bad.append((case["id"], "end", out["t"][i], stored["end_time_s"]))
        for slot, expected in enumerate(stored["metrics"]["grain_burnout_times_s"]):
            got = out["burn_t"][i, slot]
            if expected is None or not np.isclose(got, expected, rtol=tol.TIME_RTOL, atol=0):
                bad.append((case["id"], f"burnout {slot}", got, expected))
    assert bad == []
    padded = ~batch.arrays["grain_valid"]
    assert np.isnan(out["burn_t"][padded]).all()  # padded grains never burn out


def test_integrals_match_the_reference(solved):
    cases, reference, batch, out = solved
    a = batch.arrays
    offset = 2 + batch.g_max
    for i, case in enumerate(cases):
        stored = reference[case["id"]]
        if not stored["status"]["completed"]:
            continue
        metrics = stored["metrics"]
        rel(out["y"][i, offset + 3] * a["eta_cf"][i], metrics["total_impulse_ns"], f"impulse {case['id']}", tol.INTEGRAL_RTOL)
        rel(out["y"][i, offset], metrics["generated_mass_integral_kg"], f"generated {case['id']}", tol.INTEGRAL_RTOL)
        rel(out["y"][i, offset + 2], metrics["nozzle_mass_integral_kg"], f"nozzle {case['id']}", tol.INTEGRAL_RTOL)


def test_peaks_sampled_on_the_step_grid_match_the_reference(solved):
    cases, reference, batch, out = solved
    for i, case in enumerate(cases):
        stored = reference[case["id"]]
        if not stored["status"]["completed"]:
            continue
        metrics = stored["metrics"]
        for key, name in (("pmax", "peak_chamber_pressure_pa"), ("tmax", "peak_thrust_n"),
                          ("gmax", "max_generated_mass_flow_kg_s"), ("nmax", "max_nozzle_mass_flow_kg_s")):
            rel(out[key][i], metrics[name], f"{name} {case['id']}", tol.GRID_SAMPLED_RTOL)


def test_the_blowdown_cutoff_is_built_from_the_burn_stage_peak(solved):
    cases, reference, batch, out = solved
    a = batch.arrays
    for i, case in enumerate(cases):
        stored = reference[case["id"]]["status"]
        if stored["blowdown_cutoff_pressure_pa"] is None:
            continue
        pa = a["ambient_pressure"][i]
        rel(out["peak_burn"][i], stored["blowdown_reference_peak_pressure_pa"], f"peak {case['id']}", tol.GRID_SAMPLED_RTOL)
        assert out["cutoff"][i] == pa + 0.01 * max(out["peak_burn"][i] - pa, 0.0)
        rel(out["cutoff"][i], stored["blowdown_cutoff_pressure_pa"], f"cutoff {case['id']}", tol.GRID_SAMPLED_RTOL)


def test_flow_interval_trackers_match_the_reference_brackets(solved):
    cases, reference, batch, out = solved
    bad = []
    for i, case in enumerate(cases):
        stored = reference[case["id"]]
        if not stored["status"]["completed"]:
            continue
        m = stored["metrics"]
        generated_end = max(m["grain_burnout_times_s"]) if all(
            t is not None for t in m["grain_burnout_times_s"]) else out["gen_end"][i]
        for got, expected, name in (
            (out["gen_start"][i], m["propellant_burn_start_s"], "burn start"),
            (generated_end, m["propellant_burn_end_s"], "burn end"),
            (out["noz_start"][i], m["nozzle_flow_start_s"], "nozzle start"),
            (out["noz_end"][i], m["nozzle_flow_end_s"], "nozzle end"),
        ):
            if not np.isclose(got, expected, rtol=1e-5, atol=1e-12):
                bad.append((case["id"], name, got, expected))
    assert bad == []


def test_history_buffers_hold_the_accepted_points_in_order(solved):
    cases, reference, batch, out = solved
    y0 = batch.initial_state()
    for i, case in enumerate(cases):
        count = int(out["n_points"][i])
        times = out["ht"][i, :count]
        assert times[0] == 0.0 and np.array_equal(out["hy"][i, 0], y0[i]), case["id"]
        assert (np.diff(times) > 0).all(), case["id"]
        assert times[-1] == out["t"][i] and np.array_equal(out["hy"][i, count - 1], out["y"][i]), case["id"]
        assert not (out["ht"][i, count:-1]).any(), "unused slots stay empty"
        if reference[case["id"]]["status"]["completed"]:
            stored = reference[case["id"]]["history_points"]
            assert abs(count - stored) <= 0.2 * stored + 5, (case["id"], count, stored)


def test_the_burnout_state_is_stored_snapped_to_the_burnout_depth(solved):
    cases, reference, batch, out = solved
    a = batch.arrays
    checked = 0
    for i in range(len(cases)):
        for slot in range(batch.n_grains[i]):
            burnout_time = out["burn_t"][i, slot]
            if np.isnan(burnout_time):
                continue
            count = int(out["n_points"][i])
            stored = np.flatnonzero(out["ht"][i, :count] == burnout_time)
            assert stored.size == 1, (cases[i]["id"], slot)
            assert out["hy"][i, stored[0], 2 + slot] == a["burnout_depth"][i, slot]
            checked += 1
    assert checked > 100


def test_a_lane_that_runs_out_of_stored_points_fails_without_stopping_the_others(solved):
    cases, reference, batch, out = solved
    small = solver.solve_burn_and_blowdown(
        solver.numpy_driver(), batch.namespace(np), batch.initial_state(), solver.SolveConfig(max_steps=70)
    )

    points = np.array([reference[c["id"]]["history_points"] for c in cases])
    needs_more, fits = points >= 120, points <= 50  # the step counts of the two integrators differ by a few %
    assert needs_more.any() and fits.any()
    assert small["overflow"][needs_more].all()
    assert not small["tail_ok"][small["overflow"]].any()  # an overflowed lane is failed, whatever the stage
    assert not small["overflow"][fits].any()
    for i in np.flatnonzero(fits):
        assert small["t"][i] == out["t"][i]  # lanes that fit are unaffected by the failed ones


def test_lanes_are_independent_of_the_other_lanes_in_the_batch(solved):
    cases, reference, batch, out = solved
    picks = [i for i, c in enumerate(cases) if c["family"] in ("tubular", "star", "erosive", "guard-axial")][:6]
    sub = batch.select(picks)

    alone = solver.solve_burn_and_blowdown(
        solver.numpy_driver(), sub.namespace(np), sub.initial_state(), solver.SolveConfig()
    )

    for row, lane in enumerate(picks):
        rel(alone["t"][row], out["t"][lane], f"final time {cases[lane]['id']}", 1e-9)
        rel(alone["pmax"][row], out["pmax"][lane], f"peak {cases[lane]['id']}", 1e-9)


def test_the_flow_interval_tracker_equals_the_scalar_bracket_on_arbitrary_sequences():
    """``_flow_interval`` takes the neighbours of the first and last positive flow on the step grid."""
    from solidpy import BurnSimulation

    rng = np.random.default_rng(5)
    lanes, points = 400, 14
    times = np.cumsum(rng.uniform(0.01, 0.5, size=(lanes, points)), axis=1)
    times[:, 0] = 0.0
    flows = rng.uniform(0.1, 2.0, size=(lanes, points)) * (rng.random((lanes, points)) < 0.5)
    flows[:50, :3] = 0.0     # no flow at the start
    flows[50:100, -4:] = 0.0  # no flow at the end
    flows[100:130] = 0.0      # no flow at all
    flows[130:160] = 1.0      # flow everywhere

    state = {"prev_time": times[:, 0].copy()}
    for name in ("gen",):
        state.update({name + "_start": np.zeros(lanes), name + "_end": np.zeros(lanes),
                      name + "_seen": np.zeros(lanes, bool), name + "_want": np.zeros(lanes, bool)})
    everything = np.ones(lanes, bool)
    solver._track_flow(np, state, "gen", flows[:, 0], times[:, 0], everything)
    state["prev_time"] = times[:, 0].copy()
    for k in range(1, points):
        solver._track_flow(np, state, "gen", flows[:, k], times[:, k], everything)
        state["prev_time"] = times[:, k].copy()

    for lane in range(lanes):
        start, end = BurnSimulation._flow_interval(times[lane], flows[lane])
        assert (state["gen_start"][lane], state["gen_end"][lane]) == (start, end), lane
