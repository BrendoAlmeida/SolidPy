import numpy as np
import pytest
from scipy.integrate import DOP853, solve_ivp
from scipy.integrate._ivp import rk as scipy_rk

import golden_corpus as gc
from solidpy.batch import ProblemBatch
from solidpy.batch.integrators import dop853
from solidpy.batch.kernels import rhs

# ---------------------------------------------------------------------------------------------------------
# A small nonlinear system with per-lane parameters, solved by scipy one lane at a time and by the batched
# integrator all at once.

W = np.array([1.0, 2.0, 0.7, 3.0, 1.5, 2.5])
DAMPING = np.array([0.1, 0.05, 0.3, 0.02, 0.2, 0.0])
FORCING = np.array([0.5, 1.0, 0.0, 0.3, 0.8, 0.2])
Y0 = np.stack([np.linspace(0.3, 1.2, 6), np.linspace(-0.5, 0.4, 6), np.zeros(6)], axis=1)
RTOL = np.array([1e-8, 1e-6, 1e-9, 1e-7, 1e-8, 1e-10])
ATOL = np.array([1e-10, 1e-8, 1e-12, 1e-9, 1e-10, 1e-12])
MAX_STEP = np.array([0.5, 0.2, 1.0, 0.1, 0.5, 0.3])


def batched(t, y):
    return np.stack(
        [y[:, 1], -(W**2) * np.sin(y[:, 0]) - DAMPING * y[:, 1] + FORCING * np.cos(t), y[:, 0] * y[:, 1]], axis=1
    )


def scalar(lane):
    def fun(t, y):
        return [y[1], -(W[lane] ** 2) * np.sin(y[0]) - DAMPING[lane] * y[1] + FORCING[lane] * np.cos(t), y[0] * y[1]]

    return fun


def drive(fun, y0, t_bound, max_step, rtol, atol, steps=None, max_attempts=100000, size=None):
    """Run the batched integrator in lockstep, recording every accepted step of every lane.

    ``steps`` stops a lane after that many accepted steps; otherwise lanes run to ``t_bound``.
    """
    count = len(y0)
    t = np.zeros(count)
    y = y0.copy()
    f = fun(t, y)
    bound = np.full(count, float(t_bound))
    h_abs = dop853.initial_step(np, fun, t, y, f, bound, max_step, rtol, atol, size)
    rejected = np.zeros(count, dtype=bool)
    done = np.zeros(count, dtype=bool)
    record = [[(0.0, y0[i].copy())] for i in range(count)]
    for _ in range(max_attempts):
        if done.all():
            break
        out = dop853.attempt(np, fun, t, y, f, h_abs, rejected, bound, max_step, rtol, atol, ~done, size)
        accept = out["accept"]
        t = np.where(accept, out["t_new"], t)
        y = np.where(accept[:, None], out["y_new"], y)
        f = np.where(accept[:, None], out["f_new"], f)
        h_abs = np.where(out["accept"] | out["reject"], out["h_next"], h_abs)
        rejected = out["rejected_next"]
        for i in np.flatnonzero(accept):
            record[i].append((t[i], y[i].copy()))
        done |= accept & (t >= bound)
        if steps is not None:
            done |= np.array([len(r) - 1 >= steps for r in record])
        assert not out["too_small"].any()
    return record


def test_coefficients_and_constants_match_the_installed_scipy():
    assert dop853.N_STAGES == 12
    assert (dop853.SAFETY, dop853.MIN_FACTOR, dop853.MAX_FACTOR) == (
        scipy_rk.SAFETY, scipy_rk.MIN_FACTOR, scipy_rk.MAX_FACTOR
    )
    assert dop853.ERROR_ESTIMATOR_ORDER == DOP853.error_estimator_order
    assert dop853.ERROR_EXPONENT == -1 / (DOP853.error_estimator_order + 1)
    shapes = {name: getattr(dop853, name).shape for name in ("_A", "_B", "_C", "_E3", "_E5", "_D", "_A_EXTRA", "_C_EXTRA")}
    assert shapes == {"_A": (12, 12), "_B": (12,), "_C": (12,), "_E3": (13,), "_E5": (13,), "_D": (4, 16),
                      "_A_EXTRA": (3, 16), "_C_EXTRA": (3,)}
    solver = DOP853(lambda t, y: -y, 0.0, np.array([1.0]), 1.0)
    np.testing.assert_array_equal(solver.A, dop853._A)
    np.testing.assert_array_equal(solver.E5, dop853._E5)
    np.testing.assert_array_equal(solver.D, dop853._D)


def test_initial_step_is_identical_to_scipys_for_every_lane():
    t_bound = np.full(6, 10.0)
    f0 = batched(np.zeros(6), Y0)

    h = dop853.initial_step(np, batched, np.zeros(6), Y0, f0, t_bound, MAX_STEP, RTOL, ATOL)

    for lane in range(6):
        solver = DOP853(scalar(lane), 0.0, Y0[lane], 10.0, max_step=MAX_STEP[lane], rtol=RTOL[lane], atol=ATOL[lane])
        assert h[lane] == pytest.approx(solver.h_abs, rel=1e-13), lane


def test_initial_step_over_an_empty_interval_is_zero_like_scipys_not_nan():
    t0 = np.full(6, 3.0)
    f0 = batched(t0, Y0)

    h = dop853.initial_step(np, batched, t0, Y0, f0, t0.copy(), MAX_STEP, RTOL, ATOL)

    np.testing.assert_array_equal(h, np.zeros(6))


def test_accepted_steps_follow_scipys_sequence_and_lie_on_its_solution():
    record = drive(batched, Y0, 10.0, MAX_STEP, RTOL, ATOL, steps=25)

    for lane in range(6):
        solver = DOP853(scalar(lane), 0.0, Y0[lane], 10.0, max_step=MAX_STEP[lane], rtol=RTOL[lane], atol=ATOL[lane])
        reference = solve_ivp(scalar(lane), (0.0, 10.0), Y0[lane], method="DOP853", max_step=MAX_STEP[lane],
                              rtol=RTOL[lane], atol=ATOL[lane], dense_output=True)
        assert len(record[lane]) > 6, lane  # every lane takes several steps before the final time
        for step in range(1, len(record[lane])):
            solver.step()
            t, y = record[lane][step]
            # err is ~1e-8 here, the size of its own rounding noise (a near-cancelling sum), so the sum order
            # of the stages moves the step by ~1e-5 relative; a wrong controller constant moves it by >1e-2
            assert t == pytest.approx(solver.t, rel=1e-4), (lane, step)
            # the steps differ slightly in time, so the states are compared on scipy's dense solution at our t
            np.testing.assert_allclose(y, reference.sol(t), rtol=1e-4, atol=2e-5, err_msg=f"lane {lane} step {step}")


def test_integration_to_the_final_time_agrees_with_solve_ivp_lane_by_lane():
    record = drive(batched, Y0, 10.0, MAX_STEP, RTOL, ATOL)

    for lane in range(6):
        reference = solve_ivp(scalar(lane), (0.0, 10.0), Y0[lane], method="DOP853", max_step=MAX_STEP[lane],
                              rtol=RTOL[lane], atol=ATOL[lane])
        t_end, y_end = record[lane][-1]
        assert t_end == 10.0  # the last step is truncated exactly at the bound
        np.testing.assert_allclose(y_end, reference.y[:, -1], rtol=1e-5, atol=1e-7, err_msg=f"lane {lane}")


def test_dense_output_matches_scipys_polynomial_inside_a_step():
    solver_lanes = [DOP853(scalar(i), 0.0, Y0[i], 10.0, max_step=MAX_STEP[i], rtol=RTOL[i], atol=ATOL[i])
                    for i in range(6)]
    h = np.array([solver.h_abs for solver in solver_lanes])
    t = np.zeros(6)
    f = batched(t, Y0)

    y_new, f_new, K = dop853.rk_step(np, batched, t, Y0, f, h)
    F = dop853.dense_coefficients(np, batched, t, Y0, y_new, f, f_new, K, h)

    for solver in solver_lanes:
        solver.step()
    for x in (0.0, 0.1, 0.25, 0.5, 0.77, 1.0):
        got = dop853.dense_eval(np, F, Y0, np.full(6, x))
        for lane, solver in enumerate(solver_lanes):
            expected = solver.dense_output()(solver.t_old + x * solver.h_previous)
            np.testing.assert_allclose(got[lane], expected, rtol=1e-9, atol=1e-13, err_msg=f"lane {lane} x={x}")


def test_attempt_returns_auxiliary_values_from_the_final_rhs_evaluation():
    calls = []

    def endpoint_fun(t, y):
        calls.append((t.copy(), y.copy()))
        return batched(t, y), y[:, 0] + t

    t = np.zeros(len(Y0))
    h = np.full(len(Y0), 0.01)
    out = dop853.attempt(
        np, batched, t, Y0, batched(t, Y0), h, np.zeros(len(Y0), dtype=bool), np.ones(len(Y0)), MAX_STEP,
        RTOL, ATOL, np.ones(len(Y0), dtype=bool), endpoint_fun=endpoint_fun,
    )

    assert len(calls) == 1
    np.testing.assert_array_equal(calls[0][0], out["t_new"])
    np.testing.assert_array_equal(calls[0][1], out["y_new"])
    np.testing.assert_array_equal(out["f_new"], batched(out["t_new"], out["y_new"]))
    np.testing.assert_array_equal(out["endpoint_auxiliary"], out["y_new"][:, 0] + out["t_new"])


def test_a_nan_state_is_rejected_and_ends_as_a_too_small_step_not_an_infinite_loop():
    def fun(t, y):
        return np.where(np.arange(2)[:, None] == 1, np.nan, -y)

    y = np.ones((2, 1))
    t, f = np.zeros(2), fun(np.zeros(2), y)
    h_abs = np.full(2, 0.1)
    rejected = np.zeros(2, dtype=bool)
    flagged = np.zeros(2, dtype=bool)
    with np.errstate(all="ignore"):
        for _ in range(800):  # from 0.1 down to 10 ulp(0) takes ~460 rejections
            out = dop853.attempt(np, fun, t, y, f, h_abs, rejected, np.full(2, 1.0), np.full(2, 0.5),
                                 np.full(2, 1e-8), np.full(2, 1e-10), ~flagged)
            y = np.where(out["accept"][:, None], out["y_new"], y)
            t = np.where(out["accept"], out["t_new"], t)
            h_abs = np.where(out["accept"] | out["reject"], out["h_next"], h_abs)
            rejected = out["rejected_next"]
            flagged |= out["too_small"]
            if flagged[1]:
                break

    assert flagged.tolist() == [False, True]
    assert t[0] > 0.0 and t[1] == 0.0  # the healthy lane advanced, the broken one never moved


def test_lanes_outside_run_neither_accept_nor_reject():
    y = np.ones((3, 1))
    fun = lambda t, y: -y  # noqa: E731
    t, f = np.zeros(3), fun(np.zeros(3), y)

    out = dop853.attempt(np, fun, t, y, f, np.full(3, 0.1), np.zeros(3, dtype=bool), np.full(3, 1.0),
                         np.full(3, 0.5), np.full(3, 1e-8), np.full(3, 1e-10), np.array([True, False, True]))

    assert out["accept"].tolist() == [True, False, True]
    assert not out["reject"].any() and not out["too_small"].any()


def test_the_step_is_truncated_at_the_final_time():
    y = np.ones((1, 1))
    fun = lambda t, y: -y  # noqa: E731
    t = np.array([0.95])

    out = dop853.attempt(np, fun, t, y, fun(t, y), np.array([0.3]), np.zeros(1, dtype=bool), np.array([1.0]),
                         np.array([0.5]), np.array([1e-8]), np.array([1e-10]), np.ones(1, dtype=bool))

    assert out["t_new"][0] == 1.0
    assert out["h"][0] == pytest.approx(0.05, abs=1e-15)


# ---------------------------------------------------------------------------------------------------------
# The real burn right-hand side.

FAMILIES = ("tubular", "star", "ends-star", "mixed", "erosive", "efficiency", "altitude", "short")


@pytest.fixture(scope="module")
def burn_lanes():
    corpus = gc.load_corpus()["cases"]
    reference = gc.load_reference()["records"]
    lanes = []
    for family in FAMILIES:
        case = min((c for c in corpus if c["family"] == family), key=lambda c: reference[c["id"]]["history_points"])
        grain, motor, propellant, environment, kwargs = gc.build_objects(case)
        lanes.append((case, motor, propellant, environment, kwargs))
    batch = ProblemBatch.from_objects(
        [l[1] for l in lanes], [l[2] for l in lanes], [l[3] for l in lanes], [l[4] for l in lanes]
    )
    return lanes, batch


def burn_fun(batch):
    P = batch.namespace(np)
    active = P["grain_valid"]
    return lambda t, y: rhs.conservative_rhs(np, y, active, P)


def scalar_burn_simulation(lane):
    case, *_ = lane
    return gc.simulate(case)


def test_initial_step_of_the_burn_model_is_identical_to_scipys(burn_lanes):
    lanes, batch = burn_lanes
    fun = burn_fun(batch)
    y0 = batch.initial_state()
    t0 = np.zeros(len(batch))
    a = batch.arrays

    size = a["n_valid_grains"] + 7
    h = dop853.initial_step(np, fun, t0, y0, fun(t0, y0), a["burn_timeout_s"], a["max_step_size"], a["rtol"],
                            a["atol"], size)

    for i, lane in enumerate(lanes):
        simulation = scalar_burn_simulation(lane)
        n = len(simulation.motor.grains)
        active = np.ones(n, dtype=bool)
        initial = [simulation._gas_mass_initial, simulation._gas_mass_initial * simulation.initial_gas_temperature_k,
                   *([0.0] * n), *([0.0] * 5)]
        solver = DOP853(lambda t, y: simulation._conservative_rhs(t, y, active), 0.0, np.array(initial),
                        simulation.burn_timeout_s, max_step=simulation.max_step_size, rtol=simulation.rtol,
                        atol=simulation.atol)
        assert h[i] == pytest.approx(solver.h_abs, rel=1e-12), lane[0]["id"]


def test_the_first_accepted_steps_of_the_burn_model_follow_scipy(burn_lanes):
    lanes, batch = burn_lanes
    a = batch.arrays
    record = drive(burn_fun(batch), batch.initial_state(), 100.0, a["max_step_size"], a["rtol"], a["atol"], steps=4,
                   size=a["n_valid_grains"] + 7)

    for i, lane in enumerate(lanes):
        simulation = scalar_burn_simulation(lane)
        n = len(simulation.motor.grains)
        active = np.ones(n, dtype=bool)
        initial = [simulation._gas_mass_initial, simulation._gas_mass_initial * simulation.initial_gas_temperature_k,
                   *([0.0] * n), *([0.0] * 5)]
        solver = DOP853(lambda t, y: simulation._conservative_rhs(t, y, active), 0.0, np.array(initial),
                        simulation.burn_timeout_s, max_step=simulation.max_step_size, rtol=simulation.rtol,
                        atol=simulation.atol)
        for step in range(1, 5):
            solver.step()
            t, y = record[i][step]
            assert t == pytest.approx(solver.t, rel=1e-6), (lane[0]["id"], step)
            batch_y = np.concatenate([y[: 2 + n], y[2 + batch.g_max :]])
            np.testing.assert_allclose(batch_y, solver.y, rtol=1e-7, atol=1e-12, err_msg=f"{lane[0]['id']} step {step}")


def test_padding_does_not_change_the_step_sequence_when_the_real_state_size_is_given():
    """Padded state components are exactly zero; only the divisor of the error norm could notice them."""
    padded_y0 = np.concatenate([Y0, np.zeros((6, 4))], axis=1)

    def padded(t, y):
        return np.concatenate([batched(t, y[:, :3]), np.zeros((len(t), 4))], axis=1)

    plain = drive(batched, Y0, 10.0, MAX_STEP, RTOL, ATOL, steps=12)
    sized = drive(padded, padded_y0, 10.0, MAX_STEP, RTOL, ATOL, steps=12, size=np.full(6, 3))
    unsized = drive(padded, padded_y0, 10.0, MAX_STEP, RTOL, ATOL, steps=12)

    for lane in range(6):
        assert [t for t, _ in sized[lane]] == [t for t, _ in plain[lane]]
    assert [t for t, _ in unsized[0]] != [t for t, _ in plain[0]]  # the dilution is real, so the size matters
