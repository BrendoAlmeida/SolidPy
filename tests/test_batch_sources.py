from types import SimpleNamespace

import numpy as np
import pytest

import golden_corpus as gc
from solidpy import BurnSimulation
from solidpy.backends import _tolerances as tol
from solidpy.batch import ProblemBatch
from solidpy.batch.kernels import rhs, sources

RTOL = tol.KERNEL_RTOL_NUMPY
SOURCE_FAMILIES = ("igniter-scalar", "igniter-table", "igniter-after", "activation-scalar", "activation-table",
                   "ramp", "combo-ramp", "combo-table")


@pytest.fixture(scope="module")
def corpus():
    cases = gc.load_corpus()["cases"]
    return cases, gc.load_reference()["records"]


def lane_batch(cases):
    built = [gc.build_objects(c) for c in cases]
    return ProblemBatch.from_objects([b[1] for b in built], [b[2] for b in built], [b[3] for b in built],
                                     [b[4] for b in built])


@pytest.fixture(scope="module")
def source_cases(corpus):
    cases, reference = corpus
    chosen = []
    for family in SOURCE_FAMILIES:
        members = [c for c in cases if c["family"] == family]
        chosen.append(min(members, key=lambda c: reference[c["id"]]["history_points"]))
    return chosen


def scalar_namespace(case):
    kwargs = gc.build_objects(case)[4]
    return SimpleNamespace(
        igniter_mass_flow=kwargs.get("igniter_mass_flow"), igniter_burn_time=kwargs.get("igniter_burn_time", 0.0),
        burn_area_activation=kwargs.get("burn_area_activation"), ignition_ramp_time=kwargs.get("ignition_ramp_time", 0.0),
    )


def sample_times(case):
    ns = scalar_namespace(case)
    marks = {0.0, ns.igniter_burn_time, ns.ignition_ramp_time}
    for source in (ns.igniter_mass_flow, ns.burn_area_activation):
        if source is not None and not np.isscalar(source):
            marks.update(np.asarray(source, dtype=float)[:, 0].tolist())
    times = set()
    for mark in marks:
        times.update({mark, mark * (1 - 1e-12), mark * (1 + 1e-12), mark + 1e-9, max(mark - 1e-9, 0.0)})
    times.update({-0.1, 1e-12, 0.013, 0.37, 1.1, 7.7, 1e3})
    values = sorted(t for t in times if t > -1.0)
    return np.asarray(values + [(a + b) / 2 for a, b in zip(values, values[1:])])


def lane_arrays(batch, lane, count):
    """The packed arrays of one lane repeated ``count`` times: one evaluation per sampled time."""
    return {name: np.broadcast_to(array[lane], (count,) + array[lane].shape).copy()
            for name, array in batch.namespace(np).items()}


def test_every_source_kind_is_packed_with_its_mode_tables_and_boundaries(source_cases):
    batch = lane_batch(source_cases)
    a = batch.arrays
    ids = [c["id"] for c in source_cases]

    assert a["igniter_mode"].tolist() == [1, 2, 1, 0, 0, 0, 1, 2]
    assert a["activation_mode"].tolist() == [0, 0, 0, 1, 2, 0, 0, 2]
    assert (a["ignition_ramp_time"] > 0).tolist() == [False, False, False, False, False, True, True, False]
    table_lane = ids.index(next(i for i in ids if i.startswith("igniter-table")))
    knots = gc.build_objects(source_cases[table_lane])[4]["igniter_mass_flow"]
    n = int(a["igniter_table_n"][table_lane])
    assert n == len(knots)
    np.testing.assert_array_equal(a["igniter_table_t"][table_lane, :n], np.asarray(knots)[:, 0])
    np.testing.assert_array_equal(a["igniter_table_m"][table_lane, :n], np.asarray(knots)[:, 1])
    assert (np.diff(a["igniter_table_t"], axis=1) > 0).all()  # padding keeps the abscissae finite and increasing
    for lane, case in enumerate(source_cases):
        ns = scalar_namespace(case)
        expected = {ns.ignition_ramp_time, a["source_end_time"][lane]}
        for source in (ns.igniter_mass_flow, ns.burn_area_activation):
            if source is not None and not np.isscalar(source):
                expected.update(np.asarray(source, dtype=float)[:, 0].tolist())
        row = a["breakpoints"][lane]
        assert row[np.isfinite(row)].tolist() == sorted(x for x in expected if x > 0), case["id"]
    assert not np.isfinite(a["breakpoints"][ids.index(next(i for i in ids if i.startswith("activation-scalar"))), 1:]).any()


def test_the_end_of_the_igniter_follows_the_scalar_definition(source_cases):
    batch = lane_batch(source_cases)
    for lane, case in enumerate(source_cases):
        ns = scalar_namespace(case)
        simulation_like = SimpleNamespace(**vars(ns))
        expected = BurnSimulation._source_end_time(simulation_like)
        assert batch.arrays["source_end_time"][lane] == expected, case["id"]


def test_igniter_flow_matches_the_scalar_method_at_every_knot_and_edge(source_cases):
    batch = lane_batch(source_cases)
    for lane, case in enumerate(source_cases):
        ns = scalar_namespace(case)
        times = sample_times(case)

        got = sources.igniter_flow(np, times, lane_arrays(batch, lane, len(times)))

        expected = [BurnSimulation.evaluate_igniter_mass_flow(ns, float(t)) for t in times]
        np.testing.assert_allclose(got, expected, rtol=RTOL, atol=1e-300, err_msg=case["id"])


def test_activation_matches_the_scalar_method_for_none_ramp_scalar_and_table(source_cases):
    batch = lane_batch(source_cases)
    for lane, case in enumerate(source_cases):
        ns = scalar_namespace(case)
        times = sample_times(case)

        got = sources.activation(np, times, lane_arrays(batch, lane, len(times)))

        expected = [BurnSimulation.evaluate_burn_area_activation(ns, float(t), 0.0) for t in times]
        np.testing.assert_allclose(got, expected, rtol=RTOL, atol=1e-300, err_msg=case["id"])
        assert ((got >= 0.0) & (got <= 1.0)).all()


def test_a_callable_source_gives_nan_instead_of_a_plausible_value(corpus):
    cases, _ = corpus
    batch = lane_batch([next(c for c in cases if c["family"] == "igniter-callable"),
                        next(c for c in cases if c["family"] == "activation-callable")])
    t = np.full(2, 0.05)

    assert np.isnan(sources.igniter_flow(np, t, batch.namespace(np))[0])
    assert np.isnan(sources.activation(np, t, batch.namespace(np))[1])
    assert (batch.arrays["source_end_time"] >= 0).all()


def trajectory(case):
    simulation = gc.simulate(case)
    raws = [simulation._burn_raw] + ([simulation._tail_raw] if simulation.tail_off_solution is not None else [])
    times, states = simulation._join_segments(raws)
    return simulation, times, states.T


def test_quantities_and_rhs_with_sources_match_the_scalar_solver_along_trajectories(source_cases):
    for case in source_cases:
        simulation, times, states = trajectory(case)
        batch = lane_batch([case])
        view = {name: array[:, None] for name, array in batch.namespace(np).items()}
        picks = np.unique(np.concatenate([np.linspace(0, len(states) - 1, 40).astype(int), [len(states) - 1]]))
        y, t = states[picks][None], times[picks][None]

        q = rhs.state_quantities(np, y, None, view, time=t)
        derivative = rhs.conservative_rhs(np, y, None, view, time=t)

        for row, index in enumerate(picks):
            scalar = simulation._state_quantities_uncached(times[index], states[index], None)
            where = f"{case['id']} t={times[index]!r}"
            for key, expected in (("pressure", scalar["pressure"]), ("generated", scalar["generated"]),
                                  ("nozzle", scalar["nozzle"]), ("igniter", scalar["igniter"])):
                np.testing.assert_allclose(q[key][0, row], expected, rtol=RTOL, atol=1e-300, err_msg=f"{key} {where}")
            np.testing.assert_allclose(q["regression_rates"][0, row], scalar["regression_rates"], rtol=RTOL, atol=1e-300)
            np.testing.assert_allclose(q["areas"][0, row], scalar["areas"], rtol=RTOL, atol=1e-18, err_msg=where)
            expected_rhs = np.asarray(simulation._conservative_rhs(times[index], states[index], None), dtype=float)
            scale = np.abs(expected_rhs).copy()
            scale[0] = max(abs(scalar["generated"]), abs(scalar["igniter"]), abs(scalar["nozzle"]))
            scale[1] = max(abs(scalar["generated"]) * 3000.0, abs(scalar["nozzle"] * scalar["temperature"]),
                           abs(scalar["igniter"] * simulation.igniter_temperature))
            scale[2 + len(simulation.motor.grains) + 3] = abs(scalar["components"]["momentum_n"]) + abs(
                scalar["components"]["pressure_n"])
            bad = np.flatnonzero(~(np.abs(derivative[0, row] - expected_rhs) <= RTOL * (np.abs(expected_rhs) + scale)))
            assert bad.size == 0, (where, bad.tolist(), derivative[0, row][bad], expected_rhs[bad])


def test_without_time_the_lane_has_no_sources():
    case = next(c for c in gc.load_corpus()["cases"] if c["id"] == "igniter-scalar-000")
    batch = lane_batch([case])
    P = {name: array[:, None] for name, array in batch.namespace(np).items()}
    y = batch.initial_state()[:, None]

    plain = rhs.state_quantities(np, y, None, P)

    assert plain["igniter"].tolist() == [[0.0]] and plain["activation"].tolist() == [[1.0]]
