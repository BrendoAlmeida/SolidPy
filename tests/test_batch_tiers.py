import numpy as np
import pytest

import golden_corpus as gc
from solidpy import backends
from solidpy.backends import SolveOptions
from solidpy.batch import ProblemBatch
from solidpy.batch.integrators import solver
from solidpy.batch.tiers import DEFAULT_TIERS, solve_in_tiers


class FakeBatch:
    """Just enough of a ProblemBatch for the tier runner: lanes are integers and ``select`` takes a subset."""

    def __init__(self, work):
        self.work = np.asarray(work)

    def __len__(self):
        return len(self.work)

    def select(self, indices):
        return FakeBatch(self.work[np.asarray(indices, dtype=int)])


def fake_run(calls):
    def run(sub, cap):
        calls.append((cap, len(sub)))
        unfinished = np.zeros(len(sub), dtype=bool) if cap is None else sub.work > cap
        return {"value": sub.work * 10.0, "pair": np.stack([sub.work, -sub.work], axis=1), "unfinished": unfinished,
                "iterations": np.asarray(5)}  # a scalar entry that the runner must drop
    return run


def test_every_lane_ends_in_the_first_tier_that_lets_it_finish_and_order_is_kept():
    work = np.array([5, 900, 20, 70000, 300, 15, 4000, 80])
    calls = []

    merged, info = solve_in_tiers(FakeBatch(work), fake_run(calls), tiers=(100, 1000))

    assert calls == [(100, 8), (1000, 4), (None, 2)]  # the later tiers only hold the lanes still unfinished
    assert info == [(100, 8, 4), (1000, 4, 1), (None, 1, 0)] or info == [(100, 8, 4), (1000, 4, 2), (None, 2, 0)]
    np.testing.assert_array_equal(merged["value"], work * 10.0)
    np.testing.assert_array_equal(merged["pair"], np.stack([work, -work], axis=1))
    assert "iterations" not in merged and "unfinished" in merged


def test_no_tiers_is_one_uncapped_run_and_a_batch_that_finishes_early_stops_early():
    calls = []
    solve_in_tiers(FakeBatch([1, 2, 3]), fake_run(calls), tiers=())
    assert calls == [(None, 3)]

    calls.clear()
    solve_in_tiers(FakeBatch([1, 2, 3]), fake_run(calls), tiers=(10, 100))
    assert calls == [(10, 3)]  # nothing was left for the next tiers


def test_default_tiers_are_increasing_positive_caps():
    assert list(DEFAULT_TIERS) == sorted(set(DEFAULT_TIERS)) and min(DEFAULT_TIERS) > 0


def corpus_batch(ids):
    by_id = {c["id"]: c for c in gc.load_corpus()["cases"]}
    built = [gc.build_objects(by_id[i]) for i in ids]
    return ProblemBatch.from_objects([b[1] for b in built], [b[2] for b in built], [b[3] for b in built],
                                     [b[4] for b in built])


IDS = ("tubular-000", "tubular-003", "star-001", "efficiency-002", "short-001", "guard-axial-000", "mixed-002",
       "altitude-001", "erosive-004", "replicated-001")


@pytest.fixture(scope="module")
def batch():
    return corpus_batch(IDS)


def test_the_iteration_cap_flags_lanes_that_did_not_finish_and_leaves_finished_ones_exact(batch):
    P, y0 = batch.namespace(np), batch.initial_state()
    full = solver.solve_burn_and_blowdown(solver.numpy_driver(), P, y0, solver.SolveConfig())
    assert not full["unfinished"].any()

    # a cap that splits this set: the lanes need between ~50 and ~250 iterations
    for cap in (60, 90, 130, 180, 250, 400):
        capped = solver.solve_burn_and_blowdown(solver.numpy_driver(), P, y0, solver.SolveConfig(), cap)
        if capped["unfinished"].any() and not capped["unfinished"].all():
            break
    else:
        pytest.fail("no cap splits the lanes into finished and unfinished")

    done = ~capped["unfinished"]
    for name in ("t", "y", "pmax", "tmax", "gmax", "nmax", "burn_t", "cutoff", "gen_start", "noz_end"):
        np.testing.assert_array_equal(capped[name][done], full[name][done], err_msg=name)
    assert capped["burn_iterations"] <= cap and capped["tail_iterations"] <= cap
    assert (capped["n_points"][~done] < full["n_points"][~done]).all()  # the unfinished ones stopped part-way


def test_tiered_results_equal_the_uncapped_results_exactly(batch):
    backend = backends.get_backend("cpu-vectorized")

    plain = backend.solve_burn(batch, SolveOptions(tiers=()))
    tiered = backend.solve_burn(batch, SolveOptions(tiers=(60, 250)))

    assert [t[0] for t in tiered.execution["tiers"]] == [60, 250, None][: len(tiered.execution["tiers"])]
    sizes = [t[1] for t in tiered.execution["tiers"]]
    assert len(sizes) >= 2 and sizes[0] == len(batch) and sizes == sorted(sizes, reverse=True)
    for a, b in zip(plain.to_results(), tiered.to_results()):
        assert a["metrics"] == b["metrics"] and a["status"] == b["status"]
        assert a["provenance"]["physics_provider_hash"] == b["provenance"]["physics_provider_hash"]


def test_the_default_run_uses_tiers_and_reports_them(batch):
    result = backends.get_backend("cpu-vectorized").solve_burn(batch)

    assert result.execution["tiers"][0][0] == DEFAULT_TIERS[0] and result.execution["tiers"][0][1] == len(batch)
    assert all(r["status"]["completed"] for r in result.to_results())


@pytest.mark.parametrize("bad", [(0,), (5, 5), (10, 5), [10, 20], (1.5,), (True,)])
def test_invalid_tiers_are_rejected(bad):
    with pytest.raises(ValueError, match="tiers must be a tuple of increasing positive integers"):
        SolveOptions(tiers=bad)
