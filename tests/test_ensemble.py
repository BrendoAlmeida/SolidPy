import numpy as np
import pytest

import golden_corpus as gc
from solidpy import backends
from solidpy.backends import UnsupportedLane
from solidpy.ensemble import ProblemBatch, lane_cost, simulate_burn

IDS = ("tubular-000", "igniter-callable-000", "star-001", "ratetable-000", "tubular-002")
SUPPORTED_LANES = [0, 2, 4]
FALLBACK_LANES = [1, 3]


@pytest.fixture(scope="module")
def cases():
    by_id = {c["id"]: c for c in gc.load_corpus()["cases"]}
    return [by_id[i] for i in IDS]


@pytest.fixture(scope="module")
def batch(cases):
    built = [gc.build_objects(c) for c in cases]
    return ProblemBatch.from_objects(
        [b[1] for b in built], [b[2] for b in built], [b[3] for b in built], [b[4] for b in built]
    )


@pytest.fixture(scope="module")
def scalar(cases):
    return [gc.simulate(c).result for c in cases]


def test_the_default_backend_is_the_untouched_reference(batch, scalar):
    result = simulate_burn(batch)

    assert result.backend == "cpu-reference" and len(result) == len(batch)
    for got, expected in zip(result.to_results(), scalar):
        assert "execution" not in got["provenance"]
        assert got["metrics"] == expected["metrics"]
        assert got["provenance"]["physics_provider_hash"] == expected["provenance"]["physics_provider_hash"]


def test_a_backend_that_cannot_run_every_lane_falls_back_and_says_so(batch, scalar):
    result = simulate_burn(batch, backend="cpu-vectorized")
    results = result.to_results()

    assert result.execution["fallback_lanes"] == FALLBACK_LANES and result.execution["requested_backend"] == "cpu-vectorized"
    for lane in SUPPORTED_LANES:
        execution = results[lane]["provenance"]["execution"]
        assert execution["backend"] == "cpu-vectorized" and execution["fallback"] is None
        assert execution["requested_backend"] == "cpu-vectorized"
    for lane, reason in zip(FALLBACK_LANES, ("igniter_callable", "burn_rate_table")):
        execution = results[lane]["provenance"]["execution"]
        assert execution["backend"] == "cpu-reference" and execution["requested_backend"] == "cpu-vectorized"
        assert execution["fallback"] == {"lane_reason": [reason], "ran_on": "cpu-reference"}
        assert results[lane]["metrics"] == scalar[lane]["metrics"]  # the reference result, unchanged
    for lane in range(len(batch)):  # lane order is preserved whatever ran where
        assert results[lane]["provenance"]["physics_provider_hash"] == scalar[lane]["provenance"]["physics_provider_hash"]


def test_strict_mode_refuses_instead_of_falling_back(batch):
    with pytest.raises(UnsupportedLane, match=r"lane\(s\) 1: igniter_callable; 3: burn_rate_table"):
        simulate_burn(batch, backend="cpu-vectorized", strict=True)
    simulate_burn(batch.select(SUPPORTED_LANES), backend="cpu-vectorized", strict=True)  # all supported: fine


def test_the_selected_backend_is_used_when_none_is_named(batch):
    with backends.use_backend("cpu-vectorized"):
        result = simulate_burn(batch.select(SUPPORTED_LANES))

    assert result.backend == "cpu-vectorized"
    assert {r["provenance"]["execution"]["backend"] for r in result.to_results()} == {"cpu-vectorized"}


def test_chunking_and_sorting_keep_every_result_in_its_lane(batch):
    lanes = batch.select(SUPPORTED_LANES * 2)  # six lanes, two of each design
    whole = simulate_burn(lanes, backend="cpu-vectorized", sort=False).to_results()

    chunked = simulate_burn(lanes, backend="cpu-vectorized", chunk_size=2, sort=True)

    assert chunked.execution["chunks"] == 3
    for one, other in zip(whole, chunked.to_results()):
        assert one["provenance"]["physics_provider_hash"] == other["provenance"]["physics_provider_hash"]
        for key in ("total_impulse_ns", "generated_mass_integral_kg", "propellant_mass_initial_kg"):
            assert other["metrics"][key] == pytest.approx(one["metrics"][key], rel=1e-9)
        assert other["status"]["termination_reason"] == one["status"]["termination_reason"]


def test_history_reaches_the_backend_and_a_full_history_runs_in_one_uncapped_solve(batch):
    lanes = batch.select([0])

    full = simulate_burn(lanes, backend="cpu-vectorized", history="full", max_steps=900)

    assert full.to_results()[0]["history"]["time_s"][0] == 0.0 and len(full.to_results()[0]["history"]["time_s"]) > 20
    assert full.execution["tiers"] == [[(None, 1, 0)]]  # a full history is for inspection: no capped tiers


def test_a_lane_that_exhausts_the_step_budget_is_rerun_on_the_reference_and_flagged(batch, scalar):
    lanes = batch.select([0, 2])

    result = simulate_burn(lanes, backend="cpu-vectorized", max_steps=20)

    assert result.execution["fallback_lanes"] == [0, 1]
    for lane, got in zip((0, 2), result.to_results()):
        execution = got["provenance"]["execution"]
        assert execution["backend"] == "cpu-reference" and execution["requested_backend"] == "cpu-vectorized"
        assert execution["fallback"] == {"lane_reason": ["step_overflow"], "ran_on": "cpu-reference"}
        assert got["status"]["completed"] and got["metrics"] == scalar[lane]["metrics"]  # the reference result
    direct = backends.get_backend("cpu-vectorized").solve_burn(lanes, backends.SolveOptions(max_steps=20))
    assert all(r["provenance"]["execution"]["step_overflow"] for r in direct.to_results())  # the backend alone fails them


def test_auto_picks_the_reference_when_no_accelerator_is_usable(batch):
    result = simulate_burn(batch.select([0]), backend="auto")

    assert result.backend == "cpu-reference"


def test_a_non_batch_argument_is_a_type_error(cases):
    with pytest.raises(TypeError, match="needs a ProblemBatch"):
        simulate_burn(cases)


def test_lane_cost_orders_slow_and_many_grain_lanes_after_quick_ones():
    by_id = {c["id"]: c for c in gc.load_corpus()["cases"]}
    pick = [by_id[i] for i in ("short-000", "tubular-000", "long-000", "many-000")]
    built = [gc.build_objects(c) for c in pick]
    lanes = ProblemBatch.from_objects([b[1] for b in built], [b[2] for b in built], [b[3] for b in built],
                                      [b[4] for b in built])

    cost = lane_cost(lanes)

    assert np.isfinite(cost).all() and cost[0] < cost[2] and cost[0] < cost[3]


def test_lanes_without_a_power_law_burn_rate_sort_last():
    by_id = {c["id"]: c for c in gc.load_corpus()["cases"]}
    built = [gc.build_objects(by_id[i]) for i in ("ratetable-000", "tubular-000")]
    lanes = ProblemBatch.from_objects([b[1] for b in built], [b[2] for b in built], [b[3] for b in built],
                                      [b[4] for b in built])

    cost = lane_cost(lanes)

    assert cost[0] == np.finfo(float).max and cost[1] < cost[0]


@pytest.mark.parametrize("bad", [0, -1, 2.5, True])
def test_a_chunk_size_that_is_not_a_positive_integer_is_an_error(batch, bad):
    with pytest.raises(ValueError, match="chunk_size must be a positive integer or None"):
        simulate_burn(batch.select([0]), backend="cpu-vectorized", chunk_size=bad)


def test_sorting_is_skipped_when_one_chunk_holds_every_lane(batch, monkeypatch):
    from solidpy import ensemble

    def refuse(_):
        raise AssertionError("lane_cost must not run for a single chunk")

    monkeypatch.setattr(ensemble, "lane_cost", refuse)

    result = simulate_burn(batch.select(SUPPORTED_LANES), backend="cpu-vectorized")

    assert result.execution["chunks"] == 1
    with pytest.raises(AssertionError, match="must not run"):
        simulate_burn(batch.select(SUPPORTED_LANES), backend="cpu-vectorized", chunk_size=1)


class FakeAccelerator:
    name = "fake-gpu"
    api_version = backends.BACKEND_API_VERSION

    def __init__(self, device=None):
        self.device = device or "cuda:0"

    def capabilities(self):
        return backends.get_backend("cpu-vectorized").capabilities()

    def devices(self):
        return ["cuda:0"]

    def solve_burn(self, batch, options):
        return backends.get_backend("cpu-vectorized").solve_burn(batch, options)

    def provenance(self):
        return {"backend": self.name}


def test_auto_counts_only_the_lanes_the_accelerator_can_run_and_records_what_was_asked(batch, monkeypatch):
    from solidpy import ensemble

    backends.register_backend("fake-gpu", FakeAccelerator, replace=True)
    monkeypatch.setattr(ensemble, "AUTO_ACCELERATORS", ("fake-gpu",))
    monkeypatch.setattr(ensemble, "AUTO_MIN_LANES", 4)
    try:
        enough = batch.select([0, 2, 4, 0])
        too_few = batch.select([0, 1, 3, 4])  # two of the four lanes need the reference

        assert ensemble._auto_backend(enough) == "fake-gpu"
        assert ensemble._auto_backend(too_few) == "cpu-reference"
        result = simulate_burn(enough, backend="auto")
        assert result.execution["requested_backend"] == "auto" and result.execution["effective_backend"] == "fake-gpu"
        assert {r["provenance"]["execution"]["requested_backend"] for r in result.to_results()} == {"auto"}
        # an explicit device does not break a choice of the reference
        assert simulate_burn(batch.select([0]), backend="auto", device="cuda:0").backend == "cpu-reference"
    finally:
        backends.unregister_backend("fake-gpu")
