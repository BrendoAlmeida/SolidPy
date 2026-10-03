import numpy as np
import pytest

import golden_corpus as gc
from solidpy import backends
from solidpy.backends import UnsupportedLane
from solidpy.ensemble import ProblemBatch, lane_cost, simulate_burn

IDS = ("tubular-000", "igniter-scalar-000", "star-001", "ratetable-000", "tubular-002")
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
    for lane, reason in zip(FALLBACK_LANES, ("igniter_scalar", "burn_rate_table")):
        execution = results[lane]["provenance"]["execution"]
        assert execution["backend"] == "cpu-reference" and execution["requested_backend"] == "cpu-vectorized"
        assert execution["fallback"] == {"lane_reason": [reason], "ran_on": "cpu-reference"}
        assert results[lane]["metrics"] == scalar[lane]["metrics"]  # the reference result, unchanged
    for lane in range(len(batch)):  # lane order is preserved whatever ran where
        assert results[lane]["provenance"]["physics_provider_hash"] == scalar[lane]["provenance"]["physics_provider_hash"]


def test_strict_mode_refuses_instead_of_falling_back(batch):
    with pytest.raises(UnsupportedLane, match=r"lane\(s\) 1: igniter_scalar; 3: burn_rate_table"):
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


def test_history_and_step_limits_reach_the_backend(batch):
    lanes = batch.select([0])

    full = simulate_burn(lanes, backend="cpu-vectorized", history="full", max_steps=900).to_results()[0]
    short = simulate_burn(lanes, backend="cpu-vectorized", max_steps=20).to_results()[0]

    assert full["history"]["time_s"][0] == 0.0 and len(full["history"]["time_s"]) > 20
    assert short["history"] is None and short["provenance"]["execution"]["step_overflow"] is True


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
