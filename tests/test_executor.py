import threading

import pytest

import golden_corpus as gc
from solidpy import backends
from solidpy.backends import UnsupportedLane
from solidpy.ensemble import ProblemBatch, simulate_burn


def _batch(ids=("tubular-000", "igniter-callable-000", "star-001", "activation-callable-000")):
    cases = {case["id"]: case for case in gc.load_corpus()["cases"]}
    built = [gc.build_objects(cases[name]) for name in ids]
    return ProblemBatch.from_objects(
        [case[1] for case in built], [case[2] for case in built], [case[3] for case in built], [case[4] for case in built]
    )


def test_heterogeneous_backend_preserves_order_and_falls_back_for_unsupported_lanes():
    batch = _batch()

    outcome = simulate_burn(batch, backend=["cpu-vectorized"], chunk_size=1)
    results = outcome.to_results()

    assert outcome.backend == "heterogeneous"
    assert outcome.execution["effective_backend"] == "heterogeneous"
    assert outcome.execution["fallback_lanes"] == [1, 3]
    assert [result["provenance"]["execution"]["backend"] for result in results] == [
        "cpu-vectorized", "cpu-reference", "cpu-vectorized", "cpu-reference"
    ]
    assert [result["provenance"]["execution"]["requested_backend"] for result in results] == [["cpu-vectorized"]] * 4
    assert results[1]["provenance"]["execution"]["fallback"]["lane_reason"] == ["igniter_callable"]
    assert results[3]["provenance"]["execution"]["fallback"]["lane_reason"] == ["activation_callable"]


def test_heterogeneous_strict_refuses_lanes_no_selected_engine_supports():
    with pytest.raises(UnsupportedLane, match=r"lane\(s\) 1: igniter_callable; 3: activation_callable"):
        simulate_burn(_batch(), backend=["cpu-vectorized"], strict=True)


def test_separate_device_feeders_work_concurrently_and_keep_lane_order():
    batch = _batch(("tubular-000", "tubular-000", "star-001", "tubular-000"))
    barrier = threading.Barrier(2)
    vectorized = backends.get_backend("cpu-vectorized")

    class MultiDevice:
        name = "test-multi-device"
        api_version = backends.BACKEND_API_VERSION

        def __init__(self, device=None):
            self.device = device
            self._first_call = True

        def capabilities(self):
            return vectorized.capabilities()

        def devices(self):
            return ["cuda:0", "cuda:1"]

        def solve_burn(self, sub_batch, options):
            if self._first_call:
                self._first_call = False
                barrier.wait(timeout=5)
            return vectorized.solve_burn(sub_batch, options)

        def provenance(self):
            return {"backend": self.name, "device": self.device}

    backends.register_backend("test-multi-device", MultiDevice)
    try:
        outcome = simulate_burn(
            batch, backend=[("test-multi-device", "cuda:0"), ("test-multi-device", "cuda:1")], chunk_size=1
        )
    finally:
        backends.unregister_backend("test-multi-device")

    assert outcome.execution["schedule"] == "shared_dynamic_queue"
    assert len(outcome.execution["engines"]) == 2
    assert sum(engine["lanes"] for engine in outcome.execution["engines"]) == len(batch)
    assert sum(engine["chunks"] for engine in outcome.execution["engines"]) == len(batch)
    for lane, result in enumerate(outcome.to_results()):
        assert result["status"]["completed"]
        expected_mass = batch.arrays["propellant_volume"][lane] * batch.arrays["density"][lane]
        assert result["metrics"]["propellant_mass_initial_kg"] == pytest.approx(expected_mass)


def test_engine_exception_is_isolated_and_retried_on_reference():
    batch = _batch(("tubular-000",))

    class FailingBackend:
        name = "test-failing"
        api_version = backends.BACKEND_API_VERSION

        def __init__(self, device=None):
            self.device = device or "cuda:0"

        def capabilities(self):
            return backends.get_backend("cpu-vectorized").capabilities()

        def devices(self):
            return ["cuda:0"]

        def solve_burn(self, batch, options):
            raise RuntimeError("synthetic device failure")

        def provenance(self):
            return {"backend": self.name, "device": self.device}

    backends.register_backend("test-failing", FailingBackend)
    try:
        result = simulate_burn(batch, backend=["test-failing"], chunk_size=1).to_results()[0]
    finally:
        backends.unregister_backend("test-failing")

    execution = result["provenance"]["execution"]
    assert execution["backend"] == "cpu-reference"
    assert execution["fallback"]["lane_reason"] == ["engine_error"]
    assert execution["fallback"]["engine_error"] == {
        "type": "RuntimeError", "message": "synthetic device failure"
    }
