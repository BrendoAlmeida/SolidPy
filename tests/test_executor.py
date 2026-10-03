import threading

import pytest

import golden_corpus as gc
from solidpy import backends
from solidpy.backends import UnsupportedLane
from solidpy.ensemble import ProblemBatch, simulate_burn, simulate_thermal
from solidpy.executor import HeterogeneousExecutor
from thermal_cases import case_lane, pack_lanes


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


def test_reserved_cores_cap_cpu_workers_and_default_cpu_feeders_reserve_nothing(monkeypatch):
    import solidpy.executor as executor_module

    monkeypatch.setattr(executor_module, "available_cpu_count", lambda: 8)
    cpu_only = HeterogeneousExecutor(["cpu-vectorized", "cpu-reference"], default_workers=7)
    reserved = HeterogeneousExecutor(
        ["cpu-vectorized", "cpu-reference"], default_workers=7, reserved_cores=3
    )
    explicitly_limited = HeterogeneousExecutor([("cpu-reference", 6)], default_workers=1)

    assert cpu_only._reserved_default() == 0 and cpu_only._worker_limit(7) == 7
    assert reserved._worker_limit(7) == 5
    assert explicitly_limited._effective_spec(explicitly_limited.specs[0]).workers == 1


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
    assert all(engine["assigned_lanes_per_s"] > 0.0 for engine in outcome.execution["engines"])
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


def test_thermal_service_runs_on_multiple_feeders_and_preserves_lane_order():
    thermal_batch = pack_lanes([case_lane("steel"), case_lane("steel_liner"), case_lane("single_interval")])
    barrier = threading.Barrier(2)
    vectorized = backends.get_backend("cpu-vectorized")

    class MultiDeviceThermal:
        name = "test-multi-device-thermal"
        api_version = backends.BACKEND_API_VERSION

        def __init__(self, device=None):
            self.device = device
            self._first_call = True

        def capabilities(self):
            return vectorized.capabilities()

        def devices(self):
            return ["cuda:0", "cuda:1"]

        def solve_burn(self, batch, options):
            raise AssertionError("the thermal service must be selected")

        def thermal_ablation(self, batch, options):
            if self._first_call:
                self._first_call = False
                barrier.wait(timeout=5)
            return vectorized.thermal_ablation(batch, options)

        def provenance(self):
            return {"backend": self.name, "device": self.device}

    backends.register_backend("test-multi-device-thermal", MultiDeviceThermal)
    try:
        outcome = simulate_thermal(
            thermal_batch,
            backend=[("test-multi-device-thermal", "cuda:0"), ("test-multi-device-thermal", "cuda:1")],
            chunk_size=1,
        )
    finally:
        backends.unregister_backend("test-multi-device-thermal")

    expected = vectorized.thermal_ablation(thermal_batch).to_results()
    assert outcome.execution["schedule"] == "shared_dynamic_queue"
    assert sum(engine["lanes"] for engine in outcome.execution["engines"]) == len(thermal_batch)
    assert all(engine["assigned_lanes_per_s"] > 0.0 for engine in outcome.execution["engines"])
    assert set(outcome.execution["lane_backends"]) == set(range(len(thermal_batch)))
    assert {item["backend"] for item in outcome.execution["lane_backends"].values()} == {
        "test-multi-device-thermal"
    }
    assert outcome.to_results() == expected


def test_thermal_engine_error_is_retried_with_lane_provenance():
    thermal_batch = pack_lanes([case_lane("steel")])

    class FailingThermal:
        name = "test-failing-thermal"
        api_version = backends.BACKEND_API_VERSION

        def __init__(self, device=None):
            self.device = device or "cpu"

        def capabilities(self):
            return backends.get_backend("cpu-vectorized").capabilities()

        def devices(self):
            return ["cpu"]

        def solve_burn(self, batch, options):
            raise AssertionError("the thermal service must be selected")

        def thermal_ablation(self, batch, options):
            raise RuntimeError("synthetic thermal failure")

        def provenance(self):
            return {"backend": self.name, "device": self.device}

    backends.register_backend("test-failing-thermal", FailingThermal)
    try:
        outcome = simulate_thermal(thermal_batch, backend=["test-failing-thermal"], chunk_size=1)
    finally:
        backends.unregister_backend("test-failing-thermal")

    result = outcome.to_results()[0]
    assert outcome.execution["fallback_lanes"] == {0: ["engine_error"]}
    assert outcome.execution["fallback_errors"] == {
        0: {"type": "RuntimeError", "message": "synthetic thermal failure"}
    }
    assert outcome.execution["lane_backends"][0] == {
        "backend": "cpu-reference",
        "device": "cpu",
        "fallback": ["engine_error"],
        "engine_error": {"type": "RuntimeError", "message": "synthetic thermal failure"},
    }
    assert "provenance" not in result
