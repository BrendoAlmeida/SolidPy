import numpy as np
import pytest

import golden_corpus as gc
from solidpy import BurnSimulation, backends
from solidpy.backends import SolveOptions
from solidpy.batch import FEATURES, ProblemBatch
from solidpy.batch.thermal import THERMAL_FEATURES

CASE_IDS = ("tubular-000", "star-001", "ends-star-000")


def assert_same(a, b, path="result"):
    """Bit-for-bit equality of nested results (dicts, lists, tuples, arrays, scalars)."""
    if isinstance(a, dict):
        assert a.keys() == b.keys(), path
        for key in a:
            assert_same(a[key], b[key], f"{path}.{key}")
    elif isinstance(a, (list, tuple)):
        assert len(a) == len(b), path
        for i, (x, y) in enumerate(zip(a, b)):
            assert_same(x, y, f"{path}[{i}]")
    elif isinstance(a, np.ndarray):
        assert np.array_equal(a, b), path
    else:
        assert a == b, path


@pytest.fixture(scope="module")
def corpus_batch():
    cases = {case["id"]: case for case in gc.load_corpus()["cases"]}
    built = [gc.build_objects(cases[i]) for i in CASE_IDS]
    batch = ProblemBatch.from_objects(
        [b[1] for b in built], [b[2] for b in built], [b[3] for b in built], [b[4] for b in built]
    )
    return built, batch


def test_the_reference_backend_is_registered_and_is_the_default():
    assert backends.available()["cpu-reference"] == "ok"
    assert backends.current_backend()[0] == "cpu-reference"
    assert backends.get_backend().name == "cpu-reference"

    description = backends.describe("cpu-reference")

    assert description["devices"] == ["cpu"]
    assert set(description["capabilities"]) == set(FEATURES) | set(THERMAL_FEATURES)  # burn and thermal lanes
    assert description["services"] == ["thermal_ablation", "structural_response"]
    assert set(description["capabilities"].values()) == {"supported"}
    assert description["provenance"]["backend"] == "cpu-reference"
    assert description["provenance"]["dtype"] == "float64"


def test_the_default_can_be_selected_again_after_another_backend():
    backends.register_backend("temporary", type(backends.get_backend()), replace=True)
    try:
        backends.set_backend("temporary")
        assert backends.current_backend()[0] == "temporary"
        backends.set_backend("cpu-reference")
        assert backends.current_backend()[0] == "cpu-reference"
        with backends.use_backend("temporary"):
            with backends.use_backend("cpu-reference"):
                assert backends.get_backend().name == "cpu-reference"
    finally:
        backends.reset_backend()
        backends.unregister_backend("temporary")


def test_only_the_cpu_device_is_accepted():
    assert backends.get_backend("cpu-reference", device="cpu").devices() == ["cpu"]
    with pytest.raises(ValueError, match="only runs on 'cpu'"):
        backends.get_backend("cpu-reference", device="cuda:0")


def test_every_ready_backend_describes_itself(backend):
    description = backends.describe(backend)

    assert description["name"] == backend
    assert description["api_version"] == backends.BACKEND_API_VERSION
    assert description["provenance"]["backend"] == backend


def test_results_are_exactly_what_burn_simulation_produces(corpus_batch):
    built, batch = corpus_batch

    result = backends.get_backend("cpu-reference").solve_burn(batch, SolveOptions())

    assert len(result) == len(built)
    for (grain, motor, propellant, environment, kwargs), got in zip(built, result.to_results()):
        assert_same(got, BurnSimulation(grain, motor, propellant, environment, **kwargs).result)
        assert "execution" not in got["provenance"]  # the reference result is not annotated
    assert result.backend == "cpu-reference"
    assert result.execution["backend"] == "cpu-reference"


def test_a_process_pool_gives_the_same_results_in_the_same_order(corpus_batch):
    _, batch = corpus_batch
    backend = backends.get_backend("cpu-reference")

    sequential = backend.solve_burn(batch).to_results()
    pooled = backend.solve_burn(batch, SolveOptions(workers=2)).to_results()

    for one, other in zip(sequential, pooled):
        assert_same(one, other)


def test_options_default_to_the_metrics_policy_and_no_pool():
    options = SolveOptions()

    assert (options.history, options.workers) == ("metrics", None)


@pytest.mark.parametrize(
    "kwargs, message",
    [
        ({"history": "everything"}, "history must be"),
        ({"workers": 2.5}, "workers must be a positive integer"),
        ({"workers": 0}, "workers must be a positive integer"),
        ({"workers": -3}, "workers must be a positive integer"),
        ({"workers": True}, "workers must be a positive integer"),
    ],
)
def test_invalid_options_are_rejected_when_they_are_built(kwargs, message):
    with pytest.raises(ValueError, match=message):
        SolveOptions(**kwargs)


def test_the_backend_rejects_options_that_are_not_solve_options(corpus_batch):
    _, batch = corpus_batch

    with pytest.raises(TypeError, match="options must be a SolveOptions"):
        backends.get_backend("cpu-reference").solve_burn(batch, {"workers": 2})


def test_the_reference_advertises_every_history_policy_template():
    capabilities = backends.get_backend("cpu-reference").capabilities()

    assert set(capabilities.history_policies) == {"metrics", "full", "decimated:N", "uniform:N"}
    for policy in ("metrics", "full", "decimated:8", "uniform:8"):
        SolveOptions(history=policy)


@pytest.mark.parametrize("policy", ["decimated:1", "decimated:0", "uniform:x", "uniform:2.5", "other:8", "uniform:"])
def test_malformed_point_count_history_policies_are_rejected(policy):
    with pytest.raises(ValueError, match="history must be"):
        SolveOptions(history=policy)


def test_lanes_that_cannot_be_pickled_are_solved_in_process_and_the_pool_keeps_the_rest():
    cases = {case["id"]: case for case in gc.load_corpus()["cases"]}
    built = [gc.build_objects(cases[i]) for i in ("tubular-000", "tubular-001", "igniter-callable-000")]
    local = {"calls": 0}

    def igniter(t):  # a closure: not picklable
        local["calls"] += 1
        return 0.004 if t < 0.1 else 0.0

    settings = [dict(b[4]) for b in built]
    settings[2] = {**settings[2], "igniter_mass_flow": igniter, "igniter_burn_time": 0.1}
    batch = ProblemBatch.from_objects([b[1] for b in built], [b[2] for b in built], [b[3] for b in built], settings)
    backend = backends.get_backend("cpu-reference")

    pooled = backend.solve_burn(batch, SolveOptions(workers=3)).to_results()
    sequential = backend.solve_burn(batch).to_results()

    assert local["calls"] > 0  # the closure ran here, in this process
    for one, other in zip(pooled, sequential):
        assert_same(one, other)
