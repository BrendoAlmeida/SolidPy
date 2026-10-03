import numpy as np
import pytest

import golden_corpus as gc
from solidpy import BurnSimulation, backends
from solidpy.backends import SolveOptions
from solidpy.batch import FEATURES, ProblemBatch

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
    assert set(description["capabilities"]) == set(FEATURES)
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
