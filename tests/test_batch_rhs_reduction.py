import numpy as np
import pytest

import golden_corpus as gc
from solidpy.backends import SolveOptions, get_backend
from solidpy.backends._tolerances import MASS_BALANCE_ATOL_PCT
from solidpy.batch import ProblemBatch
from solidpy.batch.kernels.geometry import valid_prefix_sum


def test_valid_prefix_sum_matches_numpy_for_variable_prefixes_through_128():
    rng = np.random.default_rng(20261004)
    counts = np.arange(129)
    values = rng.normal(size=(len(counts), 136)) * 10.0 ** rng.uniform(-6.0, 6.0, size=(len(counts), 136))

    actual = valid_prefix_sum(np, values, counts)
    expected = np.asarray([np.sum(values[lane, :count]) for lane, count in enumerate(counts)])

    np.testing.assert_array_equal(actual, expected)


def test_valid_prefix_sum_is_independent_of_padding_above_128():
    rng = np.random.default_rng(20261005)
    counts = np.asarray([129, 130, 177, 200])
    prefix = rng.normal(size=(len(counts), 200))
    narrower = np.concatenate([prefix, rng.normal(size=(len(counts), 56))], axis=-1)
    wider = np.concatenate([narrower, rng.normal(size=(len(counts), 256))], axis=-1)

    narrow_sum = valid_prefix_sum(np, narrower, counts)
    wide_sum = valid_prefix_sum(np, wider, counts)
    expected = np.asarray([np.sum(narrower[lane, :count]) for lane, count in enumerate(counts)])

    np.testing.assert_array_equal(narrow_sum, wide_sum)
    np.testing.assert_allclose(narrow_sum, expected, rtol=1e-12, atol=1e-14)


def test_valid_prefix_sum_is_jittable_with_per_lane_counts():
    jax = pytest.importorskip("jax")
    jnp = jax.numpy
    values = np.arange(4 * 32, dtype=np.float64).reshape(4, 32) / 7.0
    counts = np.asarray([4, 9, 16, 24])
    enable_x64 = jax.enable_x64(True) if hasattr(jax, "enable_x64") else jax.experimental.enable_x64()
    with enable_x64:
        compiled = jax.jit(lambda x, n: valid_prefix_sum(jnp, x, n))
        actual = np.asarray(compiled(jnp.asarray(values), jnp.asarray(counts)))
    expected = np.asarray([np.sum(values[lane, :count]) for lane, count in enumerate(counts)])

    np.testing.assert_array_equal(actual, expected)


def test_efficiency_lane_mass_balance_is_invariant_to_grain_padding():
    case = next(case for case in gc.load_corpus()["cases"] if case["id"] == "efficiency-001")
    _, motor, propellant, environment, settings = gc.build_objects(case)
    backend = get_backend("cpu-vectorized")
    results = []

    for g_max in (4, 24):
        batch = ProblemBatch.from_objects(
            motor, propellant, environment, settings, g_max=g_max
        )
        result = backend.solve_burn(batch, SolveOptions()).to_results()[0]
        results.append(result)

    narrow, padded = results
    narrow_pct = narrow["metrics"]["mass_flow_balance_error_pct"]
    padded_pct = padded["metrics"]["mass_flow_balance_error_pct"]
    reference_pct = gc.load_reference()["records"][case["id"]]["metrics"]["mass_flow_balance_error_pct"]

    assert narrow["status"]["completed"] and padded["status"]["completed"]
    assert abs(padded_pct - narrow_pct) <= MASS_BALANCE_ATOL_PCT
    assert abs(padded_pct - reference_pct) <= MASS_BALANCE_ATOL_PCT


def test_eight_grain_mass_balance_stays_within_envelope_across_padding():
    case = next(case for case in gc.load_corpus()["cases"] if case["id"] == "star-029")
    _, motor, propellant, environment, settings = gc.build_objects(case)
    backend = get_backend("cpu-vectorized")
    results = []

    for g_max in (8, 24):
        batch = ProblemBatch.from_objects(
            motor, propellant, environment, settings, g_max=g_max
        )
        results.append(backend.solve_burn(batch, SolveOptions()).to_results()[0])

    measured = [result["metrics"]["mass_flow_balance_error_pct"] for result in results]
    reference = gc.load_reference()["records"][case["id"]]["metrics"]["mass_flow_balance_error_pct"]
    envelope = max(2.0 * reference, 1e-4)

    assert all(result["status"]["completed"] for result in results)
    assert abs(measured[1] - measured[0]) <= MASS_BALANCE_ATOL_PCT
    assert all(np.isfinite(value) and value <= envelope for value in measured)
