"""``simulate_thermal``: the choice of backend, the fallback to the scalar reference, chunking and strict mode."""

import numpy as np
import pytest

from solidpy import backends
from solidpy.backends import UnsupportedLane
from solidpy.backends import _tolerances as tol
from solidpy.batch.integrators import radau
from solidpy.batch.thermal import NON_FINITE_INPUT
from solidpy.ensemble import simulate_thermal
from thermal_cases import CASES, case_lane, make_curve, pack_lanes, scalar_thermal

FIXED = [case_lane(name) for name in CASES]


@pytest.fixture(scope="module")
def fixed_batch():
    return pack_lanes(FIXED)


@pytest.fixture(scope="module")
def scalar_fixed():
    return [scalar_thermal(lane) for lane in FIXED]


def close(got, want, rtol=tol.THERMAL_RTOL):
    assert set(got) == set(want)
    for key, value in want.items():
        assert got[key] == pytest.approx(value, rel=rtol, abs=1e-12), key


def test_the_router_runs_the_chosen_backend_and_reports_it(fixed_batch, scalar_fixed):
    reference = simulate_thermal(fixed_batch, backend="cpu-reference")
    vectorized = simulate_thermal(fixed_batch, backend="cpu-vectorized")

    assert reference.to_results() == scalar_fixed and reference.execution["fallback_lanes"] == {}
    assert vectorized.execution["effective_backend"] == "cpu-vectorized" and vectorized.execution["fallback_lanes"] == {}
    for got, want in zip(vectorized.to_results(), scalar_fixed):
        close(got, want)


def test_auto_keeps_a_small_batch_on_the_reference_and_the_selected_backend_is_used_for_none(fixed_batch):
    assert simulate_thermal(fixed_batch, backend="auto").execution["effective_backend"] == "cpu-reference"
    with backends.use_backend("cpu-vectorized"):
        assert simulate_thermal(fixed_batch.select([0, 1]), backend=None).execution["effective_backend"] == "cpu-vectorized"


def test_a_lane_the_backend_cannot_take_is_run_on_the_reference_or_refused_when_strict():
    ablation = np.linspace(0.0, 1e-4, 60)
    ablation[7] = np.inf  # an infinite recession: the scalar model carries it through, the batched integration does not
    lanes = [FIXED[0], dict(FIXED[0], curve=make_curve(points=60, ablation_series=ablation)), FIXED[1]]
    batch = pack_lanes(lanes)

    result = simulate_thermal(batch, backend="cpu-vectorized")

    assert result.execution["fallback_lanes"] == {1: [NON_FINITE_INPUT]}
    assert result.to_results()[1] == scalar_thermal(lanes[1])
    close(result.to_results()[0], scalar_thermal(lanes[0]))
    with pytest.raises(UnsupportedLane, match="1: non_finite_thermal_input"):
        simulate_thermal(batch, backend="cpu-vectorized", strict=True)


def test_a_lane_that_did_not_finish_is_rerun_on_the_reference_even_when_strict(fixed_batch, scalar_fixed, monkeypatch):
    monkeypatch.setattr(radau, "MAX_ATTEMPTS", 1)

    result = simulate_thermal(fixed_batch, backend="cpu-vectorized", strict=True)

    fallback = result.execution["fallback_lanes"]
    assert fallback and all(reason == ["integration_failed"] for reason in fallback.values())
    for lane in fallback:
        assert result.to_results()[lane] == scalar_fixed[lane]  # the scalar code ran it
    for lane in set(range(len(CASES))) - set(fallback):
        close(result.to_results()[lane], scalar_fixed[lane])


class NoThermalService:
    name = "no-thermal"
    api_version = backends.BACKEND_API_VERSION

    def __init__(self, device=None):
        self.device = "cpu"

    def capabilities(self):
        return backends.Capabilities(dict(backends.get_backend("cpu-vectorized").capabilities().features))

    def devices(self):
        return ["cpu"]

    def solve_burn(self, batch, options):
        raise AssertionError("not used")

    def provenance(self):
        return {"backend": self.name}


def test_a_backend_without_the_thermal_service_hands_every_lane_to_the_reference(fixed_batch, scalar_fixed):
    backends.register_backend("no-thermal", NoThermalService, replace=True)
    try:
        result = simulate_thermal(fixed_batch.select([0, 1, 2]), backend="no-thermal")

        assert result.execution["fallback_lanes"] == {0: ["service:thermal_ablation"], 1: ["service:thermal_ablation"],
                                                      2: ["service:thermal_ablation"]}
        assert result.to_results() == scalar_fixed[:3]
        with pytest.raises(UnsupportedLane, match="service:thermal_ablation"):
            simulate_thermal(fixed_batch, backend="no-thermal", strict=True)
    finally:
        backends.unregister_backend("no-thermal")


def test_chunks_keep_the_lane_order_and_the_results(fixed_batch, scalar_fixed):
    whole = simulate_thermal(fixed_batch, backend="cpu-vectorized")
    chunked = simulate_thermal(fixed_batch, backend="cpu-vectorized", chunk_size=5)
    unsorted = simulate_thermal(fixed_batch, backend="cpu-vectorized", chunk_size=np.int64(4), sort=False)

    assert whole.execution["chunks"] == 1 and chunked.execution["chunks"] == 3 and unsorted.execution["chunks"] == 4
    for result in (chunked, unsorted):
        for got, want in zip(result.to_results(), whole.to_results()):
            close(got, want, rtol=1e-12)


@pytest.mark.parametrize("bad", [0, -1, 2.5, True, "3"])
def test_chunk_size_must_be_a_positive_integer(fixed_batch, bad):
    with pytest.raises(ValueError, match="chunk_size"):
        simulate_thermal(fixed_batch, backend="cpu-vectorized", chunk_size=bad)


def test_the_router_wants_a_thermal_batch():
    with pytest.raises(TypeError, match="ThermalBatch"):
        simulate_thermal([FIXED[0]], backend="cpu-vectorized")


def test_workers_reach_the_reference(fixed_batch, scalar_fixed):
    result = simulate_thermal(fixed_batch, backend="cpu-reference", workers=2)

    assert result.to_results() == scalar_fixed


def test_a_degenerate_gas_falls_back_to_the_reference(scalar_fixed):
    lanes = [FIXED[0], dict(FIXED[0], gamma=1.0), FIXED[1]]

    result = simulate_thermal(pack_lanes(lanes), backend="cpu-vectorized")

    assert result.execution["fallback_lanes"] == {1: ["thermal_degenerate_gas"]}
    assert result.to_results()[1] == scalar_thermal(lanes[1])
    close(result.to_results()[0], scalar_fixed[0])
    close(result.to_results()[2], scalar_fixed[1])
