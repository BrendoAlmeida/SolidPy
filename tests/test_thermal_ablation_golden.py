"""``simulate_thermal_ablation`` is pinned to 1e-10 (not bit for bit, so another CPU or BLAS does not fail it), so the code around it can be restructured safely."""

import pytest

from thermal_cases import CASES, load_golden, run_case


@pytest.mark.parametrize("name", list(CASES))
def test_the_scalar_thermal_ablation_is_unchanged(name):
    assert run_case(name) == pytest.approx(load_golden()[name], rel=1e-10, abs=0.0)


def test_the_cases_cover_what_the_batched_path_must_reproduce():
    golden = load_golden()

    assert {golden[name]["simulation.advanced.metadata.thermal_node_count"] for name in golden} >= {4.0, 6.0, 7.0, 11.0}
    assert golden["steel"]["simulation.advanced.thermal.heat_load_kj_m2"] > 0.0
    assert golden["single_point"]["simulation.advanced.thermal.heat_load_kj_m2"] == 0.0
