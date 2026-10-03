"""``simulate_thermal_ablation`` is pinned bit for bit, so the code around it can be restructured safely."""

import pytest

from thermal_cases import CASES, load_golden, run_case


@pytest.mark.parametrize("name", list(CASES))
def test_the_scalar_thermal_ablation_is_unchanged(name):
    assert run_case(name) == load_golden()[name]


def test_the_cases_cover_what_the_batched_path_must_reproduce():
    golden = load_golden()

    assert {golden[name]["simulation.advanced.metadata.thermal_node_count"] for name in golden} >= {4.0, 6.0, 7.0, 11.0}
    assert golden["steel"]["simulation.advanced.thermal.heat_load_kj_m2"] > 0.0
    assert golden["single_point"]["simulation.advanced.thermal.heat_load_kj_m2"] == 0.0
