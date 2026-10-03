"""``run_advanced_physics_ensemble``: the thermal batch plus the CPU models equal ``simulate_advanced_physics`` per design."""

import pytest

from solidpy import (
    CasingMaterial, NozzleMaterial, geometry_from_components, run_detailed_ballistics, simulate_advanced_physics,
)
from solidpy.ensemble import run_advanced_physics_ensemble
from test_robustness import make_motor_stack


@pytest.fixture(scope="module")
def designs():
    """Two curves of the test motor (coarse and fine steps), the second with scenario factors, and a lined casing."""
    grain, motor, propellant, environment = make_motor_stack()
    geometry = geometry_from_components(grain, motor, propellant, casing_wall_thickness_m=0.004, dry_mass_kg=3.0)
    curves = [run_detailed_ballistics(grain, motor, propellant, environment, max_step_size=step, max_time_points=1000)
              for step in (0.03, 0.01)]
    curves[1]["scenario_factors"] = {"liner_thickness_factor": 1.4, "initial_temperature_k": 260.0,
                                     "casing_strength_factor": 0.9, "drag_coefficient_factor": 1.1}
    return geometry, curves, propellant


def scalar(geometry, curve, propellant, casing, nozzle=None):
    return simulate_advanced_physics(geometry, curve, casing_material=casing, nozzle_material=nozzle,
                                     flame_temp_k=propellant.combustion_temperature, r_specific=propellant.products_constant)


def test_the_reference_backend_gives_exactly_what_the_scalar_function_gives(designs):
    geometry, curves, propellant = designs
    casing = CasingMaterial(liner_thickness_m=0.002)

    got = run_advanced_physics_ensemble(
        geometry, curves, casing_material=casing, nozzle_material=NozzleMaterial(),
        flame_temp_k=propellant.combustion_temperature, r_specific=propellant.products_constant, backend="cpu-reference",
    )

    assert got == [scalar(geometry, curve, propellant, casing, NozzleMaterial()) for curve in curves]


def test_the_numpy_backend_agrees_on_every_key_and_the_models_after_the_thermal_follow(designs):
    geometry, curves, propellant = designs
    casing = CasingMaterial(liner_thickness_m=0.002)
    timings, execution = {}, {}

    got = run_advanced_physics_ensemble(
        geometry, curves, casing_material=casing, flame_temp_k=propellant.combustion_temperature,
        r_specific=propellant.products_constant, backend="cpu-vectorized", timings=timings, execution=execution,
    )

    for lane, curve in enumerate(curves):
        want = scalar(geometry, curve, propellant, casing)
        assert set(got[lane]) == set(want) and len(want) == 74
        for key, value in want.items():
            assert got[lane][key] == pytest.approx(value, rel=1e-8, abs=1e-9), key
    assert set(timings) == {"pack_s", "thermal_s", "models_s"} and all(v >= 0.0 for v in timings.values())
    assert execution["effective_backend"] == "cpu-vectorized" and execution["lanes"] == 2


def test_the_scenario_factors_of_a_curve_change_the_thermal_metrics_like_the_scalar_function_says(designs):
    geometry, curves, propellant = designs
    plain = dict(curves[1])
    del plain["scenario_factors"]
    casing = CasingMaterial(liner_thickness_m=0.002)

    got = run_advanced_physics_ensemble(geometry, [curves[1], plain], casing_material=casing, backend="cpu-vectorized",
                                        flame_temp_k=propellant.combustion_temperature, r_specific=propellant.products_constant)

    key = "simulation.advanced.metadata.thermal_node_count"
    assert got[0][key] > got[1][key]  # a liner 1.4 times as thick has more cells
    assert got[0]["simulation.advanced.thermal.max_casing_temp_c"] != got[1]["simulation.advanced.thermal.max_casing_temp_c"]


def test_arguments_are_one_value_for_every_lane_or_one_per_lane(designs):
    geometry, curves, propellant = designs
    materials = [CasingMaterial(), CasingMaterial(liner_thickness_m=0.002)]

    got = run_advanced_physics_ensemble(
        [geometry, geometry], curves[0], casing_material=materials, flame_temp_k=[1500.0, 1700.0], r_specific=propellant.products_constant,
        gamma=[None, 1.2], backend="cpu-reference",
    )

    want = [simulate_advanced_physics(geometry, curves[0], casing_material=m, flame_temp_k=f, r_specific=propellant.products_constant,
                                      gamma=g) for m, f, g in zip(materials, [1500.0, 1700.0], [None, 1.2])]
    assert got == want


def test_the_casing_material_is_required_as_for_the_scalar_function(designs):
    geometry, curves, _ = designs

    with pytest.raises(ValueError, match="casing_material is required"):
        run_advanced_physics_ensemble(geometry, curves, backend="cpu-reference")


def test_an_empty_ensemble_gives_an_empty_list(designs):
    geometry, _, _ = designs

    assert run_advanced_physics_ensemble(geometry, [], casing_material=CasingMaterial(), backend="cpu-reference") == []


def test_the_reference_backend_accepts_a_process_pool(designs):
    geometry, curves, propellant = designs
    casing = CasingMaterial()

    got = run_advanced_physics_ensemble(geometry, curves, casing_material=casing, flame_temp_k=propellant.combustion_temperature,
                                        r_specific=propellant.products_constant, backend="cpu-reference", workers=2)

    assert got == [scalar(geometry, curve, propellant, casing) for curve in curves]
