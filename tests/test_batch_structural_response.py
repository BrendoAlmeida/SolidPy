import numpy as np
import pytest

from solidpy import CasingMaterial, MotorGeometry
from solidpy.Multiphysics import simulate_structural_response
from solidpy.batch.kernels.structural_response import structural_response_vectorized


def make_geometry(wall_thickness_m=0.004):
    return MotorGeometry(
        motor_length_m=0.24,
        motor_inner_diameter_m=0.08,
        casing_wall_thickness_m=wall_thickness_m,
        grain_outer_diameter_m=0.07,
        grain_core_diameter_m=0.03,
        grain_gap_m=0.0,
        grain_length_each_m=0.10,
        grain_number=2,
        fill_length_m=0.20,
        throat_diameter_m=0.012,
        exit_diameter_m=0.028,
        free_volume_m3=0.001,
        propellant_mass_kg=1.0,
        dry_mass_kg=3.0,
        motor_initial_mass_kg=4.0,
        motor_final_mass_kg=3.0,
    )


@pytest.mark.parametrize(
    "geometry,material,bolts,thermal",
    [
        (
            make_geometry(),
            CasingMaterial(),
            {},
            None,
        ),
        (
            make_geometry(0.006),
            CasingMaterial(),
            {"bolt_count": 4, "bolt_diameter_m": 0.006, "bolt_strength_mpa": 800.0},
            {"simulation.advanced.thermal.casing_inner_wall_temp_c": 120.0},
        ),
        (
            make_geometry(0.006),
            CasingMaterial(
                material_family="composite",
                allowable_stress_mpa=310.0,
                ultimate_strength_mpa=760.0,
            ),
            {"bolt_count": 4, "bolt_diameter_m": 0.006},
            {"simulation.advanced.thermal.casing_inner_wall_temp_c": 600.0},
        ),
        (
            make_geometry(),
            CasingMaterial(ultimate_strength_mpa=820.0),
            {
                "bolt_count": 4,
                "bolt_diameter_m": 0.006,
                "bolt_strength_mpa": 800.0,
                "closure_bolts_applicable": False,
            },
            None,
        ),
    ],
    ids=["metal-no-bolts", "metal-configured-bolts-thermal", "composite-unavailable-bolts", "bolts-not-applicable"],
)
def test_vectorized_structural_response_matches_smc_synthetic_curve(
    geometry, material, bolts, thermal
):
    chamber_pressure_pa = np.asarray([0.0, 1.0e3, 3.4e6, 15.0e6])
    casing_strength_factor = 0.83
    actual = structural_response_vectorized(
        geometry,
        chamber_pressure_pa,
        material,
        casing_strength_factor,
        thermal=thermal,
        **bolts,
    )

    expected = []
    for peak_pressure_pa in chamber_pressure_pa:
        curve = {
            "time_s": np.asarray([0.0, 0.001, 1.0]),
            "thrust_n": np.zeros(3),
            "chamber_pressure_pa": np.asarray([0.0, peak_pressure_pa, 0.0]),
        }
        expected.append(
            simulate_structural_response(
                geometry,
                curve,
                thermal,
                casing_material=material,
                casing_strength_factor=casing_strength_factor,
                **bolts,
            )
        )

    assert set(actual) == set(expected[0])
    for key, value in actual.items():
        reference = [result[key] for result in expected]
        if reference[0] is None or isinstance(reference[0], str):
            assert value == reference[0]
            assert all(item == reference[0] for item in reference)
        else:
            assert isinstance(value, np.ndarray), key
            assert value.shape == chamber_pressure_pa.shape, key
            np.testing.assert_allclose(
                value,
                reference,
                rtol=1e-12,
                atol=1e-12,
                err_msg=key,
            )


def test_vectorized_structural_response_matches_smc_pressure_integral():
    geometry = make_geometry()
    material = CasingMaterial()
    pressures = np.asarray([1.0e6, 8.0e6])
    result = structural_response_vectorized(geometry, pressures, material)

    # SMC's [0, 0.001, 1] history has pressure [0, peak, 0], so its
    # trapezoidal integral is 0.5 * peak Pa*s (reported in MPa*s).
    np.testing.assert_allclose(
        result["simulation.advanced.structural.pressurization_impulse_mpa_s"],
        0.5 * pressures / 1e6,
        rtol=0.0,
        atol=0.0,
    )
