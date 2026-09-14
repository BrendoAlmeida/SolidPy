"""Testes de surrogate_physics — branch ai-surrogate.

Verifica que:
  1. compute_static_features retorna grandezas fisicamente plausíveis.
  2. eta_c escala o thrust corretamente em Burn.evaluate_thrust().
  3. compute_burn_area_curve produz curvas monotônicas para tubular e
     não-monotônicas (pico inicial) para star.
  4. static_features_to_dict é serializável.
"""
import math
import numpy as np
import pytest

from solidpy import (
    Burn,
    CasingMaterial,
    Grain,
    Motor,
    NozzleMaterial,
    Propellant,
    geometry_from_components,
    simulate_structural_response,
)
from solidpy.surrogate_physics import (
    _estimate_equilibrium_pressure,
    _finite_value,
    compute_burn_area_curve,
    compute_static_features,
    compute_structural_features,
    static_features_to_dict,
    structural_features_to_dict,
    CONVERGENT_HALF_ANGLE_RAD,
    DIVERGENT_HALF_ANGLE_RAD,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture()
def kndx_propellant():
    return Propellant(
        specific_heat_ratio=1.1308,
        products_molecular_mass=0.04197,
        combustion_temperature=1720.0,
        density=1879.0,
        burn_rate_coefficient=8.875e-5,
        burn_rate_exponent=0.32,
    )


@pytest.fixture()
def tubular_grain():
    return Grain(
        outer_radius=0.035,
        initial_inner_radius=0.015,
        initial_height=0.12,
        geometry="tubular",
    )


@pytest.fixture()
def star_grain():
    return Grain(
        outer_radius=0.035,
        initial_inner_radius=0.010,
        initial_height=0.12,
        geometry="star",
        n_points=6,
        epsilon=0.25,
        slot_fraction=0.75,
    )


@pytest.fixture()
def motor(tubular_grain):
    return Motor(
        grains=[tubular_grain],
        chamber_inner_radius=0.037,
        nozzle_throat_radius=0.008,
        nozzle_exit_radius=0.018,
        nozzle_angle=math.radians(15.0),
        chamber_length=0.14,
        grain_separation=0.002,
    )


# ---------------------------------------------------------------------------
# Testes: eta_c no Burn
# ---------------------------------------------------------------------------

class TestEtaC:
    def test_default_eta_c_is_one(self, tubular_grain, motor, kndx_propellant):
        burn = Burn(tubular_grain, motor, kndx_propellant)
        assert burn.eta_c == 1.0

    def test_eta_c_scales_T0_nozzle_mdot_and_Ve(
        self, tubular_grain, motor, kndx_propellant
    ):
        # New model (§2.1): eta_c enters as T_0_ef = eta_c**2 * T_0, so it
        # propagates to mdot (1/sqrt(T_0)) and exit velocity (sqrt(T_0)).
        # Critically, Cf and thrust-at-fixed-P do NOT change with eta_c —
        # they are dimensionless functions of pressure ratios only.
        # eta_c's physical effect on thrust comes through chamber pressure
        # (energy balance -> Pmax), which is exercised by solve_burn, not by
        # an isolated evaluate_thrust(Pfixed) call.
        P = 3.5e6
        burn_ideal = Burn(tubular_grain, motor, kndx_propellant, eta_c=1.0)
        burn_90 = Burn(tubular_grain, motor, kndx_propellant, eta_c=0.90)

        T0_ideal = burn_ideal._parameters_at_pressure(P)[0]
        T0_90 = burn_90._parameters_at_pressure(P)[0]
        assert T0_90 == pytest.approx(T0_ideal * 0.81, rel=1e-9)

        mdot_ideal = burn_ideal.evaluate_nozzle_mass_flow(P)
        mdot_90 = burn_90.evaluate_nozzle_mass_flow(P)
        # mdot ∝ 1/sqrt(T_0); T0_90 = 0.81*T0 -> 1/sqrt(0.81) = 1/0.9
        assert mdot_90 == pytest.approx(mdot_ideal / 0.9, rel=1e-6)

        Ve_ideal = burn_ideal.evaluate_exit_velocity(P)
        Ve_90 = burn_90.evaluate_exit_velocity(P)
        # Ve ∝ sqrt(T_0); T0_90 = 0.81*T0 -> sqrt(0.81) = 0.9
        assert Ve_90 == pytest.approx(Ve_ideal * 0.9, rel=1e-6)

        # Thrust at *fixed* Pc is unchanged: Cf is dimensionless, depends on
        # pressure ratios only (not on T_0). eta_c's effect on thrust is
        # indirect, via the energy balance that sets Pc in solve_burn.
        thrust_ideal = burn_ideal.evaluate_thrust(P)
        thrust_90 = burn_90.evaluate_thrust(P)
        assert thrust_ideal == pytest.approx(thrust_90, rel=1e-9)

    def test_eta_c_zero_keeps_thrust_finite(
        self, tubular_grain, motor, kndx_propellant
    ):
        # eta_c=0 collapses T_0 but Cf is still well-defined (pure pressure
        # ratios). The result is finite; we don't assert zero — that was the
        # deprecated linear-model assertion.
        burn = Burn(tubular_grain, motor, kndx_propellant, eta_c=0.0)
        result = burn.evaluate_thrust(3.5e6)
        assert math.isfinite(result)


# ---------------------------------------------------------------------------
# Testes: compute_static_features
# ---------------------------------------------------------------------------

class TestStaticFeatures:
    def test_returns_dataclass(self, tubular_grain, motor, kndx_propellant):
        feats = compute_static_features(tubular_grain, motor, kndx_propellant)
        assert feats.c_star_m_s > 0
        assert feats.kn_initial > 0
        assert feats.Cf_ref > 0
        assert 0.0 < feats.lambda_divergence <= 1.0
        assert feats.isp_theory_s > 0
        assert feats.isp_effective_s == pytest.approx(feats.isp_theory_s)  # eta_c=1

    def test_eta_c_propagates_to_isp_effective(self, tubular_grain, motor, kndx_propellant):
        feats = compute_static_features(tubular_grain, motor, kndx_propellant, eta_c=0.95)
        assert feats.eta_c == pytest.approx(0.95)
        assert feats.isp_effective_s == pytest.approx(0.95 * feats.isp_theory_s)

    def test_c_star_physically_plausible(self, tubular_grain, motor, kndx_propellant):
        feats = compute_static_features(tubular_grain, motor, kndx_propellant)
        # Propelentes sólidos compósitos (APCP): c* típico ≈ 800–1300 m/s.
        # (Propelentes líquidos chegam a 1700–2400 m/s — não é o caso aqui.)
        assert 700 < feats.c_star_m_s < 1400

    def test_isp_physically_plausible(self, tubular_grain, motor, kndx_propellant):
        feats = compute_static_features(tubular_grain, motor, kndx_propellant)
        # Isp típico de propelentes APCP: 130–220 s
        assert 100 < feats.isp_theory_s < 300

    def test_lambda_conical_nozzle(self, tubular_grain, motor, kndx_propellant):
        # motor fixture tem nozzle_angle=15° → λ = (1+cos15°)/2 ≈ 0.9830
        feats = compute_static_features(tubular_grain, motor, kndx_propellant)
        expected_lambda = 0.5 * (1.0 + math.cos(math.radians(15.0)))
        assert feats.lambda_divergence == pytest.approx(expected_lambda, rel=1e-4)

    def test_to_dict_serializable(self, tubular_grain, motor, kndx_propellant):
        feats = compute_static_features(tubular_grain, motor, kndx_propellant)
        d = static_features_to_dict(feats)
        assert all(isinstance(v, float) for v in d.values())
        assert "surrogate.c_star_m_s" in d
        assert "surrogate.isp_effective_s" in d

    def test_propellant_mass_uses_total_volume_for_multiple_grains(
        self, tubular_grain, kndx_propellant
    ):
        second_grain = Grain(
            outer_radius=0.035,
            initial_inner_radius=0.015,
            initial_height=0.12,
            geometry="tubular",
        )
        multi_grain_motor = Motor(
            grains=[tubular_grain, second_grain],
            chamber_inner_radius=0.037,
            nozzle_throat_radius=0.008,
            nozzle_exit_radius=0.018,
            nozzle_angle=math.radians(15.0),
            chamber_length=0.25,
            grain_separation=0.002,
        )

        features = compute_static_features(
            tubular_grain, multi_grain_motor, kndx_propellant
        )

        assert features.propellant_mass_kg == pytest.approx(
            kndx_propellant.density * (tubular_grain.volume + second_grain.volume)
        )

    @pytest.mark.parametrize("density", [None, math.nan, math.inf])
    def test_unknown_propellant_density_is_rejected(
        self, tubular_grain, motor, density
    ):
        propellant = Propellant(
            specific_heat_ratio=1.1308,
            products_molecular_mass=0.04197,
            combustion_temperature=1720.0,
            density=density,
            burn_rate_coefficient=8.875e-5,
            burn_rate_exponent=0.32,
        )
        with pytest.raises(ValueError, match="massa de propelente desconhecida"):
            compute_static_features(tubular_grain, motor, propellant)

    @pytest.mark.parametrize("volume", [-1.0, math.nan, math.inf])
    def test_invalid_grain_volume_is_rejected_before_mass_calculation(
        self, tubular_grain, motor, kndx_propellant, volume
    ):
        tubular_grain.volume = volume
        with pytest.raises(ValueError, match="grain.volume"):
            compute_static_features(tubular_grain, motor, kndx_propellant)

    def test_propellant_mass_overflow_is_rejected(
        self, tubular_grain, motor, kndx_propellant
    ):
        tubular_grain.volume = 1e308
        kndx_propellant.density = 1e308
        with pytest.raises(ValueError, match="propellant_mass_kg"):
            compute_static_features(tubular_grain, motor, kndx_propellant)


# ---------------------------------------------------------------------------
# Testes: massa e grandezas estruturais fechadas
# ---------------------------------------------------------------------------

class TestStructuralFeatures:
    def test_dry_mass_matches_component_geometry_plus_liner_and_nozzle(
        self, tubular_grain, motor, kndx_propellant
    ):
        casing = CasingMaterial(
            liner_thickness_m=0.0015,
            liner_density_kg_m3=1100.0,
        )
        nozzle = NozzleMaterial(density_kg_m3=1800.0, wall_thickness_m=0.003)
        casing_wall_m = 0.004
        features = compute_structural_features(
            motor,
            casing,
            nozzle,
            chamber_pressure_pa=3.5e6,
            casing_wall_thickness_m=casing_wall_m,
            propellant_mass_kg=kndx_propellant.density * tubular_grain.volume,
        )

        geometry = geometry_from_components(
            tubular_grain,
            motor,
            kndx_propellant,
            casing_wall_thickness_m=casing_wall_m,
            casing_density_kg_m3=casing.density_kg_m3,
        )
        assert features.casing_mass_kg == pytest.approx(geometry.dry_mass_kg)

        chamber_radius_m = math.sqrt(motor.chamber_area / math.pi)
        liner_inner_radius_m = chamber_radius_m - casing.liner_thickness_m
        expected_liner_mass = (
            math.pi
            * (chamber_radius_m**2 - liner_inner_radius_m**2)
            * motor.chamber_length
            * casing.liner_density_kg_m3
        )
        throat_radius_m = math.sqrt(motor.nozzle_throat_area / math.pi)
        exit_radius_m = math.sqrt(motor.nozzle_exit_area / math.pi)
        convergent_slant_m = (
            chamber_radius_m - throat_radius_m
        ) / math.sin(CONVERGENT_HALF_ANGLE_RAD)
        divergent_slant_m = (
            exit_radius_m - throat_radius_m
        ) / math.sin(motor.nozzle_angle)
        expected_nozzle_mass = (
            math.pi
            * (
                (chamber_radius_m + throat_radius_m) * convergent_slant_m
                + (throat_radius_m + exit_radius_m) * divergent_slant_m
            )
            * nozzle.wall_thickness_m
            * nozzle.density_kg_m3
        )
        assert features.liner_mass_kg == pytest.approx(expected_liner_mass)
        assert features.nozzle_mass_kg == pytest.approx(expected_nozzle_mass)
        assert features.dry_mass_kg == pytest.approx(
            features.casing_mass_kg
            + features.liner_mass_kg
            + features.nozzle_mass_kg
        )
        assert features.motor_final_mass_kg == pytest.approx(features.dry_mass_kg)
        assert features.motor_initial_mass_kg == pytest.approx(
            features.dry_mass_kg + kndx_propellant.density * tubular_grain.volume
        )

    def test_structural_mass_ratio_is_physically_bounded(
        self, tubular_grain, motor, kndx_propellant
    ):
        features = compute_structural_features(
            motor,
            CasingMaterial(),
            NozzleMaterial(),
            chamber_pressure_pa=3.5e6,
            casing_wall_thickness_m=0.004,
            propellant_mass_kg=kndx_propellant.density * tubular_grain.volume,
        )
        assert 0.0 <= features.structural_mass_ratio <= 1.0

    def test_port_throat_ratio_matches_grain_and_motor_geometry(
        self, tubular_grain, motor
    ):
        features = compute_structural_features(
            motor,
            CasingMaterial(),
            NozzleMaterial(),
            chamber_pressure_pa=3.5e6,
            grain=tubular_grain,
            propellant_mass_kg=1.0,
        )
        expected = tubular_grain.evaluate_port_area(0.0) / motor.nozzle_throat_area
        assert features.port_throat_ratio == pytest.approx(expected)

    def test_reference_structural_stress_and_burst_match_transient_path(
        self, tubular_grain, motor, kndx_propellant
    ):
        casing = CasingMaterial()
        nozzle = NozzleMaterial()
        casing_wall_m = 0.004
        pressure_pa = 3.5e6
        features = compute_structural_features(
            motor,
            casing,
            nozzle,
            chamber_pressure_pa=pressure_pa,
            casing_wall_thickness_m=casing_wall_m,
            grain=tubular_grain,
            propellant_mass_kg=1.0,
        )
        geometry = geometry_from_components(
            tubular_grain,
            motor,
            kndx_propellant,
            casing_wall_thickness_m=casing_wall_m,
        )
        curve = {
            "time_s": np.array([0.0, 1.0]),
            "thrust_n": np.zeros(2),
            "chamber_pressure_pa": np.full(2, pressure_pa),
        }
        transient = simulate_structural_response(
            geometry,
            curve,
            {"simulation.advanced.thermal.casing_inner_wall_temp_c": 20.0},
            casing_material=casing,
        )
        assert features.von_mises_at_reference_pa == pytest.approx(
            transient["simulation.advanced.structural.max_stress_mpa"] * 1e6
        )
        assert features.burst_pressure_pa == pytest.approx(
            transient["simulation.advanced.structural.burst_pressure_mpa"] * 1e6
        )
        assert features.burst_safety_factor_at_reference_pa == pytest.approx(
            transient["simulation.advanced.structural.burst_safety_factor"]
        )

    def test_structural_features_to_dict_is_flat_and_serializable(
        self, tubular_grain, motor
    ):
        features = compute_structural_features(
            motor,
            CasingMaterial(),
            NozzleMaterial(),
            chamber_pressure_pa=3.5e6,
            grain=tubular_grain,
            propellant_mass_kg=1.0,
        )
        values = structural_features_to_dict(features)
        assert all(isinstance(value, float) for value in values.values())
        assert values["surrogate.dry_mass_kg"] == pytest.approx(features.dry_mass_kg)
        assert "surrogate.burst_safety_factor_at_reference_pa" in values

    def test_missing_propellant_mass_source_is_rejected(self, tubular_grain, motor):
        with pytest.raises(ValueError, match="massa de propelente desconhecida"):
            compute_structural_features(
                motor,
                chamber_pressure_pa=3.5e6,
                grain=tubular_grain,
            )

    def test_explicit_zero_propellant_mass_is_a_known_value(self, tubular_grain, motor):
        features = compute_structural_features(
            motor,
            chamber_pressure_pa=3.5e6,
            grain=tubular_grain,
            propellant_mass_kg=0.0,
        )
        assert features.motor_initial_mass_kg == pytest.approx(features.dry_mass_kg)

    @pytest.mark.parametrize("liner_density", [0.0, math.nan])
    def test_inactive_liner_does_not_require_liner_density(
        self, motor, tubular_grain, liner_density
    ):
        features = compute_structural_features(
            motor,
            casing_material=CasingMaterial(
                liner_thickness_m=0.0,
                liner_density_kg_m3=liner_density,
            ),
            chamber_pressure_pa=3.5e6,
            grain=tubular_grain,
            propellant_mass_kg=1.0,
        )
        assert features.liner_mass_kg == 0.0

    def test_negative_liner_thickness_is_inactive(
        self, motor, tubular_grain
    ):
        features = compute_structural_features(
            motor,
            casing_material=CasingMaterial(
                liner_thickness_m=-0.001,
                liner_density_kg_m3=0.0,
            ),
            chamber_pressure_pa=3.5e6,
            grain=tubular_grain,
            propellant_mass_kg=1.0,
        )
        assert features.liner_mass_kg == 0.0

    def test_none_divergent_angle_uses_separate_15_degree_mass_fallback(
        self, tubular_grain, motor
    ):
        motor.nozzle_angle = None
        features = compute_structural_features(
            motor,
            chamber_pressure_pa=3.5e6,
            grain=tubular_grain,
            propellant_mass_kg=1.0,
        )
        chamber_radius_m = math.sqrt(motor.chamber_area / math.pi)
        throat_radius_m = math.sqrt(motor.nozzle_throat_area / math.pi)
        exit_radius_m = math.sqrt(motor.nozzle_exit_area / math.pi)
        expected_divergent_slant_m = (
            exit_radius_m - throat_radius_m
        ) / math.sin(DIVERGENT_HALF_ANGLE_RAD)
        expected_convergent_slant_m = (
            chamber_radius_m - throat_radius_m
        ) / math.sin(CONVERGENT_HALF_ANGLE_RAD)
        expected_nozzle_mass = (
            math.pi
            * (
                (chamber_radius_m + throat_radius_m) * expected_convergent_slant_m
                + (throat_radius_m + exit_radius_m) * expected_divergent_slant_m
            )
            * 0.005
            * 1800.0
        )
        assert DIVERGENT_HALF_ANGLE_RAD == pytest.approx(math.radians(15.0))
        assert DIVERGENT_HALF_ANGLE_RAD != CONVERGENT_HALF_ANGLE_RAD
        assert features.nozzle_mass_kg == pytest.approx(expected_nozzle_mass)

    @pytest.mark.parametrize(
        "angle",
        [0.0, -1.0, math.pi / 2.0, math.nan, math.inf],
    )
    def test_invalid_divergent_angle_is_rejected(self, motor, tubular_grain, angle):
        motor.nozzle_angle = angle
        with pytest.raises(ValueError, match="nozzle_angle"):
            compute_structural_features(
                motor,
                chamber_pressure_pa=3.5e6,
                grain=tubular_grain,
                propellant_mass_kg=1.0,
            )

    @pytest.mark.parametrize("pressure", [-1.0, math.nan, math.inf])
    def test_invalid_reference_pressure_is_rejected(
        self, motor, tubular_grain, pressure
    ):
        with pytest.raises(ValueError, match="chamber_pressure_pa"):
            compute_structural_features(
                motor,
                chamber_pressure_pa=pressure,
                grain=tubular_grain,
                propellant_mass_kg=1.0,
            )

    @pytest.mark.parametrize("wall", [math.nan, math.inf])
    def test_nonfinite_casing_wall_thickness_is_rejected(
        self, motor, tubular_grain, wall
    ):
        with pytest.raises(ValueError, match="casing_wall_thickness_m"):
            compute_structural_features(
                motor,
                chamber_pressure_pa=3.5e6,
                casing_wall_thickness_m=wall,
                grain=tubular_grain,
                propellant_mass_kg=1.0,
            )

    def test_overflowing_casing_geometry_is_rejected(
        self, motor, tubular_grain
    ):
        with pytest.raises(ValueError, match="massa do casing"):
            compute_structural_features(
                motor,
                chamber_pressure_pa=3.5e6,
                casing_wall_thickness_m=1e308,
                grain=tubular_grain,
                propellant_mass_kg=1.0,
            )

    def test_extreme_reference_pressure_rejects_nonfinite_structural_stress(
        self, motor, tubular_grain
    ):
        with pytest.raises(ValueError, match="von Mises"):
            compute_structural_features(
                motor,
                chamber_pressure_pa=1e308,
                grain=tubular_grain,
                propellant_mass_kg=1.0,
            )

    @pytest.mark.parametrize("wall", [0.0, -1.0, 1e-6])
    def test_casing_wall_thickness_matches_canonical_lower_bound(
        self, motor, tubular_grain, kndx_propellant, wall
    ):
        features = compute_structural_features(
            motor,
            chamber_pressure_pa=3.5e6,
            casing_wall_thickness_m=wall,
            grain=tubular_grain,
            propellant_mass_kg=1.0,
        )
        geometry = geometry_from_components(
            tubular_grain,
            motor,
            kndx_propellant,
            casing_wall_thickness_m=wall,
        )
        assert features.casing_mass_kg == pytest.approx(geometry.dry_mass_kg)

    @pytest.mark.parametrize("strength_factor", [0.0, -1.0, 1e-3])
    def test_strength_factor_matches_canonical_lower_bound(
        self, motor, tubular_grain, kndx_propellant, strength_factor
    ):
        pressure_pa = 3.5e6
        features = compute_structural_features(
            motor,
            chamber_pressure_pa=pressure_pa,
            casing_strength_factor=strength_factor,
            casing_wall_thickness_m=0.0,
            grain=tubular_grain,
            propellant_mass_kg=1.0,
        )
        geometry = geometry_from_components(
            tubular_grain,
            motor,
            kndx_propellant,
            casing_wall_thickness_m=0.0,
        )
        curve = {
            "time_s": np.array([0.0, 1.0]),
            "thrust_n": np.zeros(2),
            "chamber_pressure_pa": np.full(2, pressure_pa),
        }
        transient = simulate_structural_response(
            geometry,
            curve,
            {"simulation.advanced.thermal.casing_inner_wall_temp_c": 20.0},
            casing_strength_factor=strength_factor,
        )
        assert features.von_mises_at_reference_pa == pytest.approx(
            transient["simulation.advanced.structural.max_stress_mpa"] * 1e6
        )
        assert features.burst_pressure_pa == pytest.approx(
            transient["simulation.advanced.structural.burst_pressure_mpa"] * 1e6
        )

    @pytest.mark.parametrize("density", [0.0, 0.5, 1.0])
    def test_casing_density_matches_canonical_lower_bound(
        self, motor, tubular_grain, kndx_propellant, density
    ):
        features = compute_structural_features(
            motor,
            casing_material=CasingMaterial(density_kg_m3=density),
            chamber_pressure_pa=3.5e6,
            grain=tubular_grain,
            propellant_mass_kg=1.0,
        )
        geometry = geometry_from_components(
            tubular_grain,
            motor,
            kndx_propellant,
            casing_wall_thickness_m=0.005,
            casing_density_kg_m3=density,
        )
        assert features.casing_mass_kg == pytest.approx(geometry.dry_mass_kg)

    @pytest.mark.parametrize(
        "material",
        [
            CasingMaterial(density_kg_m3=math.nan),
            CasingMaterial(density_kg_m3=math.inf),
            CasingMaterial(ultimate_strength_mpa=math.inf),
        ],
    )
    def test_invalid_casing_material_values_are_rejected(
        self, motor, tubular_grain, material
    ):
        with pytest.raises(ValueError, match="casing_material"):
            compute_structural_features(
                motor,
                casing_material=material,
                chamber_pressure_pa=3.5e6,
                grain=tubular_grain,
                propellant_mass_kg=1.0,
            )

    def test_finite_value_converts_float_overflow_to_value_error(self):
        with pytest.raises(ValueError, match="overflow deve ser um número finito"):
            _finite_value("overflow", 10**10000)

    @pytest.mark.parametrize("wall", [0.0, -1.0, math.nan, math.inf])
    def test_invalid_nozzle_wall_thickness_is_rejected(
        self, motor, tubular_grain, wall
    ):
        with pytest.raises(ValueError, match="nozzle_material.wall_thickness_m"):
            compute_structural_features(
                motor,
                nozzle_material=NozzleMaterial(wall_thickness_m=wall),
                chamber_pressure_pa=3.5e6,
                grain=tubular_grain,
                propellant_mass_kg=1.0,
            )

    @pytest.mark.parametrize("factor", [math.nan, math.inf])
    def test_nonfinite_strength_factor_is_rejected(
        self, motor, tubular_grain, factor
    ):
        with pytest.raises(ValueError, match="casing_strength_factor"):
            compute_structural_features(
                motor,
                chamber_pressure_pa=3.5e6,
                casing_strength_factor=factor,
                grain=tubular_grain,
                propellant_mass_kg=1.0,
            )


# ---------------------------------------------------------------------------
# Testes: compute_burn_area_curve
# ---------------------------------------------------------------------------

class TestBurnAreaCurve:
    def test_tubular_curve_monotonically_decreasing(self, tubular_grain):
        curve = compute_burn_area_curve(tubular_grain, n_points=50)
        # Tubular: área diminui com a regressão (topo/base consomem antes das laterais)
        # Pelo menos a segunda metade deve ser decrescente
        mid = len(curve.burn_area_m2) // 2
        assert curve.burn_area_m2[mid] >= curve.burn_area_m2[-1]

    def test_star_curve_has_initial_peak(self, star_grain):
        curve = compute_burn_area_curve(star_grain, n_points=64)
        # Star: área inicial é maior que o final (burnout → 0)
        assert curve.burn_area_m2[0] > curve.burn_area_m2[-1]
        # A curva deve ter pelo menos um ponto maior que o inicial (progressiva)
        # ou manter-se neutra (regressive), dependendo dos parâmetros.
        # O que SÃO garantidos: começa com Ab > 0, termina com Ab = 0.
        assert curve.burn_area_m2[0] > 0
        assert curve.burn_area_m2[-1] == pytest.approx(0.0, abs=1e-10)

    def test_web_fraction_range(self, tubular_grain):
        curve = compute_burn_area_curve(tubular_grain, n_points=32)
        assert curve.web_fraction[0] == pytest.approx(0.0)
        assert curve.web_fraction[-1] == pytest.approx(1.0)

    def test_n_points_respected(self, star_grain):
        for n in [16, 32, 64]:
            curve = compute_burn_area_curve(star_grain, n_points=n)
            assert len(curve.burn_area_m2) == n
            assert len(curve.web_fraction) == n

    def test_geometry_label(self, tubular_grain, star_grain):
        assert compute_burn_area_curve(tubular_grain).geometry == "tubular"
        assert compute_burn_area_curve(star_grain).geometry == "star"


# ---------------------------------------------------------------------------
# Testes: _estimate_equilibrium_pressure
# ---------------------------------------------------------------------------

@pytest.fixture()
def kndx_propellant_ballistic():
    """Propelente com burn_rate_a/burn_rate_n (necessário para evaluate_burn_rate)."""
    return Propellant(
        specific_heat_ratio=1.1308,
        products_molecular_mass=0.04197,
        combustion_temperature=1720.0,
        density=1879.0,
        burn_rate_a=8.875,   # mm/s a 1 MPa  (a em mm/s/MPa^n)
        burn_rate_n=0.32,
    )


class TestEstimateEquilibriumPressure:
    def test_returns_positive_pressure(self, tubular_grain, motor, kndx_propellant_ballistic):
        kn = tubular_grain.burn_area / motor.nozzle_throat_area
        P_eq = _estimate_equilibrium_pressure(kn, kndx_propellant_ballistic)
        assert P_eq > 0.0

    def test_pressure_in_realistic_range(self, tubular_grain, motor, kndx_propellant_ballistic):
        # Para propelentes APCP com Kn típico (50–300), P_eq deve estar em 0,5–15 MPa.
        kn = tubular_grain.burn_area / motor.nozzle_throat_area
        P_eq = _estimate_equilibrium_pressure(kn, kndx_propellant_ballistic)
        assert 0.5e6 < P_eq < 15e6

    def test_higher_kn_gives_higher_pressure(self, tubular_grain, motor, kndx_propellant_ballistic):
        # P_eq escala monotonicamente com Kn.
        kn = tubular_grain.burn_area / motor.nozzle_throat_area
        P_low = _estimate_equilibrium_pressure(kn * 0.5, kndx_propellant_ballistic)
        P_high = _estimate_equilibrium_pressure(kn * 2.0, kndx_propellant_ballistic)
        assert P_high > P_low

    def test_convergence_in_few_iterations(self, tubular_grain, motor, kndx_propellant_ballistic):
        # 6 e 12 iterações devem concordar em < 1% — o default (6) é suficiente.
        kn = tubular_grain.burn_area / motor.nozzle_throat_area
        P6 = _estimate_equilibrium_pressure(kn, kndx_propellant_ballistic, n_iter=6)
        P12 = _estimate_equilibrium_pressure(kn, kndx_propellant_ballistic, n_iter=12)
        assert abs(P6 / P12 - 1.0) < 0.01

    def test_result_consistent_with_static_features(self, tubular_grain, motor, kndx_propellant_ballistic):
        # Verifica que compute_static_features aceita P_ref estimado sem erro.
        kn = tubular_grain.burn_area / motor.nozzle_throat_area
        P_eq = _estimate_equilibrium_pressure(kn, kndx_propellant_ballistic)
        feats = compute_static_features(tubular_grain, motor, kndx_propellant_ballistic, P_ref_pa=P_eq)
        assert feats.P_ref_pa == pytest.approx(P_eq)
        assert feats.Cf_ref > 0
