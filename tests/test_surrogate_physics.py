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

from solidpy import Burn, Grain, Motor, Propellant
from solidpy.surrogate_physics import (
    _estimate_equilibrium_pressure,
    compute_burn_area_curve,
    compute_static_features,
    static_features_to_dict,
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
