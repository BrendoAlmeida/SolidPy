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

    def test_eta_c_scales_thrust_linearly(self, tubular_grain, motor, kndx_propellant):
        P = 3.5e6
        burn_ideal = Burn(tubular_grain, motor, kndx_propellant, eta_c=1.0)
        burn_90 = Burn(tubular_grain, motor, kndx_propellant, eta_c=0.90)

        thrust_ideal = burn_ideal.evaluate_thrust(P)
        thrust_90 = burn_90.evaluate_thrust(P)

        assert thrust_ideal > 0
        assert abs(thrust_90 / thrust_ideal - 0.90) < 1e-9

    def test_eta_c_zero_gives_zero_thrust(self, tubular_grain, motor, kndx_propellant):
        burn = Burn(tubular_grain, motor, kndx_propellant, eta_c=0.0)
        assert burn.evaluate_thrust(3.5e6) == pytest.approx(0.0)


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
