# -*- coding: utf-8 -*-
"""Regressão para calc_cstar(): expoente correto de c* isentrópico.

O expoente de Vandenkerckhove correto para c* e' (k+1)/(2*(k-1)), nao
(k+1)/(k-1) (esse ultimo dobra o expoente e superestima c* em ~70% pra
gamma tipico de APCP). Ver Sutton & Biblarz (2010) Eq. 3-32.
"""
import math

import pytest

from solidpy import Propellant

_R_UNIVERSAL = 8.314462618


def _reference_cstar(k: float, T0: float, molecular_mass: float) -> float:
    return math.sqrt(_R_UNIVERSAL * T0 / (molecular_mass * k)) * (
        (k + 1) / 2
    ) ** ((k + 1) / (2 * (k - 1)))


@pytest.mark.parametrize("k", [1.13, 1.2, 1.25, 1.35])
def test_calc_cstar_matches_vandenkerckhove_formula(k):
    T0 = 1720.0
    molecular_mass = 0.04197
    propellant = Propellant(
        specific_heat_ratio=k,
        products_molecular_mass=molecular_mass,
        combustion_temperature=T0,
        density=1800.0,
        burn_rate_a=5.0,
        burn_rate_n=0.3,
    )
    expected = _reference_cstar(k, T0, molecular_mass)
    assert propellant.cstar == pytest.approx(expected, rel=1e-9)


def test_calc_cstar_plausible_for_apcp():
    # Propelentes solidos compositos: c* tipico ~ 800-1300 m/s (Sutton & Biblarz).
    propellant = Propellant(
        specific_heat_ratio=1.1361,
        products_molecular_mass=39.9e-3,
        combustion_temperature=1520.0,
        density=1841.0,
        burn_rate_a=5.0,
        burn_rate_n=0.3,
    )
    assert 700.0 < propellant.cstar < 1400.0
