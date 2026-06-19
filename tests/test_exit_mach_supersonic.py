# -*- coding: utf-8 -*-
"""Regression tests for evaluate_exit_mach supersonic root-finding.

Prior to the brentq fix, fsolve with x0=2 silently failed to converge for
high expansion ratios combined with low specific-heat ratio (e.g. KNSB,
γ≈1.042, ε≳16), returning ~2.0 with a residual of ~-20 instead of the true
supersonic Mach number.  These tests assert the solver now converges to the
correct supersonic root across a representative grid of (γ, ε) pairs,
including the exact failing case that triggered the bug.
"""

import math

import numpy as np
import pytest

from solidpy import Environment, Grain, Motor, Propellant
from solidpy.Burn import Burn


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _area_ratio_residual(k, expansion_ratio, mach):
    """Return |A/A*(M) - ε| (should be ~0 at the correct supersonic root)."""
    return abs(
        math.pow((k + 1) / 2, -(k + 1) / (2 * (k - 1)))
        * math.pow((1 + (k - 1) / 2 * mach**2), (k + 1) / (2 * (k - 1)))
        / mach
        - expansion_ratio
    )


def _make_burn(expansion_ratio, k=1.042):
    """Construct a minimal Burn instance with the requested expansion ratio."""
    # Throat radius 17.5 mm; exit radius chosen so (r_exit/r_throat)^2 == ε.
    throat_r = 17.5e-3
    exit_r = throat_r * math.sqrt(expansion_ratio)

    grain = Grain(
        outer_radius=71.92e-3 / 2,
        initial_inner_radius=31.92e-3 / 2,
    )
    motor = Motor(
        grain,
        grain_number=4,
        chamber_inner_radius=77.92e-3 / 2,
        nozzle_throat_radius=throat_r,
        nozzle_exit_radius=exit_r,
        chamber_length=600e-3,
    )
    propellant = Propellant(
        specific_heat_ratio=k,
        density=1700,
        products_molecular_mass=39.9e-3,
        combustion_temperature=1600,
        burn_rate_a=5.13,
        burn_rate_n=0.22,
    )
    environment = Environment()
    return Burn(grain, motor, propellant, environment)


# ---------------------------------------------------------------------------
# Parametric tests
# ---------------------------------------------------------------------------

GAMMA_VALUES = [1.042, 1.13, 1.20]
EXPANSION_RATIOS = [3, 5, 10, 15, 18, 22, 25]

RESIDUAL_TOL = 1e-6  # tight: brentq should converge to < 1e-10


@pytest.mark.parametrize("k", GAMMA_VALUES)
@pytest.mark.parametrize("eps", EXPANSION_RATIOS)
def test_supersonic_root_is_above_unity(k, eps):
    """Solved exit Mach must be > 1 (supersonic branch)."""
    burn = _make_burn(expansion_ratio=eps, k=k)
    mach = burn.evaluate_exit_mach()
    assert mach > 1.0, (
        f"k={k}, ε={eps}: exit Mach {mach:.4f} is not supersonic (≤ 1)."
    )


@pytest.mark.parametrize("k", GAMMA_VALUES)
@pytest.mark.parametrize("eps", EXPANSION_RATIOS)
def test_supersonic_root_satisfies_area_mach_equation(k, eps):
    """Solved exit Mach must satisfy the area–Mach relation to tight tolerance."""
    burn = _make_burn(expansion_ratio=eps, k=k)
    mach = burn.evaluate_exit_mach()
    residual = _area_ratio_residual(k, eps, mach)
    assert residual < RESIDUAL_TOL, (
        f"k={k}, ε={eps}: area-Mach residual {residual:.3e} exceeds {RESIDUAL_TOL}. "
        f"Exit Mach returned: {mach:.6f} (should be far from 2.0 for high ε)."
    )


def test_knsb_high_expansion_ratio_bug_case():
    """Exact reproduction case from the bug report: k=1.042, ε=21.92.

    The old fsolve(func, 2) code returned M≈2.0 with residual≈-19.8.
    The correct supersonic root is M≈3.209 (residual≈0).
    """
    k = 1.042
    eps = 21.91964757599294  # exact value from the failing production dataset

    burn = _make_burn(expansion_ratio=eps, k=k)
    mach = burn.evaluate_exit_mach()

    # Must NOT be stuck near the initial guess (old fsolve returned ~2.0).
    assert mach > 2.5, (
        f"Exit Mach {mach:.4f} is suspiciously close to the old fsolve initial "
        "guess of 2.0 — solver likely did not converge."
    )
    # Must satisfy the area-Mach equation tightly.
    residual = _area_ratio_residual(k, eps, mach)
    assert residual < RESIDUAL_TOL, (
        f"area-Mach residual {residual:.3e} is too large; expected ~ 0. "
        f"Exit Mach: {mach:.6f}."
    )
    # Cross-check: true root is known to be ~3.209.
    assert abs(mach - 3.2087) < 0.01, (
        f"Exit Mach {mach:.4f} differs from expected ~3.2087 by more than 0.01."
    )


def test_exit_mach_cache_returns_same_value(k=1.042, eps=22):
    """Cached lookup must return the identical converged value."""
    burn = _make_burn(expansion_ratio=eps, k=k)
    first = burn.evaluate_exit_mach()
    second = burn.evaluate_exit_mach()
    assert first == second


def test_exit_mach_physical_range_for_standard_motors():
    """Sanity check: Leviata motor (ε≈6.45, γ≈1.14) gives a sensible Mach."""
    # Leviata: throat_r=17.5 mm, exit_r=44.44 mm → ε=(44.44/17.5)^2 ≈ 6.45
    throat_r = 17.5e-3
    exit_r = 44.44e-3
    eps = (exit_r / throat_r) ** 2

    grain = Grain(
        outer_radius=71.92e-3 / 2,
        initial_inner_radius=31.92e-3 / 2,
    )
    motor = Motor(
        grain,
        grain_number=4,
        chamber_inner_radius=77.92e-3 / 2,
        nozzle_throat_radius=throat_r,
        nozzle_exit_radius=exit_r,
        nozzle_angle=15 * math.pi / 180,
        chamber_length=600e-3,
    )
    propellant = Propellant(
        specific_heat_ratio=1.1361,
        density=1700,
        products_molecular_mass=39.9e-3,
        combustion_temperature=1600,
        interpolation_list="data/burnrate/KNSB3.csv",
    )
    environment = Environment()
    burn = Burn(grain, motor, propellant, environment)

    mach = burn.evaluate_exit_mach()
    assert 2.0 < mach < 4.0, (
        f"Leviata exit Mach {mach:.4f} is outside the expected range [2, 4]."
    )
    residual = _area_ratio_residual(1.1361, eps, mach)
    assert residual < RESIDUAL_TOL
