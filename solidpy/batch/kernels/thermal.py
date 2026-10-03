# -*- coding: utf-8 -*-
"""Kernels of the wall conduction model of ``Multiphysics.simulate_thermal_ablation``.

The wall is a chain of finite-volume nodes whose only nonlinearity is the Bartz heat flux into the first
node, a function of that node's temperature. ``flux`` is that function and its derivative, written as the
closure ``heat_flux_and_derivative_from_hot_face`` of the scalar code is, over an array namespace ``xp``.
"""


def flux(xp, hot_face_k, bartz_base, recovery_temp_k, flame_temp_k, stagnation_factor):
    """Heat flux into the hot face [W/m2] and its derivative with respect to the hot-face temperature.

    ``bartz_base`` is the temperature-independent part of the Bartz coefficient of the time step. A hot face at or
    above the recovery temperature receives nothing, and a ``sigma`` of at most 0.1 is replaced by 0.1, as in the
    scalar code.
    """
    ratio = hot_face_k / xp.maximum(flame_temp_k, 1.0)
    base = 0.5 * ratio / xp.maximum(stagnation_factor, 1e-9) + 0.5
    safe_base = xp.where(base > 0.0, base, 1.0)  # a non-positive base only occurs in rejected Newton trials
    sigma = safe_base ** (-0.68) * stagnation_factor ** (-0.12)
    delta = recovery_temp_k - hot_face_k
    sigma_derivative = -0.34 * sigma / xp.maximum(flame_temp_k * stagnation_factor * safe_base, 1e-9)
    clipped = sigma <= 0.1
    heat_flux = xp.where(clipped, bartz_base * 0.1 * delta, bartz_base * sigma * delta)
    derivative = xp.where(clipped, -bartz_base * 0.1, bartz_base * (sigma_derivative * delta - sigma))
    no_flux = delta <= 0.0
    return xp.where(no_flux, 0.0, heat_flux), xp.where(no_flux, 0.0, derivative)
