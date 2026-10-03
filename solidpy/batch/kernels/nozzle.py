# -*- coding: utf-8 -*-
"""Nozzle kernels: mass flow and thrust components, mirroring ``Burn``.

``pressure`` is the chamber pressure, ``temperature`` the gas temperature in the chamber, ``k`` the specific
heat ratio at that pressure and ``mach`` the supersonic exit Mach number (a per-lane constant for scalar
thermochemistry, computed at packing time by the scalar root finder).
"""

import numpy as np

TINY = float(np.finfo(float).tiny)


def _denominator(xp, pressure, ambient):
    """Pressure used as a divisor: itself above ambient, otherwise a harmless finite stand-in.

    Below ambient the flow and thrust are zero and discarded by a ``where``; the stand-in only keeps the
    discarded branch free of overflow and ``inf - inf``. NaN pressures still propagate through the results.
    """
    return xp.where(pressure > ambient, pressure, 1.0 + ambient)


def critical_pressure_ratio(xp, k):
    """Ambient-to-chamber pressure ratio below which the throat is choked."""
    return (2.0 / (k + 1.0)) ** (k / (k - 1.0))


def nozzle_mass_flow(xp, pressure, temperature, k, P):
    """Mass flow through the throat (``Burn.evaluate_nozzle_mass_flow``), zero at or below ambient."""
    ratio = P["ambient_pressure"] / _denominator(xp, pressure, P["ambient_pressure"])
    choked = ratio <= critical_pressure_ratio(xp, k)
    gas_constant, throat = P["gas_constant"], P["throat_area"]
    flow_choked = (
        pressure * throat * xp.sqrt(k / (gas_constant * temperature)) * (2.0 / (k + 1.0)) ** ((k + 1.0) / (2.0 * (k - 1.0)))
    )
    unchoked_term = ratio ** (2.0 / k) - ratio ** ((k + 1.0) / k)
    flow_unchoked = throat * pressure * xp.sqrt(
        (2.0 * k / ((k - 1.0) * gas_constant * temperature)) * xp.maximum(unchoked_term, 0.0)
    )
    flow = P["discharge_coefficient"] * xp.where(choked, flow_choked, flow_unchoked)
    return xp.where(pressure <= P["ambient_pressure"], 0.0, flow)


def thrust_components(xp, pressure, temperature, nozzle_flow, k, mach, P):
    """Return ``(ideal momentum, momentum, pressure thrust, reported total)`` in newtons.

    The discharge coefficient scales the momentum only and ``eta_Cf`` scales the summed thrust
    (``Burn.evaluate_thrust_components``). All four are zero at or below ambient pressure.
    """
    ambient, gas_constant = P["ambient_pressure"], P["gas_constant"]
    above = pressure > ambient
    safe = _denominator(xp, pressure, ambient)
    choked = above & (ambient / safe <= critical_pressure_ratio(xp, k))

    exit_temperature = temperature / (1.0 + (k - 1.0) / 2.0 * mach**2)
    velocity_choked = mach * xp.sqrt(k * gas_constant * exit_temperature)
    velocity_unchoked = xp.sqrt(
        xp.maximum((2.0 * k / (k - 1.0)) * gas_constant * temperature * (1.0 - (ambient / safe) ** ((k - 1.0) / k)), 0.0)
    )
    velocity = xp.where(above, xp.where(choked, velocity_choked, velocity_unchoked), 0.0)
    exit_pressure = xp.where(choked, pressure * (1.0 + (k - 1.0) / 2.0 * mach**2) ** (-k / (k - 1.0)), ambient)

    discharge = P["discharge_coefficient"]
    ideal = P["divergence_factor"] * (nozzle_flow / discharge) * velocity
    momentum = discharge * ideal
    pressure_thrust = (exit_pressure - ambient) * P["exit_area"]
    zero = xp.zeros_like(pressure)
    ideal = xp.where(above, ideal, zero)
    momentum = xp.where(above, momentum, zero)
    pressure_thrust = xp.where(above, pressure_thrust, zero)
    return ideal, momentum, pressure_thrust, P["eta_cf"] * (momentum + pressure_thrust)
