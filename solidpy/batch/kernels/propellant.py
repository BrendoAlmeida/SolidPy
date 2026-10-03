# -*- coding: utf-8 -*-
"""Propellant kernels: burn rate with the Lenoir-Robillard erosive term, and gas properties at a pressure."""


def burn_rate(xp, pressure, port_mass_flux, P):
    """Effective burn rate in m/s (``Propellant.evaluate_burn_rate`` for a power-law propellant).

    ``pressure`` and ``port_mass_flux`` are per lane. The erosive correction ``k_e G^0.8 exp(-alpha_e r0 / G)``
    applies only where ``k_e > 0`` and ``G > 1e-3``.
    """
    r0 = P["burn_rate_a"] * (xp.maximum(pressure, 0.0) * 1e-6) ** P["burn_rate_n"] / 1000.0
    erosive = xp.maximum(
        P["erosive_coefficient"]
        * port_mass_flux**0.8
        * xp.exp(-P["erosive_alpha"] * r0 / xp.maximum(port_mass_flux, 1e-9)),
        0.0,
    )
    return r0 + xp.where((P["erosive_coefficient"] > 0.0) & (port_mass_flux > 1e-3), erosive, 0.0)


def gas_properties(xp, pressure, P):
    """Source temperature ``T_0`` (with ``eta_c**2``) and specific heat ratio ``k`` at ``pressure``.

    Lanes with scalar thermochemistry use constants. Pressure tables are a separate lane feature and are
    NaN in the packed arrays, so a lane that needs them cannot be run through these kernels by mistake.
    The constants are returned as stored (one value per lane); callers broadcast them against ``pressure``.
    """
    return P["source_temperature"], P["gamma"]
