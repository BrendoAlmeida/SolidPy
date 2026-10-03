# -*- coding: utf-8 -*-
"""Propellant kernels: burn rate with the Lenoir-Robillard erosive term, and gas properties at a pressure."""

from .tables import evaluate


def burn_rate(xp, pressure, port_mass_flux, P):
    """Effective burn rate in m/s (``Propellant.evaluate_burn_rate``): power law or tabulated, plus erosive.

    ``pressure`` and ``port_mass_flux`` are per lane. The tabulated rate is a function of the pressure in MPa, in
    mm/s. The erosive correction ``k_e G^0.8 exp(-alpha_e r0 / G)`` applies only where ``k_e > 0`` and
    ``G > 1e-3``.
    """
    megapascal = xp.maximum(pressure, 0.0) * 1e-6
    coefficients = tuple(P[f"rate_table_c{i}"] for i in range(4))
    tabulated = evaluate(xp, megapascal, P["rate_table_x"], coefficients, P["rate_table_n"], P["rate_table_below"],
                         P["rate_table_above"]) / 1000.0
    power_law = P["burn_rate_a"] * megapascal ** P["burn_rate_n"] / 1000.0
    r0 = xp.where(P["burn_rate_mode"] == 1.0, tabulated, power_law)
    erosive = xp.maximum(
        P["erosive_coefficient"]
        * port_mass_flux**0.8
        * xp.exp(-P["erosive_alpha"] * r0 / xp.maximum(port_mass_flux, 1e-9)),
        0.0,
    )
    return r0 + xp.where((P["erosive_coefficient"] > 0.0) & (port_mass_flux > 1e-3), erosive, 0.0)


def gas_properties(xp, pressure, P):
    """Source temperature ``T_0`` (with ``eta_c**2``) and specific heat ratio ``k`` at ``pressure``.

    Lanes with scalar thermochemistry use constants; lanes with a pressure table look ``Tc(p)`` and ``k(p)`` up
    (``Propellant.Tc_at_pressure`` and ``get_gamma``, evaluated at the pressure clamped to be non-negative as
    ``Burn._parameters_at_pressure`` does).
    """
    table = P["thermo_mode"] == 1.0
    p = xp.maximum(pressure, 0.0)
    count = P["thermo_n"]
    tc = evaluate(xp, p, P["thermo_x"], tuple(P[f"thermo_tc_c{i}"] for i in range(4)), count, P["thermo_tc_below"],
                  P["thermo_tc_above"])
    k = evaluate(xp, p, P["thermo_x"], tuple(P[f"thermo_k_c{i}"] for i in range(4)), count, P["thermo_k_below"],
                 P["thermo_k_above"])
    return xp.where(table, tc * P["eta_c"] ** 2, P["source_temperature"]), xp.where(table, k, P["gamma"])


def exit_mach(xp, k, P):
    """Supersonic exit Mach number for the specific heat ratio ``k``.

    A lane with scalar thermochemistry has one value, solved at packing time. For a ``k`` table the Mach number
    is a cubic of ``k`` on the grid packed from the scalar root finder (``problem._mach_parts``).
    """
    table = evaluate(xp, k, P["mach_x"], tuple(P[f"mach_c{i}"] for i in range(4)), P["mach_n"], P["mach_below"],
                     P["mach_above"])
    return xp.where(P["thermo_mode"] == 1.0, table, P["exit_mach"])
