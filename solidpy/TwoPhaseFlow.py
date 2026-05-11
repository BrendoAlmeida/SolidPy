# -*- coding: utf-8 -*-

"""Two-phase nozzle-flow post-processing scaffolding."""

from __future__ import annotations


def estimate_thermal_lag_isp_loss(
    ideal_isp_s,
    condensed_mass_fraction,
    particle_distribution,
    nozzle_state,
):
    """Estimate Isp loss caused by particle thermal lag in nozzle expansion.

    The future implementation will compare gas and condensed-phase temperature
    histories through the nozzle, estimating how much enthalpy remains trapped
    in particles such as Al2O3 or K2CO3 instead of being converted into exhaust
    kinetic energy.

    Args:
        ideal_isp_s (float): Single-phase ideal specific impulse in seconds.
        condensed_mass_fraction (float): Condensed-phase mass fraction in the
            exhaust products.
        particle_distribution: Particle size, density, and heat-capacity model.
        nozzle_state: Gas expansion state or sampled nozzle profile.

    Returns:
        dict: Planned output with thermal-lag loss in seconds and percent.

    Raises:
        NotImplementedError: Until particle heat-transfer integration is
            implemented.
    """
    raise NotImplementedError("Thermal-lag Isp loss is not implemented yet.")


def estimate_kinetic_lag_isp_loss(
    ideal_isp_s,
    condensed_mass_fraction,
    particle_distribution,
    nozzle_state,
):
    """Estimate Isp loss caused by particle velocity slip.

    The future implementation will solve or approximate particle acceleration
    through the nozzle and quantify the impulse not transferred to the gas
    stream because condensed particles leave the nozzle with a velocity lag.

    Args:
        ideal_isp_s (float): Single-phase ideal specific impulse in seconds.
        condensed_mass_fraction (float): Condensed-phase mass fraction in the
            exhaust products.
        particle_distribution: Particle size and density model.
        nozzle_state: Gas expansion state or sampled nozzle profile.

    Returns:
        dict: Planned output with kinetic-lag loss in seconds and percent.

    Raises:
        NotImplementedError: Until particle momentum-coupling integration is
            implemented.
    """
    raise NotImplementedError("Kinetic-lag Isp loss is not implemented yet.")


def estimate_two_phase_isp_loss(
    ideal_isp_s,
    condensed_mass_fraction,
    particle_distribution,
    nozzle_state,
):
    """Combine thermal and kinetic condensed-phase Isp losses.

    This post-processor will provide the main public interface for estimating
    total two-phase performance loss after a nominal single-phase nozzle
    calculation. It is intended for aluminized propellants and other
    formulations that produce condensed products such as Al2O3 or K2CO3.

    Args:
        ideal_isp_s (float): Single-phase ideal specific impulse in seconds.
        condensed_mass_fraction (float): Condensed-phase mass fraction in the
            exhaust products.
        particle_distribution: Particle size, density, heat capacity, and
            loading model.
        nozzle_state: Gas expansion state or sampled nozzle profile.

    Returns:
        dict: Planned output with total Isp loss, corrected Isp, and separate
        thermal and kinetic lag components.

    Raises:
        NotImplementedError: Until the component loss models are implemented.
    """
    raise NotImplementedError("Two-phase Isp loss is not implemented yet.")
