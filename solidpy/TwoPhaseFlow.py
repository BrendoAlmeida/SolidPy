# -*- coding: utf-8 -*-

"""Two-phase nozzle-flow post-processing scaffolding."""

from __future__ import annotations

from collections.abc import Mapping

import numpy as np


G0_M_S2 = 9.80665
DEFAULT_PARTICLE_DIAMETER_M = 5.0e-6
DEFAULT_ALUMINA_DENSITY_KG_M3 = 3950.0
DEFAULT_ALUMINA_CP_J_KG_K = 1100.0
DEFAULT_GAS_CONSTANT_J_KG_K = 355.0
DEFAULT_GAS_VISCOSITY_PA_S = 8.0e-5
DEFAULT_GAS_CONDUCTIVITY_W_M_K = 0.12
DEFAULT_GAS_PRANDTL = 0.72


def _validate_ideal_isp(ideal_isp_s):
    ideal_isp_s = float(ideal_isp_s)
    if not np.isfinite(ideal_isp_s) or ideal_isp_s <= 0.0:
        raise ValueError("ideal_isp_s must be a positive finite value.")
    return ideal_isp_s


def _validate_mass_fraction(condensed_mass_fraction):
    condensed_mass_fraction = float(condensed_mass_fraction)
    if not np.isfinite(condensed_mass_fraction):
        raise ValueError("condensed_mass_fraction must be finite.")
    if condensed_mass_fraction < 0.0 or condensed_mass_fraction > 1.0:
        raise ValueError("condensed_mass_fraction must be between 0 and 1.")
    return condensed_mass_fraction


def _profile(nozzle_state, keys, default):
    state = nozzle_state if isinstance(nozzle_state, Mapping) else {}
    value = None
    for key in keys:
        if key in state and state[key] is not None:
            value = state[key]
            break
    if value is None:
        value = default

    array = np.atleast_1d(np.asarray(value, dtype=float))
    array = array[np.isfinite(array)]
    if array.size == 0:
        array = np.atleast_1d(np.asarray(default, dtype=float))
    return array.astype(float)


def _state_scalar(nozzle_state, keys, default):
    values = _profile(nozzle_state, keys, default)
    return float(np.mean(values))


def _particle_properties(particle_distribution):
    distribution = (
        particle_distribution
        if isinstance(particle_distribution, Mapping)
        else {"diameter_m": particle_distribution}
    )

    species = str(distribution.get("species", "")).lower()
    if "k2co3" in species:
        default_density = 2430.0
        default_cp = 900.0
    else:
        default_density = DEFAULT_ALUMINA_DENSITY_KG_M3
        default_cp = DEFAULT_ALUMINA_CP_J_KG_K

    diameters = distribution.get(
        "diameters_m",
        distribution.get(
            "diameter_m",
            distribution.get("mean_diameter_m", DEFAULT_PARTICLE_DIAMETER_M),
        ),
    )
    diameters = np.atleast_1d(np.asarray(diameters, dtype=float))
    diameters = diameters[np.isfinite(diameters) & (diameters > 0.0)]
    if diameters.size == 0:
        raise ValueError("particle diameters must contain positive finite values.")

    weights = distribution.get(
        "weights",
        distribution.get("mass_fractions", np.ones_like(diameters)),
    )
    weights = np.atleast_1d(np.asarray(weights, dtype=float))
    if weights.size != diameters.size:
        weights = np.ones_like(diameters)
    weights = np.where(np.isfinite(weights) & (weights > 0.0), weights, 0.0)
    if float(np.sum(weights)) <= 0.0:
        weights = np.ones_like(diameters)
    weights = weights / np.sum(weights)

    density = float(
        distribution.get(
            "density_kg_m3",
            distribution.get("particle_density_kg_m3", default_density),
        )
    )
    specific_heat = float(
        distribution.get(
            "specific_heat_j_kg_k",
            distribution.get("cp_j_kg_k", default_cp),
        )
    )
    if not np.isfinite(density) or density <= 0.0:
        raise ValueError("particle density must be a positive finite value.")
    if not np.isfinite(specific_heat) or specific_heat <= 0.0:
        raise ValueError("particle specific heat must be a positive finite value.")

    return diameters, weights, density, specific_heat


def _gas_density_profile(nozzle_state):
    density = _profile(nozzle_state, ("gas_density_kg_m3", "density_kg_m3"), np.array([]))
    if density.size:
        return np.maximum(density, 1.0e-9)

    pressure = _profile(nozzle_state, ("pressure_pa", "gas_pressure_pa"), [3.0e6, 101325.0])
    temperature = _profile(
        nozzle_state,
        ("temperature_k", "gas_temperature_k"),
        [3400.0, 1800.0],
    )
    gas_constant = _state_scalar(
        nozzle_state,
        ("gas_constant_j_kg_k", "specific_gas_constant_j_kg_k"),
        DEFAULT_GAS_CONSTANT_J_KG_K,
    )
    if temperature.size == 1 and pressure.size > 1:
        temperature = np.full_like(pressure, float(temperature[0]))
    elif temperature.size != pressure.size:
        temperature = np.linspace(float(temperature[0]), float(temperature[-1]), pressure.size)
    return np.maximum(pressure / (gas_constant * np.maximum(temperature, 1.0)), 1.0e-9)


def _gas_velocity_profile(nozzle_state, ideal_isp_s):
    exhaust_velocity = ideal_isp_s * G0_M_S2
    velocity = _profile(
        nozzle_state,
        ("gas_velocity_m_s", "velocity_m_s", "exit_velocity_m_s"),
        [0.25 * exhaust_velocity, exhaust_velocity],
    )
    return np.maximum(velocity, 1.0)


def _residence_time_s(nozzle_state, ideal_isp_s):
    residence_time = _state_scalar(nozzle_state, ("residence_time_s",), np.nan)
    if np.isfinite(residence_time) and residence_time > 0.0:
        return residence_time

    nozzle_length_m = _state_scalar(
        nozzle_state,
        ("nozzle_length_m", "length_m"),
        0.12,
    )
    mean_velocity = float(np.mean(_gas_velocity_profile(nozzle_state, ideal_isp_s)))
    return max(nozzle_length_m / max(mean_velocity, 1.0), 1.0e-5)


def _kinetic_lag_core(ideal_isp_s, particle_distribution, nozzle_state):
    diameters, weights, particle_density, _ = _particle_properties(particle_distribution)
    gas_viscosity = _state_scalar(
        nozzle_state,
        ("gas_viscosity_pa_s", "viscosity_pa_s"),
        DEFAULT_GAS_VISCOSITY_PA_S,
    )
    gas_viscosity = max(gas_viscosity, 1.0e-8)
    residence_time = _residence_time_s(nozzle_state, ideal_isp_s)

    relaxation_times = particle_density * diameters**2 / (18.0 * gas_viscosity)
    time_ratios = relaxation_times / max(residence_time, 1.0e-9)
    slip_fractions = time_ratios * (1.0 - np.exp(-1.0 / np.maximum(time_ratios, 1.0e-12)))
    slip_fractions = np.clip(slip_fractions, 0.0, 1.0)
    mean_slip_fraction = float(np.sum(weights * slip_fractions))

    return {
        "mean_slip_fraction": mean_slip_fraction,
        "relaxation_times_s": relaxation_times,
        "residence_time_s": residence_time,
        "mean_particle_diameter_m": float(np.sum(weights * diameters)),
    }


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
    ideal_isp_s = _validate_ideal_isp(ideal_isp_s)
    condensed_mass_fraction = _validate_mass_fraction(condensed_mass_fraction)
    if condensed_mass_fraction == 0.0:
        return {
            "thermal_loss_s": 0.0,
            "thermal_loss_percent": 0.0,
            "thermal_loss_fraction": 0.0,
            "corrected_isp_s": ideal_isp_s,
            "retained_enthalpy_j_kg_particles": 0.0,
            "mean_thermal_time_constant_s": 0.0,
            "mean_nusselt_number": 0.0,
        }

    diameters, weights, particle_density, particle_cp = _particle_properties(
        particle_distribution
    )
    gas_density = float(np.mean(_gas_density_profile(nozzle_state)))
    gas_viscosity = _state_scalar(
        nozzle_state,
        ("gas_viscosity_pa_s", "viscosity_pa_s"),
        DEFAULT_GAS_VISCOSITY_PA_S,
    )
    gas_conductivity = _state_scalar(
        nozzle_state,
        ("gas_conductivity_w_m_k", "thermal_conductivity_w_m_k"),
        DEFAULT_GAS_CONDUCTIVITY_W_M_K,
    )
    gas_prandtl = _state_scalar(nozzle_state, ("gas_prandtl", "prandtl"), DEFAULT_GAS_PRANDTL)
    gas_velocity = float(np.mean(_gas_velocity_profile(nozzle_state, ideal_isp_s)))
    kinetic_core = _kinetic_lag_core(ideal_isp_s, particle_distribution, nozzle_state)
    relative_velocity = _state_scalar(
        nozzle_state,
        ("particle_relative_velocity_m_s", "relative_velocity_m_s"),
        max(0.05 * gas_velocity, kinetic_core["mean_slip_fraction"] * gas_velocity),
    )
    residence_time = _residence_time_s(nozzle_state, ideal_isp_s)

    reynolds = (
        gas_density * max(relative_velocity, 0.0) * diameters / max(gas_viscosity, 1.0e-8)
    )
    nusselt = 2.0 + 0.6 * np.sqrt(np.maximum(reynolds, 0.0)) * max(gas_prandtl, 1.0e-6) ** (1.0 / 3.0)
    heat_transfer = nusselt * max(gas_conductivity, 1.0e-9) / diameters
    thermal_times = particle_density * particle_cp * diameters / (6.0 * heat_transfer)
    retention = np.exp(-residence_time / np.maximum(thermal_times, 1.0e-12))

    temperature = _profile(
        nozzle_state,
        ("temperature_k", "gas_temperature_k"),
        [3400.0, 1800.0],
    )
    gas_temperature_drop = max(float(temperature[0] - temperature[-1]), 0.0)
    retained_enthalpy = particle_cp * gas_temperature_drop * retention
    mean_retained_enthalpy = float(np.sum(weights * retained_enthalpy))

    exhaust_velocity = ideal_isp_s * G0_M_S2
    loss_fraction = condensed_mass_fraction * mean_retained_enthalpy / max(
        exhaust_velocity**2,
        1.0,
    )
    loss_fraction = float(np.clip(loss_fraction, 0.0, 0.95))
    loss_s = ideal_isp_s * loss_fraction

    return {
        "thermal_loss_s": float(loss_s),
        "thermal_loss_percent": float(100.0 * loss_fraction),
        "thermal_loss_fraction": loss_fraction,
        "corrected_isp_s": float(ideal_isp_s - loss_s),
        "retained_enthalpy_j_kg_particles": mean_retained_enthalpy,
        "mean_thermal_time_constant_s": float(np.sum(weights * thermal_times)),
        "mean_nusselt_number": float(np.sum(weights * nusselt)),
    }


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
    ideal_isp_s = _validate_ideal_isp(ideal_isp_s)
    condensed_mass_fraction = _validate_mass_fraction(condensed_mass_fraction)
    if condensed_mass_fraction == 0.0:
        return {
            "kinetic_loss_s": 0.0,
            "kinetic_loss_percent": 0.0,
            "kinetic_loss_fraction": 0.0,
            "corrected_isp_s": ideal_isp_s,
            "velocity_slip_fraction": 0.0,
            "mean_particle_relaxation_time_s": 0.0,
            "residence_time_s": 0.0,
            "mean_particle_diameter_m": 0.0,
        }

    kinetic_core = _kinetic_lag_core(ideal_isp_s, particle_distribution, nozzle_state)
    relaxation_times = kinetic_core["relaxation_times_s"]
    diameters, weights, _, _ = _particle_properties(particle_distribution)
    mean_slip_fraction = kinetic_core["mean_slip_fraction"]

    loss_fraction = condensed_mass_fraction * mean_slip_fraction
    loss_fraction = float(np.clip(loss_fraction, 0.0, 0.95))
    loss_s = ideal_isp_s * loss_fraction

    return {
        "kinetic_loss_s": float(loss_s),
        "kinetic_loss_percent": float(100.0 * loss_fraction),
        "kinetic_loss_fraction": loss_fraction,
        "corrected_isp_s": float(ideal_isp_s - loss_s),
        "velocity_slip_fraction": mean_slip_fraction,
        "mean_particle_relaxation_time_s": float(np.sum(weights * relaxation_times)),
        "residence_time_s": kinetic_core["residence_time_s"],
        "mean_particle_diameter_m": float(np.sum(weights * diameters)),
    }


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
    ideal_isp_s = _validate_ideal_isp(ideal_isp_s)
    condensed_mass_fraction = _validate_mass_fraction(condensed_mass_fraction)
    kinetic = estimate_kinetic_lag_isp_loss(
        ideal_isp_s,
        condensed_mass_fraction,
        particle_distribution,
        nozzle_state,
    )
    thermal = estimate_thermal_lag_isp_loss(
        ideal_isp_s,
        condensed_mass_fraction,
        particle_distribution,
        nozzle_state,
    )

    kinetic_fraction = kinetic["kinetic_loss_fraction"]
    thermal_fraction = thermal["thermal_loss_fraction"]
    total_loss_fraction = 1.0 - (1.0 - kinetic_fraction) * (1.0 - thermal_fraction)
    total_loss_fraction = float(np.clip(total_loss_fraction, 0.0, 0.95))
    total_loss_s = ideal_isp_s * total_loss_fraction

    return {
        "corrected_isp_s": float(ideal_isp_s - total_loss_s),
        "total_loss_s": float(total_loss_s),
        "total_loss_percent": float(100.0 * total_loss_fraction),
        "total_loss_fraction": total_loss_fraction,
        "kinetic_loss_s": kinetic["kinetic_loss_s"],
        "kinetic_loss_percent": kinetic["kinetic_loss_percent"],
        "kinetic_loss_fraction": kinetic_fraction,
        "thermal_loss_s": thermal["thermal_loss_s"],
        "thermal_loss_percent": thermal["thermal_loss_percent"],
        "thermal_loss_fraction": thermal_fraction,
        "condensed_mass_fraction": condensed_mass_fraction,
        "components": {
            "kinetic": kinetic,
            "thermal": thermal,
        },
    }
