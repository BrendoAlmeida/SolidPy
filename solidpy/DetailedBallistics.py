# -*- coding: utf-8 -*-

"""Detailed internal ballistics post-processing for SolidPy burn simulations."""

from __future__ import annotations

import copy
import math

import numpy as np

try:
    from .Burn import BurnSimulation
    from .Environment import Environment
except ImportError:
    from Burn import BurnSimulation
    from Environment import Environment


STANDARD_GRAVITY = 9.80665


def _deduplicate_time(time, *series):
    time = np.asarray(time, dtype=float)
    keep = np.concatenate(([True], np.diff(time) > 1e-12))
    return time[keep], [np.asarray(values, dtype=float)[keep] for values in series]


def evaluate_nozzle_ablation_rate(
    chamber_pressure_pa,
    nozzle_mass_flow_kg_s,
    scale=1.0,
    pressure_exponent=0.42,
    mass_flow_exponent=0.32,
):
    """Empirical throat recession rate proxy in m/s."""
    return (
        1.8e-8
        * max(float(scale), 0.0)
        * max(float(chamber_pressure_pa), 1.0) ** float(pressure_exponent)
        * max(float(nozzle_mass_flow_kg_s), 1e-9) ** float(mass_flow_exponent)
    )


def _evaluate_remaining_propellant_mass(grain, motor, propellant, regressed_length):
    masses = []
    for regression in regressed_length:
        total_volume = 0.0
        for g in motor.grains:
            height, inner_radius, _ = g.calculate_tubular_geometry(regression)
            total_volume += math.pi * max(g.outer_radius**2 - inner_radius**2, 0.0) * height
        masses.append(max(total_volume * propellant.density, 0.0))
    return np.asarray(masses, dtype=float)


def build_detailed_ballistics(
    simulation,
    resample_step=None,
    max_time_points=None,
    nozzle_ablation_scale=1.0,
    ablation_pressure_exponent=0.42,
    ablation_mass_flow_exponent=0.32,
):
    """Build high-value internal time series from a ``BurnSimulation`` instance.

    Returns a dictionary with arrays for plotting/analysis and a nested
    ``summary`` dictionary with aggregate performance and consistency metrics.
    """
    raw_time = np.asarray(simulation.total_burn_solution[0], dtype=float)
    raw_pressure = np.asarray(simulation.total_burn_solution[1], dtype=float)
    raw_free_volume = np.asarray(simulation.total_burn_solution[2], dtype=float)
    raw_regression = np.asarray(simulation.total_burn_solution[3], dtype=float)
    raw_thrust = np.asarray(simulation.total_burn_solution[4], dtype=float)
    raw_exit_pressure = np.asarray(simulation.total_burn_solution[5], dtype=float)
    raw_exit_velocity = np.asarray(simulation.total_burn_solution[6], dtype=float)

    raw_time, deduped = _deduplicate_time(
        raw_time,
        raw_pressure,
        raw_free_volume,
        raw_regression,
        raw_thrust,
        raw_exit_pressure,
        raw_exit_velocity,
    )
    (
        raw_pressure,
        raw_free_volume,
        raw_regression,
        raw_thrust,
        raw_exit_pressure,
        raw_exit_velocity,
    ) = deduped

    if resample_step is not None and len(raw_time) > 1:
        step = max(float(resample_step), 1e-6)
        count = max(2, int(math.ceil((raw_time[-1] - raw_time[0]) / step)) + 1)
        if max_time_points is not None:
            count = min(count, max(int(max_time_points), 2))
        time_s = np.linspace(float(raw_time[0]), float(raw_time[-1]), count)
        chamber_pressure_pa = np.interp(time_s, raw_time, raw_pressure)
        free_volume_m3 = np.interp(time_s, raw_time, raw_free_volume)
        regressed_length_m = np.interp(time_s, raw_time, raw_regression)
        thrust_n = np.interp(time_s, raw_time, raw_thrust)
        exit_pressure_pa = np.interp(time_s, raw_time, raw_exit_pressure)
        exit_velocity_m_s = np.interp(time_s, raw_time, raw_exit_velocity)
    else:
        time_s = raw_time
        chamber_pressure_pa = raw_pressure
        free_volume_m3 = raw_free_volume
        regressed_length_m = raw_regression
        thrust_n = raw_thrust
        exit_pressure_pa = raw_exit_pressure
        exit_velocity_m_s = raw_exit_velocity
        if max_time_points is not None and len(time_s) > max_time_points:
            index = np.linspace(0, len(time_s) - 1, int(max_time_points), dtype=int)
            time_s = time_s[index]
            chamber_pressure_pa = chamber_pressure_pa[index]
            free_volume_m3 = free_volume_m3[index]
            regressed_length_m = regressed_length_m[index]
            thrust_n = thrust_n[index]
            exit_pressure_pa = exit_pressure_pa[index]
            exit_velocity_m_s = exit_velocity_m_s[index]

    motor = simulation.motor
    propellant = simulation.propellant
    burn_area_m2 = np.asarray(
        [simulation.compute_total_burn_area(regression) for regression in regressed_length_m],
        dtype=float,
    )

    propellant_mass_kg = _evaluate_remaining_propellant_mass(
        motor.grain, motor, propellant, regressed_length_m
    )
    if len(time_s) > 1:
        regression_rate_m_s = np.maximum(
            np.gradient(regressed_length_m, time_s, edge_order=1), 0.0
        )
        mass_generated_kg_s = np.maximum(
            -np.gradient(propellant_mass_kg, time_s, edge_order=1), 0.0
        )
    else:
        regression_rate_m_s = np.zeros_like(time_s)
        mass_generated_kg_s = np.zeros_like(time_s)

    mass_nozzle_kg_s = np.asarray(
        [
            max(float(simulation.evaluate_nozzle_mass_flow(pressure)), 0.0)
            for pressure in chamber_pressure_pa
        ],
        dtype=float,
    )

    throat_ablation_m = np.zeros_like(time_s)
    for idx in range(1, len(time_s)):
        dt = max(float(time_s[idx] - time_s[idx - 1]), 1e-9)
        throat_ablation_m[idx] = throat_ablation_m[idx - 1] + dt * evaluate_nozzle_ablation_rate(
            chamber_pressure_pa[idx - 1],
            mass_nozzle_kg_s[idx - 1],
            scale=nozzle_ablation_scale,
            pressure_exponent=ablation_pressure_exponent,
            mass_flow_exponent=ablation_mass_flow_exponent,
        )

    base_throat_radius_m = math.sqrt(max(motor.nozzle_throat_area, 0.0) / math.pi)
    throat_radius_m = base_throat_radius_m + throat_ablation_m
    throat_area_m2 = math.pi * np.maximum(throat_radius_m, 1e-9) ** 2
    throat_diameter_m = 2.0 * throat_radius_m

    cf = np.zeros_like(time_s)
    valid_cf = chamber_pressure_pa > simulation.environment_pressure * 1.0001
    cf[valid_cf] = thrust_n[valid_cf] / np.maximum(
        chamber_pressure_pa[valid_cf] * throat_area_m2[valid_cf], 1e-12
    )
    cf = np.maximum(cf, 0.0)

    active_fraction = np.asarray(
        [
            simulation.evaluate_burn_area_activation(float(time), float(regression))
            for time, regression in zip(time_s, regressed_length_m)
        ],
        dtype=float,
    )

    total_impulse_ns = float(np.trapezoid(thrust_n, time_s)) if len(time_s) > 1 else 0.0
    burn_time_s = float(time_s[-1] - time_s[0]) if len(time_s) else 0.0
    peak_thrust_n = float(np.max(thrust_n)) if len(thrust_n) else 0.0
    avg_thrust_n = total_impulse_ns / burn_time_s if burn_time_s > 0.0 else 0.0
    propellant_burned_kg = float(
        max(propellant_mass_kg[0] - propellant_mass_kg[-1], 0.0)
    )
    generated_integral_kg = (
        float(np.trapezoid(mass_generated_kg_s, time_s)) if len(time_s) > 1 else 0.0
    )
    expelled_integral_kg = (
        float(np.trapezoid(mass_nozzle_kg_s, time_s)) if len(time_s) > 1 else 0.0
    )
    mass_conservation_error_pct = (
        100.0
        * abs(generated_integral_kg - expelled_integral_kg)
        / max(generated_integral_kg, 1e-9)
        if generated_integral_kg > 1e-9
        else 0.0
    )
    isp_effective_s = (
        total_impulse_ns / (propellant_burned_kg * STANDARD_GRAVITY)
        if propellant_burned_kg > 1e-9
        else 0.0
    )
    cstar_effective_m_s = (
        float(np.trapezoid(chamber_pressure_pa * throat_area_m2, time_s))
        / max(generated_integral_kg, 1e-9)
        if generated_integral_kg > 1e-9
        else 0.0
    )
    max_dpressure_dt_pa_s = (
        float(
            np.max(
                np.abs(np.diff(chamber_pressure_pa) / np.maximum(np.diff(time_s), 1e-12))
            )
        )
        if len(time_s) > 1
        else 0.0
    )

    summary = {
        "simulation.schema_version": 3.0,
        "simulation.model_fidelity": "solidpy_detailed_ballistics",
        "simulation.nominal.burn_time_s": burn_time_s,
        "simulation.nominal.peak_thrust_n": peak_thrust_n,
        "simulation.nominal.avg_thrust_n": avg_thrust_n,
        "simulation.nominal.total_impulse_ns": total_impulse_ns,
        "simulation.nominal.isp_effective_s": isp_effective_s,
        "simulation.nominal.mass_flow_avg_kg_s": float(np.mean(mass_generated_kg_s)),
        "simulation.nominal.max_mass_flow_kg_s": float(np.max(mass_generated_kg_s)),
        "simulation.nominal.max_nozzle_mass_flow_kg_s": float(np.max(mass_nozzle_kg_s)),
        "simulation.nominal.chamber_pressure_max_mpa": float(
            np.max(chamber_pressure_pa) / 1e6
        ),
        "simulation.nominal.pressure_rise_rate_max_mpa_s": max_dpressure_dt_pa_s / 1e6,
        "simulation.nominal.mass_conservation_error_pct": mass_conservation_error_pct,
        "simulation.nominal.cstar_effective_m_s": cstar_effective_m_s,
        "simulation.nominal.final_throat_diameter_mm": 1000.0
        * float(throat_diameter_m[-1]),
        "simulation.nominal.throat_ablation_mm": 1000.0 * float(throat_ablation_m[-1]),
        "simulation.nominal.ignition_active_fraction_final": float(active_fraction[-1]),
    }

    return {
        "time_s": time_s,
        "thrust_n": thrust_n,
        "chamber_pressure_pa": chamber_pressure_pa,
        "free_volume_m3": free_volume_m3,
        "regressed_length_m": regressed_length_m,
        "burn_area_m2": burn_area_m2,
        "regression_rate_m_s": regression_rate_m_s,
        "mass_flow_kg_s": mass_generated_kg_s,
        "mass_generated_kg_s": mass_generated_kg_s,
        "mass_nozzle_kg_s": mass_nozzle_kg_s,
        "propellant_mass_kg": propellant_mass_kg,
        "exit_pressure_pa": exit_pressure_pa,
        "exit_velocity_m_s": exit_velocity_m_s,
        "throat_area_m2": throat_area_m2,
        "throat_diameter_m": throat_diameter_m,
        "throat_ablation_m": throat_ablation_m,
        "cf": cf,
        "ignition_active_fraction": active_fraction,
        "gamma": float(propellant.specific_heat_ratio),
        "summary": summary,
    }


def run_detailed_ballistics(
    grain,
    motor,
    propellant,
    environment=None,
    max_step_size=0.002,
    resample_step=None,
    max_time_points=None,
    tail_off_evaluation=True,
    nozzle_ablation_scale=1.0,
    ablation_pressure_exponent=0.42,
    ablation_mass_flow_exponent=0.32,
    **simulation_kwargs,
):
    """Run a fresh high-resolution burn and return detailed internal series."""
    cloned_grain, cloned_motor, cloned_propellant, cloned_environment = copy.deepcopy(
        (grain, motor, propellant, environment)
    )
    if cloned_environment is None:
        cloned_environment = Environment()

    simulation = BurnSimulation(
        cloned_grain,
        cloned_motor,
        cloned_propellant,
        cloned_environment,
        max_step_size=max_step_size,
        tail_off_evaluation=tail_off_evaluation,
        **simulation_kwargs,
    )
    detailed = build_detailed_ballistics(
        simulation,
        resample_step=resample_step if resample_step is not None else max_step_size,
        max_time_points=max_time_points,
        nozzle_ablation_scale=nozzle_ablation_scale,
        ablation_pressure_exponent=ablation_pressure_exponent,
        ablation_mass_flow_exponent=ablation_mass_flow_exponent,
    )
    detailed["simulation"] = simulation
    return detailed


def evaluate_combustion_stability(motor, propellant, environment=None, pressure_points=40):
    """Combustion stability indicators for a motor-propellant pair.

    Computes:
    - L* (characteristic chamber length): V_chamber / A_throat
    - Kn vs pressure sweep to locate the neutral-burn operating point
    - Pressure exponent n (from burn rate table slope) and stability verdict
    - L* adequacy relative to Summerfield criterion for KNO3-based propellants

    These are design-time analyses, independent of a full burn simulation.

    Ref: Summerfield et al. (1954); Barrère et al. (1960) §7.4;
         Sutton & Biblarz §12.3.

    Args:
        motor: Motor instance
        propellant: Propellant instance
        environment: Environment instance (for ambient pressure)
        pressure_points (int): number of pressure samples for the Kn-P sweep

    Returns:
        dict with stability metrics and arrays for plotting
    """
    if environment is None:
        try:
            from .Environment import Environment as _Env
        except ImportError:
            from Environment import Environment as _Env
        environment = _Env()

    p_amb = float(environment.atmospheric_pressure)

    # L* — characteristic chamber length
    l_star = float(motor.chamber_volume) / max(float(motor.nozzle_throat_area), 1e-9)

    # Summerfield criterion: L* >= threshold for stable combustion.
    # KNO3/KClO4 propellants: ~1.0–2.5 m.  AP-based: ~0.5–0.8 m.
    l_star_threshold = 1.5  # conservative for KNSB; adjust per propellant family

    # Kn sweep: evaluate burn area and Kn at each regression step.
    n_grains = len(motor.grains)
    web = min(g.outer_radius - g.initial_inner_radius for g in motor.grains)
    r_steps = np.linspace(0.0, web * 0.99, pressure_points)
    kn_values = []
    for r in r_steps:
        total_area = sum(
            g.evaluate_tubular_burn_area(r, update_state=False) for g in motor.grains
        )
        kn_values.append(total_area / max(float(motor.nozzle_throat_area), 1e-9))
    kn_values = np.asarray(kn_values)

    # Pressure sweep paired to regression steps (from de Saint Robert: P ~ Kn).
    # Equilibrium pressure from 0-D model: P = c* * rho_p * r(P) * Kn / A_t * A_t
    # → P^(1-n) = c* * rho_p * a * Kn  (power law r = a*P^n, r in mm/s)
    # Solve implicitly for each Kn value.
    p_sweep = np.linspace(max(p_amb * 1.1, 0.5e6), 12e6, pressure_points)
    kn_at_p_sweep = []
    for r_reg, p_est in zip(r_steps, p_sweep):
        total_area = sum(
            g.evaluate_tubular_burn_area(r_reg, update_state=False) for g in motor.grains
        )
        kn_at_p_sweep.append(total_area / max(float(motor.nozzle_throat_area), 1e-9))
    kn_at_p_sweep = np.asarray(kn_at_p_sweep)

    # Burn rate exponent n: slope of log(r) vs log(P) at mid-range pressure.
    p_lo = 2e6
    p_hi = 6e6
    r_lo = propellant.evaluate_burn_rate(p_lo)
    r_hi = propellant.evaluate_burn_rate(p_hi)
    if r_lo > 1e-9 and r_hi > 1e-9 and p_lo > 0 and p_hi > 0:
        n_exponent = math.log(r_hi / r_lo) / math.log(p_hi / p_lo)
    else:
        n_exponent = float("nan")

    # Stability verdict: motor is statically stable if n < 1.
    # For n >= 1 the propellant is in an unstable regime (mesa / plateau edge).
    pressure_stable = (not math.isnan(n_exponent)) and n_exponent < 1.0
    l_star_adequate = l_star >= l_star_threshold

    # Nominal Kn (initial state, before any regression)
    kn_initial = kn_values[0] if len(kn_values) else float("nan")
    kn_burnout = kn_values[-1] if len(kn_values) else float("nan")
    kn_type = "progressive" if kn_burnout > kn_initial else (
        "regressive" if kn_burnout < kn_initial else "neutral"
    )

    return {
        "stability.l_star_m": l_star,
        "stability.l_star_adequate": l_star_adequate,
        "stability.l_star_threshold_m": l_star_threshold,
        "stability.burn_rate_exponent_n": n_exponent,
        "stability.pressure_stable": pressure_stable,
        "stability.kn_initial": kn_initial,
        "stability.kn_burnout": kn_burnout,
        "stability.kn_profile_type": kn_type,
        "stability.n_grains": n_grains,
        # Arrays for Kn vs regression plot
        "_arrays.kn_regression_steps_m": r_steps,
        "_arrays.kn_values": kn_values,
        "_arrays.pressure_sweep_pa": p_sweep,
        "_arrays.kn_at_pressure_sweep": kn_at_p_sweep,
    }
