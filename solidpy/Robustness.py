# -*- coding: utf-8 -*-

"""Multi-scenario robustness analysis for SolidPy motors."""

from __future__ import annotations

import copy
from dataclasses import dataclass
from typing import Optional

import numpy as np

try:
    from .DetailedBallistics import run_detailed_ballistics
    from .Environment import Environment
    from ._numpy_compat import trapezoid as _trapezoid
except ImportError:
    from DetailedBallistics import run_detailed_ballistics
    from Environment import Environment
    from _numpy_compat import trapezoid as _trapezoid


BURN_RATE_TEMP_SENSITIVITY_PER_K = 0.005
BURN_RATE_TEMP_REFERENCE_K = 298.15


@dataclass(frozen=True)
class RobustnessScenario:
    """Dimensionless and environmental perturbations for one motor run."""

    scenario_id: str
    scenario_kind: str = "robustness"
    burn_rate_factor: float = 1.0
    throat_factor: float = 1.0
    isp_factor: float = 1.0
    density_factor: float = 1.0
    ambient_pressure_pa: Optional[float] = None
    initial_temperature_k: float = BURN_RATE_TEMP_REFERENCE_K
    nozzle_ablation_scale_factor: float = 1.0
    igniter_energy_factor: float = 1.0
    casing_strength_factor: float = 1.0
    liner_thickness_factor: float = 1.0
    drag_coefficient_factor: float = 1.0


def default_robustness_scenarios():
    """Return fixed extreme scenarios for deterministic robustness checks."""
    return [
        RobustnessScenario("low_burn_rate", burn_rate_factor=0.94),
        RobustnessScenario("high_burn_rate", burn_rate_factor=1.06),
        RobustnessScenario("narrow_throat", throat_factor=0.96),
        RobustnessScenario("wide_throat", throat_factor=1.04),
        RobustnessScenario("low_density", density_factor=0.98),
        RobustnessScenario("high_density", density_factor=1.02),
        RobustnessScenario("cold_start", initial_temperature_k=273.15),
        RobustnessScenario("hot_start", initial_temperature_k=323.15),
        RobustnessScenario("high_altitude", ambient_pressure_pa=85_000.0),
        RobustnessScenario("sea_level_high_pressure", ambient_pressure_pa=105_000.0),
    ]


def build_latin_hypercube_scenarios(sample_count=16, seed=20260504):
    """Build reproducible Latin-hypercube uncertainty samples."""
    sample_count = max(int(sample_count), 0)
    if sample_count == 0:
        return []

    rng = np.random.default_rng(int(seed))
    axes = [
        ("burn_rate_factor", 0.94, 1.06),
        ("throat_factor", 0.96, 1.04),
        ("isp_factor", 0.98, 1.02),
        ("density_factor", 0.98, 1.02),
        ("ambient_pressure_pa", 85_000.0, 105_000.0),
        ("initial_temperature_k", 273.15, 323.15),
        ("nozzle_ablation_scale_factor", 0.8, 1.2),
        ("igniter_energy_factor", 0.8, 1.2),
        ("casing_strength_factor", 0.95, 1.02),
        ("liner_thickness_factor", 0.95, 1.05),
        ("drag_coefficient_factor", 0.9, 1.15),
    ]
    base = (np.arange(sample_count, dtype=float) + rng.random(sample_count)) / sample_count
    samples = {}
    for name, low, high in axes:
        values = np.array(base, copy=True)
        rng.shuffle(values)
        samples[name] = low + values * (high - low)

    scenarios = []
    for idx in range(sample_count):
        payload = {name: float(values[idx]) for name, values in samples.items()}
        scenarios.append(
            RobustnessScenario(
                scenario_id=f"mc_lhs_{idx:03d}",
                scenario_kind="robustness",
                **payload,
            )
        )
    return scenarios


def _apply_scenario(grain, motor, propellant, environment, scenario):
    grain, motor, propellant, environment = copy.deepcopy(
        (grain, motor, propellant, environment)
    )
    if environment is None:
        environment = Environment()

    temperature_factor = 1.0 + BURN_RATE_TEMP_SENSITIVITY_PER_K * (
        float(scenario.initial_temperature_k) - BURN_RATE_TEMP_REFERENCE_K
    )
    burn_rate_factor = max(float(scenario.burn_rate_factor) * temperature_factor, 0.1)
    original_burn_rate = propellant.evaluate_burn_rate
    propellant.evaluate_burn_rate = (
        lambda chamber_pressure, port_mass_flux=0.0,
        _orig=original_burn_rate, _factor=burn_rate_factor:
        _factor * _orig(chamber_pressure, port_mass_flux)
    )

    density_factor = max(float(scenario.density_factor), 0.01)
    propellant.density *= density_factor

    throat_factor = max(float(scenario.throat_factor), 0.01)
    motor.nozzle_throat_area *= throat_factor**2
    motor.expansion_ratio = motor.nozzle_exit_area / max(motor.nozzle_throat_area, 1e-12)

    if scenario.ambient_pressure_pa is not None:
        environment.atmospheric_pressure = max(float(scenario.ambient_pressure_pa), 0.0)

    return grain, motor, propellant, environment


def _rescale_thrust(result, factor):
    factor = max(float(factor), 0.0)
    if factor == 1.0:
        return result

    result["thrust_n"] = np.asarray(result["thrust_n"], dtype=float) * factor
    time_s = np.asarray(result["time_s"], dtype=float)
    thrust_n = np.asarray(result["thrust_n"], dtype=float)
    propellant_mass_kg = np.asarray(result["propellant_mass_kg"], dtype=float)
    burned_kg = max(float(propellant_mass_kg[0] - propellant_mass_kg[-1]), 0.0)
    burn_time_s = float(time_s[-1] - time_s[0]) if len(time_s) else 0.0
    total_impulse = float(_trapezoid(thrust_n, time_s)) if len(time_s) > 1 else 0.0

    summary = result["summary"]
    summary["simulation.nominal.total_impulse_ns"] = total_impulse
    summary["simulation.nominal.peak_thrust_n"] = float(np.max(thrust_n))
    summary["simulation.nominal.avg_thrust_n"] = (
        total_impulse / burn_time_s if burn_time_s > 0.0 else 0.0
    )
    summary["simulation.nominal.isp_effective_s"] = (
        total_impulse / (burned_kg * 9.80665) if burned_kg > 1e-9 else 0.0
    )
    return result


def summarize_robustness(scenario_results):
    """Aggregate scenario summaries as mean/std/quantiles and validity ratios."""
    summaries = [
        result["summary"]
        for result in scenario_results
        if result.get("scenario_id") != "nominal"
    ]
    if not summaries:
        return {
            "simulation.robustness.scenario_count": 0.0,
            "simulation.robustness.valid_ratio": 1.0,
            "simulation.robustness.nonzero_curve_ratio": 1.0,
        }

    def values(key):
        return np.asarray([summary[key] for summary in summaries], dtype=float)

    burn_times = values("simulation.nominal.burn_time_s")
    peak_thrusts = values("simulation.nominal.peak_thrust_n")
    impulses = values("simulation.nominal.total_impulse_ns")
    mass_errors = np.asarray(
        [
            summary.get("simulation.nominal.mass_conservation_error_pct", np.inf)
            for summary in summaries
        ],
        dtype=float,
    )
    nonzero = (burn_times > 0.0) & (peak_thrusts > 0.0) & (impulses > 0.0)
    valid = nonzero & (mass_errors < 25.0)

    output = {
        "simulation.robustness.scenario_count": float(len(summaries)),
        "simulation.robustness.valid_ratio": float(np.mean(valid)),
        "simulation.robustness.nonzero_curve_ratio": float(np.mean(nonzero)),
    }
    for label, data in (
        ("burn_time", burn_times),
        ("peak_thrust", peak_thrusts),
        ("total_impulse", impulses),
    ):
        unit = {"burn_time": "s", "peak_thrust": "n", "total_impulse": "ns"}[label]
        output[f"simulation.robustness.{label}_mean_{unit}"] = float(np.mean(data))
        output[f"simulation.robustness.{label}_std_{unit}"] = float(np.std(data, ddof=0))
        output[f"simulation.robustness.{label}_p05_{unit}"] = float(np.quantile(data, 0.05))
        output[f"simulation.robustness.{label}_p50_{unit}"] = float(np.quantile(data, 0.50))
        output[f"simulation.robustness.{label}_p95_{unit}"] = float(np.quantile(data, 0.95))
    return output


def run_robustness_analysis(
    grain,
    motor,
    propellant,
    environment=None,
    scenarios=None,
    monte_carlo_sample_count=0,
    monte_carlo_seed=20260504,
    max_step_size=0.01,
    max_time_points=1000,
    validator=None,
    **simulation_kwargs,
):
    """Run nominal and perturbed detailed ballistics, then aggregate statistics."""
    scenario_list = list(scenarios) if scenarios is not None else default_robustness_scenarios()
    scenario_list.extend(
        build_latin_hypercube_scenarios(
            sample_count=monte_carlo_sample_count,
            seed=monte_carlo_seed,
        )
    )

    nominal = run_detailed_ballistics(
        grain,
        motor,
        propellant,
        environment,
        max_step_size=max_step_size,
        max_time_points=max_time_points,
        **simulation_kwargs,
    )
    nominal["scenario_id"] = "nominal"
    nominal["scenario_kind"] = "nominal"

    results = [nominal]
    for scenario in scenario_list:
        scenario_grain, scenario_motor, scenario_propellant, scenario_environment = (
            _apply_scenario(grain, motor, propellant, environment, scenario)
        )
        result = run_detailed_ballistics(
            scenario_grain,
            scenario_motor,
            scenario_propellant,
            scenario_environment,
            max_step_size=max_step_size,
            max_time_points=max_time_points,
            nozzle_ablation_scale=float(scenario.nozzle_ablation_scale_factor),
            igniter_mass_flow=simulation_kwargs.get("igniter_mass_flow"),
            igniter_burn_time=simulation_kwargs.get("igniter_burn_time", 0.0)
            * float(scenario.igniter_energy_factor),
            igniter_temperature=simulation_kwargs.get("igniter_temperature"),
            burn_area_activation=simulation_kwargs.get("burn_area_activation"),
            ignition_ramp_time=simulation_kwargs.get("ignition_ramp_time", 0.0),
            tail_off_method=simulation_kwargs.get("tail_off_method", "numerical"),
        )
        result = _rescale_thrust(result, scenario.isp_factor)
        result["scenario_id"] = scenario.scenario_id
        result["scenario_kind"] = scenario.scenario_kind
        result["scenario_factors"] = scenario.__dict__.copy()
        if validator is not None:
            result["valid"] = bool(validator(result))
        results.append(result)

    summary = summarize_robustness(results)
    if validator is not None:
        validation = np.asarray(
            [bool(result.get("valid", True)) for result in results if result["scenario_id"] != "nominal"],
            dtype=bool,
        )
        if len(validation):
            summary["simulation.robustness.valid_ratio"] = float(np.mean(validation))

    return {
        "nominal": nominal,
        "scenarios": results[1:],
        "summary": summary,
    }
