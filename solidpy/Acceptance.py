# -*- coding: utf-8 -*-

"""Numerical acceptance checks for reproducible v12 evaluations."""

from __future__ import annotations

import math
from collections.abc import Mapping


V12_NUMERICAL_ACCEPTANCE_POLICY = "v12_numerical_acceptance_v1"

_CONVERGENCE_LIMITS = {
    "peak_chamber_pressure_pa": ("peak_chamber_pressure_pa", 0.02, 1.0),
    "peak_thrust_n": ("peak_thrust_n", 0.02, 1e-3),
    "max_generated_mass_flow_kg_s": ("max_generated_mass_flow_kg_s", 0.02, 1e-9),
    "max_nozzle_mass_flow_kg_s": ("max_nozzle_mass_flow_kg_s", 0.02, 1e-9),
    "total_impulse_ns": ("total_impulse_ns", 0.01, 1e-3),
    "generated_mass_integral_kg": ("generated_mass_integral_kg", 0.01, 1e-9),
    "nozzle_mass_integral_kg": ("nozzle_mass_integral_kg", 0.01, 1e-9),
}


def _result_parts(result, label):
    if not isinstance(result, Mapping):
        raise TypeError(f"{label} must be a canonical simulation result mapping")
    metrics = result.get("metrics")
    status = result.get("status")
    if not isinstance(metrics, Mapping) or not isinstance(status, Mapping):
        raise ValueError(f"{label} must contain metrics and status mappings")
    return metrics, status


def _finite_metric(metrics, name):
    value = metrics.get(name)
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    value = float(value)
    return value if math.isfinite(value) else None


def evaluate_numerical_acceptance(
    coarse_result,
    refined_result,
    *,
    policy_id=V12_NUMERICAL_ACCEPTANCE_POLICY,
    scale_floors=None,
):
    """Compare two canonical results against the v12 numerical policy.

    Convergence deltas use ``abs(refined - coarse) / max(abs(refined), floor)``.
    The result is ``incomplete`` if either run lacks completed numerical
    blowdown, a required metric, or a finite mass-balance error.
    """
    if policy_id != V12_NUMERICAL_ACCEPTANCE_POLICY:
        raise ValueError(f"unsupported numerical acceptance policy: {policy_id!r}")
    if scale_floors is None:
        scale_floors = {}
    if not isinstance(scale_floors, Mapping):
        raise TypeError("scale_floors must be a mapping")

    coarse_metrics, coarse_status = _result_parts(coarse_result, "coarse_result")
    refined_metrics, refined_status = _result_parts(refined_result, "refined_result")
    coarse_provenance = coarse_result.get("provenance")
    refined_provenance = refined_result.get("provenance")
    if not isinstance(coarse_provenance, Mapping):
        coarse_provenance = {}
    if not isinstance(refined_provenance, Mapping):
        refined_provenance = {}
    required_complete = all(
        status.get("completed") is True
        and status.get("numerical_blowdown_completed") is True
        for status in (coarse_status, refined_status)
    )

    mass_balance = {}
    for label, metrics in (("coarse", coarse_metrics), ("refined", refined_metrics)):
        value = _finite_metric(metrics, "mass_flow_balance_error_pct")
        mass_balance[label] = value

    convergence = {}
    missing_metrics = []
    for output_name, (metric_name, limit, default_floor) in _CONVERGENCE_LIMITS.items():
        floor = scale_floors.get(output_name, default_floor)
        if isinstance(floor, bool) or not isinstance(floor, (int, float)):
            raise TypeError(f"scale floor for {output_name} must be numeric")
        floor = float(floor)
        if not math.isfinite(floor) or floor <= 0.0:
            raise ValueError(f"scale floor for {output_name} must be finite and positive")
        coarse_value = _finite_metric(coarse_metrics, metric_name)
        refined_value = _finite_metric(refined_metrics, metric_name)
        if coarse_value is None or refined_value is None:
            missing_metrics.append(output_name)
            convergence[output_name] = {
                "coarse": coarse_value,
                "refined": refined_value,
                "scale_floor": floor,
                "limit": limit,
                "relative_delta": None,
                "passed": None,
            }
            continue
        delta = abs(refined_value - coarse_value) / max(abs(refined_value), floor)
        convergence[output_name] = {
            "coarse": coarse_value,
            "refined": refined_value,
            "scale_floor": floor,
            "limit": limit,
            "relative_delta": delta,
            "passed": delta <= limit,
        }

    missing_balance = any(value is None for value in mass_balance.values())
    incomplete_reasons = []
    if not required_complete:
        incomplete_reasons.append("numerical_blowdown_not_completed")
    if missing_metrics:
        incomplete_reasons.append("required_metrics_missing_or_non_finite")
    if missing_balance:
        incomplete_reasons.append("mass_balance_error_missing_or_non_finite")

    provider_hashes = (
        coarse_provenance.get("physics_provider_hash"),
        refined_provenance.get("physics_provider_hash"),
    )
    if not all(isinstance(value, str) and value for value in provider_hashes):
        incomplete_reasons.append("physics_provider_hash_missing")
    elif provider_hashes[0] != provider_hashes[1]:
        incomplete_reasons.append("physical_inputs_or_provider_mismatch")

    required_provenance = (
        "eta_c_applied",
        "eta_cf_applied",
        "discharge_coefficient_applied",
        "efficiency_semantics",
        "cea_used",
    )
    if any(
        key not in provenance
        for provenance in (coarse_provenance, refined_provenance)
        for key in required_provenance
    ):
        incomplete_reasons.append("required_provenance_missing")
    else:
        efficiency_fields = required_provenance[:3]
        coarse_efficiencies = tuple(coarse_provenance[key] for key in efficiency_fields)
        refined_efficiencies = tuple(refined_provenance[key] for key in efficiency_fields)
        if coarse_efficiencies != refined_efficiencies:
            incomplete_reasons.append("applied_efficiencies_mismatch")
        if (
            coarse_provenance["efficiency_semantics"] != "native_split"
            or refined_provenance["efficiency_semantics"] != "native_split"
            or coarse_provenance["cea_used"] is not False
            or refined_provenance["cea_used"] is not False
        ):
            incomplete_reasons.append("unsupported_v12_physics_semantics")

    if incomplete_reasons:
        status = "incomplete"
        passed = False
    else:
        mass_balance_passed = all(value <= 1.0 for value in mass_balance.values())
        convergence_passed = all(item["passed"] for item in convergence.values())
        passed = mass_balance_passed and convergence_passed
        status = "passed" if passed else "failed"

    return {
        "policy_id": policy_id,
        "status": status,
        "passed": passed,
        "mass_balance_error_limit_pct": 1.0,
        "mass_balance_error_pct": mass_balance,
        "mass_balance_passed": (
            None if missing_balance else all(value <= 1.0 for value in mass_balance.values())
        ),
        "convergence": convergence,
        "missing_metrics": missing_metrics,
        "incomplete_reasons": incomplete_reasons,
        "provenance": {
            "coarse": {
                "physics_provider_hash": coarse_provenance.get("physics_provider_hash"),
                "solidpy_git_sha": coarse_provenance.get("solidpy_git_sha"),
            },
            "refined": {
                "physics_provider_hash": refined_provenance.get("physics_provider_hash"),
                "solidpy_git_sha": refined_provenance.get("solidpy_git_sha"),
            },
        },
    }
