"""Pack and assemble histories for batched detailed-ballistics post-processing."""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any, Dict, List, Sequence

import numpy as np

from .kernels.detailed_ballistics import INTERPOLATED_FIELDS


_CANONICAL_FIELDS = {
    "chamber_pressure_pa": "chamber_pressure_pa",
    "free_volume_m3": "free_volume_m3",
    "thrust_n": "thrust_n",
    "exit_pressure_pa": "exit_pressure_pa",
    "exit_velocity_m_s": "exit_velocity_m_s",
    "burn_area_m2": "burn_area_m2",
    "mass_generated_kg_s": "mdot_generated_kg_s",
    "mass_igniter_kg_s": "mdot_igniter_kg_s",
    "mass_nozzle_kg_s": "mdot_nozzle_kg_s",
    "gas_mass_kg": "gas_mass_kg",
    "momentum_ideal_n": "momentum_ideal_n",
    "momentum_n": "momentum_n",
    "pressure_thrust_n": "pressure_n",
    "generated_mass_integral_kg": "generated_mass_integral_kg",
    "igniter_mass_integral_kg": "igniter_mass_integral_kg",
    "nozzle_mass_integral_kg": "nozzle_mass_integral_kg",
    "impulse_integral_ns": "impulse_integral_ns",
    "pressure_throat_integral_ns": "pressure_throat_integral_ns",
}

_RESULT_FIELDS = (
    "time_s", "thrust_n", "chamber_pressure_pa", "free_volume_m3", "regressed_length_m", "burn_area_m2",
    "regression_rate_m_s", "mass_flow_kg_s", "mass_generated_kg_s", "mass_igniter_kg_s", "mass_nozzle_kg_s",
    "gas_mass_kg", "momentum_ideal_n", "momentum_n", "pressure_thrust_n", "generated_mass_integral_kg",
    "igniter_mass_integral_kg", "nozzle_mass_integral_kg", "impulse_integral_ns", "pressure_throat_integral_ns",
    "propellant_mass_kg", "propellant_center_of_mass_position_m", "motor_mass_kg",
    "motor_center_of_mass_position_m", "exit_pressure_pa", "exit_velocity_m_s", "throat_area_m2",
    "throat_diameter_m", "throat_ablation_m", "cf", "ignition_active_fraction",
)


def _output_grid(raw_time, metrics, options):
    """Use the scalar display-grid rules, including burnout-point replacement."""
    resample_step = options.get("resample_step")
    max_time_points = options.get("max_time_points")
    if resample_step is not None and len(raw_time) > 1:
        step = max(float(resample_step), 1e-6)
        count = max(2, int(math.ceil((raw_time[-1] - raw_time[0]) / step)) + 1)
        if max_time_points is not None:
            count = min(count, max(int(max_time_points), 2))
        time_s = np.linspace(float(raw_time[0]), float(raw_time[-1]), count)
        burnout_times = metrics["grain_burnout_times_s"]
        available = list(range(1, count - 1))
        for event_time in sorted({float(t) for t in burnout_times if t is not None}):
            if not available or not raw_time[0] < event_time < raw_time[-1]:
                continue
            index = min(available, key=lambda i: abs(time_s[i] - event_time))
            time_s[index] = event_time
            available.remove(index)
        time_s.sort()
        return time_s

    time_s = raw_time
    if max_time_points is not None and len(time_s) > max_time_points:
        index = np.linspace(0, len(time_s) - 1, int(max_time_points), dtype=int)
        time_s = time_s[index]
    return np.asarray(time_s, dtype=float)


def _interpolation_indices(source, query):
    """Precompute the brackets used by endpoint-clamped ``numpy.interp``."""
    if len(source) == 1:
        zeros = np.zeros(len(query), dtype=np.int32)
        return zeros, zeros.copy(), np.zeros(len(query), dtype=float)
    right = np.searchsorted(source, query, side="right")
    left = np.clip(right - 1, 0, len(source) - 2).astype(np.int32)
    right = left + 1
    fraction = (query - source[left]) / (source[right] - source[left])
    return left, right.astype(np.int32), np.clip(fraction, 0.0, 1.0)


def _activation_profile(view, time_s):
    """Pack supported activation rules; user callbacks stay on the scalar path."""
    source = view._activation_inputs.burn_area_activation
    if source is None:
        return 0.0, 0.0, np.zeros(1), np.zeros(1), np.zeros(len(time_s), dtype=np.int32), \
            np.zeros(len(time_s), dtype=np.int32), np.zeros(len(time_s)), False
    if callable(source):
        return 3.0, 0.0, np.zeros(1), np.zeros(1), np.zeros(len(time_s), dtype=np.int32), \
            np.zeros(len(time_s), dtype=np.int32), np.zeros(len(time_s)), True
    if np.isscalar(source):
        return 1.0, float(source), np.zeros(1), np.zeros(1), np.zeros(len(time_s), dtype=np.int32), \
            np.zeros(len(time_s), dtype=np.int32), np.zeros(len(time_s)), False
    profile = np.asarray(source, dtype=float)
    if profile.ndim != 2 or profile.shape[1] != 2 or len(profile) == 0:
        return 3.0, 0.0, np.zeros(1), np.zeros(1), np.zeros(len(time_s), dtype=np.int32), \
            np.zeros(len(time_s), dtype=np.int32), np.zeros(len(time_s)), True
    left, right, fraction = _interpolation_indices(profile[:, 0], time_s)
    return 2.0, 0.0, profile[:, 0], profile[:, 1], left, right, fraction, False


def _prepare_lane(view, options):
    """Convert one solved view into raw arrays, geometry inputs and scalar result metadata."""
    from ..DetailedBallistics import _validate_dry_hardware

    canonical = view.result
    history = canonical["history"]
    metrics = canonical["metrics"]
    raw_time = np.asarray(history["time_s"], dtype=float)
    if raw_time.ndim != 1 or not len(raw_time):
        raise ValueError("detailed ballistics requires a non-empty one-dimensional time history")
    raw_regressions = np.asarray(history["regression_m"], dtype=float)
    if raw_regressions.ndim == 1:
        raw_regressions = raw_regressions[:, None]
    if raw_regressions.ndim != 2 or raw_regressions.shape[0] != len(raw_time):
        raise ValueError("canonical regression history must be aligned with time_s")
    query = _output_grid(raw_time, metrics, options)
    if len(query) == 0 or not np.all(np.isfinite(query)) or np.any(np.diff(query) <= 0.0):
        raise ValueError("detailed ballistics requires a finite strictly increasing time domain")
    if raw_regressions.shape[1] != len(view.motor.grains):
        raise ValueError("canonical regression history must contain one column per motor grain")
    _validate_dry_hardware(view.motor)

    left, right, fraction = _interpolation_indices(raw_time, query)
    raw_values = {}
    for name, canonical_name in _CANONICAL_FIELDS.items():
        values = np.asarray(history[canonical_name], dtype=float)
        if values.ndim != 1 or len(values) != len(raw_time):
            raise ValueError(f"canonical history {canonical_name!r} must align with time_s")
        raw_values["raw_" + name] = values

    grains = list(view.motor.grains)
    if not grains:
        raise ValueError("detailed ballistics requires at least one grain")
    grain_arrays = {name: np.zeros(len(grains), dtype=bool if name in ("grain_is_star", "grain_ends_burn") else float)
                    for name in (
        "grain_is_star", "grain_inner_radius_m", "grain_outer_radius_m", "grain_height_m",
        "grain_ends_burn", "grain_star_points", "grain_epsilon", "grain_slot_fraction", "grain_stack_start_m",
    )}
    stack_start = 0.0
    unsupported = False
    for index, grain in enumerate(grains):
        if grain.geometry not in ("tubular", "star"):
            unsupported = True
        grain_arrays["grain_is_star"][index] = grain.geometry == "star"
        grain_arrays["grain_inner_radius_m"][index] = grain.initial_inner_radius
        grain_arrays["grain_outer_radius_m"][index] = grain.outer_radius
        grain_arrays["grain_height_m"][index] = grain.initial_height
        grain_arrays["grain_ends_burn"][index] = grain.ends_burn
        grain_arrays["grain_star_points"][index] = grain.n_points
        grain_arrays["grain_epsilon"][index] = grain.epsilon
        grain_arrays["grain_slot_fraction"][index] = grain.slot_fraction
        grain_arrays["grain_stack_start_m"][index] = stack_start
        stack_start += grain.initial_height + view.motor.grain_separation

    activation = _activation_profile(view, query)
    unsupported = unsupported or activation[-1]
    arrays = {
        **raw_values,
        "raw_time_s": raw_time,
        "raw_regression_m": raw_regressions,
        "query_time_s": query,
        "interp_left": left,
        "interp_right": right,
        "interp_fraction": fraction,
        "grain_valid": np.ones(len(grains), dtype=bool),
        **grain_arrays,
        "propellant_density_kg_m3": np.asarray(view.propellant.density),
        "dry_mass_kg": np.asarray(view.motor.dry_mass_kg),
        "dry_cg_m": np.asarray(view.motor.dry_center_of_mass_position_m),
        "base_throat_radius_m": np.asarray(math.sqrt(max(view.motor.nozzle_throat_area, 0.0) / math.pi)),
        "environment_pressure_pa": np.asarray(view.environment_pressure),
        "nozzle_ablation_scale": np.asarray(float(options.get("nozzle_ablation_scale", 1.0))),
        "ablation_pressure_exponent": np.asarray(float(options.get("ablation_pressure_exponent", 0.42))),
        "ablation_mass_flow_exponent": np.asarray(float(options.get("ablation_mass_flow_exponent", 0.32))),
        "activation_mode": np.asarray(activation[0]),
        "activation_value": np.asarray(activation[1]),
        "activation_profile_time": activation[2],
        "activation_profile_value": activation[3],
        "activation_left": activation[4],
        "activation_right": activation[5],
        "activation_fraction": activation[6],
        "ignition_ramp_time_s": np.asarray(view._activation_inputs.ignition_ramp_time),
        "n_source": np.asarray(len(raw_time), dtype=np.int32),
        "n_output": np.asarray(len(query), dtype=np.int32),
        "grain_count": np.asarray(len(grains), dtype=np.int32),
    }
    metadata = {
        "view": view,
        "canonical": canonical,
        "metrics": metrics,
        "raw_time": raw_time,
        "raw_pressure": raw_values["raw_chamber_pressure_pa"],
        "query_time": query,
        "grain_valid": arrays["grain_valid"],
        "activation_profile_time": activation[2],
        "gamma": float(view.propellant.specific_heat_ratio),
        "options": dict(options),
        "unsupported": unsupported,
    }
    return arrays, metadata


@dataclass
class DetailedBallisticsBatch:
    """Padded canonical histories and per-lane inputs for detailed post-processing."""

    arrays: Dict[str, np.ndarray]
    lanes: List[Dict[str, Any]]

    def __len__(self):
        return len(self.lanes)

    def namespace(self, xp):
        """Convert packed numeric arrays to NumPy, JAX NumPy or another array namespace."""
        return {name: xp.asarray(value) for name, value in self.arrays.items()}

    @property
    def unsupported_lanes(self):
        """Indices whose callable activation or custom grain geometry needs the scalar path."""
        return [i for i, lane in enumerate(self.lanes) if lane["unsupported"]]

    def select(self, indices: Sequence[int]):
        """Select lanes in order and trim each padded history axis to the selected group."""
        ids = np.asarray(indices, dtype=int)
        lanes = [self.lanes[i] for i in ids]
        arrays = {}
        source_size = max((len(lane["raw_time"]) for lane in lanes), default=1)
        output_size = max((len(lane["query_time"]) for lane in lanes), default=1)
        grain_size = max((len(lane["grain_valid"]) for lane in lanes), default=1)
        activation_size = max((len(lane["activation_profile_time"]) for lane in lanes), default=1)
        for name, value in self.arrays.items():
            selected = value[ids]
            if name.startswith("raw_") and name != "raw_regression_m":
                selected = selected[:, :source_size]
            elif name == "raw_regression_m":
                selected = selected[:, :source_size, :grain_size]
            elif name in ("query_time_s", "interp_left", "interp_right", "interp_fraction",
                          "activation_left", "activation_right", "activation_fraction"):
                selected = selected[:, :output_size]
            elif name in ("activation_profile_time", "activation_profile_value"):
                selected = selected[:, :activation_size]
            elif name.startswith("grain_") and value.ndim == 2:
                selected = selected[:, :grain_size]
            arrays[name] = selected
        return DetailedBallisticsBatch(arrays, lanes)

    @classmethod
    def from_views(cls, views, options=None):
        """Pack aligned solved views and per-lane ``build_detailed_ballistics`` options."""
        views = list(views)
        if isinstance(options, dict) or options is None:
            options = [dict(options or {}) for _ in views]
        else:
            options = [dict(value) for value in options]
        if len(options) != len(views):
            raise ValueError("detailed-ballistics options must have one mapping per lane")
        prepared = [_prepare_lane(view, option) for view, option in zip(views, options)]
        if not prepared:
            return cls({}, [])

        source_size = max(len(item[1]["raw_time"]) for item in prepared)
        output_size = max(len(item[1]["query_time"]) for item in prepared)
        grain_size = max(len(item[1]["grain_valid"]) for item in prepared)
        activation_size = max(len(item[1]["activation_profile_time"]) for item in prepared)
        arrays: Dict[str, np.ndarray] = {}
        for name in prepared[0][0]:
            sample = prepared[0][0][name]
            if name.startswith("raw_") and name != "raw_regression_m":
                arrays[name] = np.zeros((len(prepared), source_size), dtype=float)
            elif name == "raw_regression_m":
                arrays[name] = np.zeros((len(prepared), source_size, grain_size), dtype=float)
            elif name in ("query_time_s",):
                arrays[name] = np.zeros((len(prepared), output_size), dtype=float)
            elif name in ("interp_left", "interp_right", "activation_left", "activation_right"):
                arrays[name] = np.zeros((len(prepared), output_size), dtype=np.int32)
            elif name in ("interp_fraction", "activation_fraction"):
                arrays[name] = np.zeros((len(prepared), output_size), dtype=float)
            elif name in ("activation_profile_time", "activation_profile_value"):
                arrays[name] = np.zeros((len(prepared), activation_size), dtype=float)
            elif name in ("n_source", "n_output", "grain_count"):
                arrays[name] = np.zeros(len(prepared), dtype=np.int32)
            elif name.startswith("grain_"):
                arrays[name] = np.zeros((len(prepared), grain_size), dtype=bool if sample.dtype == bool else float)
            else:
                arrays[name] = np.zeros(len(prepared), dtype=float)

        lanes = []
        for lane_index, (values, metadata) in enumerate(prepared):
            source_count = len(metadata["raw_time"])
            output_count = len(metadata["query_time"])
            grain_count = len(values["grain_valid"])
            activation_count = len(metadata["activation_profile_time"])
            lanes.append(metadata)
            for name, value in values.items():
                if name.startswith("raw_") and name != "raw_regression_m":
                    arrays[name][lane_index, :source_count] = value
                    if source_count < source_size:
                        arrays[name][lane_index, source_count:] = value[-1]
                elif name == "raw_regression_m":
                    arrays[name][lane_index, :source_count, :grain_count] = value
                    if source_count < source_size:
                        arrays[name][lane_index, source_count:, :grain_count] = value[-1]
                elif name in ("query_time_s",):
                    arrays[name][lane_index, :output_count] = value
                    arrays[name][lane_index, output_count:] = value[-1]
                elif name in ("interp_left", "interp_right", "activation_left", "activation_right",
                              "interp_fraction", "activation_fraction"):
                    arrays[name][lane_index, :output_count] = value
                    if output_count < output_size:
                        arrays[name][lane_index, output_count:] = value[-1]
                elif name in ("activation_profile_time", "activation_profile_value"):
                    arrays[name][lane_index, :activation_count] = value
                    if activation_count < activation_size:
                        arrays[name][lane_index, activation_count:] = value[-1]
                elif name in ("n_source", "n_output", "grain_count"):
                    arrays[name][lane_index] = value
                elif name.startswith("grain_"):
                    arrays[name][lane_index, :grain_count] = value
                else:
                    arrays[name][lane_index] = value
            # Metadata used by selection and assembly; these stay as host references.
            metadata["query_time"] = values["query_time_s"]
            metadata["grain_valid"] = np.ones(grain_count, dtype=bool)
            metadata["activation_profile_time"] = values["activation_profile_time"]
        return cls(arrays, lanes)


def assemble_detailed_ballistics(batch: DetailedBallisticsBatch, output):
    """Restore each detailed mapping, canonical metadata and lane-sized output arrays."""
    from ..DetailedBallistics import DETAILED_BALLISTICS_SCHEMA_VERSION, _validate_result_series

    results = []
    for index, lane in enumerate(batch.lanes):
        if lane["unsupported"]:
            results.append(None)
            continue
        count = len(lane["query_time"])
        result = {name: np.asarray(output[name][index, :count], dtype=float) for name in _RESULT_FIELDS}
        canonical_metrics = lane["metrics"]
        total_impulse_ns = float(canonical_metrics["total_impulse_ns"])
        burn_time_s = float(canonical_metrics["propellant_burn_duration_s"])
        nozzle_flow_duration_s = float(canonical_metrics["nozzle_flow_duration_s"])
        propellant_burned_kg = float(canonical_metrics["propellant_mass_consumed_kg"])
        generated_integral_kg = float(canonical_metrics["generated_mass_integral_kg"])
        raw_time = lane["raw_time"]
        raw_pressure = lane["raw_pressure"]
        pressure_rise = (
            float(np.max(np.abs(np.diff(raw_pressure) / np.maximum(np.diff(raw_time), 1e-12))))
            if count > 1 else 0.0
        )
        summary = {
            "simulation.schema_version": DETAILED_BALLISTICS_SCHEMA_VERSION,
            "simulation.model_fidelity": "solidpy_detailed_ballistics",
            "simulation.nominal.burn_time_s": burn_time_s,
            "simulation.nominal.peak_thrust_n": float(canonical_metrics["peak_thrust_n"]),
            "simulation.nominal.avg_thrust_n": total_impulse_ns / nozzle_flow_duration_s
            if nozzle_flow_duration_s > 0.0 else 0.0,
            "simulation.nominal.total_impulse_ns": total_impulse_ns,
            "simulation.nominal.isp_effective_s": total_impulse_ns / (propellant_burned_kg * 9.80665)
            if propellant_burned_kg > 1e-9 else 0.0,
            "simulation.nominal.nozzle_flow_duration_s": nozzle_flow_duration_s,
            "simulation.nominal.mass_flow_avg_kg_s": float(canonical_metrics["mass_flow_avg_generated_kg_s"]),
            "simulation.nominal.mass_flow_avg_nozzle_kg_s": float(canonical_metrics["mass_flow_avg_nozzle_kg_s"]),
            "simulation.nominal.max_mass_flow_kg_s": float(canonical_metrics["max_generated_mass_flow_kg_s"]),
            "simulation.nominal.max_nozzle_mass_flow_kg_s": float(canonical_metrics["max_nozzle_mass_flow_kg_s"]),
            "simulation.nominal.chamber_pressure_max_mpa": float(canonical_metrics["peak_chamber_pressure_pa"] / 1e6),
            "simulation.nominal.pressure_rise_rate_max_mpa_s": pressure_rise / 1e6,
            "simulation.nominal.mass_conservation_error_pct": float(canonical_metrics["mass_flow_balance_error_pct"]),
            "simulation.nominal.generated_mass_integral_kg": generated_integral_kg,
            "simulation.nominal.nozzle_mass_integral_kg": float(canonical_metrics["nozzle_mass_integral_kg"]),
            "simulation.nominal.igniter_mass_injected_kg": float(canonical_metrics["igniter_mass_injected_kg"]),
            "simulation.nominal.gas_mass_initial_kg": float(canonical_metrics["gas_mass_initial_kg"]),
            "simulation.nominal.gas_mass_cutoff_kg": float(canonical_metrics["gas_mass_cutoff_kg"]),
            "simulation.nominal.cstar_effective_m_s": float(canonical_metrics["pressure_throat_integral_ns"])
            / max(generated_integral_kg, 1e-9) if generated_integral_kg > 1e-9 else 0.0,
            "simulation.nominal.final_throat_diameter_mm": 1000.0 * float(result["throat_diameter_m"][-1]),
            "simulation.nominal.throat_ablation_mm": 1000.0 * float(result["throat_ablation_m"][-1]),
            "simulation.nominal.ignition_active_fraction_final": float(result["ignition_active_fraction"][-1]),
        }
        canonical = lane["canonical"]
        result.update({
            "schema_version": DETAILED_BALLISTICS_SCHEMA_VERSION,
            "interpolation": {
                "method": "linear", "outside_domain": "endpoint_clamping", "extrapolation": False,
                "domain": "time_s",
            },
            "gamma": lane["gamma"],
            "summary": summary,
            "canonical_result": canonical,
            "status": canonical["status"],
            "provenance": canonical["provenance"],
        })
        _validate_result_series(result, result["time_s"])
        results.append(result)
    return results


def scalar_detailed_ballistics(batch: DetailedBallisticsBatch):
    """Evaluate lanes through the public scalar detailed-ballistics implementation."""
    from ..DetailedBallistics import build_detailed_ballistics

    return [build_detailed_ballistics(lane["view"], **lane["options"]) for lane in batch.lanes]
