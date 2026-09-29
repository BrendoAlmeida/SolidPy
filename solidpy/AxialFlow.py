"""Axial source-throughput diagnostics for a single-outlet grain stack."""

from numbers import Integral, Real

import numpy as np


AXIAL_BACKEND = "source_accumulation_single_outlet_v1"


def _history_arrays(motor, history):
    arrays = {}
    names = ("time_s", "regression_m", "mdot_generated_grains_kg_s", "mdot_igniter_kg_s")
    for name in names:
        if name not in history:
            raise ValueError(f"axial diagnostic requires history[{name!r}]")
        try:
            arrays[name] = np.asarray(history[name], dtype=float)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"{name} must contain finite real numbers") from exc
        if not np.all(np.isfinite(arrays[name])) or np.any(arrays[name] < 0.0):
            raise ValueError(f"{name} must contain finite non-negative values")
    time = arrays["time_s"]
    if time.ndim != 1 or not len(time) or np.any(np.diff(time) <= 0.0):
        raise ValueError("time_s must be a nonempty strictly increasing vector")
    for name in ("regression_m", "mdot_generated_grains_kg_s"):
        if arrays[name].shape != (len(time), len(motor.grains)):
            raise ValueError(f"{name} must have shape (time, grains)")
    if arrays["mdot_igniter_kg_s"].shape != time.shape:
        raise ValueError("mdot_igniter_kg_s must have shape (time,)")
    limits = np.asarray([g.burnout_regression_m for g in motor.grains])
    if np.any(arrays["regression_m"] > limits * (1 + 64 * np.finfo(float).eps)):
        raise ValueError("regression_m exceeds grain burnout regression")
    return arrays


def evaluate_axial_mass_flux(
    motor, history, *, stations_per_grain=5, nozzle_direction="negative",
    flow_arrangement="single_outlet",
):
    """Accumulate upstream sources and divide by instantaneous local port area.

    Coordinates follow ``Motor.grain_axial_positions_m``: positive z points
    away from the default nozzle. ``nozzle_direction='positive'`` reverses the
    outlet direction. Each active grain distributes lateral sources uniformly
    along its remaining length and places burning end-face sources at the faces.
    Stations include both faces and use the outlet-side trace of point sources.

    This diagnostic does not allocate gas storage or drainage spatially. It
    therefore represents source throughput, not transient nozzle discharge or
    CFD velocity. Igniter mass enters at the closed upstream end. Only a single
    outlet is supported; split-flow arrangements require a different backend.
    """
    if (isinstance(stations_per_grain, (bool, np.bool_))
            or not isinstance(stations_per_grain, Integral) or stations_per_grain < 2):
        raise ValueError("stations_per_grain must be an integer >= 2")
    if nozzle_direction not in {"negative", "positive"}:
        raise ValueError("nozzle_direction must be negative or positive")
    if flow_arrangement != "single_outlet":
        raise ValueError("axial backend supports only single_outlet flow arrangements")
    if any(g.geometry not in {"tubular", "star"} for g in motor.grains):
        raise ValueError("axial backend supports only tubular and star grain geometry")
    arrays = _history_arrays(motor, history)
    time = arrays["time_s"]
    regression = arrays["regression_m"]
    generated = arrays["mdot_generated_grains_kg_s"]
    order = np.argsort(motor.grain_axial_positions_m)
    if nozzle_direction == "negative":
        order = order[::-1]
    fractions = np.linspace(0.0, 1.0, stations_per_grain)
    coordinate_fractions = 1.0 - fractions if nozzle_direction == "negative" else fractions
    shape = (len(time), len(order) * stations_per_grain)
    positions, areas, flow = (np.empty(shape) for _ in range(3))
    upstream = arrays["mdot_igniter_kg_s"].copy()
    face_sources = np.zeros_like(generated)
    lateral_sources = np.zeros_like(generated)
    upstream_sources = np.zeros_like(generated)
    grain_indices, station_ids = [], []
    for ordered_index, grain_index in enumerate(order):
        grain = motor.grains[grain_index]
        block = slice(ordered_index * stations_per_grain, (ordered_index + 1) * stations_per_grain)
        depths = np.minimum(regression[:, grain_index], grain.burnout_regression_m)
        active = depths < grain.burnout_regression_m
        start = motor.grain_axial_positions_m[grain_index]
        if grain.ends_burn:
            lengths = np.full(len(time), grain.initial_height)
            starts = np.full(len(time), start)
        else:
            lengths = np.maximum(grain.initial_height - 2.0 * depths, 0.0)
            starts = start + depths
        positions[:, block] = starts[:, None] + lengths[:, None] * coordinate_fractions
        port = np.asarray([grain.evaluate_port_area(r) for r in depths])
        port = np.where(active, port, motor.chamber_area)
        if np.any(port <= 0.0) or not np.all(np.isfinite(port)):
            raise ValueError("instantaneous port area must be finite and positive")
        areas[:, block] = port[:, None]
        burn_area = np.asarray([grain.evaluate_burn_area(r) for r in depths])
        end_area = np.zeros(len(time)) if grain.ends_burn else np.where(
            active, np.maximum(np.pi * grain.outer_radius**2 - port, 0.0), 0.0,
        )
        if np.any((~active) & (generated[:, grain_index] > 0.0)):
            raise ValueError("burned-out grains cannot retain generated mass sources")
        face_fraction = np.divide(end_area, burn_area, out=np.zeros(len(time)), where=burn_area > 0)
        face = generated[:, grain_index] * face_fraction
        lateral = generated[:, grain_index] - 2.0 * face
        upstream_sources[:, grain_index] = upstream
        face_sources[:, grain_index] = face
        lateral_sources[:, grain_index] = lateral
        local = upstream[:, None] + face[:, None] + lateral[:, None] * fractions
        local[:, -1] += face
        flow[:, block] = local
        upstream += generated[:, grain_index]
        grain_indices.extend([int(grain_index)] * stations_per_grain)
        station_ids.extend(f"grain:{grain_index}:station:{j}" for j in range(stations_per_grain))
    flux = flow / areas
    time_index, station_index = np.unravel_index(np.argmax(flux), flux.shape)
    metrics = {
        "max_axial_mass_flux_kg_m2_s": float(flux[time_index, station_index]),
        "max_axial_mass_flux_time_s": float(time[time_index]),
        "max_axial_mass_flux_grain_index": grain_indices[station_index],
        "max_axial_mass_flux_station_id": station_ids[station_index],
        "max_axial_mass_flux_position_m": float(positions[time_index, station_index]),
        "max_axial_mass_flux_time_index": int(time_index),
        "max_axial_mass_flux_station_index": int(station_index),
    }
    return {
        "backend": AXIAL_BACKEND, "status": "uncalibrated_diagnostic",
        "nozzle_direction": nozzle_direction, "flow_arrangement": flow_arrangement,
        "coordinate_convention": "positive_z_from_initial_nozzle_side_stack_reference",
        "stations_per_grain": int(stations_per_grain), "grain_order": order.tolist(),
        "station_ids": station_ids, "grain_indices": np.asarray(grain_indices),
        "station_fraction_toward_outlet": np.tile(fractions, len(order)),
        "time_s": time.copy(), "positions_m": positions, "port_area_m2": areas,
        "mdot_axial_kg_s": flow, "mass_flux_kg_m2_s": flux,
        "mdot_upstream_grains_kg_s": upstream_sources,
        "mdot_lateral_grains_kg_s": lateral_sources,
        "mdot_end_face_grains_kg_s": face_sources,
        "geometry": [{"geometry_model": g.geometry_model, "outer_radius_m": g.outer_radius,
                      "initial_inner_radius_m": g.initial_inner_radius, "initial_height_m": g.initial_height,
                      "ends_inhibited": g.ends_burn, "n_points": g.n_points,
                      "epsilon_rad": g.epsilon, "slot_fraction": g.slot_fraction,
                      "start_position_m": motor.grain_axial_positions_m[i]}
                     for i, g in enumerate(motor.grains)],
        "chamber_area_m2": motor.chamber_area,
        "assumptions": {
            "source_distribution": "uniform_lateral_and_point_end_faces",
            "point_source_trace": "outlet_side",
            "igniter_location": "closed_upstream_end",
            "gas_storage_distribution": "not_modeled",
            "gas_drainage_distribution": "not_modeled",
            "empty_grain_port": "chamber_cross_section",
            "erosion_feedback": False, "physical_validity_gate": False,
        },
        "metrics": metrics, "convergence": {"status": "not_evaluated"},
    }


def compare_axial_diagnostics(coarse, refined, *, scale_floor=1e-12):
    """Report temporal/profile and spatial-sampling errors without a validity gate.

    Peak deltas compare each complete sampled history. Profile errors compare
    matching coarse stations on the refined time grid
    within the common time interval. Linear temporal interpolation includes
    source discontinuities, so their sampling error remains visible. Spatial
    peak errors also compare the maximum of each grain's two endpoint traces.
    For this backend the interior profile is affine and nondecreasing toward
    the outlet; including both endpoints captures the exact spatial maximum.
    """
    if isinstance(scale_floor, (bool, np.bool_)) or not isinstance(scale_floor, Real):
        raise ValueError("scale_floor must be finite and positive")
    try:
        scale_floor = float(scale_floor)
    except (TypeError, ValueError) as exc:
        raise ValueError("scale_floor must be finite and positive") from exc
    if not np.isfinite(scale_floor) or scale_floor <= 0.0:
        raise ValueError("scale_floor must be finite and positive")
    for name in ("backend", "nozzle_direction", "flow_arrangement", "geometry", "chamber_area_m2", "assumptions"):
        if coarse[name] != refined[name]:
            raise ValueError(f"axial diagnostics differ in {name}")
    if coarse["backend"] != AXIAL_BACKEND:
        raise ValueError("unsupported axial diagnostic backend")
    shared = refined["time_s"][(refined["time_s"] >= coarse["time_s"][0])
                               & (refined["time_s"] <= coarse["time_s"][-1])]
    if not len(shared):
        raise ValueError("axial diagnostics have no common time interval")
    peak_coarse = coarse["metrics"]["max_axial_mass_flux_kg_m2_s"]
    peak_refined = refined["metrics"]["max_axial_mass_flux_kg_m2_s"]
    profile_error = 0.0
    for station, grain_index in enumerate(coarse["grain_indices"]):
        fraction = coarse["station_fraction_toward_outlet"][station]
        area_station = int(np.flatnonzero(refined["grain_indices"] == grain_index)[0])
        refined_profile = (refined["mdot_upstream_grains_kg_s"][:, grain_index]
                           + refined["mdot_end_face_grains_kg_s"][:, grain_index]
                           + fraction * refined["mdot_lateral_grains_kg_s"][:, grain_index])
        if fraction == 1.0:
            refined_profile = refined_profile + refined["mdot_end_face_grains_kg_s"][:, grain_index]
        refined_profile = refined_profile / refined["port_area_m2"][:, area_station]
        a = np.interp(shared, coarse["time_s"], coarse["mass_flux_kg_m2_s"][:, station])
        b = np.interp(shared, refined["time_s"], refined_profile)
        profile_error = max(profile_error, float(np.max(np.abs(a - b))))
    spatial_errors = {}
    for label, diagnostic in (("coarse", coarse), ("refined", refined)):
        count = diagnostic["stations_per_grain"]
        endpoints = np.arange(count - 1, len(diagnostic["station_ids"]), count)
        exact_peak = float(np.max(diagnostic["mass_flux_kg_m2_s"][:, endpoints]))
        sampled_peak = diagnostic["metrics"]["max_axial_mass_flux_kg_m2_s"]
        spatial_errors[label] = abs(sampled_peak - exact_peak) / max(abs(exact_peak), scale_floor)
    return {
        "status": "evaluated_diagnostic_only", "scale_floor_kg_m2_s": scale_floor,
        "temporal": {
            "peak_relative_delta": abs(peak_refined - peak_coarse) / max(abs(peak_refined), scale_floor),
            "profile_relative_max_error": profile_error / max(abs(peak_refined), scale_floor),
            "comparison_start_s": float(shared[0]), "comparison_end_s": float(shared[-1]),
            "comparison_samples": len(shared), "method": "linear_time_interpolation_at_matching_stations",
            "peak_interval": "each_complete_sampled_history",
        },
        "spatial": {
            "coarse_stations_per_grain": coarse["stations_per_grain"],
            "refined_stations_per_grain": refined["stations_per_grain"],
            "peak_relative_errors": spatial_errors,
            "method": "affine_interior_profile_with_exact_outlet_endpoint_maximum",
        },
        "physical_validity_gate": False,
    }
