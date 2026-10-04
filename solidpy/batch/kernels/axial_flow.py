"""Batched axial source-throughput metrics for native burn histories.

The public scalar implementation in :mod:`solidpy.AxialFlow` remains the oracle.  This module contains only
array-namespace kernels so the same calculation runs with NumPy and JAX.
"""

from __future__ import annotations

import math

from . import geometry, rhs


_PI = math.pi
_STATIONS_PER_GRAIN = 5


def axial_mass_flux_metrics(
    xp,
    P,
    time_s,
    regression_m,
    generated_grains_kg_s,
    igniter_kg_s,
    n_points,
    grain_order,
    grain_start_positions_m,
    chamber_area_m2,
    n_grains,
    geometry_supported,
):
    """Return the scalar axial evaluator's maximum metrics for a batch of native histories.

    ``time_s`` has shape ``[B, T]``; regression and generated source histories have shape ``[B, T, G]``.
    ``grain_order`` and ``grain_start_positions_m`` describe the outlet-to-upstream layout for each lane.
    The station loop is static and small (five stations): retaining only each grain's current maximum avoids
    materializing a ``[B, T, G, stations]`` tensor.  Strict ``>`` updates preserve the first station on ties.
    """
    lanes, time_count = time_s.shape
    grains = regression_m.shape[-1]
    order = xp.asarray(grain_order)
    starts = xp.asarray(grain_start_positions_m)
    n_grains = xp.asarray(n_grains)
    n_points = xp.asarray(n_points)
    valid_time = xp.arange(time_count)[None, :] < n_points[:, None]
    valid_grain = xp.arange(grains)[None, :] < n_grains[:, None]
    history_mask = valid_time[:, :, None] & valid_grain[:, None, :]

    # These checks mirror _history_arrays.  Invalid lanes are sent back through the scalar oracle by assembly,
    # which retains its precise ValueError/unsupported-result behavior.
    finite_nonnegative_time = xp.isfinite(time_s) & (time_s >= 0.0)
    time_ok = xp.all(xp.where(valid_time, finite_nonnegative_time, True), axis=1) & (n_points > 0)
    if time_count > 1:
        adjacent_valid = xp.arange(time_count - 1)[None, :] < (n_points[:, None] - 1)
        increasing = time_s[:, 1:] > time_s[:, :-1]
        time_ok = time_ok & xp.all(xp.where(adjacent_valid, increasing, True), axis=1)

    regression_ok = xp.all(
        xp.where(history_mask, xp.isfinite(regression_m) & (regression_m >= 0.0), True), axis=(1, 2)
    )
    generated_ok = xp.all(
        xp.where(history_mask, xp.isfinite(generated_grains_kg_s) & (generated_grains_kg_s >= 0.0), True),
        axis=(1, 2),
    )
    igniter_ok = xp.all(
        xp.where(valid_time, xp.isfinite(igniter_kg_s) & (igniter_kg_s >= 0.0), True), axis=1
    )

    # Reorder the padded grain axis exactly as evaluate_axial_mass_flux does for nozzle_direction="negative".
    ordered_regression = xp.take_along_axis(
        regression_m, xp.broadcast_to(order[:, None, :], regression_m.shape), axis=2,
    )
    ordered_generated = xp.take_along_axis(
        generated_grains_kg_s, xp.broadcast_to(order[:, None, :], generated_grains_kg_s.shape), axis=2,
    )
    ordered_generated = xp.where(valid_grain[:, None, :], ordered_generated, 0.0)
    ordered_P = {}
    for name in (
        "outer_radius", "inner_radius0", "height0", "ends_burn", "is_star", "n_points", "epsilon",
        "slot_fraction", "slot_floor_radius", "slot_floor_depth", "burnout_depth", "grain_valid",
    ):
        ordered_P[name] = xp.take_along_axis(P[name], order, axis=1)[:, None, :]

    burnout = ordered_P["burnout_depth"]
    depth = xp.minimum(ordered_regression, burnout)
    active = depth < burnout
    port = geometry.port_area(xp, depth, ordered_P)
    port = xp.where(active, port, chamber_area_m2[:, None, None])
    burn_area = geometry.burn_area(xp, depth, ordered_P)
    outer = ordered_P["outer_radius"]
    ends_burn = ordered_P["ends_burn"]
    end_area = xp.where(
        ends_burn,
        0.0,
        xp.where(active, xp.maximum(_PI * outer**2 - port, 0.0), 0.0),
    )
    safe_burn_area = xp.where(burn_area > 0.0, burn_area, 1.0)
    face_fraction = xp.where(burn_area > 0.0, end_area / safe_burn_area, 0.0)
    face = ordered_generated * face_fraction
    lateral = ordered_generated - 2.0 * face

    regression_within_burnout = xp.all(
        xp.where(
            history_mask,
            ordered_regression <= burnout * (1.0 + 64.0 * xp.finfo(xp.float64).eps),
            True,
        ),
        axis=(1, 2),
    )
    port_valid = xp.isfinite(port) & (port > 0.0)
    port_ok = xp.all(xp.where(history_mask, port_valid, True), axis=(1, 2))
    no_burned_out_sources = xp.all(
        xp.where(history_mask, active | (ordered_generated <= 0.0), True), axis=(1, 2)
    )

    # The scalar evaluator adds sources grain-by-grain.  The short static recurrence preserves that floating-point
    # order (an xp.cumsum may use a different reduction tree on JAX and can change exact argmax ties).
    upstream_values = []
    upstream = igniter_kg_s
    for grain_slot in range(grains):
        upstream_values.append(upstream)
        upstream = upstream + ordered_generated[..., grain_slot]
    upstream = xp.stack(upstream_values, axis=-1)

    lengths = xp.where(
        ends_burn,
        ordered_P["height0"],
        xp.maximum(ordered_P["height0"] - 2.0 * depth, 0.0),
    )
    starts_at_time = xp.where(ends_burn, starts[:, None, :], starts[:, None, :] + depth)

    station_flux = xp.full((lanes, time_count, grains), -xp.inf, dtype=xp.float64)
    station_index = xp.zeros((lanes, time_count, grains), dtype=xp.int32)
    station_position = xp.zeros((lanes, time_count, grains), dtype=xp.float64)
    for station in range(_STATIONS_PER_GRAIN):
        fraction = station / (_STATIONS_PER_GRAIN - 1)
        flow = upstream + face + lateral * fraction
        if station == _STATIONS_PER_GRAIN - 1:
            flow = flow + face
        safe_port = xp.where(port_valid, port, 1.0)
        candidate_flux = flow / safe_port
        candidate_valid = history_mask & port_valid & xp.isfinite(candidate_flux)
        candidate_flux = xp.where(candidate_valid, candidate_flux, -xp.inf)
        improves = candidate_flux > station_flux
        station_flux = xp.where(improves, candidate_flux, station_flux)
        station_index = xp.where(improves, station, station_index)
        coordinate_fraction = 1.0 - fraction
        candidate_position = starts_at_time + lengths * coordinate_fraction
        station_position = xp.where(improves, candidate_position, station_position)

    maximum = xp.max(station_flux.reshape((lanes, -1)), axis=1)
    at_maximum = history_mask & xp.isfinite(station_flux) & (station_flux == maximum[:, None, None])
    earliest_time_mask = xp.any(at_maximum, axis=2)
    time_index = xp.argmax(earliest_time_mask, axis=1)
    time_axis = xp.arange(time_count)[None, :]
    at_first_time = at_maximum & (time_axis[:, :, None] == time_index[:, None, None])
    global_station_index = xp.arange(grains, dtype=xp.int32)[None, None, :] * _STATIONS_PER_GRAIN + station_index
    station_sentinel = xp.asarray(2**31 - 1, dtype=xp.int32)
    max_station_index = xp.min(xp.where(at_first_time, global_station_index, station_sentinel), axis=(1, 2))
    selected_grain_slot = xp.minimum(max_station_index // _STATIONS_PER_GRAIN, max(grains - 1, 0))
    lane_axis = xp.arange(lanes)
    selected_time = time_s[lane_axis, time_index]
    selected_position = station_position[lane_axis, time_index, selected_grain_slot]
    selected_grain_index = order[lane_axis, selected_grain_slot]

    valid = (
        time_ok & regression_ok & generated_ok & igniter_ok & regression_within_burnout & port_ok
        & no_burned_out_sources & xp.asarray(geometry_supported, dtype=bool) & xp.isfinite(maximum)
    )
    return {
        "max_flux": xp.where(valid, maximum, 0.0),
        "time_s": xp.where(valid, selected_time, 0.0),
        "grain_index": xp.where(valid, selected_grain_index, 0).astype(xp.int32),
        "station_index": xp.where(valid, max_station_index, 0).astype(xp.int32),
        "position_m": xp.where(valid, selected_position, 0.0),
        "valid": valid,
        "time_index": xp.where(valid, time_index, 0).astype(xp.int32),
    }


def axial_mass_flux_metrics_from_states(
    xp,
    P,
    time_s,
    state,
    n_points,
    grain_order,
    grain_start_positions_m,
    chamber_area_m2,
    n_grains,
    geometry_supported,
):
    """Derive accepted-history source rates and run the axial metrics kernel in ``xp``."""
    history_P = {name: value[:, None, ...] for name, value in P.items()}
    quantities = rhs.state_quantities(xp, state, None, history_P, time=time_s)
    grain_count = P["grain_valid"].shape[-1]
    return axial_mass_flux_metrics(
        xp, P, time_s, state[..., 2 : 2 + grain_count], quantities["generated_grains"], quantities["igniter"],
        n_points, grain_order, grain_start_positions_m, chamber_area_m2, n_grains, geometry_supported,
    )
