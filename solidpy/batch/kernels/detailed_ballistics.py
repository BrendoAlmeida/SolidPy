"""Array kernels for lane-batched detailed-ballistics post-processing."""

from __future__ import annotations


INTERPOLATED_FIELDS = (
    "chamber_pressure_pa", "free_volume_m3", "thrust_n", "exit_pressure_pa", "exit_velocity_m_s",
    "burn_area_m2", "mass_generated_kg_s", "mass_igniter_kg_s", "mass_nozzle_kg_s", "gas_mass_kg",
    "momentum_ideal_n", "momentum_n", "pressure_thrust_n", "generated_mass_integral_kg",
    "igniter_mass_integral_kg", "nozzle_mass_integral_kg", "impulse_integral_ns",
    "pressure_throat_integral_ns",
)


def _interpolate_1d(values, left, right, fraction, xp):
    lo = xp.take_along_axis(values, left, axis=1)
    hi = xp.take_along_axis(values, right, axis=1)
    return lo + fraction * (hi - lo)


def detailed_ballistics(arrays, xp):
    """Evaluate the numeric history, grain-mass, center-of-mass and nozzle series for lanes."""
    a = arrays
    batch, output_points = a["query_time_s"].shape
    grain_count = a["grain_valid"].shape[1]
    output_mask = xp.arange(output_points)[None, :] < a["n_output"][:, None]

    output = {"time_s": a["query_time_s"]}
    for name in INTERPOLATED_FIELDS:
        output[name] = _interpolate_1d(
            a["raw_" + name], a["interp_left"], a["interp_right"], a["interp_fraction"], xp
        )
    regression_index = xp.broadcast_to(
        a["interp_left"][:, :, None], (batch, output_points, grain_count)
    )
    regression_right = xp.broadcast_to(
        a["interp_right"][:, :, None], (batch, output_points, grain_count)
    )
    regression_fraction = a["interp_fraction"][:, :, None]
    regression_lo = xp.take_along_axis(a["raw_regression_m"], regression_index, axis=1)
    regression_hi = xp.take_along_axis(a["raw_regression_m"], regression_right, axis=1)
    regressions = regression_lo + regression_fraction * (regression_hi - regression_lo)

    grain_valid = a["grain_valid"][:, None, :]
    regression = xp.maximum(regressions, 0.0)
    ri = a["grain_inner_radius_m"][:, None, :]
    ro = a["grain_outer_radius_m"][:, None, :]
    height0 = a["grain_height_m"][:, None, :]
    web = xp.maximum(ro - ri, 0.0)
    bore = xp.minimum(ri + regression, ro)

    tubular_height = xp.where(
        a["grain_ends_burn"][:, None, :], height0, xp.maximum(height0 - 2.0 * regression, 0.0)
    )
    tubular_burned = xp.where(
        a["grain_ends_burn"][:, None, :], regression >= web,
        (regression >= web) | (regression >= height0 / 2.0),
    )
    # DetailedBallistics uses remaining cross-sectional volume area here, not burn surface area.
    tubular_area = xp.pi * xp.maximum(ro**2 - bore**2, 0.0)

    slot_floor = ri + a["grain_slot_fraction"][:, None, :] * web
    floor_web = xp.maximum(ro - slot_floor, 0.0)
    slot_radius = xp.minimum(slot_floor + regression, ro)
    star_area_before_floor = xp.pi * xp.maximum(ro**2 - bore**2, 0.0) - (
        a["grain_star_points"][:, None, :] * a["grain_epsilon"][:, None, :]
        * xp.maximum(slot_radius**2 - bore**2, 0.0)
    )
    star_area_after_floor = (
        xp.pi - a["grain_star_points"][:, None, :] * a["grain_epsilon"][:, None, :]
    ) * xp.maximum(ro**2 - bore**2, 0.0)
    star_area = xp.where(regression < floor_web, star_area_before_floor, star_area_after_floor)
    star_height = xp.where(
        a["grain_ends_burn"][:, None, :], height0, xp.maximum(height0 - 2.0 * regression, 0.0)
    )
    star_burned = (regression >= web) | (
        (~a["grain_ends_burn"][:, None, :]) & (regression >= height0 / 2.0)
    )
    star_area = xp.where(star_burned, 0.0, xp.maximum(star_area, 0.0))

    grain_height = xp.where(a["grain_is_star"][:, None, :], star_height, tubular_height)
    grain_area = xp.where(a["grain_is_star"][:, None, :], star_area, tubular_area)
    grain_height = xp.where(grain_valid, grain_height, 0.0)
    grain_area = xp.where(grain_valid, xp.maximum(grain_area, 0.0), 0.0)
    grain_mass = grain_height * grain_area * a["propellant_density_kg_m3"][:, None, None]
    centroid = a["grain_stack_start_m"][:, None, :] + 0.5 * grain_height
    centroid = centroid + xp.where(
        (~a["grain_ends_burn"][:, None, :]) & (grain_height > 0.0), regression, 0.0
    )
    propellant_mass = xp.sum(grain_mass, axis=2)
    propellant_moment = xp.sum(grain_mass * centroid, axis=2)
    propellant_cg = xp.where(propellant_mass > 0.0, propellant_moment / xp.maximum(propellant_mass, 1e-300), 0.0)
    motor_mass = propellant_mass + a["dry_mass_kg"][:, None]
    motor_cg = xp.where(
        motor_mass > 0.0,
        (propellant_moment + a["dry_mass_kg"][:, None] * a["dry_cg_m"][:, None])
        / xp.maximum(motor_mass, 1e-300),
        a["dry_cg_m"][:, None],
    )
    propellant_cg = xp.where(propellant_mass > 0.0, propellant_cg, a["dry_cg_m"][:, None])
    regressed_length = xp.sum(xp.where(grain_valid, regressions, 0.0), axis=2) / xp.maximum(
        a["grain_count"][:, None], 1
    )

    if output_points > 1:
        times = a["query_time_s"]
        first_dt = xp.maximum(times[:, 1:2] - times[:, :1], 1e-300)
        first = (regressed_length[:, 1:2] - regressed_length[:, :1]) / first_dt
        last_index = xp.maximum(a["n_output"] - 1, 0)[:, None]
        previous_index = xp.maximum(a["n_output"] - 2, 0)[:, None]
        last_time = xp.take_along_axis(times, last_index, axis=1)
        previous_time = xp.take_along_axis(times, previous_index, axis=1)
        last_regression = xp.take_along_axis(regressed_length, last_index, axis=1)
        previous_regression = xp.take_along_axis(regressed_length, previous_index, axis=1)
        last = (last_regression - previous_regression) / xp.maximum(last_time - previous_time, 1e-300)
        if output_points > 2:
            h_left = xp.maximum(times[:, 1:-1] - times[:, :-2], 1e-100)
            h_right = xp.maximum(times[:, 2:] - times[:, 1:-1], 1e-100)
            middle = (
                -h_right / (h_left * (h_left + h_right)) * regressed_length[:, :-2]
                + (h_right - h_left) / (h_left * h_right) * regressed_length[:, 1:-1]
                + h_left / (h_right * (h_left + h_right)) * regressed_length[:, 2:]
            )
            middle_mask = xp.arange(1, output_points - 1)[None, :] < (a["n_output"][:, None] - 1)
            middle = xp.where(middle_mask, middle, 0.0)
            regression_rate = xp.concatenate((first, middle, last), axis=1)
        else:
            regression_rate = xp.concatenate((first, last), axis=1)
        regression_rate = xp.where(output_mask, xp.maximum(regression_rate, 0.0), 0.0)
    else:
        regression_rate = xp.zeros_like(regressed_length)

    raw_time = a["raw_time_s"]
    if raw_time.shape[1] > 1:
        raw_dt = xp.maximum(raw_time[:, 1:] - raw_time[:, :-1], 1e-9)
        raw_pressure = a["raw_chamber_pressure_pa"]
        raw_nozzle_flow = a["raw_mass_nozzle_kg_s"]
        rate = (
            1.8e-8 * xp.maximum(a["nozzle_ablation_scale"], 0.0)[:, None]
            * xp.maximum(raw_pressure[:, :-1], 1.0) ** a["ablation_pressure_exponent"][:, None]
            * xp.maximum(raw_nozzle_flow[:, :-1], 1e-9) ** a["ablation_mass_flow_exponent"][:, None]
        )
        interval_mask = xp.arange(raw_time.shape[1] - 1)[None, :] < (a["n_source"][:, None] - 1)
        increment = xp.where(interval_mask, raw_dt * rate, 0.0)
        raw_ablation = xp.concatenate(
            (xp.zeros((batch, 1), dtype=raw_time.dtype), xp.cumsum(increment, axis=1)), axis=1
        )
        throat_ablation = _interpolate_1d(
            raw_ablation, a["interp_left"], a["interp_right"], a["interp_fraction"], xp
        )
    else:
        throat_ablation = xp.zeros_like(a["query_time_s"])

    throat_radius = a["base_throat_radius_m"][:, None] + throat_ablation
    throat_diameter = 2.0 * throat_radius
    throat_area = xp.pi * xp.maximum(throat_radius, 1e-9) ** 2
    chamber_pressure = output["chamber_pressure_pa"]
    thrust = output["thrust_n"]
    valid_cf = chamber_pressure > a["environment_pressure_pa"][:, None] * 1.0001
    cf = xp.where(
        valid_cf,
        thrust / xp.maximum(chamber_pressure * throat_area, 1e-12),
        0.0,
    )
    cf = xp.maximum(cf, 0.0)

    activation = xp.where(
        a["activation_mode"][:, None] == 0.0,
        xp.where(
            a["ignition_ramp_time_s"][:, None] > 0.0,
            xp.clip(a["query_time_s"] / xp.maximum(a["ignition_ramp_time_s"][:, None], 1e-300), 0.0, 1.0) ** 2
            * (3.0 - 2.0 * xp.clip(
                a["query_time_s"] / xp.maximum(a["ignition_ramp_time_s"][:, None], 1e-300), 0.0, 1.0
            )),
            1.0,
        ),
        xp.clip(a["activation_value"][:, None], 0.0, 1.0),
    )
    profile_left = xp.take_along_axis(a["activation_profile_value"], a["activation_left"], axis=1)
    profile_right = xp.take_along_axis(a["activation_profile_value"], a["activation_right"], axis=1)
    profile = xp.clip(profile_left + a["activation_fraction"] * (profile_right - profile_left), 0.0, 1.0)
    activation = xp.where(a["activation_mode"][:, None] == 2.0, profile, activation)
    activation = xp.where(output_mask, activation, 0.0)

    output.update({
        "regressed_length_m": regressed_length,
        "burn_area_m2": output["burn_area_m2"],
        "regression_rate_m_s": regression_rate,
        "mass_flow_kg_s": output["mass_generated_kg_s"],
        "propellant_mass_kg": propellant_mass,
        "propellant_center_of_mass_position_m": propellant_cg,
        "motor_mass_kg": motor_mass,
        "motor_center_of_mass_position_m": motor_cg,
        "throat_area_m2": throat_area,
        "throat_diameter_m": throat_diameter,
        "throat_ablation_m": throat_ablation,
        "cf": cf,
        "ignition_active_fraction": activation,
    })
    return {name: xp.where(output_mask, values, 0.0) for name, values in output.items() if name != "query_time_s"}
