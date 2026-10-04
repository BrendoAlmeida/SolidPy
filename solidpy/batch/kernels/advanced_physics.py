"""Array kernels for the transient structural, CFD and ignition proxies."""

from __future__ import annotations


STRUCTURAL_PREFIX = "simulation.advanced.structural."
CFD_PREFIX = "simulation.advanced.cfd."
IGNITION_PREFIX = "simulation.advanced.ignition."


def advanced_physics_proxies(arrays, xp):
    """Evaluate structural, flow and ignition proxies over lanes and padded time histories."""
    a = arrays
    lane = xp.arange(a["time_s"].shape[1])[None, :] < a["n_points"][:, None]
    interval = lane[:, 1:]
    time_s = a["time_s"]
    thrust = a["thrust_n"]
    pressure_in = a["pressure_pa"]
    throat_area = xp.pi * (0.5 * xp.maximum(a["throat_diameter_m"], 1e-6)) ** 2
    pressure = xp.maximum(pressure_in, 0.0)
    pressure = xp.where(
        pressure <= 0.0,
        xp.maximum(thrust, 0.0) / xp.maximum(throat_area[:, None], 1e-9),
        pressure,
    )
    pressure = xp.where(lane, pressure, 0.0)

    wall = xp.maximum(a["wall_thickness_m"], 1e-5)
    inner = xp.maximum(a["motor_inner_diameter_m"] / 2.0, 1e-5)
    outer = inner + wall
    ri2 = inner**2
    ro2 = outer**2
    use_lame = wall / xp.maximum(inner, 1e-9) > 0.1
    denom = xp.maximum(ro2 - ri2, 1e-9)
    lame_hoop = pressure * ri2[:, None] * (ro2 + ri2)[:, None] / denom[:, None] / xp.maximum(ri2[:, None], 1e-9)
    lame_axial = pressure * ri2[:, None] / denom[:, None]
    thin_hoop = pressure * inner[:, None] / wall[:, None]
    thin_axial = pressure * inner[:, None] / (2.0 * wall[:, None])
    hoop = xp.where(use_lame[:, None], lame_hoop, thin_hoop)
    axial = xp.where(use_lame[:, None], lame_axial, thin_axial)
    radial = -pressure
    von_mises = xp.sqrt(xp.maximum(0.5 * (
        (hoop - radial) ** 2 + (radial - axial) ** 2 + (axial - hoop) ** 2
    ), 0.0))
    modulus = xp.maximum(a["modulus_pa"], 1.0)
    strain = (hoop - a["poisson_ratio"][:, None] * axial) / modulus[:, None]
    bulge = pressure * inner[:, None] ** 2 * (1.0 - 0.5 * a["poisson_ratio"][:, None]) / xp.maximum(
        modulus[:, None] * wall[:, None], 1.0
    )
    max_von_mises = xp.max(xp.where(lane, von_mises, 0.0), axis=1)
    max_hoop = xp.max(xp.where(lane, hoop, 0.0), axis=1)
    max_axial = xp.max(xp.where(lane, axial, 0.0), axis=1)
    max_strain = xp.max(xp.where(lane, strain, 0.0), axis=1)
    max_bulge = xp.max(xp.where(lane, bulge, 0.0), axis=1)
    max_pressure = xp.max(xp.where(lane, pressure, 0.0), axis=1)
    max_thrust = xp.maximum(xp.max(xp.where(lane, thrust, 0.0), axis=1), 0.0)
    dt = time_s[:, 1:] - time_s[:, :-1]
    pressure_integral = xp.sum(
        xp.where(interval, 0.5 * (pressure[:, :-1] + pressure[:, 1:]) * dt, 0.0), axis=1
    )

    r_mid = (inner + outer) / 2.0
    i_tube = xp.pi * r_mid**3 * wall
    rho_a = a["casing_density_kg_m3"] * 2.0 * xp.pi * r_mid * wall
    first_mode = (1.875**2) / (2.0 * xp.pi * xp.maximum(a["motor_length_m"] ** 2, 1e-9)) * xp.sqrt(
        a["modulus_pa"] * i_tube / xp.maximum(rho_a, 1e-9)
    )
    critical_buckling = 0.605 * a["modulus_pa"] * (wall / xp.maximum(outer, 1e-9)) / xp.maximum(
        xp.sqrt(xp.maximum(1.0 - a["poisson_ratio"] ** 2, 1e-9)), 1e-9
    )
    metal_sf = a["yield_pa"] / xp.maximum(max_von_mises, 1.0)
    composite_margin = a["allowable_pa"] / xp.maximum(max_hoop, 1.0)
    governing = xp.where(a["is_composite"], composite_margin, metal_sf)
    burst_pressure = (2.0 / xp.sqrt(3.0)) * a["ultimate_pa"] * xp.log1p(wall / inner)
    yield_pressure = a["ultimate_pa"] * xp.maximum(outer**2 - inner**2, 0.0) / (
        xp.sqrt(3.0) * outer**2
    )
    blow_force = max_pressure * xp.pi * 0.25 * xp.maximum(a["motor_inner_diameter_m"], 0.0) ** 2
    thermal_onset = 0.6 * a["max_service_temp_c"]
    thermal_fraction = 1.0 - xp.maximum(
        0.0,
        (a["wall_temperature_c"] - thermal_onset)
        / xp.maximum(a["max_service_temp_c"] - thermal_onset, 1.0),
    )
    thermal_margin = xp.where(a["has_wall_temperature"], thermal_fraction, 0.0)

    port_diameter = xp.maximum(a["grain_core_diameter_m"], 1e-6)
    port_area = xp.pi * (0.5 * port_diameter) ** 2
    flame = a["flame_temp_k"]
    viscosity_temp = xp.maximum(flame, 100.0)
    viscosity = 1.716e-5 * (viscosity_temp / 273.15) ** 1.5 * (273.15 + 110.4) / (viscosity_temp + 110.4)
    rho = xp.maximum(
        xp.maximum(pressure_in, 0.0) / xp.maximum(a["r_specific"][:, None] * flame[:, None], 1e-9), 0.03
    )
    velocity = xp.maximum(a["mass_flow_kg_s"], 0.0) / xp.maximum(rho * port_area[:, None], 1e-9)
    reynolds = rho * velocity * port_diameter[:, None] / xp.maximum(viscosity[:, None], 1e-9)
    erosion = (
        a["grain_gap_m"][:, None]
        / xp.maximum(port_diameter[:, None], 1e-9)
        * xp.sqrt(xp.maximum(reynolds, 1.0))
        * (1.0 + a["throat_ablation_mm"][:, None] / 10.0)
    )
    max_reynolds = xp.max(xp.where(lane, reynolds, 0.0), axis=1)
    peak_erosion = xp.max(xp.where(lane, erosion, 0.0), axis=1)
    ld_ratio = a["fill_length_m"] / xp.maximum(port_diameter, 1e-9)
    efficiency = xp.minimum(
        0.997,
        0.89 + 0.013 * xp.log10(xp.maximum(max_reynolds, 10.0)) + 0.02 * xp.minimum(ld_ratio, 2.0),
    )

    pressure_max = xp.max(xp.where(lane, pressure_in, 0.0), axis=1)
    threshold = xp.maximum(0.5e6, pressure_max * 0.05)
    hit_time = xp.min(xp.where(lane & (pressure_in >= threshold[:, None]), time_s, xp.inf), axis=1)
    found = xp.isfinite(hit_time)
    dt_safe = xp.maximum(dt, 1e-9)
    ramp_series = (pressure_in[:, 1:] - pressure_in[:, :-1]) / dt_safe
    ramp_series = xp.where(interval, ramp_series, 0.0)
    ramp = xp.max(ramp_series, axis=1) / 1e6
    burn_time = xp.maximum(time_s[:, -1] - time_s[:, 0], 1e-9)
    fallback_ramp = (pressure_integral / 1e6 / burn_time) / xp.maximum(0.18 * burn_time, xp.maximum(
        0.003, a["free_volume_m3"] / xp.maximum((0.03 + 200.0 * throat_area) * 2.2, 1e-6)
    ))
    ramp = xp.where(a["n_points"] > 1, ramp, fallback_ramp)
    base_delay = xp.maximum(0.003, a["free_volume_m3"] / xp.maximum((0.03 + 200.0 * throat_area) * 2.2, 1e-6))
    delay = xp.where(a["n_points"] > 1, xp.where(found, xp.maximum(0.003, hit_time), base_delay), base_delay)
    core_area = xp.pi * xp.maximum(a["grain_core_diameter_m"] * 0.5, 1e-6) ** 2
    core_filling = xp.minimum(
        1.0, core_area * a["fill_length_m"] / xp.maximum(a["free_volume_m3"], 1e-9)
    )
    energy_kj = a["heat_load_kj_m2"] * throat_area

    return {
        STRUCTURAL_PREFIX + "max_stress_mpa": max_von_mises / 1e6,
        STRUCTURAL_PREFIX + "safety_factor": governing,
        STRUCTURAL_PREFIX + "metal_equivalent_sf": metal_sf,
        STRUCTURAL_PREFIX + "composite_case_margin": composite_margin,
        STRUCTURAL_PREFIX + "governing_margin": governing,
        STRUCTURAL_PREFIX + "casing_radial_bulge_mm": 1000.0 * max_bulge,
        STRUCTURAL_PREFIX + "first_mode_hz": first_mode,
        STRUCTURAL_PREFIX + "acceleration_crack_index": max_thrust / xp.maximum(a["dry_mass_kg"] * 9.80665, 1.0),
        STRUCTURAL_PREFIX + "max_hoop_stress_mpa": max_hoop / 1e6,
        STRUCTURAL_PREFIX + "max_axial_stress_mpa": max_axial / 1e6,
        STRUCTURAL_PREFIX + "max_hoop_strain_microstrain": max_strain * 1e6,
        STRUCTURAL_PREFIX + "buckling_margin": critical_buckling / xp.maximum(max_pressure, 1.0),
        STRUCTURAL_PREFIX + "low_cycle_fatigue_damage": max_von_mises / xp.maximum(a["yield_pa"], 1.0),
        STRUCTURAL_PREFIX + "pressurization_impulse_mpa_s": pressure_integral / 1e6,
        STRUCTURAL_PREFIX + "thermal_service_margin": thermal_margin,
        STRUCTURAL_PREFIX + "thermoelastic_margin": thermal_margin,
        STRUCTURAL_PREFIX + "burst_pressure_mpa": burst_pressure / 1e6,
        STRUCTURAL_PREFIX + "casing_burst_pressure_mpa": burst_pressure / 1e6,
        STRUCTURAL_PREFIX + "burst_safety_factor": burst_pressure / xp.maximum(max_pressure, 1.0),
        STRUCTURAL_PREFIX + "yield_pressure_mpa": yield_pressure / 1e6,
        STRUCTURAL_PREFIX + "ultimate_elastic_limit_pressure_mpa": yield_pressure / 1e6,
        STRUCTURAL_PREFIX + "closure_bolt_blow_force_n": blow_force,
        CFD_PREFIX + "combustion_efficiency_proxy": efficiency,
        CFD_PREFIX + "reynolds_proxy": max_reynolds,
        CFD_PREFIX + "combustion_ld_ratio": ld_ratio,
        CFD_PREFIX + "gap_erosion_risk": peak_erosion,
        CFD_PREFIX + "exhaust_torque_vector_n_m": xp.zeros_like(max_reynolds),
        IGNITION_PREFIX + "delay_s": delay,
        IGNITION_PREFIX + "pressurization_rate_mpa_s": ramp,
        IGNITION_PREFIX + "hard_start_index": ramp / 500.0,
        IGNITION_PREFIX + "core_filling_index": core_filling,
        IGNITION_PREFIX + "energy_proxy_j": energy_kj * 1000.0,
        IGNITION_PREFIX + "energy_proxy_kj": energy_kj,
    }
