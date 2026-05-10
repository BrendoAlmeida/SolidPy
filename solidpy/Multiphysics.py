# -*- coding: utf-8 -*-

"""Advanced thermal, structural, flow-proxy, ignition and 1D-flight models."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional

import numpy as np
from scipy.integrate import solve_ivp


G0_M_S2 = 9.80665
SEA_LEVEL_DENSITY_KG_M3 = 1.225
SPEED_OF_SOUND_M_S = 343.0  # sea-level reference; use _isa_speed_of_sound() for altitude


def _isa_speed_of_sound(altitude_m):
    """ISA speed of sound as a function of altitude.

    Troposphere (0–11 km): T = 288.15 - 0.0065*h, a = sqrt(1.4*287.05*T).
    Stratosphere (11–20 km): T = 216.65 K, a = 295.07 m/s.
    Above 20 km: constant stratosphere value (conservative).
    Ref: ISO 2533:1975 (ISA standard atmosphere).
    """
    alt = max(float(altitude_m), 0.0)
    if alt <= 11000.0:
        temp_k = 288.15 - 0.0065 * alt
    elif alt <= 20000.0:
        temp_k = 216.65
    else:
        temp_k = 216.65
    return math.sqrt(1.4 * 287.05 * temp_k)


@dataclass(frozen=True)
class MotorGeometry:
    motor_length_m: float
    motor_inner_diameter_m: float
    casing_wall_thickness_m: float
    grain_outer_diameter_m: float
    grain_core_diameter_m: float
    grain_gap_m: float
    grain_length_each_m: float
    grain_number: int
    fill_length_m: float
    throat_diameter_m: float
    exit_diameter_m: float
    free_volume_m3: float
    propellant_mass_kg: float
    dry_mass_kg: float
    motor_initial_mass_kg: float
    motor_final_mass_kg: float


@dataclass(frozen=True)
class CasingMaterial:
    density_kg_m3: float = 7850.0
    modulus_gpa: float = 205.0
    yield_strength_mpa: float = 620.0
    allowable_stress_mpa: Optional[float] = None
    thermal_conductivity_w_mk: float = 16.0
    heat_capacity_j_kgk: float = 520.0
    max_service_temp_c: float = 450.0
    poisson_ratio: float = 0.29
    material_family: str = "metal"
    liner_thickness_m: float = 0.0
    liner_k_w_mk: float = 0.25
    liner_density_kg_m3: float = 1100.0
    liner_cp_j_kgk: float = 1600.0

    @property
    def resolved_allowable_stress_mpa(self):
        if self.allowable_stress_mpa is not None:
            return float(self.allowable_stress_mpa)
        return float(self.yield_strength_mpa)


@dataclass(frozen=True)
class NozzleMaterial:
    density_kg_m3: float = 1800.0
    thermal_conductivity_w_mk: float = 80.0
    heat_capacity_j_kgk: float = 710.0
    max_service_temp_c: float = 3000.0
    ablation_rate_scale: float = 1.0
    ablation_pressure_exponent: float = 0.42
    ablation_mass_flux_exponent: float = 0.32


def geometry_from_components(
    grain,
    motor,
    propellant,
    casing_wall_thickness_m,
    dry_mass_kg=None,
    casing_density_kg_m3=7850.0,
):
    """Derive an advanced-model geometry from SolidPy components."""
    motor_inner_radius = math.sqrt(max(motor.chamber_area, 0.0) / math.pi)
    throat_radius = math.sqrt(max(motor.nozzle_throat_area, 0.0) / math.pi)
    exit_radius = math.sqrt(max(motor.nozzle_exit_area, 0.0) / math.pi)
    casing_wall_thickness_m = max(float(casing_wall_thickness_m), 1e-5)
    propellant_mass_kg = (
        sum(g.volume for g in motor.grains) * max(float(propellant.density), 0.0)
    )

    outer_radius = motor_inner_radius + casing_wall_thickness_m
    casing_volume = (
        math.pi * max(outer_radius**2 - motor_inner_radius**2, 0.0) * motor.chamber_length
    )
    casing_mass = casing_volume * max(float(casing_density_kg_m3), 1.0)
    dry_mass = casing_mass if dry_mass_kg is None else max(float(dry_mass_kg), 0.0)

    return MotorGeometry(
        motor_length_m=float(motor.chamber_length),
        motor_inner_diameter_m=2.0 * motor_inner_radius,
        casing_wall_thickness_m=casing_wall_thickness_m,
        grain_outer_diameter_m=2.0 * grain.outer_radius,
        grain_core_diameter_m=2.0 * grain.initial_inner_radius,
        grain_gap_m=max(motor_inner_radius - grain.outer_radius, 0.0),
        grain_length_each_m=float(grain.initial_height),
        grain_number=int(motor.grain_number),
        fill_length_m=float(motor.chamber_length),
        throat_diameter_m=2.0 * throat_radius,
        exit_diameter_m=2.0 * exit_radius,
        free_volume_m3=float(motor.free_volume),
        propellant_mass_kg=float(propellant_mass_kg),
        dry_mass_kg=float(dry_mass),
        motor_initial_mass_kg=float(dry_mass + propellant_mass_kg),
        motor_final_mass_kg=float(dry_mass),
    )


def _sutherland_viscosity(temp_k):
    mu_ref = 1.716e-5
    t_ref = 273.15
    s_const = 110.4
    temp_k = max(float(temp_k), 100.0)
    return mu_ref * (temp_k / t_ref) ** 1.5 * (t_ref + s_const) / (temp_k + s_const)


def _series(curve, key, default):
    if key in curve:
        return np.asarray(curve[key], dtype=float)
    return np.asarray(default, dtype=float)


def simulate_thermal_ablation(
    geometry,
    curve,
    casing_material=None,
    nozzle_material=None,
    flame_temp_k=2800.0,
    r_specific=287.0,
    gamma=None,
    initial_temperature_k=298.15,
    liner_thickness_factor=1.0,
):
    """Transient 1D wall conduction plus empirical throat ablation."""
    casing_material = casing_material or CasingMaterial()
    nozzle_material = nozzle_material or NozzleMaterial()
    gamma = float(_resolve_gamma(curve, gamma))

    time_s = np.asarray(curve["time_s"], dtype=float)
    thrust_n = np.asarray(curve["thrust_n"], dtype=float)
    mass_flow_kg_s = _series(curve, "mass_nozzle_kg_s", curve["mass_flow_kg_s"])
    chamber_pressure_pa = _series(
        curve,
        "chamber_pressure_pa",
        np.zeros_like(time_s),
    )
    throat_diameter_m = _series(
        curve,
        "throat_diameter_m",
        np.full_like(time_s, geometry.throat_diameter_m),
    )
    throat_ablation_series_m = _series(
        curve,
        "throat_ablation_m",
        np.full_like(time_s, np.nan),
    )

    wall_thickness_m = max(float(geometry.casing_wall_thickness_m), 1e-4)
    liner_thickness_m = (
        max(float(casing_material.liner_thickness_m), 0.0)
        * max(float(liner_thickness_factor), 0.0)
    )

    layer_dx = []
    layer_k = []
    layer_rho_cp = []
    if liner_thickness_m > 1e-4:
        n_liner = max(2, int(math.ceil(liner_thickness_m / 0.001)))
        layer_dx.extend([liner_thickness_m / n_liner] * n_liner)
        layer_k.extend([max(casing_material.liner_k_w_mk, 1e-4)] * n_liner)
        layer_rho_cp.extend(
            [
                max(casing_material.liner_density_kg_m3 * casing_material.liner_cp_j_kgk, 1.0)
            ]
            * n_liner
        )
        casing_inner_node = n_liner
        liner_interface_node = n_liner - 1
    else:
        casing_inner_node = 0
        liner_interface_node = 0

    n_casing = max(4, int(math.ceil(wall_thickness_m / 0.001)))
    layer_dx.extend([wall_thickness_m / n_casing] * n_casing)
    layer_k.extend([casing_material.thermal_conductivity_w_mk] * n_casing)
    layer_rho_cp.extend(
        [casing_material.density_kg_m3 * casing_material.heat_capacity_j_kgk] * n_casing
    )

    dx_arr = np.asarray(layer_dx, dtype=float)
    k_arr = np.asarray(layer_k, dtype=float)
    rho_cp_arr = np.asarray(layer_rho_cp, dtype=float)
    temperatures_k = np.full(len(dx_arr), max(float(initial_temperature_k), 150.0))
    h_outer_w_m2k = 18.0

    throat_radius_0 = 0.5 * max(float(geometry.throat_diameter_m), 1e-6)
    throat_ablation_m = 0.0
    max_heat_flux_w_m2 = 0.0
    max_recovery_temp_k = float(initial_temperature_k)
    max_hot_face_k = float(initial_temperature_k)
    max_interface_k = float(initial_temperature_k)
    max_inner_wall_k = float(initial_temperature_k)
    max_outer_wall_k = float(initial_temperature_k)
    max_wall_gradient_k_m = 0.0
    integrated_heat_j_m2 = 0.0

    for idx in range(1, len(time_s)):
        dt_s = max(float(time_s[idx] - time_s[idx - 1]), 1e-5)
        throat_radius_eff = max(float(throat_diameter_m[idx]) * 0.5, 1e-6)
        throat_area_eff = math.pi * throat_radius_eff**2
        pressure_now = max(float(chamber_pressure_pa[idx]), 0.0)
        if pressure_now <= 0.0:
            pressure_now = max(float(thrust_n[idx]), 0.0) / max(throat_area_eff, 1e-9)
        mass_flow_now = max(float(mass_flow_kg_s[idx]), 0.0)

        rho_chamber = max(pressure_now / max(r_specific * flame_temp_k, 1e-9), 1e-6)
        throat_density_ratio = (2.0 / (gamma + 1.0)) ** (1.0 / max(gamma - 1.0, 1e-9))
        rho_throat = max(rho_chamber * throat_density_ratio, 1e-6)
        gas_velocity = mass_flow_now / max(rho_throat * throat_area_eff, 1e-9)

        # Eckert recovery temperature for turbulent flow at the throat (M=1).
        # T_r = T_0 * (1 + r*(k-1)/2 * M²) / (1 + (k-1)/2 * M²)
        # r = Pr^(1/3) ≈ 0.89 for combustion gases; at throat M=1.
        # Ref: Eckert (1955); Sutton & Biblarz §8.3.
        prandtl_recovery = 0.89
        t_ratio_num = 1.0 + prandtl_recovery * (gamma - 1.0) / 2.0
        t_ratio_den = 1.0 + (gamma - 1.0) / 2.0
        recovery_temp_k = flame_temp_k * t_ratio_num / max(t_ratio_den, 1e-9)
        max_recovery_temp_k = max(max_recovery_temp_k, recovery_temp_k)

        heat_flux_w_m2 = max(
            0.0,
            1.35 * pressure_now**0.8 / max((2.0 * throat_radius_eff) ** 0.2, 1e-3),
        )
        max_heat_flux_w_m2 = max(max_heat_flux_w_m2, heat_flux_w_m2)
        integrated_heat_j_m2 += heat_flux_w_m2 * dt_s

        if not np.isnan(throat_ablation_series_m[idx]):
            throat_ablation_m = max(float(throat_ablation_series_m[idx]), 0.0)
        else:
            ablation_rate = (
                1.8e-8
                * nozzle_material.ablation_rate_scale
                * max(pressure_now, 1.0) ** nozzle_material.ablation_pressure_exponent
                * max(mass_flow_now, 1e-9) ** nozzle_material.ablation_mass_flux_exponent
            )
            throat_ablation_m += ablation_rate * dt_s

        alpha_cells = k_arr / np.maximum(rho_cp_arr, 1e-9)
        stable_dt = 0.35 * float(np.min(dx_arr**2 / np.maximum(alpha_cells, 1e-12)))
        substeps = max(1, int(math.ceil(dt_s / max(stable_dt, 1e-6))))
        local_dt = dt_s / substeps

        for _ in range(substeps):
            previous = temperatures_k.copy()
            right_flux = np.zeros_like(temperatures_k)
            for node in range(len(temperatures_k) - 1):
                k_face = (
                    2.0
                    * k_arr[node]
                    * k_arr[node + 1]
                    / max(k_arr[node] + k_arr[node + 1], 1e-9)
                )
                distance = 0.5 * (dx_arr[node] + dx_arr[node + 1])
                right_flux[node] = k_face * (previous[node] - previous[node + 1]) / max(
                    distance, 1e-9
                )
            right_flux[-1] = h_outer_w_m2k * (previous[-1] - initial_temperature_k)
            for node in range(len(temperatures_k)):
                left_flux = heat_flux_w_m2 if node == 0 else right_flux[node - 1]
                temperatures_k[node] = previous[node] + local_dt * (
                    left_flux - right_flux[node]
                ) / max(rho_cp_arr[node] * dx_arr[node], 1e-9)

        max_inner_wall_k = max(max_inner_wall_k, float(temperatures_k[casing_inner_node]))
        max_outer_wall_k = max(max_outer_wall_k, float(temperatures_k[-1]))
        max_wall_gradient_k_m = max(
            max_wall_gradient_k_m,
            float(temperatures_k[0] - temperatures_k[-1])
            / max(liner_thickness_m + wall_thickness_m, 1e-9),
        )
        if liner_thickness_m > 1e-4:
            max_hot_face_k = max(max_hot_face_k, float(temperatures_k[0]))
            max_interface_k = max(
                max_interface_k,
                0.5 * float(temperatures_k[liner_interface_node] + temperatures_k[casing_inner_node]),
            )
        else:
            max_hot_face_k = max_inner_wall_k
            max_interface_k = max_inner_wall_k

    burn_duration_s = max(float(time_s[-1] - time_s[0]), 1e-6) if len(time_s) else 1e-6
    alpha_casing = casing_material.thermal_conductivity_w_mk / max(
        casing_material.density_kg_m3 * casing_material.heat_capacity_j_kgk, 1e-9
    )
    penetration_depth_m = 2.0 * math.sqrt(alpha_casing * burn_duration_s)
    final_throat_diameter_mm = 1000.0 * 2.0 * (throat_radius_0 + throat_ablation_m)

    return {
        "simulation.advanced.thermal.throat_heat_flux_kw_m2": max_heat_flux_w_m2 / 1000.0,
        "simulation.advanced.thermal.wall_temp_gradient_k_m": max_wall_gradient_k_m,
        "simulation.advanced.thermal.throat_recession_rate_mm_s": 1000.0
        * throat_ablation_m
        / burn_duration_s,
        "simulation.advanced.thermal.throat_ablation_mm": 1000.0 * throat_ablation_m,
        "simulation.advanced.thermal.throat_growth_pct": 100.0
        * (2.0 * throat_ablation_m)
        / max(geometry.throat_diameter_m, 1e-9),
        "simulation.advanced.thermal.max_casing_temp_c": max_outer_wall_k - 273.15,
        "simulation.advanced.thermal.max_inner_wall_temp_c": max_inner_wall_k - 273.15,
        "simulation.advanced.thermal.liner_hot_face_temp_c": max_hot_face_k - 273.15,
        "simulation.advanced.thermal.liner_casing_interface_temp_c": max_interface_k - 273.15,
        "simulation.advanced.thermal.casing_inner_wall_temp_c": max_inner_wall_k - 273.15,
        "simulation.advanced.thermal.casing_outer_wall_temp_c": max_outer_wall_k - 273.15,
        "simulation.advanced.thermal.heat_load_kj_m2": integrated_heat_j_m2 / 1000.0,
        "simulation.advanced.thermal.penetration_depth_mm": 1000.0 * penetration_depth_m,
        "simulation.advanced.thermal.final_throat_diameter_mm": final_throat_diameter_mm,
        "simulation.advanced.thermal.biot_number": h_outer_w_m2k
        * wall_thickness_m
        / max(casing_material.thermal_conductivity_w_mk, 1e-9),
        "simulation.advanced.metadata.max_chamber_temperature_k": max_recovery_temp_k,
        "simulation.advanced.metadata.thermal_node_count": float(len(dx_arr)),
    }


def simulate_structural_response(
    geometry,
    curve,
    thermal,
    casing_material=None,
    casing_strength_factor=1.0,
):
    """Thin-wall casing stress, deformation, buckling and fatigue proxies."""
    casing_material = casing_material or CasingMaterial()
    time_s = np.asarray(curve["time_s"], dtype=float)
    thrust_n = np.asarray(curve["thrust_n"], dtype=float)
    chamber_pressure_pa = _series(
        curve,
        "chamber_pressure_pa",
        np.zeros_like(time_s),
    )
    throat_area = math.pi * (0.5 * max(geometry.throat_diameter_m, 1e-6)) ** 2

    wall = max(geometry.casing_wall_thickness_m, 1e-5)
    inner_radius = max(geometry.motor_inner_diameter_m / 2.0, 1e-5)
    outer_radius = inner_radius + wall
    modulus_pa = casing_material.modulus_gpa * 1e9
    yield_pa = casing_material.yield_strength_mpa * 1e6 * max(casing_strength_factor, 0.01)
    allowable_pa = (
        casing_material.resolved_allowable_stress_mpa * 1e6 * max(casing_strength_factor, 0.01)
    )
    poisson = casing_material.poisson_ratio

    # Select thin-wall (Barlow) or thick-wall (Lamé) depending on t/r ratio.
    # Lamé is exact for both regimes; thin-wall underestimates σ when t/r > 0.1.
    # Ref: Lamé (1852); Shigley §3-15.
    ri2 = inner_radius**2
    ro2 = outer_radius**2
    use_lame = (wall / max(inner_radius, 1e-9)) > 0.1

    max_von_mises = 0.0
    max_hoop = 0.0
    max_axial = 0.0
    max_strain = 0.0
    max_bulge = 0.0
    pressure_integral = 0.0
    max_pressure = 0.0
    for idx in range(len(time_s)):
        pressure = max(float(chamber_pressure_pa[idx]), 0.0)
        if pressure <= 0.0:
            pressure = max(float(thrust_n[idx]), 0.0) / max(throat_area, 1e-9)
        if use_lame:
            # Lamé solution: maximum hoop stress at inner wall (r = r_i).
            hoop = pressure * ri2 * (ro2 + ri2) / max(ro2 - ri2, 1e-9) / max(ri2, 1e-9)
            axial = pressure * ri2 / max(ro2 - ri2, 1e-9)
        else:
            hoop = pressure * inner_radius / wall
            axial = pressure * inner_radius / (2.0 * wall)
        von_mises = math.sqrt(max(hoop**2 + axial**2 - hoop * axial, 0.0))
        strain = (hoop - poisson * axial) / max(modulus_pa, 1.0)
        bulge = pressure * inner_radius**2 * (1.0 - 0.5 * poisson) / max(
            modulus_pa * wall, 1.0
        )

        max_von_mises = max(max_von_mises, von_mises)
        max_hoop = max(max_hoop, hoop)
        max_axial = max(max_axial, axial)
        max_strain = max(max_strain, strain)
        max_bulge = max(max_bulge, bulge)
        max_pressure = max(max_pressure, pressure)
        if idx > 0:
            pressure_integral += pressure * max(float(time_s[idx] - time_s[idx - 1]), 1e-9)

    r_mid = (inner_radius + outer_radius) / 2.0
    i_tube = math.pi * r_mid**3 * wall
    rho_a = casing_material.density_kg_m3 * 2.0 * math.pi * r_mid * wall
    first_mode_hz = (
        (1.875**2)
        / (2.0 * math.pi * max(geometry.motor_length_m**2, 1e-9))
        * math.sqrt(modulus_pa * i_tube / max(rho_a, 1e-9))
    )
    critical_buckling_pa = (
        0.605
        * modulus_pa
        * (wall / max(outer_radius, 1e-9))
        / max(math.sqrt(max(1.0 - poisson**2, 1e-9)), 1e-9)
    )
    metal_sf = yield_pa / max(max_von_mises, 1.0)
    composite_margin = allowable_pa / max(max_hoop, 1.0)
    governing_margin = (
        composite_margin if casing_material.material_family == "composite" else metal_sf
    )
    onset_temp_c = 0.6 * casing_material.max_service_temp_c
    inner_wall_temp_c = thermal["simulation.advanced.thermal.casing_inner_wall_temp_c"]
    thermoelastic_margin = 1.0 - max(
        0.0,
        (inner_wall_temp_c - onset_temp_c)
        / max(casing_material.max_service_temp_c - onset_temp_c, 1.0),
    )

    return {
        "simulation.advanced.structural.max_stress_mpa": max_von_mises / 1e6,
        "simulation.advanced.structural.safety_factor": governing_margin,
        "simulation.advanced.structural.metal_equivalent_sf": metal_sf,
        "simulation.advanced.structural.composite_case_margin": composite_margin,
        "simulation.advanced.structural.governing_margin": governing_margin,
        "simulation.advanced.structural.casing_radial_bulge_mm": 1000.0 * max_bulge,
        "simulation.advanced.structural.first_mode_hz": first_mode_hz,
        "simulation.advanced.structural.acceleration_crack_index": max(float(np.max(thrust_n)), 0.0)
        / max(geometry.dry_mass_kg * G0_M_S2, 1.0),
        "simulation.advanced.structural.max_hoop_stress_mpa": max_hoop / 1e6,
        "simulation.advanced.structural.max_axial_stress_mpa": max_axial / 1e6,
        "simulation.advanced.structural.max_hoop_strain_microstrain": max_strain * 1e6,
        "simulation.advanced.structural.buckling_margin": critical_buckling_pa
        / max(max_pressure, 1.0),
        "simulation.advanced.structural.low_cycle_fatigue_damage": max_von_mises
        / max(yield_pa, 1.0),
        "simulation.advanced.structural.pressurization_impulse_mpa_s": pressure_integral / 1e6,
        "simulation.advanced.structural.thermoelastic_margin": thermoelastic_margin,
    }


def simulate_cfd_proxies(
    geometry,
    curve,
    thermal,
    r_specific=287.0,
    flame_temp_k=2200.0,
    gamma=None,
):
    """Fast internal-flow proxies for Reynolds, residence and erosion risk."""
    time_s = np.asarray(curve["time_s"], dtype=float)
    mass_flow = _series(curve, "mass_nozzle_kg_s", curve["mass_flow_kg_s"])
    pressure = _series(curve, "chamber_pressure_pa", np.zeros_like(time_s))
    port_diameter = max(geometry.grain_core_diameter_m, 1e-6)
    port_area = math.pi * (0.5 * port_diameter) ** 2
    max_reynolds = 0.0
    peak_gap_erosion = 0.0
    ld_ratio = geometry.fill_length_m / max(port_diameter, 1e-9)
    for idx in range(len(time_s)):
        rho = max(float(pressure[idx]) / max(r_specific * flame_temp_k, 1e-9), 0.03)
        velocity = max(float(mass_flow[idx]), 0.0) / max(rho * port_area, 1e-9)
        reynolds = rho * velocity * port_diameter / max(_sutherland_viscosity(flame_temp_k), 1e-9)
        gap_erosion = (
            geometry.grain_gap_m
            / max(port_diameter, 1e-9)
            * math.sqrt(max(reynolds, 1.0))
            * (1.0 + thermal["simulation.advanced.thermal.throat_ablation_mm"] / 10.0)
        )
        max_reynolds = max(max_reynolds, reynolds)
        peak_gap_erosion = max(peak_gap_erosion, gap_erosion)

    # Higher L/D increases residence time and combustion efficiency.
    # Ref: Rocketdyne/Thiokol empirical L* correlations.
    efficiency = min(
        0.997,
        0.89 + 0.013 * math.log10(max(max_reynolds, 10.0)) + 0.02 * min(ld_ratio, 2.0),
    )
    return {
        "simulation.advanced.cfd.combustion_efficiency_proxy": efficiency,
        "simulation.advanced.cfd.reynolds_proxy": max_reynolds,
        "simulation.advanced.cfd.combustion_ld_ratio": ld_ratio,
        "simulation.advanced.cfd.gap_erosion_risk": peak_gap_erosion,
        "simulation.advanced.cfd.exhaust_torque_vector_n_m": 0.0,
    }


def simulate_ignition_proxy(geometry, curve, thermal, structural):
    """Operational ignition metrics from free volume and pressure ramp."""
    time_s = np.asarray(curve["time_s"], dtype=float)
    pressure = _series(curve, "chamber_pressure_pa", np.zeros_like(time_s))
    throat_area = math.pi * max(geometry.throat_diameter_m * 0.5, 1e-6) ** 2
    igniter_mass_flow = 0.03 + 200.0 * throat_area
    delay_s = max(0.003, geometry.free_volume_m3 / max(igniter_mass_flow * 2.2, 1e-6))
    if len(time_s) > 1 and len(pressure) == len(time_s):
        ramp = float(np.max(np.diff(pressure) / np.maximum(np.diff(time_s), 1e-9))) / 1e6
        threshold = max(0.5e6, float(np.max(pressure)) * 0.05)
        above = np.flatnonzero(pressure >= threshold)
        if len(above):
            delay_s = max(0.003, float(time_s[int(above[0])]))
    else:
        burn_time = max(float(time_s[-1] - time_s[0]), 1e-9) if len(time_s) else 1e-9
        avg_pressure_mpa = structural[
            "simulation.advanced.structural.pressurization_impulse_mpa_s"
        ] / burn_time
        ramp = avg_pressure_mpa / max(0.18 * burn_time, delay_s)

    core_area = math.pi * max(geometry.grain_core_diameter_m * 0.5, 1e-6) ** 2
    core_filling = min(1.0, core_area * geometry.fill_length_m / max(geometry.free_volume_m3, 1e-9))
    energy_kj = thermal["simulation.advanced.thermal.heat_load_kj_m2"] * throat_area
    return {
        "simulation.advanced.ignition.delay_s": delay_s,
        "simulation.advanced.ignition.pressurization_rate_mpa_s": ramp,
        "simulation.advanced.ignition.hard_start_index": ramp / 500.0,
        "simulation.advanced.ignition.core_filling_index": core_filling,
        "simulation.advanced.ignition.energy_proxy_j": energy_kj * 1000.0,
        "simulation.advanced.ignition.energy_proxy_kj": energy_kj,
    }


def simulate_flight_1d(
    geometry,
    curve,
    drag_coefficient_factor=1.0,
    airframe_to_motor_mass_ratio=2.4,
    rail_length_m=5.0,
):
    """Vertical 1D flight proxy with drag and post-burn coast."""
    time_s = np.asarray(curve["time_s"], dtype=float)
    thrust_n = np.asarray(curve["thrust_n"], dtype=float)
    propellant_mass = _series(
        curve,
        "propellant_mass_kg",
        np.linspace(geometry.propellant_mass_kg, 0.0, len(time_s)),
    )
    dry_airframe_mass = geometry.motor_final_mass_kg * max(float(airframe_to_motor_mass_ratio), 0.0)
    body_diameter = max(
        geometry.motor_inner_diameter_m + 2.0 * geometry.casing_wall_thickness_m,
        1e-3,
    )
    reference_area = math.pi * (0.5 * body_diameter) ** 2

    def cd_for_mach(mach):
        """Piecewise Cd vs Mach for a generic sounding rocket body."""
        if mach < 0.75:
            cd = 0.48
        elif mach < 1.2:
            cd = 0.48 + 0.35 * (mach - 0.75) / 0.45
        elif mach < 2.0:
            cd = 0.83 - 0.18 * (mach - 1.2) / 0.8
        else:
            cd = 0.65
        return max(0.2, cd * max(float(drag_coefficient_factor), 0.01))

    altitude = 0.0
    velocity = 0.0
    distance = 0.0
    max_altitude = 0.0
    max_velocity = 0.0
    max_mach = 0.0
    burnout_altitude = 0.0
    burnout_velocity = 0.0
    rail_exit_velocity = 0.0
    rail_exit_twr = 0.0
    rail_exited = False

    for idx in range(1, len(time_s)):
        dt = max(float(time_s[idx] - time_s[idx - 1]), 1e-5)
        motor_mass = geometry.motor_final_mass_kg + max(float(propellant_mass[idx]), 0.0)
        rocket_mass = dry_airframe_mass + motor_mass
        density = SEA_LEVEL_DENSITY_KG_M3 * math.exp(-altitude / 8500.0)
        a_local = _isa_speed_of_sound(altitude)
        mach_local = abs(velocity) / max(a_local, 1.0)
        drag = 0.5 * density * velocity**2 * cd_for_mach(mach_local) * reference_area
        drag *= -1.0 if velocity < 0.0 else 1.0
        thrust = max(float(thrust_n[idx]), 0.0)
        acceleration = (thrust - drag - rocket_mass * G0_M_S2) / max(rocket_mass, 1e-9)
        velocity += acceleration * dt
        altitude = max(0.0, altitude + velocity * dt)
        distance += max(velocity, 0.0) * dt
        max_altitude = max(max_altitude, altitude)
        max_velocity = max(max_velocity, abs(velocity))
        max_mach = max(max_mach, mach_local)
        if thrust > 0.0:
            burnout_altitude = altitude
            burnout_velocity = velocity
        if not rail_exited and distance >= rail_length_m:
            rail_exited = True
            rail_exit_velocity = velocity
            rail_exit_twr = thrust / max(rocket_mass * G0_M_S2, 1.0)

    total_impulse = float(np.trapezoid(thrust_n, time_s)) if len(time_s) > 1 else 0.0
    propellant_burned = max(geometry.motor_initial_mass_kg - geometry.motor_final_mass_kg, 1e-9)
    isp_s = total_impulse / (propellant_burned * G0_M_S2)
    m0 = dry_airframe_mass + geometry.motor_initial_mass_kg
    mf = dry_airframe_mass + geometry.motor_final_mass_kg
    delta_v = isp_s * G0_M_S2 * math.log(max(m0 / max(mf, 1e-9), 1.0))

    # Coast phase: solve ballistic ODE with DOP853 instead of fixed-step Euler.
    # State: [altitude_m, velocity_m_s]
    coast_mass = dry_airframe_mass + geometry.motor_final_mass_kg

    def coast_ode(t, y):
        alt, vel = y
        alt = max(alt, 0.0)
        dens = SEA_LEVEL_DENSITY_KG_M3 * math.exp(-alt / 8500.0)
        a_c = _isa_speed_of_sound(alt)
        mach_c = abs(vel) / max(a_c, 1.0)
        drag_mag = 0.5 * dens * vel**2 * cd_for_mach(mach_c) * reference_area
        drag_force = math.copysign(drag_mag, -vel)
        dvdt = (drag_force - coast_mass * G0_M_S2) / max(coast_mass, 1e-9)
        return [vel, dvdt]

    def apogee_event(t, y):
        return y[1]

    apogee_event.terminal = True
    apogee_event.direction = -1

    coast_sol = solve_ivp(
        coast_ode,
        (0.0, 300.0),
        [altitude, velocity],
        method="DOP853",
        events=apogee_event,
        max_step=1.0,
        atol=1.0,
        rtol=1e-4,
    )
    if coast_sol.y.shape[1] > 0:
        max_altitude = max(max_altitude, float(np.max(coast_sol.y[0])))

    ballistic_coefficient = (dry_airframe_mass + geometry.motor_initial_mass_kg) / max(
        cd_for_mach(max_mach) * reference_area, 1e-9
    )
    return {
        "simulation.advanced.flight.max_altitude_m": max_altitude,
        "simulation.advanced.flight.max_velocity_mach": max_mach,
        "simulation.advanced.flight.delta_v_m_s": delta_v,
        "simulation.advanced.flight.rail_exit_twr": rail_exit_twr,
        "simulation.advanced.flight.burnout_altitude_m": burnout_altitude,
        "simulation.advanced.flight.burnout_velocity_m_s": burnout_velocity,
        "simulation.advanced.flight.rail_exit_velocity_m_s": rail_exit_velocity,
        "simulation.advanced.flight.ballistic_coefficient": ballistic_coefficient,
        "simulation.advanced.flight.drag_model_code": 2.0,
        "simulation.advanced.metadata.reference_airframe_mass_kg": dry_airframe_mass
        + geometry.motor_initial_mass_kg,
        "simulation.advanced.metadata.propellant_loading_kg": max(float(propellant_mass[0]), 0.0),
    }


def _resolve_gamma(curve, gamma_arg):
    """Return the specific heat ratio to use across advanced models.

    Priority: explicit argument > curve["gamma"] > fallback 1.3.
    Using the propellant's actual k avoids the old hard-coded 1.21 default
    that was inconsistent with, e.g., KNSB (k=1.1361).
    """
    if gamma_arg is not None:
        return float(gamma_arg)
    if isinstance(curve, dict) and "gamma" in curve:
        return float(curve["gamma"])
    return 1.3


def simulate_advanced_physics(
    geometry,
    curve,
    casing_material=None,
    nozzle_material=None,
    flame_temp_k=2800.0,
    r_specific=287.0,
    gamma=None,
):
    """Run all advanced components and return a flat metrics dictionary."""
    scenario_factors = curve.get("scenario_factors", {}) if isinstance(curve, dict) else {}
    gamma_resolved = _resolve_gamma(curve, gamma)
    thermal = simulate_thermal_ablation(
        geometry,
        curve,
        casing_material=casing_material,
        nozzle_material=nozzle_material,
        flame_temp_k=flame_temp_k,
        r_specific=r_specific,
        gamma=gamma_resolved,
        liner_thickness_factor=float(scenario_factors.get("liner_thickness_factor", 1.0) or 1.0),
        initial_temperature_k=float(scenario_factors.get("initial_temperature_k", 298.15) or 298.15),
    )
    structural = simulate_structural_response(
        geometry,
        curve,
        thermal,
        casing_material=casing_material,
        casing_strength_factor=float(scenario_factors.get("casing_strength_factor", 1.0) or 1.0),
    )
    cfd = simulate_cfd_proxies(
        geometry,
        curve,
        thermal,
        r_specific=r_specific,
        flame_temp_k=flame_temp_k,
        gamma=gamma_resolved,
    )
    ignition = simulate_ignition_proxy(geometry, curve, thermal, structural)
    flight = simulate_flight_1d(
        geometry,
        curve,
        drag_coefficient_factor=float(scenario_factors.get("drag_coefficient_factor", 1.0) or 1.0),
    )
    total_impulse = (
        float(np.trapezoid(curve["thrust_n"], curve["time_s"]))
        if len(curve.get("time_s", [])) > 1
        else 0.0
    )
    metadata = {
        "simulation.schema_version": 3.0,
        "simulation.advanced.metadata.model_fidelity": 3.0,
        "simulation.advanced.metadata.backend": 2.0,
        "simulation.advanced.metadata.impulse_density_ns_kg": total_impulse
        / max(geometry.propellant_mass_kg, 1e-9),
        "simulation.advanced.metadata.ballistic_loading_kg_m3": geometry.propellant_mass_kg
        / max(geometry.free_volume_m3, 1e-9),
    }
    return {
        **thermal,
        **structural,
        **cfd,
        **ignition,
        **flight,
        **metadata,
    }


def simulate_advanced_components(
    geometry,
    curve,
    casing_material=None,
    nozzle_material=None,
    flame_temp_k=2800.0,
    r_specific=287.0,
    gamma=None,
):
    """Return advanced simulations grouped by component and as a flat bundle."""
    gamma_resolved = _resolve_gamma(curve, gamma)
    thermal = simulate_thermal_ablation(
        geometry,
        curve,
        casing_material=casing_material,
        nozzle_material=nozzle_material,
        flame_temp_k=flame_temp_k,
        r_specific=r_specific,
        gamma=gamma_resolved,
    )
    structural = simulate_structural_response(
        geometry,
        curve,
        thermal,
        casing_material=casing_material,
    )
    cfd = simulate_cfd_proxies(geometry, curve, thermal, r_specific=r_specific, flame_temp_k=flame_temp_k, gamma=gamma_resolved)
    ignition = simulate_ignition_proxy(geometry, curve, thermal, structural)
    flight = simulate_flight_1d(geometry, curve)
    return {
        "thermal": thermal,
        "structural": structural,
        "cfd": cfd,
        "ignition": ignition,
        "flight": flight,
        "nominal_advanced": {
            **thermal,
            **structural,
            **cfd,
            **ignition,
            **flight,
        },
    }
