# -*- coding: utf-8 -*-

"""Advanced thermal, structural, flow-proxy, ignition and 1D-flight models."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional

import numpy as np
from scipy import sparse
from scipy.integrate import solve_ivp

try:
    from ._numpy_compat import trapezoid as _trapezoid
except ImportError:
    from _numpy_compat import trapezoid as _trapezoid


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
    ultimate_strength_mpa: Optional[float] = None

    @property
    def resolved_allowable_stress_mpa(self):
        if self.allowable_stress_mpa is not None:
            return float(self.allowable_stress_mpa)
        return float(self.yield_strength_mpa)

    @property
    def resolved_ultimate_strength_mpa(self):
        if self.ultimate_strength_mpa is not None:
            return float(self.ultimate_strength_mpa)
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


def _build_wall_conduction_operator(
    dx_arr,
    k_arr,
    rho_cp_arr,
    h_outer_w_m2k,
    ambient_temperature_k,
):
    """Build the linear finite-volume operator for 1D wall conduction."""
    node_count = len(dx_arr)
    thermal_mass = np.maximum(rho_cp_arr * dx_arr, 1e-9)
    conductance = np.zeros(max(node_count - 1, 0), dtype=float)

    for node in range(node_count - 1):
        k_face = (
            2.0
            * k_arr[node]
            * k_arr[node + 1]
            / max(k_arr[node] + k_arr[node + 1], 1e-9)
        )
        distance = 0.5 * (dx_arr[node] + dx_arr[node + 1])
        conductance[node] = k_face / max(distance, 1e-9)

    diagonal = np.zeros(node_count, dtype=float)
    lower = np.zeros(max(node_count - 1, 0), dtype=float)
    upper = np.zeros(max(node_count - 1, 0), dtype=float)

    if node_count == 1:
        diagonal[0] = -h_outer_w_m2k / thermal_mass[0]
    else:
        diagonal[0] = -conductance[0] / thermal_mass[0]
        upper[0] = conductance[0] / thermal_mass[0]

        for node in range(1, node_count - 1):
            left = conductance[node - 1]
            right = conductance[node]
            lower[node - 1] = left / thermal_mass[node]
            diagonal[node] = -(left + right) / thermal_mass[node]
            upper[node] = right / thermal_mass[node]

        lower[-1] = conductance[-1] / thermal_mass[-1]
        diagonal[-1] = -(conductance[-1] + h_outer_w_m2k) / thermal_mass[-1]

    source_base = np.zeros(node_count, dtype=float)
    source_base[-1] += h_outer_w_m2k * ambient_temperature_k / thermal_mass[-1]
    inner_heat_flux_source = np.zeros(node_count, dtype=float)
    inner_heat_flux_source[0] = 1.0 / thermal_mass[0]

    jacobian = sparse.diags(
        [lower, diagonal, upper],
        offsets=[-1, 0, 1],
        shape=(node_count, node_count),
        format="csc",
    )
    return jacobian, source_base, inner_heat_flux_source


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
    initial_temperature_k = float(initial_temperature_k)

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
    (
        conduction_jacobian,
        conduction_source_base,
        inner_heat_flux_source,
    ) = _build_wall_conduction_operator(
        dx_arr,
        k_arr,
        rho_cp_arr,
        h_outer_w_m2k,
        initial_temperature_k,
    )

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

        # Bartz (1957) convective heat transfer at the nozzle throat.
        # h = (0.026/D_t^0.2)*(mu^0.2*cp/Pr^0.6)*(Pc/c*)^0.8*(Dt/Rc)^0.1*sigma
        # At throat: A_t/A = 1; assume Rc = D_t (typical SRM nozzle curvature).
        # Ref: Bartz (1957); Sutton & Biblarz §8.3.
        mu_gas = _sutherland_viscosity(flame_temp_k)
        cp_gas = gamma * r_specific / max(gamma - 1.0, 1e-9)
        prandtl = 0.82  # representative for hot combustion gases
        mass_flux = mass_flow_now / max(throat_area_eff, 1e-9)  # = Pc/c*
        r_curvature = max(2.0 * throat_radius_eff, 1e-9)        # Rc = D_t

        # Wall temperature at inner surface (from conduction solver state).
        T_w = float(temperatures_k[casing_inner_node])
        # σ: stagnation-temperature wall-correction factor at M=1 (Bartz eq.6).
        stag_factor = (gamma + 1.0) / 2.0  # = 1 + (k-1)/2 at M=1
        tw_t0_ratio = T_w / max(flame_temp_k, 1.0)
        sigma = (
            (0.5 * tw_t0_ratio / max(stag_factor, 1e-9) + 0.5) ** (-0.68)
            * stag_factor ** (-0.12)
        )
        h_bartz = (
            0.026
            / max((2.0 * throat_radius_eff) ** 0.2, 1e-4)
            * (mu_gas ** 0.2 * cp_gas / max(prandtl ** 0.6, 1e-9))
            * max(mass_flux, 1e-9) ** 0.8
            * (2.0 * throat_radius_eff / max(r_curvature, 1e-9)) ** 0.1
            * max(sigma, 0.1)
        )
        heat_flux_w_m2 = max(0.0, h_bartz * max(recovery_temp_k - T_w, 0.0))
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

        # Wall diffusion is stiff for fine meshes. Radau advances this linear
        # finite-volume system implicitly, removing the explicit CFL substep
        # restriction while the sparse Jacobian preserves the tridiagonal form.
        conduction_source = (
            conduction_source_base + inner_heat_flux_source * heat_flux_w_m2
        )

        def conduction_rhs(_time, wall_temperatures_k):
            return conduction_jacobian.dot(wall_temperatures_k) + conduction_source

        conduction_solution = solve_ivp(
            conduction_rhs,
            (0.0, dt_s),
            temperatures_k,
            method="Radau",
            jac=conduction_jacobian,
            atol=1e-6,
            rtol=1e-5,
        )
        if not conduction_solution.success:
            raise RuntimeError(
                "Thermal conduction solver failed: "
                f"{conduction_solution.message}"
            )
        temperatures_k = conduction_solution.y[:, -1]

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
    *,
    bolt_count=0,
    bolt_diameter_m=0.0,
    bolt_strength_mpa=0.0,
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
        # Radial stress at the inner wall equals the internal pressure with
        # opposite sign (Lamé boundary condition: sigma_r(r_i) = -P). The
        # previous biaxial formula (sqrt(sh^2 + sa^2 - sh*sa)) discarded it
        # and reported an optimistic von Mises, especially for thick walls
        # (asymptotic ~42% under-estimate as wall -> infinity). Use the exact
        # triaxial form so FS no longer approves vessels that should fail.
        # Ref: von Mises (1913); Shigley §3-15.
        radial = -pressure
        von_mises = math.sqrt(
            max(
                0.5
                * (
                    (hoop - radial) ** 2
                    + (radial - axial) ** 2
                    + (axial - hoop) ** 2
                ),
                0.0,
            )
        )
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

    # Burst pressure per Tresca criterion for a closed-end thick cylinder
    # under internal pressure. Uses ultimate strength (Su) as the flow limit
    # instead of the yield strength used above; this is the analytical
    # limit-load solution for a perfectly plastic material, matching what
    # the HTML simulator reports as pBurst in structuralAnalysis().
    # Ref: Tresca; Mendelson §8; simulador_balistica_interna_v7.html (~L1700).
    ultimate_pa = (
        casing_material.resolved_ultimate_strength_mpa * 1e6 * max(casing_strength_factor, 0.01)
    )
    burst_pressure_pa = (
        (2.0 / math.sqrt(3.0))
        * ultimate_pa
        * math.log(max(outer_radius / max(inner_radius, 1e-9), 1.0))
    )
    burst_safety_factor = burst_pressure_pa / max(max_pressure, 1.0)
    yield_pressure_pa = (
        ultimate_pa
        * max(outer_radius**2 - inner_radius**2, 0.0)
        / max(math.sqrt(3.0) * outer_radius**2, 1.0)
    )

    # Closure-bolt shear and bearing against the casing wall.
    # blowF is the axial pull on the forward closure under MEOP; the cross
    # section used here is the chamber bore (upper bound — no deduction for
    # the nozzle opening), matching the HTML structuralAnalysis().
    #   shear: tau_bolt = blowF / (n_bolts · π/4 · d²); τ_ult ≈ 0.6·Su_bolt
    #   bearing: the load path through the wall itself.
    #     The wall in bearing is *confined* (the bolt squeezes it against
    #     the head), so MMPDS/MIL-HDBK-5 allow Fbru ≈ 1.5–1.8·Su once
    #     edge distance e/D ≥ 2; 1.5·Su is the conservative lower bound.
    # Ref: simulador_balistica_interna_v7.html (~L1718); MIL-HDBK-5.
    blow_force_n = max_pressure * math.pi * 0.25 * max(geometry.motor_inner_diameter_m, 0.0) ** 2
    bolt_shear_sf = float("inf")
    bolt_bearing_sf = float("inf")
    bolt_shear_stress_mpa = 0.0
    bolt_bearing_stress_mpa = 0.0
    if bolt_count > 0 and bolt_diameter_m > 0.0:
        shear_area_m2 = bolt_count * math.pi * 0.25 * bolt_diameter_m**2
        bolt_shear_stress_pa = blow_force_n / max(shear_area_m2, 1e-9)
        bolt_shear_stress_mpa = bolt_shear_stress_pa / 1e6
        bolt_shear_sf = (
            0.6 * bolt_strength_mpa * 1e6 / max(bolt_shear_stress_pa, 1.0)
        )
        bearing_area_m2 = bolt_count * bolt_diameter_m * wall
        bolt_bearing_stress_pa = blow_force_n / max(bearing_area_m2, 1e-9)
        bolt_bearing_stress_mpa = bolt_bearing_stress_pa / 1e6
        bolt_bearing_sf = (
            1.5
            * casing_material.resolved_ultimate_strength_mpa
            * 1e6
            * max(casing_strength_factor, 0.01)
            / max(bolt_bearing_stress_pa, 1.0)
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
        "simulation.advanced.structural.burst_pressure_mpa": burst_pressure_pa / 1e6,
        "simulation.advanced.structural.burst_safety_factor": burst_safety_factor,
        "simulation.advanced.structural.yield_pressure_mpa": yield_pressure_pa / 1e6,
        "simulation.advanced.structural.closure_bolt_blow_force_n": blow_force_n,
        "simulation.advanced.structural.closure_bolt_shear_safety_factor": bolt_shear_sf,
        "simulation.advanced.structural.closure_bolt_bearing_safety_factor": bolt_bearing_sf,
        "simulation.advanced.structural.closure_bolt_shear_stress_mpa": bolt_shear_stress_mpa,
        "simulation.advanced.structural.closure_bolt_bearing_stress_mpa": bolt_bearing_stress_mpa,
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
    nose_half_angle_rad=None,
    body_fineness_ratio=10.0,
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

    use_component_cd = nose_half_angle_rad is not None

    def cd_for_mach(mach, burning=False, alt=0.0):
        """Return total Cd: component-based (if nose_half_angle_rad given) or piecewise."""
        if use_component_cd:
            result = evaluate_cd_by_components(
                mach=mach,
                nose_half_angle_rad=nose_half_angle_rad,
                body_fineness_ratio=body_fineness_ratio,
                is_burning=burning,
                altitude_m=alt,
                body_length_m=body_fineness_ratio * body_diameter,
            )
            cd = result["cd_total"]
        else:
            # Legacy piecewise model (backward-compatible default)
            if mach < 0.75:
                cd = 0.48
            elif mach < 1.2:
                cd = 0.48 + 0.35 * (mach - 0.75) / 0.45
            elif mach < 2.0:
                cd = 0.83 - 0.18 * (mach - 1.2) / 0.8
            else:
                cd = 0.65
        return max(0.05, cd * max(float(drag_coefficient_factor), 0.01))

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
        thrust = max(float(thrust_n[idx]), 0.0)
        drag = 0.5 * density * velocity**2 * cd_for_mach(mach_local, burning=thrust > 0.0, alt=altitude) * reference_area
        drag *= -1.0 if velocity < 0.0 else 1.0
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

    total_impulse = float(_trapezoid(thrust_n, time_s)) if len(time_s) > 1 else 0.0
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
        drag_mag = 0.5 * dens * vel**2 * cd_for_mach(mach_c, burning=False, alt=alt) * reference_area
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
        cd_for_mach(max_mach, burning=False, alt=0.0) * reference_area, 1e-9
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


def evaluate_cd_by_components(
    mach,
    nose_half_angle_rad=0.15,
    body_fineness_ratio=10.0,
    is_burning=False,
    altitude_m=0.0,
    body_length_m=None,
):
    """Estimate total drag coefficient from physical components.

    Implements the Missile DATCOM / RASAero II component build-up approach:
      Cd = Cd_wave  +  Cd_friction  +  Cd_base

    Wave drag — linearised supersonic theory (Missile DATCOM §4.1.3):
      Supersonic  (M ≥ 1.05): Cd_wave = 2·θ² / √(M²−1)  (conical nose)
      Transonic (0.80–1.05): smooth cosine rise to the supersonic value.
      Subsonic   (M < 0.80):  Cd_wave = 0 for slender body.

    Skin-friction drag — Schlichting turbulent flat-plate with compressibility:
      Cf = 0.455 / log10(Re)^2.58 / (1 + 0.144·M²)^0.65  (Van Driest II)
      Cd_friction = Cf · 4 · (L/D)                        (wetted area factor)

    Base drag — from Hoerner & RASAero II:
      Burning motor (plume fills base): Cd_base ≈ 0.
      Coast:   Cd_base = 0.12 … 0.18 subsonic,
               Cd_base = 0.25/M² supersonic.

    Args:
        mach               — current Mach number (≥ 0).
        nose_half_angle_rad— half-cone angle of the nose [rad] (default 0.15 ≈ 8.6°).
        body_fineness_ratio— overall body L/D (default 10).
        is_burning         — True while motor is producing thrust (suppresses base drag).
        altitude_m         — altitude for ISA density/viscosity (default 0).
        body_length_m      — absolute body length [m]; if None, L = L/D · 0.15 m.

    Returns:
        dict with keys: cd_wave, cd_friction, cd_base, cd_total.
    """
    M = max(float(mach), 0.0)
    theta = max(float(nose_half_angle_rad), 1e-3)
    L_D = max(float(body_fineness_ratio), 0.5)
    alt = max(float(altitude_m), 0.0)

    # ── Wave drag ──────────────────────────────────────────────────────────────
    M_super = max(1.05, M)
    cd_wave_super = 2.0 * theta ** 2 / math.sqrt(max(M_super ** 2 - 1.0, 1e-4))

    if M < 0.80:
        cd_wave = 0.0
    elif M < 1.05:
        # Smooth cosine onset from 0 at M=0.80 to full value at M=1.05
        t = (M - 0.80) / 0.25
        cd_wave = cd_wave_super * 0.5 * (1.0 - math.cos(math.pi * t))
    else:
        cd_wave = 2.0 * theta ** 2 / math.sqrt(M ** 2 - 1.0)

    # ── Skin-friction drag ────────────────────────────────────────────────────
    if alt <= 11000.0:
        T_alt = 288.15 - 0.0065 * alt
    else:
        T_alt = 216.65
    rho = SEA_LEVEL_DENSITY_KG_M3 * (T_alt / 288.15) ** 4.256
    mu = _sutherland_viscosity(T_alt)

    a_local = _isa_speed_of_sound(alt)
    V = M * a_local

    L_body = float(body_length_m) if body_length_m is not None else L_D * 0.15
    if V > 0.01 and mu > 0.0:
        Re_L = max(rho * V * L_body / mu, 1e4)
        Cf_inc = 0.455 / (math.log10(Re_L) ** 2.58)
        # Compressibility correction (Van Driest II)
        Cf = Cf_inc / (1.0 + 0.144 * M ** 2) ** 0.65
    else:
        Cf = 0.004

    # Wetted area ratio relative to π*(D/2)²: 4·(L/D) for a cylinder
    cd_friction = Cf * 4.0 * L_D

    # ── Base drag ─────────────────────────────────────────────────────────────
    if is_burning:
        cd_base = 0.0        # plume suppresses base drag (Hoerner §20, RASAero)
    elif M < 0.5:
        cd_base = 0.12
    elif M < 1.0:
        # Subsonic rise toward transonic (roughly linear)
        cd_base = 0.12 + 0.06 * (M - 0.5) / 0.5
    else:
        # Supersonic: Cd_base ~ 1/(M² · von Karman) (Hoerner §21-2)
        cd_base = 0.25 / max(M ** 2, 0.01)

    cd_total = cd_wave + cd_friction + cd_base
    return {
        "cd_wave": float(cd_wave),
        "cd_friction": float(cd_friction),
        "cd_base": float(cd_base),
        "cd_total": float(cd_total),
    }


def evaluate_barrowman_stability(
    nose_length_m,
    body_diameter_m,
    n_fins=4,
    fin_root_chord_m=0.08,
    fin_tip_chord_m=0.04,
    fin_span_m=0.06,
    fin_sweep_angle_rad=0.52,
    cg_from_nose_m=None,
    body_length_m=None,
    propellant_mass_kg=0.0,
    empty_mass_kg=1.0,
    nose_shape="conical",
):
    """Estimate centre-of-pressure using the Barrowman equations (1966).

    Implements the classic subsonic Barrowman method as used in OpenRocket and
    described in Niskanen (2009) §3.  Computes the normal-force coefficient
    slope (CNα) and CP location for a nose cone + cylindrical body + trapezoidal
    fins configuration.

    Barrowman is valid for M < 0.6 and small angle of attack (α < 10°).  For
    quick transonic/supersonic estimates the CP is held at the subsonic value
    (conservative for stability analysis).

    Args:
        nose_length_m        — axial length of the nose cone [m].
        body_diameter_m      — reference body diameter [m].
        n_fins               — number of identical fins (default 4).
        fin_root_chord_m     — fin chord at the root [m] (default 0.08).
        fin_tip_chord_m      — fin chord at the tip [m] (default 0.04).
        fin_span_m           — fin half-span from body (normal to axis) [m].
        fin_sweep_angle_rad  — leading-edge sweep angle measured from body axis
                               [rad] (0 = unswept, ~0.5 rad = 30° sweep).
        cg_from_nose_m       — distance of CG from nose tip [m]; if None the
                               CG is estimated from body geometry.
        body_length_m        — total body length from nose tip to base [m];
                               if None, body_length = nose_length + 3*diameter.
        propellant_mass_kg   — current propellant mass for CG shift estimate [kg].
        empty_mass_kg        — empty vehicle mass [kg].
        nose_shape           — "conical" (default) or "ogive"; ogive CP at 0.467*L_nose.

    Returns:
        dict with keys:
            cp_from_nose_m          — CP distance from nose tip [m].
            cg_from_nose_m          — CG distance from nose tip [m].
            static_margin_cal       — (CP − CG) / d_ref in calibres (>1 = stable).
            cn_alpha_nose           — nose normal-force slope.
            cn_alpha_fins           — fin normal-force slope (total, all fins).
            cn_alpha_total          — total CNα of the rocket.
            is_stable               — True when static margin ≥ 1 calibre.
    """
    d = max(float(body_diameter_m), 1e-3)
    R = d / 2.0
    A_ref = math.pi * R ** 2
    L_nose = max(float(nose_length_m), d * 0.5)

    if body_length_m is not None:
        L_body = max(float(body_length_m), L_nose)
    else:
        L_body = L_nose + 3.0 * d

    # ── Nose cone contribution ────────────────────────────────────────────────
    # CNα_nose = 2  (slender-body theory, any axisymmetric nose)
    cn_alpha_nose = 2.0
    # CP location:
    if nose_shape == "ogive":
        xcp_nose = 0.467 * L_nose   # OpenRocket: tangent ogive CP at 0.467*L_nose
    else:
        xcp_nose = L_nose / 3.0     # conical: 1/3 from base = 2/3 from tip

    # Wait: Barrowman measures from nose TIP, cone CP = L_nose / 3 from tip? No.
    # For a CONE: cp_from_tip = (2/3)*L_nose  (at 2/3 of the way from tip to base)
    # Actually Barrowman: for conical nose, XCP = 0.666*L_nose from nose tip.
    # For ogive: XCP ≈ 0.466*L_nose from nose tip.
    if nose_shape == "ogive":
        xcp_nose = 0.466 * L_nose
    else:
        xcp_nose = 0.666 * L_nose   # Barrowman (1966) eq.(6)

    # ── Fin contribution (Barrowman trapezoidal fin equations) ───────────────
    N = max(int(n_fins), 3)
    Cr = max(float(fin_root_chord_m), 1e-4)    # root chord
    Ct = max(float(fin_tip_chord_m), 0.0)       # tip chord  (0 = triangular)
    s = max(float(fin_span_m), 1e-4)            # fin span (one side, from body)
    Lambda = float(fin_sweep_angle_rad)          # leading-edge sweep from axis

    # Mid-chord sweep angle
    Lambda_mid = math.atan(math.tan(Lambda) - (Cr - Ct) / (2.0 * s))

    # Fin aspect ratio (for one fin panel, both sides)
    A_fin = (Cr + Ct) / 2.0 * s         # planform area of one fin
    AR = 2.0 * s ** 2 / max(A_fin, 1e-9)  # full-span aspect ratio

    # CNα per fin panel (Barrowman eq. 7)
    # CNα_fin = 4*N*s²/d² / (1 + √(1 + (2*s/(Cr+Ct))²)) (Barrowman simplified)
    fin_ratio = 1.0 + s / max(s + R, 1e-9)  # interference factor K_fin
    cn_alpha_1fin = (4.0 * s ** 2 / max(d ** 2, 1e-9)) / (
        1.0 + math.sqrt(1.0 + (2.0 * s / max(Cr + Ct, 1e-9)) ** 2)
    )
    cn_alpha_fins = N * fin_ratio * cn_alpha_1fin   # all fins

    # CP of fins from nose tip = x_fin_le + (Cr/3)*(Cr+2*Ct)/(Cr+Ct) + Ym*tan(Lambda_mid)
    # x_fin_le (leading edge of root at body): take at base of rocket
    x_fin_le = L_body - Cr       # root LE at body base minus root chord
    Ym = s / 3.0 * (Cr + 2 * Ct) / max(Cr + Ct, 1e-9)   # centroid span location
    xcp_fins = x_fin_le + Cr / 3.0 * (Cr + 2 * Ct) / max(Cr + Ct, 1e-9) + Ym * math.tan(Lambda_mid)

    # ── Total CP (area-weighted Barrowman) ────────────────────────────────────
    cn_alpha_total = cn_alpha_nose + cn_alpha_fins
    cp_from_nose = (cn_alpha_nose * xcp_nose + cn_alpha_fins * xcp_fins) / max(cn_alpha_total, 1e-9)

    # ── CG estimate ──────────────────────────────────────────────────────────
    if cg_from_nose_m is not None:
        cg = float(cg_from_nose_m)
    else:
        # Simple CG estimate: assume uniform mass distribution along body length
        # shifted forward by propellant loading fraction
        total_mass = max(empty_mass_kg + propellant_mass_kg, 1e-9)
        cg_empty = 0.45 * L_body        # typical empty CG at 45% from nose
        cg_prop = 0.60 * L_body         # propellant CG near aft (60% from nose)
        cg = (empty_mass_kg * cg_empty + propellant_mass_kg * cg_prop) / total_mass

    # ── Static margin ─────────────────────────────────────────────────────────
    static_margin_cal = (cp_from_nose - cg) / max(d, 1e-9)   # in calibres
    is_stable = static_margin_cal >= 1.0

    return {
        "cp_from_nose_m": float(cp_from_nose),
        "cg_from_nose_m": float(cg),
        "static_margin_cal": float(static_margin_cal),
        "cn_alpha_nose": float(cn_alpha_nose),
        "cn_alpha_fins": float(cn_alpha_fins),
        "cn_alpha_total": float(cn_alpha_total),
        "is_stable": bool(is_stable),
    }


def simulate_flight_3dof(
    geometry,
    curve,
    launch_angle_deg=90.0,
    launch_azimuth_deg=0.0,
    wind_speed_m_s=0.0,
    wind_direction_deg=0.0,
    rail_length_m=5.0,
    airframe_to_motor_mass_ratio=2.4,
    drag_coefficient_factor=1.0,
    nose_half_angle_rad=None,
    body_fineness_ratio=10.0,
    static_margin_cal=2.0,
    barrowman_result=None,
    max_time_s=300.0,
):
    """3-DOF point-mass trajectory simulation with wind, varying atmosphere, and stability.

    Integrates the equations of motion in an inertial XYZ frame where:
      X — horizontal downrange (launch azimuth direction)
      Y — horizontal crossrange (perpendicular to azimuth)
      Z — vertical (up positive)

    During powered flight the rocket is assumed to maintain its launch angle
    (rail guidance approximation during motor burn + a simple weathercocking
    correction for wind).  After burnout the trajectory is ballistic with full
    3-DOF dynamics.  Stability assessment uses the Barrowman static margin
    (from barrowman_result or the static_margin_cal parameter).

    Wind model — constant horizontal wind vector at all altitudes (simple model;
    no Dryden turbulence).

    State vector: [x, y, z, vx, vy, vz]  (positions [m], velocities [m/s])

    Args:
        geometry              — MotorGeometry instance.
        curve                 — Burn simulation output dict (needs time_s, thrust_n,
                                propellant_mass_kg).
        launch_angle_deg      — elevation angle from horizontal [deg] (default 90 = vertical).
        launch_azimuth_deg    — azimuth of launch direction from North [deg].
        wind_speed_m_s        — constant horizontal wind speed [m/s] (default 0).
        wind_direction_deg    — direction wind is BLOWING FROM [deg from North].
        rail_length_m         — launch rail length; vehicle follows rail during this phase.
        airframe_to_motor_mass_ratio — airframe dry mass / motor final mass (default 2.4).
        drag_coefficient_factor      — scale factor applied to Cd (robustness).
        nose_half_angle_rad   — if given, use component Cd model; else piecewise.
        body_fineness_ratio   — L/D for friction drag (used with component model).
        static_margin_cal     — assumed static margin in calibres (used when
                                barrowman_result is None) for weathercocking estimate.
        barrowman_result      — output of evaluate_barrowman_stability(), used to
                                compute the weathercocking correction angle.
        max_time_s            — maximum integration time [s] (default 300).

    Returns:
        dict with simulation summary and trajectory arrays.
    """
    import math
    # ── Parse inputs ──────────────────────────────────────────────────────────
    theta0 = math.radians(max(0.0, min(float(launch_angle_deg), 90.0)))  # elevation
    azimuth = math.radians(float(launch_azimuth_deg))

    # Launch direction unit vector (X downrange, Z up)
    # X = sin(90-theta)*cos(azimuth), Y = sin(90-theta)*sin(azimuth), Z = sin(theta)
    # But we'll treat X as the primary downrange horizontal.
    sin_theta = math.sin(theta0)
    cos_theta = math.cos(theta0)
    # Downrange = projected horizontal of launch direction
    launch_dir = np.array([cos_theta * math.cos(azimuth),
                           cos_theta * math.sin(azimuth),
                           sin_theta])   # unit vector

    # Wind velocity vector in NEU frame (X=North, Y=East, Z=Up).
    # "wind_direction_deg" is the compass bearing FROM WHICH the wind blows.
    # Wind blows TOWARD (bearing + 180) mod 360.
    # At compass bearing B: North component = cos(B_rad), East component = sin(B_rad).
    w_speed = max(float(wind_speed_m_s), 0.0)
    wind_to_bearing_rad = math.radians((float(wind_direction_deg) + 180.0) % 360.0)
    wind_vec = np.array([w_speed * math.cos(wind_to_bearing_rad),
                         w_speed * math.sin(wind_to_bearing_rad),
                         0.0])

    # Static margin for weathercocking
    if barrowman_result is not None:
        sm = float(barrowman_result.get("static_margin_cal", static_margin_cal))
    else:
        sm = max(float(static_margin_cal), 0.0)

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
    use_component_cd = nose_half_angle_rad is not None

    def get_cd(mach, burning, alt):
        if use_component_cd:
            r = evaluate_cd_by_components(
                mach=mach,
                nose_half_angle_rad=nose_half_angle_rad,
                body_fineness_ratio=body_fineness_ratio,
                is_burning=burning,
                altitude_m=alt,
                body_length_m=body_fineness_ratio * body_diameter,
            )
            cd = r["cd_total"]
        else:
            if mach < 0.75:
                cd = 0.48
            elif mach < 1.2:
                cd = 0.48 + 0.35 * (mach - 0.75) / 0.45
            elif mach < 2.0:
                cd = 0.83 - 0.18 * (mach - 1.2) / 0.8
            else:
                cd = 0.65
        return max(0.05, cd * max(float(drag_coefficient_factor), 0.01))

    # ── Powered-phase integration (Euler, same time steps as burn curve) ──────
    pos = np.zeros(3)    # [x, y, z]
    vel = np.zeros(3)    # [vx, vy, vz]
    distance_along_rail = 0.0
    rail_exited = False

    metrics = {
        "max_altitude_m": 0.0,
        "max_speed_m_s": 0.0,
        "max_mach": 0.0,
        "downrange_at_apogee_m": 0.0,
        "crossrange_at_apogee_m": 0.0,
        "burnout_altitude_m": 0.0,
        "burnout_speed_m_s": 0.0,
        "rail_exit_velocity_m_s": 0.0,
        "rail_exit_twr": 0.0,
        "time_to_apogee_s": 0.0,
    }
    traj_t = [0.0]
    traj_pos = [pos.copy()]
    traj_vel = [vel.copy()]

    for idx in range(1, len(time_s)):
        dt = max(float(time_s[idx] - time_s[idx - 1]), 1e-5)
        alt = max(float(pos[2]), 0.0)
        prop_mass = max(float(propellant_mass[idx]), 0.0)
        motor_mass = geometry.motor_final_mass_kg + prop_mass
        rocket_mass = dry_airframe_mass + motor_mass
        thrust_mag = max(float(thrust_n[idx]), 0.0)
        burning = thrust_mag > 0.0

        # Apparent velocity (relative to air = vel - wind)
        vel_apparent = vel - wind_vec
        speed_app = float(np.linalg.norm(vel_apparent))
        a_local = _isa_speed_of_sound(alt)
        mach = speed_app / max(a_local, 1.0)

        rho = SEA_LEVEL_DENSITY_KG_M3 * math.exp(-alt / 8500.0)
        cd = get_cd(mach, burning, alt)
        drag_mag = 0.5 * rho * speed_app ** 2 * cd * reference_area

        # Weathercocking: small restoring moment from crosswind rotates trajectory
        # toward wind. Simple linear approximation: angle correction proportional
        # to wind angle of attack divided by static margin.
        if not rail_exited and sm > 0.0 and speed_app > 1.0:
            crosswind = wind_vec - np.dot(wind_vec, launch_dir) * launch_dir
            cross_speed = float(np.linalg.norm(crosswind))
            aoa_rad = math.atan2(cross_speed, max(speed_app, 1.0))
            # Weathercocking correction to velocity direction (proportional to AoA / SM)
            # This is a simplified linear model, not full moment integration.
            correction_rate = min(aoa_rad / max(sm, 0.1), 0.02)   # rad per step, capped
        else:
            correction_rate = 0.0

        # Force vector
        if speed_app > 1e-6:
            drag_vec = -drag_mag * vel_apparent / speed_app
        else:
            drag_vec = np.zeros(3)

        if rail_exited:
            # Free flight: thrust follows current velocity direction
            if speed_app > 1e-6:
                thrust_dir = vel_apparent / speed_app
            else:
                thrust_dir = launch_dir.copy()
        else:
            # On rail: constrained to launch direction
            thrust_dir = launch_dir.copy()

        gravity = np.array([0.0, 0.0, -G0_M_S2 * rocket_mass])
        thrust_vec = thrust_mag * thrust_dir
        accel = (thrust_vec + drag_vec + gravity) / max(rocket_mass, 1e-9)

        vel = vel + accel * dt
        pos = pos + vel * dt
        pos[2] = max(pos[2], 0.0)

        # Rail exit check
        if not rail_exited:
            distance_along_rail += float(np.linalg.norm(vel)) * dt
            if distance_along_rail >= rail_length_m:
                rail_exited = True
                metrics["rail_exit_velocity_m_s"] = float(np.linalg.norm(vel))
                metrics["rail_exit_twr"] = thrust_mag / max(rocket_mass * G0_M_S2, 1.0)

        speed = float(np.linalg.norm(vel))
        metrics["max_altitude_m"] = max(metrics["max_altitude_m"], float(pos[2]))
        metrics["max_speed_m_s"] = max(metrics["max_speed_m_s"], speed)
        metrics["max_mach"] = max(metrics["max_mach"], mach)

        if burning:
            metrics["burnout_altitude_m"] = float(pos[2])
            metrics["burnout_speed_m_s"] = speed

        traj_t.append(float(time_s[idx]))
        traj_pos.append(pos.copy())
        traj_vel.append(vel.copy())

    # ── Ballistic coast (DOP853 ODE) ─────────────────────────────────────────
    coast_mass = dry_airframe_mass + geometry.motor_final_mass_kg

    def coast_ode_3dof(t, state):
        x, y, z, vx, vy, vz = state
        alt = max(z, 0.0)
        vel_3 = np.array([vx, vy, vz])
        vel_app = vel_3 - wind_vec
        speed_app = float(np.linalg.norm(vel_app))
        a_c = _isa_speed_of_sound(alt)
        mach_c = speed_app / max(a_c, 1.0)
        rho_c = SEA_LEVEL_DENSITY_KG_M3 * math.exp(-alt / 8500.0)
        cd_c = get_cd(mach_c, False, alt)
        drag_c = 0.5 * rho_c * speed_app ** 2 * cd_c * reference_area
        if speed_app > 1e-6:
            drag_vec_c = -drag_c * vel_app / speed_app
        else:
            drag_vec_c = np.zeros(3)
        accel_c = (drag_vec_c + np.array([0.0, 0.0, -G0_M_S2 * coast_mass])) / max(coast_mass, 1e-9)
        return [vx, vy, vz, float(accel_c[0]), float(accel_c[1]), float(accel_c[2])]

    def ground_event(t, state):
        return state[2]

    ground_event.terminal = True
    ground_event.direction = -1

    t0_coast = float(traj_t[-1])
    y0_coast = list(traj_pos[-1]) + list(traj_vel[-1])
    coast_sol = solve_ivp(
        coast_ode_3dof,
        (t0_coast, t0_coast + max_time_s),
        y0_coast,
        method="DOP853",
        events=ground_event,
        max_step=2.0,
        atol=1.0,
        rtol=1e-4,
    )

    # Extract coast trajectory
    if coast_sol.y.shape[1] > 0:
        coast_z = coast_sol.y[2]
        coast_x = coast_sol.y[0]
        coast_y = coast_sol.y[1]
        metrics["max_altitude_m"] = max(metrics["max_altitude_m"], float(np.max(coast_z)))

        # Apogee time and position
        if len(coast_z) > 1:
            apogee_idx = int(np.argmax(coast_z))
            metrics["time_to_apogee_s"] = float(coast_sol.t[apogee_idx])
            metrics["downrange_at_apogee_m"] = float(coast_x[apogee_idx])
            metrics["crossrange_at_apogee_m"] = float(coast_y[apogee_idx])

        # Landing position (last coast point)
        landing_x = float(coast_x[-1])
        landing_y = float(coast_y[-1])
        landing_t = float(coast_sol.t[-1])
    else:
        landing_x = float(traj_pos[-1][0])
        landing_y = float(traj_pos[-1][1])
        landing_t = t0_coast

    return {
        "simulation.3dof.max_altitude_m": metrics["max_altitude_m"],
        "simulation.3dof.max_speed_m_s": metrics["max_speed_m_s"],
        "simulation.3dof.max_mach": metrics["max_mach"],
        "simulation.3dof.time_to_apogee_s": metrics["time_to_apogee_s"],
        "simulation.3dof.downrange_at_apogee_m": metrics["downrange_at_apogee_m"],
        "simulation.3dof.crossrange_at_apogee_m": metrics["crossrange_at_apogee_m"],
        "simulation.3dof.burnout_altitude_m": metrics["burnout_altitude_m"],
        "simulation.3dof.burnout_speed_m_s": metrics["burnout_speed_m_s"],
        "simulation.3dof.landing_downrange_m": landing_x,
        "simulation.3dof.landing_crossrange_m": landing_y,
        "simulation.3dof.flight_time_s": landing_t,
        "simulation.3dof.rail_exit_velocity_m_s": metrics["rail_exit_velocity_m_s"],
        "simulation.3dof.rail_exit_twr": metrics["rail_exit_twr"],
        "simulation.3dof.wind_speed_m_s": float(w_speed),
        "simulation.3dof.launch_angle_deg": float(launch_angle_deg),
        "simulation.3dof.drag_model": "component" if use_component_cd else "piecewise",
        "trajectory": {
            "time_s": np.array(traj_t),
            "x_m": np.array([p[0] for p in traj_pos]),
            "y_m": np.array([p[1] for p in traj_pos]),
            "z_m": np.array([p[2] for p in traj_pos]),
        },
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
        float(_trapezoid(curve["thrust_n"], curve["time_s"]))
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


class StructuralMonteCarlo:
    """Monte Carlo for casing / closure-bolt failure probability.

    Provides a "structural-only" dispersion analysis focused on the
    probability that the casing or the closure bolts fail given uncertainty
    in the inputs that set peak chamber pressure (the dominant driver of
    the structural margins once geometry and material are fixed).

    Two sampling modes are supported:

    1. ``peak_pressure_distribution`` (cheap): supply a callable that returns
       a single sampled P_máx (Pa) for each iteration. Useful when the user
       already has Pmax statistics from a ballistics Monte Carlo.

    2. ``ballistics_callable`` (expensive): supply a callable that receives
       the perturbed-parameter dict and returns a ``curve`` dict with the
       keys ``"time_s"`` and ``"chamber_pressure_pa"``. Peak pressure is
       then extracted internally via :func:`Export.refine_peak_parabolic`-
       equivalent argmax, calling ``simulate_structural_response`` with the
       full curve so the metal fatigue / bulge / thermo-elastic metrics
       are also accurately sampled.

    Parameter sampling uses the same ``DispersionAnalysis``-style convention
    as ``MonteCarlo.py``: ``parameter_sigmas`` maps ``name -> sigma`` (float)
    or ``name -> {"mean": ..., "sigma": ...}``. Sampling is gaussian;
    "perturb_peak_pressure" inside ``parameter_sigmas`` is a reserved key
    that, when present, is added to the raw peak pressure of each sample
    (useful to inject Pmax scatter on top of (a, n, rho, T0, Dt, eta_c)
    perturbations).

    Failure criterion (any criterion below):

    - burst_safety_factor        < 1.0  (casing bulk rupture, Tresca)
    - safety_factor / governing_margin < 1.0  (casing wall yield, von Mises)
    - closure_bolt_shear_sf      < 1.0  (closure bolt shear)
    - closure_bolt_bearing_sf    < 1.0  (closure bolt bearing pressure)
    """

    def __init__(
        self,
        geometry,
        casing_material,
        *,
        peak_pressure_distribution=None,
        ballistics_callable=None,
        parameter_sigmas=None,
        bolt_count=0,
        bolt_diameter_m=0.0,
        bolt_strength_mpa=0.0,
        casing_strength_factor=1.0,
        thermal=None,
        random_seed=None,
    ):
        if (peak_pressure_distribution is None) == (ballistics_callable is None):
            raise ValueError(
                "Provide exactly one of peak_pressure_distribution or "
                "ballistics_callable."
            )
        self.geometry = geometry
        self.casing_material = casing_material or CasingMaterial()
        self.peak_pressure_distribution = peak_pressure_distribution
        self.ballistics_callable = ballistics_callable
        self.parameter_sigmas = dict(parameter_sigmas or {})
        self.bolt_count = bolt_count
        self.bolt_diameter_m = bolt_diameter_m
        self.bolt_strength_mpa = bolt_strength_mpa
        self.casing_strength_factor = casing_strength_factor
        # ``thermal`` is required by simulate_structural_response for the
        # thermo-elastic margin. Provide a neutral default so the structural
        # path does not silently crash when the user has not run a thermal
        # analysis before the structural Monte Carlo.
        self.thermal = thermal or {
            "simulation.advanced.thermal.casing_inner_wall_temp_c": 20.0,
            "simulation.advanced.thermal.throat_ablation_mm": 0.0,
        }
        self.random_seed = random_seed

    def _sample_parameters(self, n_iterations, rng):
        """Build n perturbed-parameter dicts using the DispersionAnalysis
        convention (mean=0 default, normal noise with sigma=parameter_sigmas[k]).
        When ``"perturb_peak_pressure"`` is in sigmas, it is returned separately
        so callers can add it on top of the raw Pmax without confusing the
        ballistics callable (which only consumes "real" propellant params).
        """
        samples = []
        pmax_perturbs = []
        for _ in range(n_iterations):
            sample = {}
            pmax_perturb = 0.0
            for name, spec in self.parameter_sigmas.items():
                if isinstance(spec, dict):
                    mean = float(spec.get("mean", spec.get("nominal", 0.0)))
                    sigma = float(spec.get("sigma", spec.get("std", spec.get("stddev", 0.0))))
                else:
                    mean = 0.0
                    sigma = float(spec)
                value = mean + sigma * rng.standard_normal() if sigma > 0.0 else mean
                if name == "perturb_peak_pressure":
                    pmax_perturb = value
                else:
                    sample[name] = value
            samples.append(sample)
            pmax_perturbs.append(pmax_perturb)
        return samples, pmax_perturbs

    @staticmethod
    def _extract_peak_pressure(curve):
        """Pull peak chamber pressure (Pa) from a curve dict."""
        t = np.asarray(curve.get("time_s", []), dtype=float)
        p = np.asarray(curve.get("chamber_pressure_pa", []), dtype=float)
        if t.size == 0 or p.size == 0:
            return 0.0
        return float(np.max(p))

    def _run_single(self, sample, pmax_perturb):
        """One Monte Carlo iteration: perturb -> resolve Pmax -> structural -> FS.

        Returns (peak_pressure, structural_dict) or None if the iteration
        failed (e.g. ballistics callable raised — treat as a no-vote).
        """
        try:
            if self.peak_pressure_distribution is not None:
                peak_pressure = float(self.peak_pressure_distribution())
            else:
                curve = self.ballistics_callable(sample)
                peak_pressure = self._extract_peak_pressure(curve)
            peak_pressure = max(peak_pressure + pmax_perturb, 0.0)

            # Synthesise a 1-point curve, since every structural metric that
            # is failure-relevant (max_pressure, max_von_mises, blow_force,
            # bolt stresses) sees only the peak pressure through the path
            # we already extracted. Full-pressure-history is consumed only
            # by the pressurization impulse and the fatigue proxy, neither of
            # which feeds the standard failure criteria.
            synthetic_curve = {
                "time_s": np.array([0.0, 0.001, 1.0]),
                "thrust_n": np.array([0.0, 0.0, 0.0]),
                "chamber_pressure_pa": np.array([0.0, peak_pressure, 0.0]),
                "mass_flow_kg_s": np.array([0.0, 0.0, 0.0]),
            }
            structural = simulate_structural_response(
                self.geometry,
                synthetic_curve,
                self.thermal,
                casing_material=self.casing_material,
                casing_strength_factor=self.casing_strength_factor,
                bolt_count=self.bolt_count,
                bolt_diameter_m=self.bolt_diameter_m,
                bolt_strength_mpa=self.bolt_strength_mpa,
            )
            return peak_pressure, structural
        except Exception:
            return None

    def run(self, n_iterations):
        """Execute n_iterations samples serially and return a summary dict.

        Returns a dict with:
            - failure_probability:        any-criterion <1.0 fraction
            - failure_probability_casing: governing margin <1.0
            - failure_probability_burst:  burst_safety_factor <1.0
            - failure_probability_bolts:  shear OR bearing <1.0
            - peak_pressure_pa:           list of sampled Pmax (Pa)
            - burst_sf, governing_sf, bolt_shear_sf, bolt_bearing_sf: lists
            - samples:                    the perturbed-parameter dicts
            - n_iterations, n_evaluated
        """
        rng = np.random.default_rng(self.random_seed)
        samples, pmax_perturbs = self._sample_parameters(n_iterations, rng)
        peak_pressures = []
        burst_sf_list = []
        governing_sf_list = []
        bolt_shear_sf_list = []
        bolt_bearing_sf_list = []
        failures_casing = 0
        failures_burst = 0
        failures_bolts = 0
        n_evaluated = 0
        for sample, pmax_perturb in zip(samples, pmax_perturbs):
            result = self._run_single(sample, pmax_perturb)
            if result is None:
                continue
            peak_pressure, structural = result
            n_evaluated += 1
            peak_pressures.append(peak_pressure)
            burst_sf = structural["simulation.advanced.structural.burst_safety_factor"]
            governing_sf = structural["simulation.advanced.structural.safety_factor"]
            bolt_shear_sf = structural["simulation.advanced.structural.closure_bolt_shear_safety_factor"]
            bolt_bearing_sf = structural["simulation.advanced.structural.closure_bolt_bearing_safety_factor"]
            burst_sf_list.append(burst_sf)
            governing_sf_list.append(governing_sf)
            bolt_shear_sf_list.append(bolt_shear_sf)
            bolt_bearing_sf_list.append(bolt_bearing_sf)
            failed_casing = (governing_sf < 1.0) or (burst_sf < 1.0)
            failed_burst = burst_sf < 1.0
            # ``inf`` SF means bolts were not configured; don't count as failure.
            bolt_configured = self.bolt_count > 0 and self.bolt_diameter_m > 0.0
            failed_bolts = (
                bolt_configured
                and (bolt_shear_sf < 1.0 or bolt_bearing_sf < 1.0)
            )
            if failed_casing:
                failures_casing += 1
            if failed_burst:
                failures_burst += 1
            if failed_bolts:
                failures_bolts += 1
        # Use a proper union rather than summing single-criterion failures
        # because casing and bolt failures can co-occur on the same sample.
        failures_any = 0
        for i in range(n_evaluated):
            fcasing = (governing_sf_list[i] < 1.0) or (burst_sf_list[i] < 1.0)
            bolt_configured = self.bolt_count > 0 and self.bolt_diameter_m > 0.0
            fbolts = (
                bolt_configured
                and (bolt_shear_sf_list[i] < 1.0 or bolt_bearing_sf_list[i] < 1.0)
            )
            if fcasing or fbolts:
                failures_any += 1
        n = max(n_evaluated, 1)
        return {
            "n_iterations": n_iterations,
            "n_evaluated": n_evaluated,
            "failure_probability": failures_any / n,
            "failure_probability_casing": failures_casing / n,
            "failure_probability_burst": failures_burst / n,
            "failure_probability_bolts": failures_bolts / n,
            "peak_pressure_pa": peak_pressures,
            "burst_safety_factor": burst_sf_list,
            "governing_safety_factor": governing_sf_list,
            "bolt_shear_safety_factor": bolt_shear_sf_list,
            "bolt_bearing_safety_factor": bolt_bearing_sf_list,
            "samples": samples,
        }
