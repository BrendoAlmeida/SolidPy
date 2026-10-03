# -*- coding: utf-8 -*-
"""Batched structural-response kernel for StructuralMonteCarlo scenarios.

The sample axis is evaluated with ``xp`` so the same equations can run with NumPy or
``jax.numpy``. Geometry, material properties, and bolt configuration are scalar inputs
prepared by the caller; peak chamber pressure is a one-dimensional array.
"""

from __future__ import annotations

import math

import numpy as np


_PREFIX = "simulation.advanced.structural."


def _per_lane(reference, value, xp):
    """Broadcast a scalar structural value over the pressure sample axis."""
    return xp.zeros_like(reference) + value


def structural_response_vectorized(
    geometry,
    chamber_pressure_pa,
    casing_material,
    casing_strength_factor=1.0,
    *,
    bolt_count=0,
    bolt_diameter_m=0.0,
    bolt_strength_mpa=0.0,
    closure_bolts_applicable=True,
    thermal=None,
    xp=np,
):
    """Evaluate structural metrics for independent peak-pressure scenarios.

    This mirrors ``simulate_structural_response`` for StructuralMonteCarlo's synthetic
    history: ``time_s=[0, 0.001, 1]``, zero thrust, and chamber pressure
    ``[0, peak_pressure, 0]``. Every numeric result is returned on the sample axis;
    unavailable bolt metrics and absent thermal metrics are ``None``. Inputs are
    expected to have been validated by the caller.

    ``chamber_pressure_pa`` contains one peak pressure per sample and is used directly
    through namespace operations; it is never converted to a host array, so it may be
    a JAX tracer. ``thermal`` follows the scalar model's mapping contract.
    """
    pressure = xp.maximum(chamber_pressure_pa, 0.0)

    wall = max(geometry.casing_wall_thickness_m, 1e-5)
    inner_radius = max(geometry.motor_inner_diameter_m / 2.0, 1e-5)
    outer_radius = inner_radius + wall
    modulus_pa = casing_material.modulus_gpa * 1e9
    strength_factor = max(casing_strength_factor, 0.01)
    yield_pa = casing_material.yield_strength_mpa * 1e6 * strength_factor
    allowable_pa = casing_material.resolved_allowable_stress_mpa * 1e6 * strength_factor
    poisson = casing_material.poisson_ratio

    # Preserve the thin-wall transition and Lamé equations of the scalar model.
    ri2 = inner_radius**2
    ro2 = outer_radius**2
    if (wall / max(inner_radius, 1e-9)) > 0.1:
        hoop = (
            pressure * ri2 * (ro2 + ri2)
            / max(ro2 - ri2, 1e-9)
            / max(ri2, 1e-9)
        )
        axial = pressure * ri2 / max(ro2 - ri2, 1e-9)
    else:
        hoop = pressure * inner_radius / wall
        axial = pressure * inner_radius / (2.0 * wall)
    radial = -pressure
    von_mises = xp.sqrt(
        xp.maximum(
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

    metal_sf = yield_pa / xp.maximum(von_mises, 1.0)
    composite_margin = allowable_pa / xp.maximum(hoop, 1.0)
    governing_margin = (
        composite_margin
        if casing_material.material_family == "composite"
        else metal_sf
    )

    ultimate_mpa = casing_material.resolved_ultimate_strength_mpa
    ultimate_pa = ultimate_mpa * 1e6 * strength_factor
    burst_pressure_pa = (
        (2.0 / math.sqrt(3.0))
        * ultimate_mpa
        * 1e6
        * strength_factor
        * math.log1p(wall / inner_radius)
    )
    yield_pressure_pa = (
        ultimate_pa
        * max(outer_radius**2 - inner_radius**2, 0.0)
        / (math.sqrt(3.0) * outer_radius**2)
    )

    blow_force_n = (
        pressure
        * math.pi
        * 0.25
        * max(geometry.motor_inner_diameter_m, 0.0) ** 2
    )
    if not closure_bolts_applicable:
        bolt_status = "not_configured"
        bolt_applicability = "not_applicable"
        bolt_reason = "closure_bolts_not_applicable"
    elif bolt_count == 0:
        bolt_status = "not_configured"
        bolt_applicability = "not_modeled"
        bolt_reason = "closure_bolts_not_configured"
    elif bolt_diameter_m == 0 or bolt_strength_mpa == 0:
        bolt_status = "model_not_available"
        bolt_applicability = "not_modeled"
        bolt_reason = "incomplete_closure_bolt_properties"
    else:
        bolt_status = "configured"
        bolt_applicability = "applicable"
        bolt_reason = None

    if bolt_status == "configured":
        shear_area_m2 = bolt_count * math.pi * 0.25 * bolt_diameter_m**2
        bolt_shear_stress_pa = blow_force_n / max(shear_area_m2, 1e-9)
        bolt_shear_stress_mpa = bolt_shear_stress_pa / 1e6
        bolt_shear_sf = (
            0.6 * bolt_strength_mpa * 1e6 / xp.maximum(bolt_shear_stress_pa, 1.0)
        )
        bearing_area_m2 = bolt_count * bolt_diameter_m * wall
        bolt_bearing_stress_pa = blow_force_n / max(bearing_area_m2, 1e-9)
        bolt_bearing_stress_mpa = bolt_bearing_stress_pa / 1e6
        bolt_bearing_sf = (
            1.5
            * ultimate_mpa
            * 1e6
            * strength_factor
            / xp.maximum(bolt_bearing_stress_pa, 1.0)
        )
    else:
        bolt_shear_stress_mpa = None
        bolt_shear_sf = None
        bolt_bearing_stress_mpa = None
        bolt_bearing_sf = None

    wall_temperature = (thermal or {}).get(
        "simulation.advanced.thermal.casing_inner_wall_temp_c"
    )
    if wall_temperature is None:
        thermal_service_margin = None
        thermal_status = "not_modeled"
    else:
        onset_temp_c = 0.6 * casing_material.max_service_temp_c
        thermal_service_margin = 1.0 - max(
            0.0,
            (wall_temperature - onset_temp_c)
            / max(casing_material.max_service_temp_c - onset_temp_c, 1.0),
        )
        thermal_service_margin = _per_lane(pressure, thermal_service_margin, xp)
        thermal_status = "computed"

    # The synthetic pressure history integrates to 0.5 * peak over its one-second
    # duration; its zero thrust makes the acceleration crack index exactly zero.
    pressure_integral = (
        0.5 * (0.0 + pressure) * 0.001
        + 0.5 * (pressure + 0.0) * 0.999
    )
    zero = xp.zeros_like(pressure)
    return {
        _PREFIX + "max_stress_mpa": von_mises / 1e6,
        _PREFIX + "safety_factor": governing_margin,
        _PREFIX + "metal_equivalent_sf": metal_sf,
        _PREFIX + "composite_case_margin": composite_margin,
        _PREFIX + "governing_margin": governing_margin,
        _PREFIX + "casing_radial_bulge_mm": 1000.0 * bulge,
        _PREFIX + "first_mode_hz": _per_lane(pressure, first_mode_hz, xp),
        _PREFIX + "acceleration_crack_index": zero,
        _PREFIX + "max_hoop_stress_mpa": hoop / 1e6,
        _PREFIX + "max_axial_stress_mpa": axial / 1e6,
        _PREFIX + "max_hoop_strain_microstrain": strain * 1e6,
        _PREFIX + "buckling_margin": critical_buckling_pa / xp.maximum(pressure, 1.0),
        _PREFIX + "low_cycle_fatigue_damage": von_mises / max(yield_pa, 1.0),
        _PREFIX + "pressurization_impulse_mpa_s": pressure_integral / 1e6,
        _PREFIX + "thermal_service_margin": thermal_service_margin,
        _PREFIX + "thermoelastic_margin": thermal_service_margin,
        _PREFIX + "thermal_service_status": thermal_status,
        _PREFIX + "burst_pressure_mpa": _per_lane(pressure, burst_pressure_pa / 1e6, xp),
        _PREFIX + "casing_burst_pressure_mpa": _per_lane(pressure, burst_pressure_pa / 1e6, xp),
        _PREFIX + "burst_safety_factor": burst_pressure_pa / xp.maximum(pressure, 1.0),
        _PREFIX + "yield_pressure_mpa": _per_lane(pressure, yield_pressure_pa / 1e6, xp),
        _PREFIX + "ultimate_elastic_limit_pressure_mpa": _per_lane(
            pressure, yield_pressure_pa / 1e6, xp
        ),
        _PREFIX + "closure_bolt_blow_force_n": blow_force_n,
        _PREFIX + "closure_bolt_shear_safety_factor": bolt_shear_sf,
        _PREFIX + "closure_bolt_bearing_safety_factor": bolt_bearing_sf,
        _PREFIX + "closure_bolt_shear_stress_mpa": bolt_shear_stress_mpa,
        _PREFIX + "closure_bolt_bearing_stress_mpa": bolt_bearing_stress_mpa,
        _PREFIX + "closure_bolt_status": bolt_status,
        _PREFIX + "closure_bolt_applicability": bolt_applicability,
        _PREFIX + "closure_bolt_reason": bolt_reason,
    }
