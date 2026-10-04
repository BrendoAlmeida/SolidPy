"""Pack the transient structural, CFD and ignition proxies for an advanced-physics ensemble."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Sequence

import numpy as np

from ..Multiphysics import (
    _closure_bolt_configuration, _require_casing_material, _series, _structural_number,
)
from .thermal import _lane_count, _per_lane


LANE_ARRAYS = (
    "time_s", "thrust_n", "pressure_pa", "mass_flow_kg_s", "n_points",
    "throat_diameter_m", "motor_inner_diameter_m", "wall_thickness_m", "motor_length_m",
    "dry_mass_kg", "grain_core_diameter_m", "grain_gap_m", "fill_length_m", "free_volume_m3",
    "heat_load_kj_m2", "throat_ablation_mm", "wall_temperature_c", "has_wall_temperature",
    "flame_temp_k", "r_specific", "casing_density_kg_m3", "modulus_pa", "yield_pa",
    "allowable_pa", "ultimate_pa", "poisson_ratio", "max_service_temp_c", "is_composite",
    "strength_factor",
)


@dataclass
class AdvancedPhysicsBatch:
    """Padded per-lane inputs for the three transient advanced-physics proxies.

    ``lanes`` retains the original geometry, curve, thermal mapping and material references for scalar fallback.
    Numerical arrays are snapshots and do not change if those objects are later mutated.
    """

    arrays: Dict[str, np.ndarray]
    lanes: List[Dict[str, Any]]

    def __len__(self) -> int:
        return len(self.lanes)

    @property
    def t_max(self) -> int:
        return int(self.arrays["time_s"].shape[1])

    def namespace(self, xp) -> Dict[str, Any]:
        """Return the numeric arrays converted to the requested array namespace."""
        return {name: xp.asarray(value) for name, value in self.arrays.items()}

    def select(self, indices: Sequence[int]) -> "AdvancedPhysicsBatch":
        """Select lanes in order and trim the padded time axis to the longest selected history."""
        index = np.asarray(indices, dtype=int)
        arrays = {name: self.arrays[name][index] for name in LANE_ARRAYS}
        n = max(int(arrays["n_points"].max()) if len(index) else 1, 2)
        for name in ("time_s", "thrust_n", "pressure_pa", "mass_flow_kg_s"):
            arrays[name] = arrays[name][:, :n]
        return AdvancedPhysicsBatch(arrays, [self.lanes[i] for i in index])

    @classmethod
    def from_objects(
        cls,
        geometries,
        curves,
        thermals,
        casing_materials=None,
        *,
        flame_temp_k=2800.0,
        r_specific=287.0,
    ) -> "AdvancedPhysicsBatch":
        """Pack aligned lane inputs, validating the transient structural contract up front."""
        count = _lane_count(geometries, curves, thermals, casing_materials, flame_temp_k, r_specific)
        geometries = _per_lane(geometries, count, "geometries")
        curves = _per_lane(curves, count, "curves")
        thermals = _per_lane(thermals, count, "thermals")
        casings = [_require_casing_material(m) for m in _per_lane(casing_materials, count, "casing_materials")]
        flames = _per_lane(flame_temp_k, count, "flame_temp_k")
        specifics = _per_lane(r_specific, count, "r_specific")

        lanes: List[Dict[str, Any]] = []
        parts: List[Dict[str, Any]] = []
        for i, (geometry, curve, thermal, casing, flame, specific) in enumerate(
            zip(geometries, curves, thermals, casings, flames, specifics)
        ):
            scenario = curve.get("scenario_factors", {}) if isinstance(curve, dict) else {}
            scenario_strength = float(scenario.get("casing_strength_factor", 1.0) or 1.0)
            strength_factor = max(
                _structural_number("casing_strength_factor", scenario_strength, positive=True),
                0.01,
            )
            time_s = np.asarray(curve["time_s"], dtype=float)
            thrust = np.asarray(curve["thrust_n"], dtype=float)
            pressure = _series(curve, "chamber_pressure_pa", np.zeros_like(time_s))
            if (time_s.ndim != 1 or not len(time_s) or thrust.shape != time_s.shape
                    or pressure.shape != time_s.shape or np.any(np.diff(time_s) <= 0)
                    or not all(np.all(np.isfinite(values)) for values in (time_s, thrust, pressure))
                    or np.any(pressure < 0)):
                raise ValueError("structural curve must contain aligned finite arrays on increasing time_s")

            # The default expression is intentionally evaluated even when mass_nozzle_kg_s exists, matching the
            # scalar proxy's requirement that mass_flow_kg_s be present on the curve.
            mass_flow = _series(curve, "mass_nozzle_kg_s", curve["mass_flow_kg_s"])
            if mass_flow.ndim != 1 or len(mass_flow) != len(time_s):
                raise ValueError("CFD mass-flow series must be aligned with time_s")
            for name, value in (
                ("motor_inner_diameter_m", geometry.motor_inner_diameter_m),
                ("casing_wall_thickness_m", geometry.casing_wall_thickness_m),
                ("motor_length_m", geometry.motor_length_m),
            ):
                _structural_number(name, value, positive=True)
            for name, value in (
                ("grain_core_diameter_m", geometry.grain_core_diameter_m),
                ("grain_gap_m", geometry.grain_gap_m),
                ("fill_length_m", geometry.fill_length_m),
                ("free_volume_m3", geometry.free_volume_m3),
                ("throat_diameter_m", geometry.throat_diameter_m),
                ("dry_mass_kg", geometry.dry_mass_kg),
            ):
                _structural_number(name, value)

            wall_temperature = thermal.get("simulation.advanced.thermal.casing_inner_wall_temp_c")
            if wall_temperature is not None:
                _structural_number("casing_inner_wall_temp_c", wall_temperature)
            heat_load = _structural_number(
                "heat_load_kj_m2", thermal["simulation.advanced.thermal.heat_load_kj_m2"]
            )
            throat_ablation = _structural_number(
                "throat_ablation_mm", thermal["simulation.advanced.thermal.throat_ablation_mm"]
            )
            flame = _structural_number("flame_temp_k", flame)
            specific = _structural_number("r_specific", specific, positive=True)

            lane = {
                "geometry": geometry, "curve": curve, "thermal": thermal, "casing_material": casing,
                "flame_temp_k": flame, "r_specific": specific, "strength_factor": strength_factor,
            }
            lanes.append(lane)
            parts.append({
                "time_s": time_s, "thrust_n": thrust, "pressure_pa": pressure, "mass_flow_kg_s": mass_flow,
                "n_points": len(time_s), "throat_diameter_m": geometry.throat_diameter_m,
                "motor_inner_diameter_m": geometry.motor_inner_diameter_m,
                "wall_thickness_m": geometry.casing_wall_thickness_m, "motor_length_m": geometry.motor_length_m,
                "dry_mass_kg": geometry.dry_mass_kg, "grain_core_diameter_m": geometry.grain_core_diameter_m,
                "grain_gap_m": geometry.grain_gap_m, "fill_length_m": geometry.fill_length_m,
                "free_volume_m3": geometry.free_volume_m3, "heat_load_kj_m2": heat_load,
                "throat_ablation_mm": throat_ablation,
                "wall_temperature_c": 0.0 if wall_temperature is None else float(wall_temperature),
                "has_wall_temperature": wall_temperature is not None, "flame_temp_k": flame,
                "r_specific": specific, "casing_density_kg_m3": casing.density_kg_m3,
                "modulus_pa": casing.modulus_gpa * 1e9,
                "yield_pa": casing.yield_strength_mpa * 1e6 * strength_factor,
                "allowable_pa": casing.resolved_allowable_stress_mpa * 1e6 * strength_factor,
                "ultimate_pa": casing.resolved_ultimate_strength_mpa * 1e6 * strength_factor,
                "poisson_ratio": casing.poisson_ratio, "max_service_temp_c": casing.max_service_temp_c,
                "is_composite": casing.material_family == "composite", "strength_factor": strength_factor,
            })

        t_max = max(max((part["n_points"] for part in parts), default=1), 2)
        arrays: Dict[str, np.ndarray] = {}
        for name in LANE_ARRAYS:
            if name in {"time_s", "thrust_n", "pressure_pa", "mass_flow_kg_s"}:
                arrays[name] = np.zeros((count, t_max), dtype=float)
            else:
                arrays[name] = np.asarray([part[name] for part in parts])
        for i, part in enumerate(parts):
            n = part["n_points"]
            for name in ("time_s", "thrust_n", "pressure_pa", "mass_flow_kg_s"):
                values = part[name]
                arrays[name][i, :n] = values
                if n < t_max:
                    arrays[name][i, n:] = values[-1] if name == "time_s" else 0.0
        return cls(arrays, lanes)


def assemble_advanced_physics_proxies(batch: AdvancedPhysicsBatch, output) -> List[Dict[str, Any]]:
    """Convert numeric kernel arrays to per-lane mappings with the scalar category and optional-value fields."""
    from .kernels.advanced_physics import STRUCTURAL_PREFIX

    count = len(batch)
    results: List[Dict[str, Any]] = [dict() for _ in range(count)]
    for name, values in output.items():
        values = np.asarray(values)
        if values.shape != (count,) or not np.all(np.isfinite(values)):
            raise ValueError(f"advanced proxy output {name!r} must be one finite value per lane")
        for lane in range(count):
            results[lane][name] = float(values[lane])

    status, applicability, reason = _closure_bolt_configuration(0, 0.0, 0.0, True)
    for lane, result in enumerate(results):
        thermal = batch.lanes[lane]["thermal"]
        wall_temperature = thermal.get("simulation.advanced.thermal.casing_inner_wall_temp_c")
        margin = None
        if wall_temperature is not None:
            material = batch.lanes[lane]["casing_material"]
            onset = 0.6 * material.max_service_temp_c
            margin = 1.0 - max(
                0.0,
                (float(wall_temperature) - onset) / max(material.max_service_temp_c - onset, 1.0),
            )
        result.update({
            STRUCTURAL_PREFIX + "thermal_service_margin": margin,
            STRUCTURAL_PREFIX + "thermoelastic_margin": margin,
            STRUCTURAL_PREFIX + "thermal_service_status": "computed" if margin is not None else "not_modeled",
            STRUCTURAL_PREFIX + "closure_bolt_status": status,
            STRUCTURAL_PREFIX + "closure_bolt_applicability": applicability,
            STRUCTURAL_PREFIX + "closure_bolt_reason": reason,
            STRUCTURAL_PREFIX + "closure_bolt_shear_safety_factor": None,
            STRUCTURAL_PREFIX + "closure_bolt_bearing_safety_factor": None,
            STRUCTURAL_PREFIX + "closure_bolt_shear_stress_mpa": None,
            STRUCTURAL_PREFIX + "closure_bolt_bearing_stress_mpa": None,
        })
    return results


def scalar_advanced_physics_proxies(batch: AdvancedPhysicsBatch) -> List[Dict[str, Any]]:
    """Evaluate proxy lanes through the scalar reference functions."""
    from ..Multiphysics import simulate_cfd_proxies, simulate_ignition_proxy, simulate_structural_response

    results = []
    for lane in batch.lanes:
        structural = simulate_structural_response(
            lane["geometry"], lane["curve"], lane["thermal"],
            casing_material=lane["casing_material"], casing_strength_factor=lane["strength_factor"],
        )
        results.append({
            **structural,
            **simulate_cfd_proxies(
                lane["geometry"], lane["curve"], lane["thermal"],
                r_specific=lane["r_specific"], flame_temp_k=lane["flame_temp_k"],
            ),
            **simulate_ignition_proxy(lane["geometry"], lane["curve"], lane["thermal"], structural),
        })
    return results
