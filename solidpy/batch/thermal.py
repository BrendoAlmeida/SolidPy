# -*- coding: utf-8 -*-
"""Pack the inputs of ``Multiphysics.simulate_thermal_ablation`` into a structure-of-arrays batch (one lane per curve).

A lane is a motor geometry, a ballistic curve, a casing and nozzle material and the thermal settings. Packing
builds on the host, in float64 and with the scalar code's own helpers (``_wall_layers`` and the finite-volume
operator), everything that does not depend on the wall temperature: the operator of each wall, and for every
time step of the curve the Bartz coefficient and the step length. What is left for the accelerator is the
integration of the wall temperatures, which is the work that costs time. The throat ablation does not feed back
into the conduction, so it is a sum over the series and is also finished at pack time.

Array schema (``B`` = lane, ``N`` = padded wall cells, ``T`` = padded time steps):

* ``lower[B, N-1]``, ``diag[B, N]``, ``upper[B, N-1]``, ``source[B, N]``, ``y0[B, N]``, ``e0[B]``: the diagonals of the
  tridiagonal conduction operator, its constant source, the initial temperatures and the inverse thermal mass of the
  hot-face cell, which scales the heat flux into it. Padded cells have zero entries and zero temperature.
* ``dt[B, T]``, ``bartz[B, T]``: step length and temperature-independent Bartz coefficient of each step.
* lane constants: ``n_nodes``, ``n_intervals``, ``inner_node``, ``interface_node``, ``has_liner``, ``thickness_m``,
  ``flame_temp_k``, ``stagnation``, ``recovery_temp_k`` and the pack-time results ``throat_ablation_m``,
  ``max_recovery_temp_k``, ``burn_duration_s``, ``wall_thickness_m``, ``initial_temp_k``.

A lane whose inputs the batched integration cannot reproduce (a series that is not finite, series of different
lengths) carries a feature in ``ThermalBatch.features`` and is left to the scalar reference by the router.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Dict, FrozenSet, List, Mapping, Optional, Sequence

import numpy as np

from ..Multiphysics import (
    OUTER_WALL_H_W_M2K, CasingMaterial, NozzleMaterial, _build_wall_conduction_operator, _resolve_gamma, _series,
    _sutherland_viscosity, _thermal_report, _wall_layers,
)

NON_FINITE_INPUT = "non_finite_thermal_input"
SERIES_MISMATCH = "thermal_series_mismatch"
THERMAL_FEATURES = (NON_FINITE_INPUT, SERIES_MISMATCH)

#: Names of the per-lane arrays, in the order ``select`` slices them.
LANE_ARRAYS = (
    "lower", "diag", "upper", "source", "y0", "e0", "dt", "bartz", "n_nodes", "n_intervals", "inner_node",
    "interface_node", "has_liner", "thickness_m", "flame_temp_k", "stagnation", "recovery_temp_k", "throat_ablation_m", "max_recovery_temp_k",
    "burn_duration_s", "wall_thickness_m", "initial_temp_k",
)


def _per_lane(value, count: int, name: str) -> List[Any]:
    """``value`` broadcast to ``count`` lanes, or checked to hold one entry per lane."""
    if isinstance(value, Mapping) or not isinstance(value, (list, tuple, np.ndarray)):
        return [value] * count
    if isinstance(value, np.ndarray) and value.ndim == 0:
        return [value.item()] * count
    if len(value) != count:
        raise ValueError(f"{name} has {len(value)} entries for {count} lanes")
    return list(value)


def _lane_count(*values) -> int:
    counts = {len(v) for v in values if isinstance(v, (list, tuple)) or (isinstance(v, np.ndarray) and v.ndim)}
    if not counts:
        return 1
    if len(counts) > 1:
        raise ValueError(f"the per-lane arguments disagree on the number of lanes: {sorted(counts)}")
    return counts.pop()


def _finite(*values) -> bool:
    return all(np.all(np.isfinite(np.asarray(v, dtype=float))) for v in values)


def interval_coefficients(time_s, thrust_n, mass_flow_kg_s, pressure_pa, throat_diameter_m, ablation_series_m, flame_temp_k,
                          r_specific, gamma, ablation_scale, pressure_exponent, mass_flux_exponent):
    """Step length, Bartz coefficient and throat ablation of every time step of one lane, as the scalar loop computes them.

    The series have one entry per time point; step ``i`` (1 to ``len - 1``) uses the values at point ``i`` and the length
    ``t[i] - t[i-1]``, floored at 1e-5 s. Returns ``(dt, bartz_base, recovery_temp_k, throat_ablation_m)``, the last
    being the recession after the final step.
    """
    dt = np.maximum(np.diff(time_s), 1e-5)
    radius = np.maximum(throat_diameter_m[1:] * 0.5, 1e-6)
    area = math.pi * radius**2
    pressure = np.maximum(pressure_pa[1:], 0.0)
    pressure = np.where(pressure <= 0.0, np.maximum(thrust_n[1:], 0.0) / np.maximum(area, 1e-9), pressure)
    mass_flow = np.maximum(mass_flow_kg_s[1:], 0.0)

    recovery = flame_temp_k * (1.0 + 0.89 * (gamma - 1.0) / 2.0) / max(1.0 + (gamma - 1.0) / 2.0, 1e-9)
    cp_gas = gamma * r_specific / max(gamma - 1.0, 1e-9)
    mass_flux = mass_flow / np.maximum(area, 1e-9)
    curvature = np.maximum(2.0 * radius, 1e-9)
    bartz = (
        0.026
        / np.maximum((2.0 * radius) ** 0.2, 1e-4)
        * (_sutherland_viscosity(flame_temp_k) ** 0.2 * cp_gas / max(0.82**0.6, 1e-9))
        * np.maximum(mass_flux, 1e-9) ** 0.8
        * (2.0 * radius / np.maximum(curvature, 1e-9)) ** 0.1
    )

    rate = 1.8e-8 * ablation_scale * np.maximum(pressure, 1.0) ** pressure_exponent * np.maximum(mass_flow, 1e-9) ** mass_flux_exponent
    # the scalar loop sets the recession from the series where it has a value and otherwise adds rate * dt: that is the
    # last series value (or zero) followed by the sum of the later steps, accumulated in the same order
    series = ablation_series_m[1:]
    given = np.flatnonzero(~np.isnan(series))
    first = int(given[-1]) + 1 if len(given) else 0
    base = max(float(series[given[-1]]), 0.0) if len(given) else 0.0
    ablation = float(np.cumsum(np.concatenate([[base], (rate * dt)[first:]]))[-1])
    return dt, bartz, recovery, ablation


@dataclass
class ThermalBatch:
    """Many independent wall-conduction problems as arrays; ``lanes`` keeps each lane's scalar inputs."""

    arrays: Dict[str, np.ndarray]
    lanes: List[Dict[str, Any]]
    features: List[FrozenSet[str]] = field(default_factory=list)

    def __len__(self) -> int:
        return len(self.lanes)

    @property
    def n_max(self) -> int:
        return int(self.arrays["diag"].shape[1])

    @property
    def t_max(self) -> int:
        return int(self.arrays["dt"].shape[1])

    def namespace(self, xp) -> Dict[str, Any]:
        """The arrays as ``xp`` arrays."""
        return {name: xp.asarray(array) for name, array in self.arrays.items()}

    def unsupported(self, capabilities) -> List[List[str]]:
        """For each lane, the features ``capabilities`` does not fully support."""
        return [capabilities.missing(sorted(features)) for features in self.features]

    def select(self, indices: Sequence[int], trim: bool = True) -> "ThermalBatch":
        """The lanes ``indices`` in that order (an index may repeat), the padded axes trimmed to what they need unless ``trim`` is false."""
        index = np.asarray(indices, dtype=int)
        arrays = {name: self.arrays[name][index] for name in LANE_ARRAYS}
        if trim:
            n = int(arrays["n_nodes"].max()) if len(index) else 1
            t = max(int(arrays["n_intervals"].max()) if len(index) else 1, 1)
            for name in ("lower", "upper"):
                arrays[name] = arrays[name][:, : n - 1]
            for name in ("diag", "source", "y0"):
                arrays[name] = arrays[name][:, :n]
            for name in ("dt", "bartz"):
                arrays[name] = arrays[name][:, :t]
        return ThermalBatch(arrays, [self.lanes[i] for i in index], [self.features[i] for i in index])

    def with_padding(self, nodes: int, steps: int) -> "ThermalBatch":
        """The same lanes with the node axis padded to ``nodes`` and the step axis to ``steps`` (no smaller than now)."""
        if nodes < self.n_max or steps < self.t_max:
            raise ValueError("padding cannot shrink an axis")
        a = self.arrays
        count = len(self)
        arrays = dict(a)
        for name in ("lower", "upper"):
            arrays[name] = np.zeros((count, nodes - 1))
            arrays[name][:, : self.n_max - 1] = a[name]
        for name in ("diag", "source", "y0"):
            arrays[name] = np.zeros((count, nodes))
            arrays[name][:, : self.n_max] = a[name]
        for name in ("dt", "bartz"):
            arrays[name] = np.full((count, steps), 1.0 if name == "dt" else 0.0)
            arrays[name][:, : self.t_max] = a[name]
        return ThermalBatch(arrays, self.lanes, self.features)

    @classmethod
    def from_objects(
        cls,
        geometries,
        curves,
        casing_materials=None,
        nozzle_materials=None,
        *,
        flame_temp_k=2800.0,
        r_specific=287.0,
        gamma=None,
        initial_temperature_k=298.15,
        liner_thickness_factor=1.0,
    ) -> "ThermalBatch":
        """Pack lanes with the arguments of ``simulate_thermal_ablation``; each may be one value for all lanes or one per lane.

        Raises ``KeyError`` for a curve without ``time_s``, ``thrust_n`` or a mass flow, as the scalar function does
        when it reads them (here at pack time rather than when the lane is solved).
        """
        count = _lane_count(geometries, curves, casing_materials, nozzle_materials, flame_temp_k, r_specific, gamma,
                            initial_temperature_k, liner_thickness_factor)
        geometries = _per_lane(geometries, count, "geometries")
        curves = _per_lane(curves, count, "curves")
        casings = [m or CasingMaterial() for m in _per_lane(casing_materials, count, "casing_materials")]
        nozzles = [m or NozzleMaterial() for m in _per_lane(nozzle_materials, count, "nozzle_materials")]
        flames = _per_lane(flame_temp_k, count, "flame_temp_k")
        specifics = _per_lane(r_specific, count, "r_specific")
        gammas = _per_lane(gamma, count, "gamma")
        starts = _per_lane(initial_temperature_k, count, "initial_temperature_k")
        liners = _per_lane(liner_thickness_factor, count, "liner_thickness_factor")

        lanes: List[Dict[str, Any]] = []
        features: List[FrozenSet[str]] = []
        parts: List[Dict[str, Any]] = []
        for i in range(count):
            gamma_i = float(_resolve_gamma(curves[i], gammas[i]))
            lane = dict(geometry=geometries[i], curve=curves[i], casing_material=casings[i], nozzle_material=nozzles[i],
                        flame_temp_k=flames[i], r_specific=specifics[i], gamma=gamma_i,
                        initial_temperature_k=float(starts[i]), liner_thickness_factor=liners[i])
            part, lane_features = _pack_lane(lane)
            lanes.append(lane)
            features.append(lane_features)
            parts.append(part)

        nodes = max((p["n_nodes"] for p in parts), default=4)
        steps = max(max((p["n_intervals"] for p in parts), default=0), 1)
        arrays = {
            "lower": np.zeros((count, nodes - 1)), "diag": np.zeros((count, nodes)), "upper": np.zeros((count, nodes - 1)),
            "source": np.zeros((count, nodes)), "y0": np.zeros((count, nodes)),
            "dt": np.ones((count, steps)), "bartz": np.zeros((count, steps)),
        }
        for name in LANE_ARRAYS:
            if name not in arrays:
                arrays[name] = np.asarray([p[name] for p in parts])
        for i, p in enumerate(parts):
            n, m = p["n_nodes"], p["n_intervals"]
            arrays["lower"][i, : n - 1] = p["lower"]
            arrays["diag"][i, :n] = p["diag"]
            arrays["upper"][i, : n - 1] = p["upper"]
            arrays["source"][i, :n] = p["source"]
            arrays["y0"][i, :n] = p["y0"]
            arrays["dt"][i, :m] = p["dt"]
            arrays["bartz"][i, :m] = p["bartz"]
        return cls(arrays, lanes, features)


def _pack_lane(lane: Dict[str, Any]):
    """The arrays of one lane and its features; a lane the batched solve cannot take gets harmless placeholder arrays."""
    geometry, curve = lane["geometry"], lane["curve"]
    casing, nozzle = lane["casing_material"], lane["nozzle_material"]
    flame, gamma = float(lane["flame_temp_k"]), float(lane["gamma"])
    start = float(lane["initial_temperature_k"])
    time_s = np.asarray(curve["time_s"], dtype=float)
    thrust = np.asarray(curve["thrust_n"], dtype=float)
    mass_flow = _series(curve, "mass_nozzle_kg_s", curve["mass_flow_kg_s"])
    pressure = _series(curve, "chamber_pressure_pa", np.zeros_like(time_s))
    throat = _series(curve, "throat_diameter_m", np.full_like(time_s, geometry.throat_diameter_m))
    ablation = _series(curve, "throat_ablation_m", np.full_like(time_s, np.nan))

    features = set()
    series = (time_s, thrust, mass_flow, pressure, throat, ablation)
    if any(s.ndim != 1 or s.shape != time_s.shape for s in series):
        features.add(SERIES_MISMATCH)
    else:
        if not _finite(time_s, thrust, mass_flow, pressure, throat) or np.any(np.isinf(ablation)):
            features.add(NON_FINITE_INPUT)
    material = (casing.density_kg_m3, casing.heat_capacity_j_kgk, casing.thermal_conductivity_w_mk, casing.liner_thickness_m,
                casing.liner_k_w_mk, casing.liner_density_kg_m3, casing.liner_cp_j_kgk, geometry.casing_wall_thickness_m,
                geometry.throat_diameter_m, nozzle.ablation_rate_scale, nozzle.ablation_pressure_exponent,
                nozzle.ablation_mass_flux_exponent, lane["r_specific"], lane["liner_thickness_factor"])
    if not _finite(flame, gamma, start, *material):
        features.add(NON_FINITE_INPUT)

    if features:  # harmless placeholders: the router never sends this lane to a batched solve
        dx, k, rho_cp = np.full(4, 1e-3), np.full(4, 16.0), np.full(4, 7850.0 * 520.0)
        inner = interface = 0
        liner_m, wall_m = 0.0, 4e-3
        start, flame, gamma = 298.15, 1600.0, 1.2
        dt, bartz, recovery, ablated = np.ones(max(len(time_s) - 1, 0)), np.zeros(max(len(time_s) - 1, 0)), flame, 0.0
        burn_duration = 1e-6
    else:
        dx, k, rho_cp, inner, interface, liner_m, wall_m = _wall_layers(geometry, casing, lane["liner_thickness_factor"])
        dt, bartz, recovery, ablated = interval_coefficients(
            time_s, thrust, mass_flow, pressure, throat, ablation, flame, float(lane["r_specific"]), gamma,
            nozzle.ablation_rate_scale, nozzle.ablation_pressure_exponent, nozzle.ablation_mass_flux_exponent,
        )
        burn_duration = max(float(time_s[-1] - time_s[0]), 1e-6) if len(time_s) else 1e-6
    jacobian, source, heat_flux_source = _build_wall_conduction_operator(dx, k, rho_cp, OUTER_WALL_H_W_M2K, start)
    nodes = len(dx)
    intervals = len(dt)
    part = {
        "lower": jacobian.diagonal(-1), "diag": jacobian.diagonal(0), "upper": jacobian.diagonal(1), "source": source, "y0": np.full(nodes, max(start, 150.0)),
        "e0": float(heat_flux_source[0]), "dt": dt, "bartz": bartz,
        "n_nodes": nodes, "n_intervals": intervals, "inner_node": inner, "interface_node": interface,
        "has_liner": liner_m > 1e-4, "thickness_m": liner_m + wall_m, "flame_temp_k": flame,
        "stagnation": (gamma + 1.0) / 2.0, "recovery_temp_k": recovery, "throat_ablation_m": ablated,
        "max_recovery_temp_k": max(start, recovery) if intervals else start, "burn_duration_s": burn_duration,
        "wall_thickness_m": wall_m, "initial_temp_k": start,
    }
    return part, frozenset(features)


def assemble(batch: ThermalBatch, out: Dict[str, np.ndarray]) -> List[Optional[Dict[str, float]]]:
    """The metrics mappings of ``simulate_thermal_ablation`` from the extremes ``solve_thermal`` returned.

    A lane whose integration failed or whose extremes are not finite gives ``None``: the router reruns it on the scalar code.
    """
    a = batch.arrays
    results: List[Optional[Dict[str, float]]] = []
    for lane, spec in enumerate(batch.lanes):
        values = {name: float(out[name][lane]) for name in
                  ("max_heat_flux", "heat", "max_inner", "max_outer", "max_hot", "max_interface", "max_gradient")}
        if out["failed"][lane] or not all(math.isfinite(v) for v in values.values()):
            results.append(None)
            continue
        liner = bool(a["has_liner"][lane])
        results.append(_thermal_report(
            spec["geometry"], spec["casing_material"],
            burn_duration_s=float(a["burn_duration_s"][lane]),
            throat_ablation_m=float(a["throat_ablation_m"][lane]),
            max_heat_flux_w_m2=values["max_heat_flux"],
            max_wall_gradient_k_m=values["max_gradient"],
            max_outer_wall_k=values["max_outer"],
            max_inner_wall_k=values["max_inner"],
            max_hot_face_k=values["max_hot"] if liner else values["max_inner"],
            max_interface_k=values["max_interface"] if liner else values["max_inner"],
            integrated_heat_j_m2=values["heat"],
            max_recovery_temp_k=float(a["max_recovery_temp_k"][lane]),
            wall_thickness_m=float(a["wall_thickness_m"][lane]),
            node_count=int(a["n_nodes"][lane]),
        ))
    return results
