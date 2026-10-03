# -*- coding: utf-8 -*-
"""Canonical result mappings from the outputs of the batched solver.

Each lane gets the mapping ``BurnSimulation.result`` holds: ``history``, ``metrics``, ``status``,
``efficiencies`` and ``provenance``. The metrics are the scalar ones (``Burn._build_result``), computed from
the running reductions and the final state. ``provenance`` keeps every scalar key and adds an ``execution``
block; in particular ``physics_provider_hash`` has the reference definition (resolved inputs plus the bytes of
``Burn.py``, ``Grain.py`` and ``Propellant.py``), so ``evaluate_numerical_acceptance`` can compare a batched
result with a reference one. The hash of the kernel and integrator sources is in ``execution``.
"""

from __future__ import annotations

import copy
import functools
import hashlib
import importlib
import json
import platform
import subprocess
import threading
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

from ..backends import _tolerances
from . import problem as pb
from .kernels import geometry, rhs

# ``solidpy.Burn`` is the class once the package is imported, so the module is looked up by name
_burn_module = importlib.import_module("solidpy.Burn")
SCALAR_SOURCES = ("Burn.py", "Grain.py", "Propellant.py")
HISTORY_CHANNELS = (
    "time_s", "chamber_pressure_pa", "free_volume_m3", "regression_m", "regression_rate_m_s", "burn_area_m2",
    "burn_area_grains_m2", "mdot_generated_kg_s", "mdot_generated_grains_kg_s", "mdot_igniter_kg_s",
    "mdot_nozzle_kg_s", "gas_mass_kg", "gas_temperature_k", "thrust_n", "momentum_ideal_n", "momentum_n",
    "pressure_n", "exit_pressure_pa", "exit_velocity_m_s", "generated_mass_integral_kg",
    "igniter_mass_integral_kg", "nozzle_mass_integral_kg", "impulse_integral_ns", "pressure_throat_integral_ns",
)


_GIT_TTL_S = 5.0
_git_cache: Dict[str, Any] = {"stamp": None, "at": -1e9, "value": None}
_git_lock = threading.Lock()


def _source_stamp(paths) -> tuple:
    stats = [(p, p.stat()) for p in paths]
    return tuple((str(p), st.st_mtime_ns, st.st_size) for p, st in stats)


def _git_stamp() -> tuple:
    """Changes when HEAD moves: the modification times of ``.git/HEAD`` and its reflog."""
    root = Path(_burn_module.__file__).resolve().parent.parent
    stamp = []
    for name in (".git/HEAD", ".git/logs/HEAD"):
        try:
            stamp.append((name, (root / name).stat().st_mtime_ns))
        except OSError:
            stamp.append((name, None))
    return tuple(stamp)


@functools.lru_cache(maxsize=4)
def _kernel_hash(stamp: tuple) -> str:
    digest = hashlib.sha256()
    for name, _, _ in stamp:
        path = Path(name)
        digest.update(path.name.encode())
        digest.update(path.read_bytes().replace(b"\r\n", b"\n"))
    return digest.hexdigest()


def kernel_source_hash() -> str:
    """SHA-256 of the kernel and integrator sources that produce a batched result (cached until a file changes)."""
    root = Path(__file__).parent
    return _kernel_hash(_source_stamp(sorted(list((root / "kernels").glob("*.py")) + list((root / "integrators").glob("*.py")))))


def _git_sha() -> Optional[str]:
    """The checkout's commit, found the way ``Burn._build_result`` finds it.

    It is read again as soon as ``.git/HEAD`` or its reflog changes, and at most every 5 s otherwise (a worktree,
    where those files are elsewhere, relies on the time-to-live alone).
    """
    stamp, now = _git_stamp(), time.monotonic()
    with _git_lock:
        if stamp != _git_cache["stamp"] or now - _git_cache["at"] > _GIT_TTL_S:
            _git_cache.update(stamp=stamp, at=now, value=_read_git_sha())
        return _git_cache["value"]


def _read_git_sha() -> Optional[str]:
    try:
        checkout_root = Path(_burn_module.__file__).resolve().parent.parent
        top = subprocess.run(["git", "rev-parse", "--show-toplevel"], cwd=Path(_burn_module.__file__).parent,
                             capture_output=True, text=True, timeout=1, check=True).stdout.strip()
        if Path(top).resolve() != checkout_root:
            return None
        return subprocess.run(["git", "rev-parse", "HEAD"], cwd=checkout_root, capture_output=True, text=True,
                              timeout=1, check=True).stdout.strip()
    except (OSError, subprocess.SubprocessError):
        return None


def _source_description(name: str, source, igniter_burn_time: float):
    if callable(source):
        return {"type": "callable", "identity": getattr(source, "__qualname__", type(source).__qualname__),
                "module": getattr(source, "__module__", None),
                "support_s": igniter_burn_time if name == "igniter_mass_flow" else None}
    if source is None:
        return None
    if np.isscalar(source):
        return float(source)
    return np.asarray(source).tolist()


def resolved_inputs(motor, propellant, environment, settings, row) -> Dict[str, Any]:
    """The ``provenance["resolved_inputs"]`` mapping of ``Burn._build_result`` for one lane.

    ``row`` holds the packed scalars of the lane (``initial_gas_temperature_k`` and the resolved igniter
    temperature come from there so they are the numbers the solver used).
    """
    resolved = {
        "combustion_temperature_k": float(propellant.combustion_temperature),
        "gas_constant_j_kg_k": float(propellant.products_constant),
        "propellant_density_kg_m3": float(propellant.density),
        "specific_heat_ratio": float(propellant.specific_heat_ratio),
        "initial_gas_temperature_k": float(row["source_temperature"]),
        "igniter_temperature_k": float(row["igniter_temperature"]),
        "connected_chamber_volume_m3": motor.chamber_volume,
        "physical_chamber_length_m": motor.chamber_length,
        "grain_geometry_models": [g.geometry_model for g in motor.grains],
        "grain_geometry": [
            {"outer_radius_m": g.outer_radius, "inner_radius_m": g.initial_inner_radius, "height_m": g.initial_height,
             "ends_inhibited": g.ends_burn, "geometry": g.geometry, "n_points": g.n_points, "epsilon_rad": g.epsilon,
             "slot_fraction": g.slot_fraction}
            for g in motor.grains
        ],
        "nozzle_throat_area_m2": motor.nozzle_throat_area,
        "nozzle_exit_area_m2": motor.nozzle_exit_area,
        "nozzle_angle_rad": motor.nozzle_angle,
        "ambient_pressure_pa": environment.atmospheric_pressure,
        "burn_rate_a": getattr(propellant, "burn_rate_a", None),
        "burn_rate_n": getattr(propellant, "burn_rate_n", None),
        "erosive_burning_coefficient": getattr(propellant, "erosive_burning_coefficient", 0.0),
        "erosive_alpha": getattr(propellant, "erosive_alpha", 35.0),
        "efficiencies": {"eta_c": float(row["eta_c"]), "eta_Cf": float(row["eta_cf"]),
                         "discharge_coefficient": float(row["discharge_coefficient"])},
        "igniter_burn_time_s": settings["igniter_burn_time"],
        "ignition_ramp_time_s": settings["ignition_ramp_time"],
    }
    interpolator = propellant._burn_rate_interpolator
    if interpolator is not None:
        resolved["burn_rate_pressure_table_mpa"] = interpolator.x.tolist()
        resolved["burn_rate_table_mm_s"] = interpolator.y.tolist()
    for name in ("igniter_mass_flow", "burn_area_activation"):
        resolved[name] = _source_description(name, settings[name], settings["igniter_burn_time"])
    return resolved


@functools.lru_cache(maxsize=4)
def _source_bytes(stamp: tuple) -> bytes:
    return b"".join(Path(name).read_bytes() for name, _, _ in stamp)


def _scalar_source_bytes() -> bytes:
    """The bytes of the three reference files, in hash order (read again only when one of them changes)."""
    root = Path(_burn_module.__file__).parent
    return _source_bytes(_source_stamp([root / name for name in SCALAR_SOURCES]))


def _physics_hash(resolved: Dict[str, Any], sources: bytes) -> str:
    digest = hashlib.sha256()
    digest.update(json.dumps(resolved, sort_keys=True, default=float).encode())
    digest.update(sources)
    return digest.hexdigest()


def termination_reasons(out, tail_off) -> List[str]:
    """The scalar ``termination_reason`` of every lane (before the thermochemistry override)."""
    reasons = []
    for lane in range(len(out["t"])):
        if not out["burn_ok"][lane]:
            reason = "solver_failure"
        elif not out["burned_out"][lane]:
            reason = "burn_timeout"
        elif not tail_off[lane]:
            reason = "tail_off_omitted"
        elif not out["tail_ok"][lane]:
            reason = "solver_failure"
        else:
            reason = "completed" if out["reached_cutoff"][lane] else "blowdown_timeout"
        reasons.append(reason)
    return reasons


def _history(batch, out, lane, namespace) -> Dict[str, Any]:
    """The stored accepted points of one lane as the canonical ``history`` mapping.

    ``namespace`` is ``batch.namespace(np)``, built once by the caller for all lanes.
    """
    G = batch.g_max
    n_grains = int(batch.n_grains[lane])
    count = int(out["n_points"][lane])
    a = batch.arrays
    # one lane, one time axis: lane arrays [1, 1], grain arrays [1, 1, G]
    view = {name: array[lane : lane + 1, None] for name, array in namespace.items()}
    y = out["hy"][lane, :count][None]
    q = rhs.state_quantities(np, y, None, view, detail=True, time=out["ht"][lane, :count][None])
    q = {key: value[0] for key, value in q.items()}
    offset = 2 + G
    grains = slice(0, n_grains)
    return {
        "time_s": out["ht"][lane, :count].copy(),
        "chamber_pressure_pa": q["pressure"],
        "free_volume_m3": q["volume"],
        "regression_m": y[0, :, 2 : 2 + n_grains],
        "regression_rate_m_s": q["regression_rates"][:, grains],
        "burn_area_m2": np.sum(q["areas"][:, grains], axis=1),
        "burn_area_grains_m2": q["areas"][:, grains],
        "mdot_generated_kg_s": q["generated"],
        "mdot_generated_grains_kg_s": q["generated_grains"][:, grains],
        "mdot_igniter_kg_s": q["igniter"],
        "mdot_nozzle_kg_s": q["nozzle"],
        "gas_mass_kg": y[0, :, 0],
        "gas_temperature_k": q["temperature"],
        "thrust_n": q["thrust"],
        "momentum_ideal_n": q["momentum_ideal"],
        "momentum_n": q["momentum"],
        "pressure_n": q["pressure_thrust"],
        "exit_pressure_pa": q["exit_pressure"],
        "exit_velocity_m_s": q["exit_velocity"],
        "generated_mass_integral_kg": y[0, :, offset],
        "igniter_mass_integral_kg": y[0, :, offset + 1],
        "nozzle_mass_integral_kg": y[0, :, offset + 2],
        "impulse_integral_ns": y[0, :, offset + 3] * a["eta_cf"][lane],
        "pressure_throat_integral_ns": y[0, :, offset + 4],
    }


def assemble(batch, out, history: str = "metrics", execution: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
    """One canonical result mapping per lane from the solver outputs ``out``.

    ``history`` is ``"metrics"`` (``result["history"]`` is ``None``) or ``"full"`` (the solver must have been run
    with ``keep_history``). ``execution`` is the backend's part of ``provenance["execution"]``.
    """
    a = batch.arrays
    G = batch.g_max
    offset = 2 + G
    y, y0 = out["y"], out["y0"]
    P = batch.namespace(np)
    tail_off = a["tail_off_evaluation"] > 0.5

    solid_remaining = geometry.ordered_sum(np, geometry.remaining_volume(np, y[:, 2 : 2 + G], P))
    consumed = a["density"] * (a["propellant_volume"] - solid_remaining)
    generated, igniter, nozzle = y[:, offset], y[:, offset + 1], y[:, offset + 2]
    residual = consumed + igniter - nozzle - (y[:, 0] - y0[:, 0])
    reasons = termination_reasons(out, tail_off)
    git_sha = _git_sha()
    kernel_hash = kernel_source_hash()
    sources = _scalar_source_bytes()
    execution = dict(execution or {})
    results = []
    for lane in range(len(batch)):
        n_grains = int(batch.n_grains[lane])
        burnout = [None if np.isnan(t) else float(t) for t in out["burn_t"][lane, :n_grains]]
        all_burned = all(t is not None for t in burnout)
        burn_start, burn_end = float(out["gen_start"][lane]), float(out["gen_end"][lane])
        if all_burned:
            burn_end = max(burnout)
        noz_start, noz_end = float(out["noz_start"][lane]), float(out["noz_end"][lane])
        burn_duration, nozzle_duration = burn_end - burn_start, noz_end - noz_start
        g, n, c = float(generated[lane]), float(nozzle[lane]), float(consumed[lane])
        metrics = {
            "propellant_mass_consumed_kg": c, "integrated_generated_mass_kg": g, "generated_mass_integral_kg": g,
            "propellant_mass_initial_kg": float(a["propellant_volume"][lane] * a["density"][lane]),
            "propellant_mass_remaining_kg": float(solid_remaining[lane] * a["density"][lane]),
            "propellant_burn_start_s": burn_start, "propellant_burn_end_s": burn_end,
            "propellant_burn_duration_s": burn_duration,
            "nozzle_flow_start_s": noz_start, "nozzle_flow_end_s": noz_end, "nozzle_flow_duration_s": nozzle_duration,
            "mass_flow_avg_generated_kg_s": g / burn_duration if burn_duration > 0 else 0.0,
            "max_generated_mass_flow_kg_s": float(out["gmax"][lane]),
            "mass_flow_avg_nozzle_kg_s": n / nozzle_duration if nozzle_duration > 0 else 0.0,
            "max_nozzle_mass_flow_kg_s": float(out["nmax"][lane]),
            "nozzle_mass_integral_kg": n, "igniter_mass_injected_kg": float(igniter[lane]),
            "gas_mass_initial_kg": float(y0[lane, 0]), "gas_mass_cutoff_kg": float(y[lane, 0]),
            "other_declared_outflows_kg": 0.0, "mass_balance_residual_kg": float(residual[lane]),
            "mass_flow_balance_error_pct": float(100 * abs(residual[lane]) / max(c + float(igniter[lane]), 1e-15)),
            "total_impulse_ns": float(y[lane, offset + 3] * a["eta_cf"][lane]),
            "peak_chamber_pressure_pa": float(out["pmax"][lane]), "peak_thrust_n": float(out["tmax"][lane]),
            "pressure_throat_integral_ns": float(y[lane, offset + 4]),
            "grain_burnout_times_s": tuple(burnout),
        }
        reason = reasons[lane]
        scalar = pb.THERMO_SCALAR in batch.lane_features[lane]
        numerical_completed = reason == "completed"
        if not scalar:
            reason = "unsupported_thermochemistry"
        # the scalar code sets the cutoff once the source-only stage has succeeded, even if the blowdown then fails
        stage_two = bool(out["burned_out"][lane]) and bool(tail_off[lane]) and bool(out["source_ok"][lane])
        row = {name: array[lane] for name, array in a.items() if array.ndim == 1}
        motor, propellant, environment = batch.motors[lane], batch.propellants[lane], batch.environments[lane]
        settings = batch.settings[lane]
        resolved = resolved_inputs(motor, propellant, environment, settings, row)
        cea_used = propellant.cea_formulation is not None or propellant._cea_obj is not None
        status = {
            "completed": numerical_completed and scalar, "termination_reason": reason,
            "numerical_blowdown_completed": numerical_completed, "scalar_contract_supported": scalar,
            "burnout_completed": all_burned,
            "blowdown_cutoff_pressure_pa": float(out["cutoff"][lane]) if stage_two else None,
            "blowdown_reference_peak_pressure_pa": float(out["peak_burn"][lane]) if stage_two else None,
        }
        applied = {"eta_c": float(a["eta_c"][lane]), "eta_Cf": float(a["eta_cf"][lane]),
                   "discharge_coefficient": float(a["discharge_coefficient"][lane])}
        provenance = {
            "eta_c_applied": applied["eta_c"], "eta_cf_applied": applied["eta_Cf"],
            "discharge_coefficient_applied": applied["discharge_coefficient"],
            "efficiency_semantics": "native_split", "cea_used": cea_used,
            "thermochemistry_source": "cea_legacy" if cea_used else ("scalar" if scalar else "pressure_table_legacy"),
            "physics_provider_hash": _physics_hash(resolved, sources), "solidpy_git_sha": git_sha,
            "solidpy_git_sha_status": "available" if git_sha else "unavailable", "resolved_inputs": resolved,
            "gas_temperature_model": "prescribed_source_temperature_mixing_v1",
            "activation_model": "uniform_front_rate_scaling_v1",
            "flow_interval_method": "adaptive_positive_source_bracket",
            "integration_method": "adaptive_ode_quadrature",
            "solver_settings": {"method": "DOP853", "rtol": settings["rtol"], "atol": settings["atol"],
                                "max_step_size_s": settings["max_step_size"],
                                "burn_timeout_s": settings["burn_timeout_s"],
                                "tail_off_timeout_s": settings["tail_off_timeout_s"],
                                "tail_off_method": settings["tail_off_method"]},
            "execution": {
                **copy.deepcopy(execution),  # nested mappings (library versions) must not be shared between lanes
                "integrator": {"name": "dop853_batched", "rtol": settings["rtol"], "atol": settings["atol"],
                               "max_step_s": settings["max_step_size"]},
                "history": history, "fallback": None, "kernel_source_hash": kernel_hash,
                "tolerances_version": _tolerances.TOLERANCES_VERSION,
                "step_overflow": bool(out["overflow"][lane]),
            },
        }
        results.append({
            "history": _history(batch, out, lane, P) if history == "full" else None,
            "metrics": metrics, "status": status,
            "efficiencies": {**applied, "efficiency_semantics": "native_split"}, "provenance": provenance,
        })
    return results


def library_versions() -> Dict[str, str]:
    import numpy
    import scipy

    return {"python": platform.python_version(), "numpy": numpy.__version__, "scipy": scipy.__version__}
