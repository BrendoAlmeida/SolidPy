# -*- coding: utf-8 -*-
"""High-level entry point for simulating many motors through a chosen backend.

    from solidpy.ensemble import ProblemBatch, simulate_burn

    batch = ProblemBatch.from_objects(motors, propellants, environments, settings)
    results = simulate_burn(batch, backend="cpu-vectorized").to_results()

``results[i]`` is always the canonical result of lane ``i``. A lane the chosen backend cannot run falls back
to the scalar reference (or raises ``UnsupportedLane`` with ``strict=True``), and the result says so in
``provenance["execution"]["fallback"]``. The reference path itself is untouched: ``BurnSimulation`` still
works exactly as before.
"""

from __future__ import annotations

import copy
import numbers
import time
from typing import Any, Dict, List, Optional

import numpy as np

from . import backends
from .backends import SolveOptions, UnsupportedLane
from .backends._protocol import refused_lanes, unsupported_lane_error
from .batch import ProblemBatch
from .batch.kernels.tables import evaluate as evaluate_table
from .batch.result import BatchResult

__all__ = ["ProblemBatch", "SolveOptions", "UnsupportedLane", "lane_cost", "run_robustness_ensemble", "simulate_burn"]

#: Backends ``backend="auto"`` may pick, best first, and the smallest batch worth sending to one.
AUTO_ACCELERATORS = ("jax",)
AUTO_MIN_LANES = 2048


def lane_cost(batch: ProblemBatch) -> np.ndarray:
    """A cheap estimate of the work of each lane, used only to group lanes of similar cost.

    Lockstep batches run as many iterations as their slowest lane, so lanes with a similar number of steps
    belong together. The estimate is the burn time (deepest grain over the equilibrium burn rate) in units of
    the largest step, plus a term per grain for the restart at each burnout. A tabulated rate is read at 3 MPa.
    Lanes without a usable burn rate (NaN) sort last.
    """
    a = batch.arrays
    with np.errstate(all="ignore"):  # lanes the kernels cannot describe have NaN inputs
        cstar = np.sqrt(a["gas_constant"] * a["source_temperature"] / a["gamma"]) * (
            (a["gamma"] + 1.0) / 2.0
        ) ** ((a["gamma"] + 1.0) / (2.0 * (a["gamma"] - 1.0)))
        burning_area = np.where(a["grain_valid"], np.pi * (a["outer_radius"] ** 2 - a["inner_radius0"] ** 2), 0.0).sum(axis=1)
        kn = np.maximum(burning_area, 1e-12) / a["throat_area"]
        n = a["burn_rate_n"]
        pressure = (a["density"] * a["burn_rate_a"] * (1e-6) ** n * kn * cstar / (1000.0 * a["discharge_coefficient"])) ** (
            1.0 / (1.0 - n)
        )
        rate = a["burn_rate_a"] * np.maximum(pressure * 1e-6, 1e-12) ** n / 1000.0
        tabulated = evaluate_table(np, np.full(len(batch), 3.0), a["rate_table_x"],
                                   tuple(a[f"rate_table_c{i}"] for i in range(4)), a["rate_table_n"],
                                   a["rate_table_below"], a["rate_table_above"]) / 1000.0  # at a typical 3 MPa
        rate = np.where(a["burn_rate_mode"] == 1.0, tabulated, rate) * a["burn_rate_factor"]
        depth = np.where(a["grain_valid"], a["burnout_depth"], 0.0).max(axis=1)
        cost = depth / rate / a["max_step_size"] + 40.0 * a["n_valid_grains"]
    return np.where(np.isfinite(cost), cost, np.finfo(float).max)


def _auto_backend(batch: ProblemBatch, device: Optional[str] = None) -> str:
    """An accelerator when one is usable and enough lanes can run on it to amortise it, else the reference.

    An explicit ``device`` restricts the choice to a backend that lists it, and ``"cpu"`` never selects an accelerator.
    """
    if len(batch) < AUTO_MIN_LANES or device in ("cpu", "cpu:0"):
        return "cpu-reference"
    status = backends.available()
    for name in AUTO_ACCELERATORS:
        if status.get(name) != "ok":
            continue
        try:
            accelerator = backends.get_backend(name)
            devices = accelerator.devices()
        except ImportError:
            continue
        usable = (device in devices) if device is not None else any(not d.startswith("cpu") for d in devices)
        if usable and len(batch) - len(refused_lanes(batch, accelerator.capabilities())) >= AUTO_MIN_LANES:
            return name
    return "cpu-reference"


def _chunks(order: np.ndarray, chunk_size: Optional[int]) -> List[np.ndarray]:
    if chunk_size is None or chunk_size >= len(order):
        return [order]
    return [order[i : i + chunk_size] for i in range(0, len(order), chunk_size)]


def _execution(result: Dict[str, Any]) -> Dict[str, Any]:
    """``result["provenance"]["execution"]``, created when a third-party backend leaves it out."""
    return result.setdefault("provenance", {}).setdefault("execution", {})


def simulate_burn(
    batch: ProblemBatch,
    backend: Optional[str] = None,
    device: Optional[str] = None,
    *,
    history: str = "metrics",
    strict: bool = False,
    workers: Optional[int] = None,
    max_steps: Optional[int] = None,
    tiers: Optional[tuple] = None,
    chunk_size: Optional[int] = None,
    sort: bool = True,
) -> BatchResult:
    """Simulate every lane of ``batch`` and return their canonical results in lane order.

    ``backend`` is a backend name, ``"auto"``, or ``None`` for the one selected with ``set_backend``,
    ``use_backend`` or the environment (default ``"cpu-reference"``). ``history`` is ``"metrics"`` or ``"full"``.
    ``chunk_size`` bounds how many lanes one solve holds (memory); ``sort`` groups lanes of similar estimated
    cost into the same chunk, which matters when lanes need very different numbers of steps (and is skipped when
    one chunk holds every lane). A lane that exhausts the step budget of a batched backend is rerun on the
    reference and flagged ``step_overflow`` in ``provenance["execution"]["fallback"]``. ``tiers`` are the
    iteration caps a batched backend runs before its uncapped tier (``None``: ``batch.tiers.DEFAULT_TIERS``,
    ``()``: one uncapped solve); see ``solidpy.batch.tiers``.
    """
    if not isinstance(batch, ProblemBatch):
        raise TypeError("simulate_burn needs a ProblemBatch; build one with ProblemBatch.from_objects")
    if chunk_size is not None and (
        isinstance(chunk_size, bool) or not isinstance(chunk_size, numbers.Integral) or chunk_size < 1
    ):
        raise ValueError(f"chunk_size must be a positive integer or None, got {chunk_size!r}")
    chunk_size = None if chunk_size is None else int(chunk_size)
    requested = backend
    if backend == "auto":
        backend = _auto_backend(batch, device)
        if device is not None and backend == "cpu-reference":
            device = None  # "auto" chose the reference, which has no other device
    if backend is None:
        backend, selected_device = backends.current_backend()
        requested = backend
        device = device if device is not None else selected_device
    chosen = backends.get_backend(backend, device)
    options = SolveOptions(history=history, workers=workers, max_steps=max_steps, tiers=tiers)

    refused = refused_lanes(batch, chosen.capabilities())
    if refused and strict:
        raise unsupported_lane_error(backend, refused)

    results: List[Optional[Dict[str, Any]]] = [None] * len(batch)
    supported = np.asarray([lane for lane in range(len(batch)) if lane not in refused], dtype=int)
    chunks: List[np.ndarray] = []
    tier_log: List[Any] = []
    if len(supported):
        # sorting matters when lanes are split into several launches: by chunk_size, or by the backend itself
        launch_limit = getattr(chosen, "max_lanes", None)
        several = any(limit is not None and limit < len(supported) for limit in (chunk_size, launch_limit))
        order = supported[np.argsort(lane_cost(batch.select(supported)), kind="stable")] if sort and several else supported
        chunks = _chunks(order, chunk_size)
        for chunk in chunks:
            outcome = chosen.solve_burn(batch.select(chunk), options)
            tier_log.append(outcome.execution.get("tiers"))
            returned = outcome.to_results()
            if len(returned) != len(chunk):
                raise RuntimeError(f"backend {backend!r} returned {len(returned)} results for {len(chunk)} lanes")
            for lane, result in zip(chunk, returned):
                results[int(lane)] = result
    reasons: Dict[int, List[str]] = {lane: list(features) for lane, features in refused.items()}
    if backend != "cpu-reference":
        # a lane that ran out of its step budget is a failure of the budget, not of the physics: rerun it on the reference
        for lane, result in enumerate(results):
            if result is not None and _execution(result).get("step_overflow"):
                reasons[lane] = ["step_overflow"]
    if reasons:
        reference = backends.get_backend("cpu-reference")
        lanes = np.asarray(sorted(reasons), dtype=int)
        solved = reference.solve_burn(batch.select(lanes), options).to_results()
        for lane, result in zip(lanes, solved):
            execution = dict(reference.provenance())
            execution["fallback"] = {"lane_reason": reasons[int(lane)], "ran_on": "cpu-reference"}
            result.setdefault("provenance", {})["execution"] = execution
            results[int(lane)] = result
    if backend != "cpu-reference":
        for result in results:
            _execution(result)["requested_backend"] = requested

    summary = {"requested_backend": requested, "effective_backend": backend, "lanes": len(batch),
               "fallback_lanes": sorted(int(i) for i in reasons), "chunks": len(chunks), "tiers": tier_log}
    return BatchResult(results, backend, summary)


#: What a lane keeps with ``keep_series=False``: the scalar outputs and the identification of the scenario.
_SCALAR_KEYS = ("summary", "status", "provenance", "schema_version", "gamma", "scenario_id", "scenario_kind",
                "scenario_factors", "valid")

#: Arguments of ``run_detailed_ballistics`` that shape its post-processing and not the burn.
_DETAIL_ARGUMENTS = ("resample_step", "nozzle_ablation_scale", "ablation_pressure_exponent", "ablation_mass_flow_exponent")


def _robustness_design(entry):
    entry = tuple(entry)
    if not 3 <= len(entry) <= 5:
        raise ValueError("a design is (grain, motor, propellant[, environment[, simulation_kwargs]])")
    grain, motor, propellant = entry[:3]
    environment = entry[3] if len(entry) > 3 else None
    own = dict(entry[4]) if len(entry) > 4 and entry[4] else {}
    return grain, motor, propellant, environment, own


def run_robustness_ensemble(
    designs,
    *,
    scenarios=None,
    monte_carlo_sample_count: int = 0,
    monte_carlo_seed: int = 20260504,
    max_step_size: float = 0.01,
    max_time_points: Optional[int] = 1000,
    validator=None,
    backend: Optional[str] = "auto",
    device: Optional[str] = None,
    strict: bool = False,
    workers: Optional[int] = None,
    max_steps: Optional[int] = None,
    chunk_lanes: int = 4096,
    timings: Optional[Dict[str, float]] = None,
    keep_series: bool = True,
    **simulation_kwargs: Any,
) -> List[Dict[str, Any]]:
    """Robustness analysis of many designs: every (design, scenario) pair is a lane of one batch.

    Returns one report per design, in the form ``run_robustness_analysis`` returns for it. ``designs`` holds
    ``(grain, motor, propellant[, environment[, simulation_kwargs]])`` tuples; ``simulation_kwargs`` given here
    apply to every design and the ones in a design's tuple override them (that includes ``max_step_size`` and
    ``max_time_points``). The scenarios are the same for every
    design (the default ones, or ``scenarios``, plus ``monte_carlo_sample_count`` Latin-hypercube samples).

    The burns run on ``backend`` (a name, ``"auto"``, or ``None`` for the selected one; see ``simulate_burn``) with
    the full history; the detailed ballistics of every lane is then built on the CPU from the lane's result, so the
    reports are what the scalar path gives within the numerical tolerances of ``solidpy.backends._tolerances``.
    ``chunk_lanes`` bounds how many lanes are solved and held at once. Invalid inputs raise when the batch is packed,
    before any burn is solved; the scalar path raises as it reaches them. A dict passed as ``timings`` receives the
    seconds spent packing (``pack_s``), solving (``solve_s``), building the detailed ballistics (``postprocess_s``) and
    assembling the reports (``report_s``).

    Every lane of a report holds its full series and canonical history, about 200 kB each for a four-grain design, as
    the scalar path's does.
    For thousands of lanes pass ``keep_series=False``: each lane then keeps only ``summary``, ``status``, ``provenance``
    and the scenario fields (and ``valid``), and the report, its statistics and its validity ratio are unchanged.
    """
    from .batch.simulation_view import SimulationView
    from .DetailedBallistics import _validate_dry_hardware, build_detailed_ballistics
    from .Environment import Environment
    from .Robustness import (
        _build_report, _rescale_thrust, _scenario_objects, build_latin_hypercube_scenarios, default_robustness_scenarios,
    )

    if isinstance(chunk_lanes, bool) or not isinstance(chunk_lanes, numbers.Integral) or chunk_lanes < 1:
        raise ValueError(f"chunk_lanes must be a positive integer, got {chunk_lanes!r}")
    parsed = [_robustness_design(entry) for entry in designs]
    scenario_list = list(scenarios) if scenarios is not None else default_robustness_scenarios()
    scenario_list.extend(build_latin_hypercube_scenarios(sample_count=monte_carlo_sample_count, seed=monte_carlo_seed))
    lanes_per_design = 1 + len(scenario_list)
    for _, motor, _, _, _ in parsed:
        _validate_dry_hardware(motor)

    clock = {"pack_s": 0.0, "solve_s": 0.0, "postprocess_s": 0.0, "report_s": 0.0}
    reports: List[Dict[str, Any]] = []
    designs_per_chunk = max(1, int(chunk_lanes) // lanes_per_design)
    for start in range(0, len(parsed), designs_per_chunk):
        chunk = parsed[start : start + designs_per_chunk]
        mark = time.perf_counter()
        motors, propellants, environments, settings, factors, details = [], [], [], [], [], []
        for grain, motor, propellant, environment, own in chunk:
            merged = {**simulation_kwargs, **own}
            step = merged.pop("max_step_size", max_step_size)
            detail = {name: merged.pop(name) for name in _DETAIL_ARGUMENTS if name in merged}
            detail.setdefault("resample_step", step)
            detail["max_time_points"] = merged.pop("max_time_points", max_time_points)
            burn_kwargs = {"max_step_size": step, "tail_off_evaluation": merged.pop("tail_off_evaluation", True), **merged}
            nominal = copy.deepcopy((grain, motor, propellant, environment))
            lanes = [(nominal[1], nominal[2], nominal[3] if nominal[3] is not None else Environment(), burn_kwargs, 1.0,
                      detail)]
            for scenario in scenario_list:
                _, scenario_motor, scenario_propellant, scenario_environment, factor = _scenario_objects(
                    grain, motor, propellant, environment, scenario
                )
                lane_kwargs = dict(burn_kwargs)
                lane_kwargs.update({
                    "igniter_mass_flow": merged.get("igniter_mass_flow"),
                    "igniter_burn_time": merged.get("igniter_burn_time", 0.0) * float(scenario.igniter_energy_factor),
                    "igniter_temperature": merged.get("igniter_temperature"),
                    "burn_area_activation": merged.get("burn_area_activation"),
                    "ignition_ramp_time": merged.get("ignition_ramp_time", 0.0),
                    "tail_off_method": merged.get("tail_off_method", "numerical"),
                })
                lane_detail = dict(detail, nozzle_ablation_scale=float(scenario.nozzle_ablation_scale_factor))
                lanes.append((scenario_motor, scenario_propellant, scenario_environment, lane_kwargs, factor, lane_detail))
            for motor_i, propellant_i, environment_i, kwargs_i, factor_i, detail_i in lanes:
                motors.append(motor_i)
                propellants.append(propellant_i)
                environments.append(environment_i)
                settings.append(kwargs_i)
                factors.append(factor_i)
                details.append(detail_i)

        batch = ProblemBatch.from_objects(motors, propellants, environments, settings, burn_rate_factor=factors)
        clock["pack_s"] += time.perf_counter() - mark
        mark = time.perf_counter()
        solved = simulate_burn(batch, backend=backend, device=device, history="full", strict=strict, workers=workers,
                               max_steps=max_steps).to_results()
        clock["solve_s"] += time.perf_counter() - mark
        mark = time.perf_counter()
        results: List[Dict[str, Any]] = []
        for lane in range(len(solved)):
            scenario = None if lane % lanes_per_design == 0 else scenario_list[lane % lanes_per_design - 1]
            view = SimulationView.from_lane(batch, lane, solved[lane])
            solved[lane] = None  # the view holds it; with keep_series=False nothing does once the lane is summarised
            result = build_detailed_ballistics(view, **details[lane])
            result["simulation"] = view
            if scenario is None:
                result["scenario_id"] = "nominal"
                result["scenario_kind"] = "nominal"
            else:
                _rescale_thrust(result, scenario.isp_factor)
                result["scenario_id"] = scenario.scenario_id
                result["scenario_kind"] = scenario.scenario_kind
                result["scenario_factors"] = scenario.__dict__.copy()
                if validator is not None:
                    result["valid"] = bool(validator(result))
            results.append(result if keep_series else {key: result[key] for key in _SCALAR_KEYS if key in result})
        clock["postprocess_s"] += time.perf_counter() - mark
        mark = time.perf_counter()
        for design in range(len(chunk)):
            reports.append(_build_report(results[design * lanes_per_design : (design + 1) * lanes_per_design], validator))
        clock["report_s"] += time.perf_counter() - mark
    if timings is not None:
        timings.update(clock)
    return reports
