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
from .batch.thermal import ThermalBatch

__all__ = [
    "ProblemBatch", "SolveOptions", "ThermalBatch", "UnsupportedLane", "lane_cost", "run_advanced_physics_ensemble",
    "run_robustness_ensemble", "simulate_burn", "simulate_thermal",
]

#: Backends ``backend="auto"`` may pick, best first, and the smallest batch worth sending to one.
AUTO_ACCELERATORS = ("jax",)
AUTO_MIN_LANES = 2048
#: A thermal lane costs about a tenth of a second on the scalar code, so an accelerator pays off from far fewer lanes.
AUTO_MIN_THERMAL_LANES = 128


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
        coefficient = a["burn_rate_a"] * a["burn_rate_factor"]  # the factor scales the rate, so the pressure it reaches
        pressure = (a["density"] * coefficient * (1e-6) ** n * kn * cstar / (1000.0 * a["discharge_coefficient"])) ** (
            1.0 / (1.0 - n)
        )
        rate = coefficient * np.maximum(pressure * 1e-6, 1e-12) ** n / 1000.0
        tabulated = evaluate_table(np, np.full(len(batch), 3.0), a["rate_table_x"],
                                   tuple(a[f"rate_table_c{i}"] for i in range(4)), a["rate_table_n"],
                                   a["rate_table_below"], a["rate_table_above"]) / 1000.0  # at a typical 3 MPa
        rate = np.where(a["burn_rate_mode"] == 1.0, tabulated * a["burn_rate_factor"], rate)
        depth = np.where(a["grain_valid"], a["burnout_depth"], 0.0).max(axis=1)
        cost = depth / rate / a["max_step_size"] + 40.0 * a["n_valid_grains"]
    return np.where(np.isfinite(cost), cost, np.finfo(float).max)


def _auto_backend(batch, device: Optional[str] = None, service: Optional[str] = None,
                  minimum: Optional[int] = None) -> str:
    """An accelerator when one is usable and enough lanes can run on it to amortise it, else the reference.

    An explicit ``device`` restricts the choice to a backend that lists it, and ``"cpu"`` never selects an accelerator.
    With a ``service`` the backend must provide it; ``minimum`` defaults to ``AUTO_MIN_LANES``.
    """
    minimum = AUTO_MIN_LANES if minimum is None else minimum
    if len(batch) < minimum or device in ("cpu", "cpu:0"):
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
        capabilities = accelerator.capabilities()
        usable = (device in devices) if device is not None else any(not d.startswith("cpu") for d in devices)
        if service is not None and not capabilities.provides(service):
            continue
        if usable and len(batch) - len(refused_lanes(batch, capabilities)) >= minimum:
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
            execution["scenario_inputs"] = {"burn_rate_factor": float(batch.arrays["burn_rate_factor"][int(lane)])}
            result.setdefault("provenance", {})["execution"] = execution
            results[int(lane)] = result
    if backend != "cpu-reference":
        for result in results:
            _execution(result)["requested_backend"] = requested

    summary = {"requested_backend": requested, "effective_backend": backend, "lanes": len(batch),
               "fallback_lanes": sorted(int(i) for i in reasons), "chunks": len(chunks), "tiers": tier_log}
    return BatchResult(results, backend, summary)


THERMAL_SERVICE = "thermal_ablation"


def simulate_thermal(
    batch: ThermalBatch,
    backend: Optional[str] = None,
    device: Optional[str] = None,
    *,
    strict: bool = False,
    workers: Optional[int] = None,
    chunk_size: Optional[int] = None,
    sort: bool = True,
) -> BatchResult:
    """The wall conduction and throat ablation of every lane of ``batch``, in lane order.

    ``results[i]`` is the mapping ``Multiphysics.simulate_thermal_ablation`` returns for lane ``i``. ``backend`` is a
    name, ``"auto"`` (an accelerator from 128 lanes, else the reference) or ``None`` for the selected one. A lane the
    backend cannot take (a series that is not finite, series of different lengths, a backend without the thermal
    service) are rerun on the scalar reference, or raise ``UnsupportedLane`` with ``strict=True``. A lane whose integration
    did not finish is rerun on the reference even with ``strict=True`` (the scalar code decides what happens to it, as
    ``simulate_burn`` does for a lane that ran out of steps). ``result.execution["fallback_lanes"]`` lists the lanes rerun,
    with their reasons, and ``result.execution["backend_executions"]`` holds what the backend reported for each of its
    chunks (integrator, tolerances, Radau steps and attempts, device, versions); ``effective_backend`` is the backend that
    was asked to solve, so a batch whose lanes all fell back still names it. ``workers`` is the process count of the
    reference, ``chunk_size`` bounds the lanes of one solve, and ``sort`` groups lanes of similar length into the same
    chunk when there are several. The lanes keep references to the objects they were packed from, which the reference
    reads when it reruns a lane: do not modify them between packing and solving.
    """
    if not isinstance(batch, ThermalBatch):
        raise TypeError("simulate_thermal needs a ThermalBatch; build one with ThermalBatch.from_objects")
    if chunk_size is not None and (
        isinstance(chunk_size, bool) or not isinstance(chunk_size, numbers.Integral) or chunk_size < 1
    ):
        raise ValueError(f"chunk_size must be a positive integer or None, got {chunk_size!r}")
    chunk_size = None if chunk_size is None else int(chunk_size)
    requested = backend
    if backend == "auto":
        backend = _auto_backend(batch, device, THERMAL_SERVICE, AUTO_MIN_THERMAL_LANES)
        if device is not None and backend == "cpu-reference":
            device = None
    if backend is None:
        backend, selected_device = backends.current_backend()
        requested = backend
        device = device if device is not None else selected_device
    chosen = backends.get_backend(backend, device)
    options = SolveOptions(workers=workers)

    capabilities = chosen.capabilities()
    if capabilities.provides(THERMAL_SERVICE):
        refused = refused_lanes(batch, capabilities)
    else:
        refused = {lane: [f"service:{THERMAL_SERVICE}"] for lane in range(len(batch))}
    if refused and strict:
        raise unsupported_lane_error(backend, refused)

    results: List[Optional[Dict[str, Any]]] = [None] * len(batch)
    supported = np.asarray([lane for lane in range(len(batch)) if lane not in refused], dtype=int)
    chunks: List[np.ndarray] = []
    executions: List[Dict[str, Any]] = []
    if len(supported):
        launch_limit = getattr(chosen, "max_lanes", None)
        several = any(limit is not None and limit < len(supported) for limit in (chunk_size, launch_limit))
        order = supported[np.argsort(batch.arrays["n_intervals"][supported], kind="stable")] if sort and several else supported
        chunks = _chunks(order, chunk_size)
        for chunk in chunks:
            outcome = chosen.thermal_ablation(batch.select(chunk), options)
            executions.append(outcome.execution)
            returned = outcome.to_results()
            if len(returned) != len(chunk):
                raise RuntimeError(f"backend {backend!r} returned {len(returned)} results for {len(chunk)} lanes")
            for lane, result in zip(chunk, returned):
                results[int(lane)] = result
    reasons: Dict[int, List[str]] = {lane: list(features) for lane, features in refused.items()}
    for lane in supported:
        if results[int(lane)] is None:  # the integration did not finish: the scalar code decides what happens
            reasons[int(lane)] = ["integration_failed"]
    if reasons:
        reference = backends.get_backend("cpu-reference")
        lanes = np.asarray(sorted(reasons), dtype=int)
        for lane, result in zip(lanes, reference.thermal_ablation(batch.select(lanes), options).to_results()):
            results[int(lane)] = result
    summary = {"requested_backend": requested, "effective_backend": backend, "lanes": len(batch),
               "fallback_lanes": {int(i): reasons[i] for i in sorted(reasons)}, "chunks": len(chunks),
               "backend_executions": executions}
    return BatchResult(results, backend, summary)


def run_advanced_physics_ensemble(
    geometries,
    curves,
    *,
    casing_material=None,
    nozzle_material=None,
    flame_temp_k=2800.0,
    r_specific=287.0,
    gamma=None,
    backend: Optional[str] = "auto",
    device: Optional[str] = None,
    strict: bool = False,
    workers: Optional[int] = None,
    chunk_size: Optional[int] = None,
    timings: Optional[Dict[str, float]] = None,
    execution: Optional[Dict[str, Any]] = None,
) -> List[Dict[str, float]]:
    """``simulate_advanced_physics`` for many designs: the thermal ablation of all of them is one batch.

    ``geometries`` and ``curves`` are lists with one entry per lane (one geometry or curve is broadcast), and
    ``casing_material``, ``nozzle_material``, ``flame_temp_k``, ``r_specific`` and ``gamma`` are one value for every lane
    or one per lane. Returns one flat metrics mapping per lane, the one ``simulate_advanced_physics`` returns for it: the
    wall conduction, which costs most of the advanced physics, runs through ``simulate_thermal`` on ``backend`` and the
    structural, CFD, ignition and flight models then run on the CPU for each lane from its thermal metrics. The scenario
    factors in ``curve["scenario_factors"]`` are read as ``simulate_advanced_physics`` reads them. ``workers`` is the process
    count of the ``cpu-reference`` thermal batch; the models after the thermal one run serially in this process, and at
    thousands of lanes they are most of the time (5 ms per lane against 0.3 ms for the batched thermal ablation on the GPU).
    A dict passed as
    ``timings`` receives the seconds packing (``pack_s``), in the thermal batch (``thermal_s``) and in the other models
    (``models_s``); one passed as ``execution`` receives the summary of ``simulate_thermal``.
    """
    from .batch.thermal import _lane_count, _per_lane
    from .Multiphysics import _advanced_after_thermal, _require_casing_material, _resolve_gamma, _scenario_thermal_inputs

    mark = time.perf_counter()
    count = _lane_count(geometries, curves, casing_material, nozzle_material, flame_temp_k, r_specific, gamma)
    geometries = _per_lane(geometries, count, "geometries")
    curves = _per_lane(curves, count, "curves")
    casings = [_require_casing_material(m) for m in _per_lane(casing_material, count, "casing_material")]
    flames = _per_lane(flame_temp_k, count, "flame_temp_k")
    specifics = _per_lane(r_specific, count, "r_specific")
    gammas = [float(_resolve_gamma(curve, g)) for curve, g in zip(curves, _per_lane(gamma, count, "gamma"))]
    scenario = [_scenario_thermal_inputs(curve) for curve in curves]
    batch = ThermalBatch.from_objects(
        geometries, curves, casings, nozzle_material, flame_temp_k=flames, r_specific=specifics, gamma=gammas,
        initial_temperature_k=[temperature for _, temperature in scenario],
        liner_thickness_factor=[liner for liner, _ in scenario],
    )
    packed = time.perf_counter()
    outcome = simulate_thermal(batch, backend=backend, device=device, strict=strict, workers=workers, chunk_size=chunk_size)
    solved = time.perf_counter()
    thermal = outcome.to_results()
    results = [
        _advanced_after_thermal(geometries[i], curves[i], thermal[i], casings[i], flames[i], specifics[i], gammas[i])
        for i in range(count)
    ]
    if timings is not None:
        timings.update(pack_s=packed - mark, thermal_s=solved - packed, models_s=time.perf_counter() - solved)
    if execution is not None:
        execution.update(outcome.execution)
    return results


#: What a lane drops with ``keep_series=False``, besides every array: the canonical history and the simulation view.
_HEAVY_KEYS = ("canonical_result", "simulation")

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
    Invalid inputs raise when the batch is packed,
    before any burn is solved; the scalar path raises as it reaches them. A dict passed as ``timings`` receives the
    seconds spent packing (``pack_s``), solving (``solve_s``), building the detailed ballistics (``postprocess_s``) and
    assembling the reports (``report_s``).

    Every lane of a report holds its full series and canonical history, about 200 kB each for a four-grain design, as
    the scalar path's does.
    For thousands of lanes pass ``keep_series=False``: each lane then drops every array, its canonical history and its
    simulation view and keeps its scalar outputs (``summary``, ``status``, ``provenance``, the scenario fields,
    ``valid``), and the report, its statistics and its validity ratio are unchanged.

    ``chunk_lanes`` is the most lanes a launch holds, except that a design is never split: a chunk is at least one design
    (27 lanes with the defaults, more with many Latin-hypercube samples). ``workers`` is the process count of the
    ``cpu-reference`` backend; the detailed ballistics of the lanes is built serially in this process.
    """
    from .batch.simulation_view import SimulationView
    from .DetailedBallistics import _validate_dry_hardware, build_detailed_ballistics
    from .Environment import Environment
    from .Robustness import (
        _build_report, _finish_scenario_result, _scenario_objects, _scenario_simulation_kwargs,
        build_latin_hypercube_scenarios, default_robustness_scenarios,
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
                lane_kwargs = {"max_step_size": step, "tail_off_evaluation": burn_kwargs["tail_off_evaluation"],
                               **_scenario_simulation_kwargs(merged, scenario)}
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
                _finish_scenario_result(result, scenario, validator)
            results.append(result if keep_series else {key: value for key, value in result.items()
                                                       if key not in _HEAVY_KEYS and not isinstance(value, np.ndarray)})
        clock["postprocess_s"] += time.perf_counter() - mark
        mark = time.perf_counter()
        for design in range(len(chunk)):
            reports.append(_build_report(results[design * lanes_per_design : (design + 1) * lanes_per_design], validator))
        clock["report_s"] += time.perf_counter() - mark
    if timings is not None:
        timings.update(clock)
    return reports
