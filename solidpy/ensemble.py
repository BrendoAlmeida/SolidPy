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
from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait
from itertools import islice
from typing import Any, Callable, Dict, Iterable, Iterator, List, Optional, Tuple

import numpy as np

from . import backends
from .backends import SolveOptions, UnsupportedLane
from .backends._protocol import refused_lanes, unsupported_lane_error
from .batch import ProblemBatch
from .batch.kernels.tables import evaluate as evaluate_table
from .batch.result import BatchResult, DeviceBatchResult
from .batch.thermal import ThermalBatch
from ._parallel import (
    available_cpu_count, process_worker_count, process_worker_initializer, safe_process_context, spawn_pickle_safe,
)

__all__ = [
    "ProblemBatch", "SolveOptions", "ThermalBatch", "UnsupportedLane", "lane_cost", "run_advanced_physics_ensemble",
    "run_robustness_ensemble", "simulate_burn", "simulate_thermal",
    "simulate_burn_device",
]

#: Backends ``backend="auto"`` may pick, best first, and the smallest batch worth sending to one.
AUTO_ACCELERATORS = ("jax",)
AUTO_MIN_LANES = 2048
#: A thermal lane costs about a tenth of a second on the scalar code, so an accelerator pays off from far fewer lanes.
AUTO_MIN_THERMAL_LANES = 128
#: Bound the input histories and completed results retained behind a slow earlier batch.
MAX_POSTPROCESS_IN_FLIGHT_JOBS = 128


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


def _validate_workers(workers: Optional[int]) -> Optional[int]:
    """Normalize the process count shared by reference solving and CPU post-processing."""
    if workers is None:
        return None
    if isinstance(workers, bool) or not isinstance(workers, numbers.Integral) or workers < 1:
        raise ValueError(f"workers must be a positive integer or None, got {workers!r}")
    return int(workers)


def _pipeline_worker_counts(workers: Optional[int]) -> Tuple[Optional[int], Optional[int]]:
    """Keep one reference worker available while the requested pool processes completed chunks."""
    if workers is None or workers <= 1:
        return workers, workers
    return 1, workers


def _process_reserved_cores(backend, device, requested, lane_count, service=None) -> int:
    """Choose how many host cores process pools should leave for accelerator feeder threads."""
    if requested is not None:
        if not isinstance(backend, (list, tuple)):
            raise ValueError("reserved_cores applies only when backend is a heterogeneous engine list")
        if isinstance(requested, bool) or not isinstance(requested, numbers.Integral) or requested < 0:
            raise ValueError(f"reserved_cores must be a non-negative integer or None, got {requested!r}")
        return int(requested)
    if backend is None:
        backend, selected = backends.current_backend()
        device = device if device is not None else selected
    if isinstance(backend, (list, tuple)):
        from .executor import HeterogeneousExecutor

        return HeterogeneousExecutor(backend)._reserved_default()
    if backend == "auto":
        minimum = AUTO_MIN_THERMAL_LANES if service == THERMAL_SERVICE else AUTO_MIN_LANES
        if lane_count < minimum or device in ("cpu", "cpu:0"):
            return 0
        for name in AUTO_ACCELERATORS:
            if backends.available().get(name) != "ok":
                continue
            try:
                candidate = backends.get_backend(name)
                devices = candidate.devices()
            except ImportError:
                continue
            if service is not None and not candidate.capabilities().provides(service):
                continue
            usable = (device in devices) if device is not None else any(
                not item.startswith("cpu") for item in devices
            )
            if usable:
                return 0 if device is not None and device.startswith("cpu") else 1
        return 0
    if backend == "cpu-reference":
        return 0
    candidate = backends.get_backend(backend, device)
    selected_device = str(getattr(candidate, "device", device or "cpu"))
    return 0 if selected_device.startswith("cpu") else 1


def _pickle_safe(value) -> bool:
    """Check spawn importability and serialization without copying numeric histories."""
    return spawn_pickle_safe(value, skip_numeric_arrays=True)


def _detailed_ballistics_process_safe(job) -> bool:
    """Check picklability without serializing the large history arrays in a ``SimulationView``."""
    if len(job) > 6 and job[6] is not None:
        return False
    _, view, scenario, options = job[:4]
    return _pickle_safe((view.motor, view.propellant, view.environment_pressure, view._activation_inputs,
                         view.result, scenario, options))


def _bounded_process_batches(
    worker: Callable[[List[Any]], List[Any]], jobs: Iterable[Any], job_count: int, workers: Optional[int],
    *, process_safe: Optional[Callable[[Any], bool]] = None, batch_size: int = 8, reserved_cores: int = 0,
    schedule: Optional[Dict[str, int]] = None,
) -> Iterator[Tuple[Any, Any]]:
    """Yield ``(job, result)`` in input order with a bounded, lazily filled process window.

    ``worker`` accepts a list of jobs and returns same-order results. At most eight batches per process and 128 jobs
    overall are retained. Unserializable jobs run locally in input order. Results from faster later batches are
    buffered only inside that bounded window, keeping histories from accumulating across the ensemble.
    """
    if job_count < 1:
        return
    if schedule is not None:
        schedule["process_batches"] = 0
    job_iterator = iter(jobs)
    if workers is None or workers <= 1 or job_count == 1:
        while True:
            batch = list(islice(job_iterator, 1))
            if not batch:
                break
            results = worker(batch)
            if len(results) != len(batch):
                raise RuntimeError("post-processing worker returned a different number of results than jobs")
            yield from zip(batch, results)
        return

    worker_budget = max(1, available_cpu_count() - int(reserved_cores))
    process_count = process_worker_count(min(workers, worker_budget), job_count)
    batch_size = min(batch_size, max(1, job_count // (2 * process_count)))
    window = min(8 * process_count, max(1, MAX_POSTPROCESS_IN_FLIGHT_JOBS // batch_size))
    slots = {}
    next_submit = 0
    next_output = 0

    def submit_one(pool):
        nonlocal next_submit
        batch = list(islice(job_iterator, batch_size))
        if not batch:
            return False
        safe = process_safe is None or all(process_safe(job) for job in batch)
        future = pool.submit(worker, batch) if safe else None
        if future is not None and schedule is not None:
            schedule["process_batches"] += 1
        slots[next_submit] = (batch, future, None)
        next_submit += 1
        return True

    # JAX may have initialized CUDA in the parent; use isolated workers rather than fork the runtime state.
    with ProcessPoolExecutor(
        max_workers=process_count, mp_context=safe_process_context(), initializer=process_worker_initializer
    ) as pool:
        while next_submit < window and submit_one(pool):
            pass
        while next_output < next_submit:
            batch, future, results = slots[next_output]
            if future is None:
                results = worker(batch)
            elif results is None:
                waiting = [entry[1] for entry in slots.values() if entry[1] is not None and entry[2] is None]
                completed, _ = wait(waiting, return_when=FIRST_COMPLETED)
                for sequence, (other_batch, other_future, _) in tuple(slots.items()):
                    if other_future in completed:
                        slots[sequence] = (other_batch, other_future, other_future.result())
                continue
            if len(results) != len(batch):
                raise RuntimeError("post-processing worker returned a different number of results than jobs")
            del slots[next_output]
            next_output += 1
            yield from zip(batch, results)
            batch = None
            results = None
            while next_submit - next_output < window and submit_one(pool):
                pass


def _timed_advanced_after_thermal_batch(jobs):
    """Return each advanced-physics result with its worker-side CPU time."""
    from .Multiphysics import _advanced_after_thermal

    results = []
    for job in jobs:
        started = time.perf_counter()
        results.append((_advanced_after_thermal(*job), time.perf_counter() - started))
    return results


def _advanced_flight_curve(curve):
    """Keep only the curve channels used after the advanced proxy batch has run."""
    if not isinstance(curve, dict):
        return curve
    keys = ("time_s", "thrust_n", "propellant_mass_kg", "scenario_factors")
    return {key: curve[key] for key in keys if key in curve}


def _advanced_proxy_batch(
    geometries, curves, thermals, casings, flames, specifics, *, effective_backend, device, execution,
):
    """Run supported advanced proxy lanes through the backend that solved their thermal lane.

    Reference lanes are left as ``None`` so the existing scalar post-processor evaluates them in its worker. Backend
    failures follow the same fallback and preserve the original result path.
    """
    from .batch.advanced_physics import AdvancedPhysicsBatch

    count = len(geometries)
    results: List[Optional[Dict[str, Any]]] = [None] * count
    if not count:
        return results, {"service": "advanced_physics_proxies", "lanes": 0, "accelerated_lanes": 0,
                         "scalar_lanes": 0, "backend_lanes": {}, "fallback_errors": {}}

    lane_backends = execution.get("lane_backends", {})
    fallback_lanes = execution.get("fallback_lanes", {})
    if isinstance(fallback_lanes, dict):
        fallback_indices = {int(index) for index in fallback_lanes}
    else:
        fallback_indices = {int(index) for index in fallback_lanes}
    groups: Dict[Tuple[str, Optional[str]], List[int]] = {}
    for lane in range(count):
        info = lane_backends.get(lane, lane_backends.get(str(lane))) if lane_backends else None
        if info is not None:
            name, selected_device = info.get("backend", "cpu-reference"), info.get("device")
        elif lane in fallback_indices:
            name, selected_device = "cpu-reference", None
        elif effective_backend in ("heterogeneous", "auto", None):
            name, selected_device = "cpu-reference", None
        else:
            name, selected_device = effective_backend, device
        groups.setdefault((name, selected_device), []).append(lane)

    backend_lanes: Dict[str, int] = {}
    fallback_errors: Dict[int, str] = {}
    accelerated_lanes = scalar_lanes = 0
    for (name, selected_device), indices in groups.items():
        backend_lanes[name] = backend_lanes.get(name, 0) + len(indices)
        if name == "cpu-reference":
            scalar_lanes += len(indices)
            continue
        try:
            selected = backends.get_backend(name, selected_device)
            if not selected.capabilities().provides("advanced_physics_proxies"):
                scalar_lanes += len(indices)
                continue
        except Exception as exc:
            selected = None
            lookup_error = f"{type(exc).__name__}: {exc}"
        else:
            lookup_error = None

        backend_limit = getattr(selected, "max_lanes", None) if selected is not None else None
        limit = min(2048, int(backend_limit)) if backend_limit is not None else 2048
        for start in range(0, len(indices), max(int(limit), 1)):
            current = indices[start : start + max(int(limit), 1)]
            batch = AdvancedPhysicsBatch.from_objects(
                [geometries[i] for i in current], [curves[i] for i in current],
                [thermals[i] for i in current], [casings[i] for i in current],
                flame_temp_k=[flames[i] for i in current], r_specific=[specifics[i] for i in current],
            )
            try:
                if lookup_error is not None:
                    raise RuntimeError(lookup_error)
                returned = selected.advanced_physics_proxies(batch, SolveOptions()).to_results()
                if len(returned) != len(current):
                    raise RuntimeError("backend returned an unexpected number of advanced proxy results")
            except Exception as exc:
                fallback_errors.update({i: f"{type(exc).__name__}: {exc}" for i in current})
                scalar_lanes += len(current)
                continue
            for lane, value in zip(current, returned):
                results[lane] = value
            accelerated_lanes += len(current)

    return results, {
        "service": "advanced_physics_proxies", "lanes": count, "accelerated_lanes": accelerated_lanes,
        "scalar_lanes": scalar_lanes, "backend_lanes": backend_lanes, "fallback_errors": fallback_errors,
    }


def _timed_detailed_ballistics_batch(jobs):
    """Build each lane's detailed ballistics and report its worker-side CPU time."""
    from .DetailedBallistics import build_detailed_ballistics

    results = []
    for job in jobs:
        if len(job) > 6 and job[6] is not None:
            results.append((job[6], 0.0))
            continue
        started = time.perf_counter()
        results.append((build_detailed_ballistics(job[1], **job[3]), time.perf_counter() - started))
    return results


def _validate_detailed_ballistics_service_result(result, lane):
    """Require the complete scalar schema and preserve each lane's canonical references."""
    from .DetailedBallistics import _validate_result_series
    from .batch.detailed_ballistics import _RESULT_FIELDS

    expected = set(_RESULT_FIELDS) | {
        "schema_version", "interpolation", "gamma", "summary", "canonical_result", "status", "provenance",
    }
    if not isinstance(result, dict) or result.keys() != expected:
        raise ValueError("backend returned an invalid detailed-ballistics result schema")
    if result["canonical_result"] is not lane["canonical"]:
        raise ValueError("backend changed the canonical-result reference of a detailed-ballistics lane")
    if result["status"] is not lane["canonical"]["status"] or result["provenance"] is not lane["canonical"]["provenance"]:
        raise ValueError("backend changed the status or provenance reference of a detailed-ballistics lane")
    _validate_result_series(result, result["time_s"])


def _detailed_ballistics_batch_block(views, options, *, effective_backend, device, execution):
    """Route supported detailed-history lanes through the solver's backend and retain scalar fallbacks."""
    from .batch.detailed_ballistics import DetailedBallisticsBatch

    count = len(views)
    results: List[Optional[Dict[str, Any]]] = [None] * count
    summary = {
        "service": "detailed_ballistics", "lanes": count, "accelerated_lanes": 0,
        "scalar_lanes": 0, "backend_lanes": {}, "fallback_errors": {},
    }
    if not count:
        return results, summary
    batch = DetailedBallisticsBatch.from_views(views, options)
    unsupported = set(batch.unsupported_lanes)
    lane_backends = execution.get("lane_backends", {})
    fallback_lanes = execution.get("fallback_lanes", {})
    if isinstance(fallback_lanes, dict):
        fallback_indices = {int(index) for index in fallback_lanes}
    else:
        fallback_indices = {int(index) for index in fallback_lanes}

    groups: Dict[Tuple[str, Optional[str]], List[int]] = {}
    for lane in range(count):
        info = lane_backends.get(lane, lane_backends.get(str(lane))) if lane_backends else None
        if info is not None:
            name, selected_device = info.get("backend", "cpu-reference"), info.get("device")
        elif lane in fallback_indices or effective_backend in ("heterogeneous", "auto", None):
            name, selected_device = "cpu-reference", None
        else:
            name, selected_device = effective_backend, device
        groups.setdefault((name, selected_device), []).append(lane)

    for (name, selected_device), indices in groups.items():
        summary["backend_lanes"][name] = summary["backend_lanes"].get(name, 0) + len(indices)
        eligible = [lane for lane in indices if lane not in unsupported]
        summary["scalar_lanes"] += len(indices) - len(eligible)
        if name == "cpu-reference" or not eligible:
            summary["scalar_lanes"] += len(eligible)
            continue
        try:
            selected = backends.get_backend(name, selected_device)
            if not selected.capabilities().provides("detailed_ballistics"):
                summary["scalar_lanes"] += len(eligible)
                continue
        except Exception as exc:
            selected = None
            lookup_error = f"{type(exc).__name__}: {exc}"
        else:
            lookup_error = None

        backend_limit = getattr(selected, "max_lanes", None) if selected is not None else None
        limit = min(2048, int(backend_limit)) if backend_limit is not None else 2048
        for start in range(0, len(eligible), max(limit, 1)):
            current = eligible[start : start + max(limit, 1)]
            sub_batch = batch.select(current)
            try:
                if lookup_error is not None:
                    raise RuntimeError(lookup_error)
                returned = selected.detailed_ballistics(sub_batch, SolveOptions()).to_results()
                if len(returned) != len(current):
                    raise RuntimeError("backend returned an unexpected number of detailed-ballistics results")
            except Exception as exc:
                summary["scalar_lanes"] += len(current)
                summary["fallback_errors"].update({
                    lane: f"{type(exc).__name__}: {exc}" for lane in current
                })
                continue
            for local_lane, (lane, value) in enumerate(zip(current, returned)):
                try:
                    if value is None:
                        raise RuntimeError("backend marked a supported detailed-ballistics lane unsupported")
                    _validate_detailed_ballistics_service_result(value, sub_batch.lanes[local_lane])
                except Exception as exc:
                    summary["scalar_lanes"] += 1
                    summary["fallback_errors"][lane] = f"{type(exc).__name__}: {exc}"
                else:
                    results[lane] = value
                    summary["accelerated_lanes"] += 1
    return results, summary


def _iter_detailed_ballistics_batches(views, options, *, effective_backend, device, execution):
    """Yield detailed results in memory-bounded blocks of at most 512 histories."""
    count = len(views)
    lane_backends = execution.get("lane_backends", {})
    fallback_lanes = execution.get("fallback_lanes", {})
    fallback_items = fallback_lanes.items() if isinstance(fallback_lanes, dict) else ((lane, None) for lane in fallback_lanes)
    fallback_items = {int(lane): detail for lane, detail in fallback_items}

    for start in range(0, count, 512):
        end = min(start + 512, count)
        started = time.perf_counter()
        block_execution = dict(execution)
        block_execution["lane_backends"] = {
            int(lane) - start: detail for lane, detail in lane_backends.items()
            if start <= int(lane) < end
        }
        block_execution["fallback_lanes"] = {
            lane - start: detail for lane, detail in fallback_items.items() if start <= lane < end
        }
        block_results, block_summary = _detailed_ballistics_batch_block(
            views[start:end], options[start:end], effective_backend=effective_backend, device=device,
            execution=block_execution,
        )
        yield start, block_results, block_summary, time.perf_counter() - started


def _execution(result: Dict[str, Any]) -> Dict[str, Any]:
    """``result["provenance"]["execution"]``, created when a third-party backend leaves it out."""
    return result.setdefault("provenance", {}).setdefault("execution", {})


def simulate_burn(
    batch: ProblemBatch,
    backend: Optional[Any] = None,
    device: Optional[str] = None,
    *,
    history: str = "metrics",
    strict: bool = False,
    workers: Optional[int] = None,
    reserved_cores: Optional[int] = None,
    max_steps: Optional[int] = None,
    tiers: Optional[tuple] = None,
    continuous_peak_diagnostics: bool = False,
    chunk_size: Optional[int] = None,
    sort: bool = True,
) -> BatchResult:
    """Simulate every lane of ``batch`` and return their canonical results in lane order.

    ``backend`` is a backend name, ``"auto"``, ``None`` for the selected backend, or a list of engine specs such as
    ``[("jax", "cuda:0"), ("cpu-reference", 6)]`` to run several devices and CPU workers concurrently. An engine
    spec is a backend name, ``(name, device)``, ``(name, workers)`` for the reference backend, or
    ``(name, device, workers)``. Unsupported lanes use the scalar reference unless ``strict=True``. ``history`` is
    ``"metrics"``, ``"full"``,
    ``"decimated:N"`` (up to N native accepted points) or ``"uniform:N"`` (N points on a uniform time grid),
    where N is an integer of at least 2.
    ``continuous_peak_diagnostics=True`` estimates maxima from each accepted DOP853 dense step on the
    ``cpu-vectorized`` and ``jax`` backends and records them under
    ``provenance["execution"]["continuous_peaks"]``. It is an opt-in diagnostic; canonical peak metrics remain
    the accepted-point values. ``cpu-reference`` results, including fallback lanes, omit this diagnostic.
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
    if isinstance(backend, (list, tuple)):
        from .executor import HeterogeneousExecutor

        if device is not None:
            raise ValueError("device cannot be combined with heterogeneous backend specs; put it in each engine spec")
        if reserved_cores is not None and (
            isinstance(reserved_cores, bool) or not isinstance(reserved_cores, numbers.Integral) or reserved_cores < 0
        ):
            raise ValueError(f"reserved_cores must be a non-negative integer or None, got {reserved_cores!r}")
        options = SolveOptions(
            history=history, workers=workers, max_steps=max_steps, tiers=tiers,
            continuous_peak_diagnostics=continuous_peak_diagnostics,
        )
        return HeterogeneousExecutor(backend, default_workers=workers, reserved_cores=reserved_cores).solve_burn(
            batch, options, requested=backend, device=device, chunk_size=chunk_size, sort=sort, strict=strict
        )
    if reserved_cores is not None:
        raise ValueError("reserved_cores applies only when backend is a heterogeneous engine list")
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
    options = SolveOptions(
        history=history, workers=workers, max_steps=max_steps, tiers=tiers,
        continuous_peak_diagnostics=continuous_peak_diagnostics,
    )

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


def simulate_burn_device(
    batch: ProblemBatch,
    backend: str = "jax",
    device: Optional[str] = None,
    *,
    history: str = "full",
    max_steps: Optional[int] = None,
    tiers: Optional[tuple] = None,
    continuous_peak_diagnostics: bool = False,
) -> DeviceBatchResult:
    """Solve a fully supported JAX batch and stream its device-resident output chunks.

    This lower-level entry point requires JAX and refuses unsupported lanes instead of silently
    switching them to the CPU reference. Use ``iter_device_batches`` for fused accelerator work,
    ``iter_results`` to materialize bounded host chunks, or ``to_results`` for the regular API.
    Step-overflow lanes are rerun on the CPU reference during host materialization, matching
    ``simulate_burn``'s recovery behavior.
    """
    if not isinstance(batch, ProblemBatch):
        raise TypeError("simulate_burn_device needs a ProblemBatch")
    if backend != "jax":
        raise ValueError("simulate_burn_device currently requires backend='jax'")
    chosen = backends.get_backend(backend, device)
    options = SolveOptions(
        history=history,
        max_steps=max_steps,
        tiers=tiers,
        continuous_peak_diagnostics=continuous_peak_diagnostics,
    )
    return chosen.solve_burn_device(batch, options)


THERMAL_SERVICE = "thermal_ablation"


def simulate_thermal(
    batch: ThermalBatch,
    backend: Optional[Any] = None,
    device: Optional[str] = None,
    *,
    strict: bool = False,
    workers: Optional[int] = None,
    reserved_cores: Optional[int] = None,
    chunk_size: Optional[int] = None,
    sort: bool = True,
) -> BatchResult:
    """The wall conduction and throat ablation of every lane of ``batch``, in lane order.

    ``results[i]`` is the mapping ``Multiphysics.simulate_thermal_ablation`` returns for lane ``i``. ``backend`` is a
    name, ``"auto"`` (an accelerator from 128 lanes, else the reference), ``None`` for the selected one, or a list of
    backend engine specs for heterogeneous scheduling. A lane the
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
    if isinstance(backend, (list, tuple)):
        from .executor import HeterogeneousExecutor

        if device is not None:
            raise ValueError("device cannot be combined with heterogeneous backend specs; put it in each engine spec")
        if reserved_cores is not None and (
            isinstance(reserved_cores, bool) or not isinstance(reserved_cores, numbers.Integral) or reserved_cores < 0
        ):
            raise ValueError(f"reserved_cores must be a non-negative integer or None, got {reserved_cores!r}")
        options = SolveOptions(workers=workers)
        return HeterogeneousExecutor(backend, default_workers=workers, reserved_cores=reserved_cores).solve_thermal(
            batch, options, requested=backend, chunk_size=chunk_size, sort=sort, strict=strict
        )
    if reserved_cores is not None:
        raise ValueError("reserved_cores applies only when backend is a heterogeneous engine list")
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
    backend: Optional[Any] = "auto",
    device: Optional[str] = None,
    strict: bool = False,
    workers: Optional[int] = None,
    reserved_cores: Optional[int] = None,
    chunk_size: Optional[int] = None,
    timings: Optional[Dict[str, float]] = None,
    execution: Optional[Dict[str, Any]] = None,
) -> List[Dict[str, float]]:
    """``simulate_advanced_physics`` for many designs: the thermal ablation of all of them is one batch.

    ``geometries`` and ``curves`` are lists with one entry per lane (one geometry or curve is broadcast), and
    ``casing_material``, ``nozzle_material``, ``flame_temp_k``, ``r_specific`` and ``gamma`` are one value for every lane
    or one per lane. Returns one flat metrics mapping per lane, the one ``simulate_advanced_physics`` returns for it: the
    wall conduction and the transient structural, CFD and ignition proxies run through the selected backend services
    when available; flight and scalar fallback lanes run on the CPU. The scenario
    factors in ``curve["scenario_factors"]`` are read as ``simulate_advanced_physics`` reads them. ``workers`` sets the
    CPU model process count. ``None`` or ``1`` keeps post-processing serial; values above one run it in an ordered
    process pool. While the bounded thermal pipeline is active, reference fallbacks use one process so the requested
    workers stay available for post-processing. ``reserved_cores`` is available with a heterogeneous engine list; by
    default the executor leaves one host core for each non-CPU feeder. At thousands of lanes, those models can dominate
    (about 5 ms per lane against 0.3 ms for batched thermal ablation on the GPU). CPU process pools use ``spawn``, so
    scripts must call this function under ``if __name__ == "__main__":`` when ``workers > 1``.
    A dict passed as ``timings`` receives cumulative seconds packing (``pack_s``), in thermal solves (``thermal_s``),
    in advanced proxy batches (``proxy_s``) and in CPU models or scalar fallbacks (``models_s``); these stages can
    overlap in the producer-consumer schedule. One passed as ``execution`` receives thermal and proxy backend summaries.
    """
    workers = _validate_workers(workers)
    from .batch.thermal import _lane_count, _per_lane
    from .Multiphysics import _require_casing_material, _resolve_gamma, _scenario_thermal_inputs

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
        defer_dynamic_coefficients=(backend == "jax"),
    )
    packed = time.perf_counter()
    requested_backend = backend
    selected_backend, selected_device = backend, device
    if backend == "auto":
        selected_backend = _auto_backend(batch, device, THERMAL_SERVICE, AUTO_MIN_THERMAL_LANES)
        if selected_backend == "cpu-reference":
            selected_device = None
    elif backend is None:
        selected_backend, configured_device = backends.current_backend()
        selected_device = device if device is not None else configured_device
        requested_backend = selected_backend

    heterogeneous = isinstance(selected_backend, (list, tuple))
    reference_only = selected_backend == "cpu-reference"
    if chunk_size is not None and (
        isinstance(chunk_size, bool) or not isinstance(chunk_size, numbers.Integral) or chunk_size < 1
    ):
        raise ValueError(f"chunk_size must be a positive integer or None, got {chunk_size!r}")
    pipeline = workers is not None and workers > 1 and not reference_only
    thermal_workers, postprocess_workers = _pipeline_worker_counts(workers) if pipeline else (workers, workers)
    postprocess_reserved_cores = _process_reserved_cores(
        selected_backend, selected_device, reserved_cores, count, service=THERMAL_SERVICE
    ) + (2 if pipeline else 0)
    thermal_elapsed = 0.0
    proxy_elapsed = 0.0
    model_elapsed = 0.0
    chunk_pack_elapsed = 0.0
    final_execution: Dict[str, Any]

    if pipeline and count:
        thermal_chunk_size = int(chunk_size) if chunk_size is not None else 2048
        execution_chunks = []
        fallback_lanes: Dict[int, Any] = {}
        fallback_errors: Dict[int, Any] = {}
        lane_backends: Dict[int, Any] = {}
        postprocess_schedule: Dict[str, int] = {}
        proxy_execution_chunks = []

        def jobs():
            nonlocal thermal_elapsed, proxy_elapsed, chunk_pack_elapsed
            for start in range(0, count, thermal_chunk_size):
                end = min(start + thermal_chunk_size, count)
                chunk_start = time.perf_counter()
                sub_batch = batch.select(np.arange(start, end))
                chunk_pack_elapsed += time.perf_counter() - chunk_start
                solve_start = time.perf_counter()
                outcome = simulate_thermal(
                    sub_batch, backend=selected_backend, device=selected_device, strict=strict,
                    workers=thermal_workers,
                    reserved_cores=reserved_cores, chunk_size=chunk_size, sort=True,
                )
                thermal_elapsed += time.perf_counter() - solve_start
                execution_chunks.append(outcome.execution)
                for lane, reasons in outcome.execution.get("fallback_lanes", {}).items():
                    fallback_lanes[start + int(lane)] = reasons
                for lane, error in outcome.execution.get("fallback_errors", {}).items():
                    fallback_errors[start + int(lane)] = error
                for lane, backend_info in outcome.execution.get("lane_backends", {}).items():
                    lane_backends[start + int(lane)] = backend_info
                thermal = outcome.to_results()
                proxy_start = time.perf_counter()
                proxies, proxy_summary = _advanced_proxy_batch(
                    geometries[start:end], curves[start:end], thermal, casings[start:end], flames[start:end],
                    specifics[start:end], effective_backend=outcome.backend, device=selected_device,
                    execution=outcome.execution,
                )
                proxy_elapsed += time.perf_counter() - proxy_start
                proxy_execution_chunks.append(proxy_summary)
                for local_lane, result in enumerate(thermal):
                    i = start + local_lane
                    proxies_for_lane = proxies[local_lane]
                    curve_for_models = curves[i]
                    if proxies_for_lane is not None:
                        curve_for_models = _advanced_flight_curve(curves[i])
                    yield (
                        geometries[i], curve_for_models, result, casings[i], flames[i], specifics[i], gammas[i],
                        proxies_for_lane,
                    )

        results = []
        for _, (result, elapsed) in _bounded_process_batches(
            _timed_advanced_after_thermal_batch, jobs(), count, postprocess_workers, process_safe=_pickle_safe,
            reserved_cores=postprocess_reserved_cores, schedule=postprocess_schedule,
        ):
            results.append(result)
            model_elapsed += elapsed
        final_execution = {
            "requested_backend": requested_backend,
            "effective_backend": "heterogeneous" if heterogeneous else selected_backend,
            "lanes": count,
            "chunks": len(execution_chunks),
            "fallback_lanes": fallback_lanes,
            "fallback_errors": fallback_errors,
            "lane_backends": lane_backends,
            "backend_executions": [
                execution for chunk_execution in execution_chunks
                for execution in chunk_execution.get("backend_executions", [chunk_execution])
            ],
            "schedule": "thermal_postprocess_pipeline",
            "overlap": len(execution_chunks) > 1 and postprocess_schedule["process_batches"] > 0,
            "advanced_proxies": {
                "service": "advanced_physics_proxies",
                "lanes": sum(item["lanes"] for item in proxy_execution_chunks),
                "accelerated_lanes": sum(item["accelerated_lanes"] for item in proxy_execution_chunks),
                "scalar_lanes": sum(item["scalar_lanes"] for item in proxy_execution_chunks),
                "backend_lanes": {
                    name: sum(item["backend_lanes"].get(name, 0) for item in proxy_execution_chunks)
                    for name in {name for item in proxy_execution_chunks for name in item["backend_lanes"]}
                },
                "fallback_errors": {
                    start + lane: error
                    for start, item in zip(
                        range(0, count, thermal_chunk_size), proxy_execution_chunks
                    )
                    for lane, error in item["fallback_errors"].items()
                },
            },
        }
    else:
        outcome = simulate_thermal(
            batch, backend=backend, device=device, strict=strict, workers=workers, reserved_cores=reserved_cores,
            chunk_size=chunk_size,
        )
        solved = time.perf_counter()
        thermal = outcome.to_results()
        proxy_start = time.perf_counter()
        proxies, proxy_execution = _advanced_proxy_batch(
            geometries, curves, thermal, casings, flames, specifics, effective_backend=outcome.backend,
            device=selected_device, execution=outcome.execution,
        )
        proxy_elapsed = time.perf_counter() - proxy_start
        jobs = (
            (geometries[i], curves[i], thermal[i], casings[i], flames[i], specifics[i], gammas[i], proxies[i])
            for i in range(count)
        )
        results = []
        for _, (result, elapsed) in _bounded_process_batches(
            _timed_advanced_after_thermal_batch, jobs, count, workers, process_safe=_pickle_safe,
            reserved_cores=postprocess_reserved_cores,
        ):
            results.append(result)
            model_elapsed += elapsed
        thermal_elapsed = solved - packed
        final_execution = dict(outcome.execution)
        final_execution["advanced_proxies"] = proxy_execution

    if timings is not None:
        timings.update(pack_s=packed - mark + chunk_pack_elapsed,
                       thermal_s=thermal_elapsed, proxy_s=proxy_elapsed, models_s=model_elapsed)
    if execution is not None:
        execution.update(final_execution)
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
    backend: Optional[Any] = "auto",
    device: Optional[str] = None,
    strict: bool = False,
    workers: Optional[int] = None,
    reserved_cores: Optional[int] = None,
    max_steps: Optional[int] = None,
    chunk_lanes: int = 4096,
    timings: Optional[Dict[str, float]] = None,
    execution: Optional[Dict[str, Any]] = None,
    keep_series: bool = True,
    **simulation_kwargs: Any,
) -> List[Dict[str, Any]]:
    """Robustness analysis of many designs: every (design, scenario) pair is a lane of one batch.

    Returns one report per design, in the form ``run_robustness_analysis`` returns for it. ``designs`` holds
    ``(grain, motor, propellant[, environment[, simulation_kwargs]])`` tuples; ``simulation_kwargs`` given here
    apply to every design and the ones in a design's tuple override them (that includes ``max_step_size`` and
    ``max_time_points``). The scenarios are the same for every
    design (the default ones, or ``scenarios``, plus ``monte_carlo_sample_count`` Latin-hypercube samples).

    The burns run on ``backend`` (a name, ``"auto"``, a heterogeneous engine list, or ``None`` for the selected one;
    see ``simulate_burn``) with
    the full history; detailed ballistics then runs as lanes through the selected backend's ``detailed_ballistics``
    service when available, with unsupported lanes sent through the scalar implementation. Reports match the scalar
    result within the numerical tolerances of ``solidpy.backends._tolerances``.
    Invalid inputs raise when the batch is packed,
    before any burn is solved; the scalar path raises as it reaches them. A dict passed as ``timings`` receives the
    seconds spent packing (``pack_s``), solving chunks (``solve_s``), batched detailed-ballistics services
    (``detail_s``), scalar fallback work (``postprocess_s``) and assembling reports (``report_s``). A dict passed as
    ``execution`` receives the solve-chunk count, process-batch count, lane service routes and whether solve and
    fallback work overlapped.

    Every lane of a report holds its full series and canonical history, about 200 kB each for a four-grain design, as
    the scalar path's does.
    For thousands of lanes pass ``keep_series=False``: each lane then drops every array, its canonical history and its
    simulation view and keeps its scalar outputs (``summary``, ``status``, ``provenance``, the scenario fields,
    ``valid``), and the report, its statistics and its validity ratio are unchanged.

    ``chunk_lanes`` is the most lanes a launch holds, except that a design is never split: a chunk is at least one design
    (27 lanes with the defaults, more with many Latin-hypercube samples). ``workers`` is the process count for detailed-
    ballistics post-processing; while chunks overlap, reference solves use one process. ``reserved_cores`` optionally
    reserves host cores for accelerator feeders when using a heterogeneous list. ``None`` or ``1`` keeps post-processing serial.
    Process pools use ``spawn``, so scripts must call this function under ``if __name__ == "__main__":`` when
    ``workers > 1``.
    """
    from .batch.simulation_view import SimulationView
    from .DetailedBallistics import _validate_dry_hardware
    from .Environment import Environment
    from .Robustness import (
        _build_report, _finish_scenario_result, _scenario_objects, _scenario_simulation_kwargs,
        build_latin_hypercube_scenarios, default_robustness_scenarios,
    )

    workers = _validate_workers(workers)
    if isinstance(chunk_lanes, bool) or not isinstance(chunk_lanes, numbers.Integral) or chunk_lanes < 1:
        raise ValueError(f"chunk_lanes must be a positive integer, got {chunk_lanes!r}")
    parsed = [_robustness_design(entry) for entry in designs]
    scenario_list = list(scenarios) if scenarios is not None else default_robustness_scenarios()
    scenario_list.extend(build_latin_hypercube_scenarios(sample_count=monte_carlo_sample_count, seed=monte_carlo_seed))
    lanes_per_design = 1 + len(scenario_list)
    for _, motor, _, _, _ in parsed:
        _validate_dry_hardware(motor)

    clock = {"pack_s": 0.0, "solve_s": 0.0, "detail_s": 0.0, "postprocess_s": 0.0, "report_s": 0.0}
    reports: List[Dict[str, Any]] = []
    designs_per_chunk = max(1, int(chunk_lanes) // lanes_per_design)
    total_lanes = len(parsed) * lanes_per_design
    solver_workers, postprocess_workers = _pipeline_worker_counts(workers)
    solve_chunks = 0
    detail_summaries = []

    def lane_jobs():
        nonlocal solve_chunks
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
                burn_kwargs = {
                    "max_step_size": step, "tail_off_evaluation": merged.pop("tail_off_evaluation", True), **merged
                }
                nominal = copy.deepcopy((grain, motor, propellant, environment))
                lanes = [(nominal[1], nominal[2], nominal[3] if nominal[3] is not None else Environment(),
                          burn_kwargs, 1.0, detail)]
                for scenario in scenario_list:
                    _, scenario_motor, scenario_propellant, scenario_environment, factor = _scenario_objects(
                        grain, motor, propellant, environment, scenario
                    )
                    lane_kwargs = {
                        "max_step_size": step, "tail_off_evaluation": burn_kwargs["tail_off_evaluation"],
                        **_scenario_simulation_kwargs(merged, scenario),
                    }
                    lane_detail = dict(detail, nozzle_ablation_scale=float(scenario.nozzle_ablation_scale_factor))
                    lanes.append((scenario_motor, scenario_propellant, scenario_environment, lane_kwargs, factor,
                                  lane_detail))
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
            outcome = simulate_burn(
                batch, backend=backend, device=device, history="full", strict=strict, workers=solver_workers,
                reserved_cores=reserved_cores, max_steps=max_steps,
            )
            solved = outcome.to_results()
            clock["solve_s"] += time.perf_counter() - mark
            solve_chunks += 1
            views = []
            for lane in range(len(solved)):
                views.append(SimulationView.from_lane(batch, lane, solved[lane]))
            solved = None  # the detailed batch or the bounded window now owns each lane's history
            for block_start, detailed_results, detail_summary, elapsed in _iter_detailed_ballistics_batches(
                views, details, effective_backend=outcome.backend, device=device, execution=outcome.execution,
            ):
                clock["detail_s"] += elapsed
                detail_summaries.append((start * lanes_per_design + block_start, detail_summary))
                for offset, detailed_result in enumerate(detailed_results):
                    lane = block_start + offset
                    local_design, scenario_index = divmod(lane, lanes_per_design)
                    scenario = None if scenario_index == 0 else scenario_list[scenario_index - 1]
                    yield (
                        lane, views[lane], scenario, details[lane], start + local_design, scenario_index,
                        detailed_result,
                    )
            views = None
            solved = None

    design_results: List[Dict[str, Any]] = []
    postprocess_schedule: Dict[str, int] = {}
    postprocess_reserved_cores = _process_reserved_cores(backend, device, reserved_cores, total_lanes)
    if workers is not None and workers > 1:
        postprocess_reserved_cores += 2  # keep a producer core and one reference fallback worker available
    for job, (result, elapsed) in _bounded_process_batches(
        _timed_detailed_ballistics_batch, lane_jobs(), total_lanes, postprocess_workers,
        process_safe=_detailed_ballistics_process_safe, reserved_cores=postprocess_reserved_cores,
        schedule=postprocess_schedule,
    ):
        _, view, scenario, _, design_index, local_lane, _ = job
        if design_index != len(reports):
            raise RuntimeError(
                f"robustness post-processing changed design order: expected {len(reports)}, got {design_index}"
            )
        clock["postprocess_s"] += elapsed
        if keep_series:
            result["simulation"] = view
        if scenario is None:
            result["scenario_id"] = "nominal"
            result["scenario_kind"] = "nominal"
        else:
            _finish_scenario_result(result, scenario, validator)
        design_results.append(
            result if keep_series else {
                key: value for key, value in result.items()
                if key not in _HEAVY_KEYS and not isinstance(value, np.ndarray)
            }
        )
        if local_lane == lanes_per_design - 1:
            mark = time.perf_counter()
            reports.append(_build_report(design_results, validator))
            clock["report_s"] += time.perf_counter() - mark
            design_results = []
    if timings is not None:
        timings.update(clock)
    if execution is not None:
        detailed_execution = {
            "service": "detailed_ballistics",
            "lanes": sum(summary["lanes"] for _, summary in detail_summaries),
            "accelerated_lanes": sum(summary["accelerated_lanes"] for _, summary in detail_summaries),
            "scalar_lanes": sum(summary["scalar_lanes"] for _, summary in detail_summaries),
            "backend_lanes": {
                name: sum(summary["backend_lanes"].get(name, 0) for _, summary in detail_summaries)
                for name in {name for _, summary in detail_summaries for name in summary["backend_lanes"]}
            },
            "fallback_errors": {
                start + lane: error
                for start, summary in detail_summaries
                for lane, error in summary["fallback_errors"].items()
            },
        }
        execution.update(
            schedule="burn_postprocess_pipeline",
            lanes=total_lanes,
            chunks=solve_chunks,
            process_batches=postprocess_schedule.get("process_batches", 0),
            overlap=solve_chunks > 1 and postprocess_schedule.get("process_batches", 0) > 0,
            detailed_ballistics=detailed_execution,
        )
    return reports
