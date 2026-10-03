"""Concurrent lane scheduling across CPU and accelerator backend instances.

``HeterogeneousExecutor`` is used by :func:`solidpy.ensemble.simulate_burn` when a caller supplies more than one
backend. Each backend instance has one feeder thread; feeders pull compatible lanes from a shared, cost-ordered
queue, so faster engines naturally take more chunks and independent devices can solve at the same time.
"""

from __future__ import annotations

import dataclasses
import numbers
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np

from . import backends
from .backends._protocol import SolveOptions, refused_lanes, unsupported_lane_error
from .batch.result import BatchResult


@dataclasses.dataclass(frozen=True)
class EngineSpec:
    """One backend feeder. ``workers`` sets the reference backend's process count."""

    name: str
    device: Optional[str] = None
    workers: Optional[int] = None

    def __post_init__(self):
        if not isinstance(self.name, str) or not self.name or self.name == "auto":
            raise ValueError("heterogeneous engine names must be non-empty backend names other than 'auto'")
        if self.device is not None and not isinstance(self.device, str):
            raise ValueError("engine device must be a string or None")
        if self.workers is not None and (
            isinstance(self.workers, bool) or not isinstance(self.workers, numbers.Integral) or self.workers < 1
        ):
            raise ValueError(f"engine workers must be a positive integer or None, got {self.workers!r}")
        if self.workers is not None and self.name != "cpu-reference":
            raise ValueError("per-engine workers are currently supported only for 'cpu-reference'")


def _parse_spec(value) -> EngineSpec:
    if isinstance(value, EngineSpec):
        return value
    if isinstance(value, str):
        return EngineSpec(value)
    if not isinstance(value, (tuple, list)) or not 2 <= len(value) <= 3:
        raise TypeError(f"engine specs must be a name or (name, device/workers[, workers]), got {value!r}")
    name, second, *third = value
    if len(value) == 2 and (second is None or isinstance(second, str)):
        return EngineSpec(name, device=second)
    if len(value) == 2 and isinstance(second, numbers.Integral) and not isinstance(second, bool):
        return EngineSpec(name, workers=int(second))
    if len(value) == 3:
        workers = third[0]
        return EngineSpec(name, device=second, workers=workers)
    raise TypeError(f"engine specs must be (name, device) or (name, workers), got {value!r}")


def parse_engine_specs(specs) -> Tuple[EngineSpec, ...]:
    """Normalize and validate the public ``backend=[...]`` engine specification."""
    if not isinstance(specs, (list, tuple)) or not specs:
        raise TypeError("heterogeneous backend must be a non-empty list of engine specs")
    # A two-item tuple can itself be a single engine spec; lists always describe the engine collection.
    if isinstance(specs, tuple) and len(specs) in (2, 3) and isinstance(specs[0], str):
        values = [specs]
    else:
        values = list(specs)
    engines = tuple(_parse_spec(item) for item in values)
    keys = [(engine.name, engine.device) for engine in engines]
    if len(set(keys)) != len(keys):
        raise ValueError("each backend/device pair may appear only once in a heterogeneous backend")
    return engines


def _backend_options(options: SolveOptions, spec: EngineSpec, default_workers: Optional[int]) -> SolveOptions:
    if spec.name == "cpu-reference":
        workers = spec.workers if spec.workers is not None else default_workers
        return dataclasses.replace(options, workers=workers)
    return options


def _preferred_chunk_size(
    backend, spec: EngineSpec, lane_count: int, requested: Optional[int], engine_count: int,
    default_workers: Optional[int] = None,
) -> int:
    limit = getattr(backend, "max_lanes", None)
    if requested is not None:
        return max(1, min(int(requested), int(limit) if limit is not None else lane_count))
    if engine_count == 1:
        return max(1, lane_count)
    if spec.name == "cpu-reference":
        workers = int(spec.workers or default_workers or 1)
        return max(1, min(lane_count, 64 * workers))
    if limit is not None:
        return max(1, min(int(limit), 2048, lane_count))
    return max(1, min(256, lane_count))


class _LaneQueue:
    """A shared ordered queue that lets each feeder claim only lanes it can execute."""

    def __init__(self, order: np.ndarray, eligible: Sequence[np.ndarray]):
        self.order = np.asarray(order, dtype=int)
        self.eligible = eligible
        self.pending = np.ones(len(order), dtype=bool)
        self.cursors = [0] * len(eligible)
        self.lock = threading.Lock()

    def take(self, engine_index: int, limit: int) -> np.ndarray:
        chosen = []
        with self.lock:
            cursor = self.cursors[engine_index]
            allowed = self.eligible[engine_index]
            while cursor < len(self.order) and len(chosen) < limit:
                lane = int(self.order[cursor])
                cursor += 1
                if self.pending[lane] and allowed[lane]:
                    self.pending[lane] = False
                    chosen.append(lane)
            self.cursors[engine_index] = cursor
        return np.asarray(chosen, dtype=int)


def _execution(result: Dict[str, Any]) -> Dict[str, Any]:
    return result.setdefault("provenance", {}).setdefault("execution", {})


class HeterogeneousExecutor:
    """Solve one ``ProblemBatch`` across backend instances while preserving lane order.

    The scheduler launches one feeder thread for each backend/device. A shared queue provides bounded chunks and
    lets a feeder that finishes early claim more work. Unsupported lanes use the CPU reference automatically unless
    ``strict=True``. A chunk exception, missing result or step overflow is isolated and retried on the reference.
    """

    def __init__(self, specs, *, default_workers: Optional[int] = None):
        self.specs = parse_engine_specs(specs)
        self.default_workers = default_workers
        self.backends = tuple(backends.get_backend(spec.name, spec.device) for spec in self.specs)

    def solve_burn(
        self,
        batch,
        options: SolveOptions,
        *,
        requested: Any = None,
        device: Optional[str] = None,
        chunk_size: Optional[int] = None,
        sort: bool = True,
        strict: bool = False,
    ) -> BatchResult:
        if device is not None:
            raise ValueError("device cannot be combined with heterogeneous backend specs; put it in each engine spec")
        if not isinstance(options, SolveOptions):
            raise TypeError(f"options must be a SolveOptions, got {type(options).__name__}")
        if chunk_size is not None and (
            isinstance(chunk_size, bool) or not isinstance(chunk_size, numbers.Integral) or chunk_size < 1
        ):
            raise ValueError(f"chunk_size must be a positive integer or None, got {chunk_size!r}")

        lane_count = len(batch)
        if not lane_count:
            return BatchResult([], "heterogeneous", {
                "requested_backend": requested, "effective_backend": "heterogeneous", "lanes": 0,
                "engines": [], "fallback_lanes": [], "chunks": 0,
            })

        capabilities = [backend.capabilities() for backend in self.backends]
        refused = [refused_lanes(batch, capability) for capability in capabilities]
        eligible = [np.asarray([lane not in missing for lane in range(lane_count)], dtype=bool) for missing in refused]
        covered = np.logical_or.reduce(eligible)
        uncovered = np.flatnonzero(~covered)
        if len(uncovered) and strict:
            missing = {}
            for lane in uncovered:
                features = []
                for feature in refused[0].get(int(lane), []):
                    if all(feature in mapping.get(int(lane), []) for mapping in refused[1:]):
                        features.append(feature)
                missing[int(lane)] = features or ["no_selected_backend_supports_lane"]
            raise unsupported_lane_error("heterogeneous", missing)

        specs = list(self.specs)
        backend_instances = list(self.backends)
        fallback_only = [False] * len(specs)
        if len(uncovered):
            specs.append(EngineSpec("cpu-reference", workers=self.default_workers))
            backend_instances.append(backends.get_backend("cpu-reference"))
            eligible.append(np.isin(np.arange(lane_count), uncovered))
            fallback_only.append(True)

        smallest_chunk = min(
            _preferred_chunk_size(b, s, lane_count, chunk_size, len(specs), self.default_workers)
            for b, s in zip(backend_instances, specs)
        )
        if sort and lane_count > smallest_chunk:
            from .ensemble import lane_cost

            order = np.argsort(lane_cost(batch), kind="stable")
        else:
            order = np.arange(lane_count)
        work = _LaneQueue(order, eligible)
        results: List[Optional[Dict[str, Any]]] = [None] * lane_count
        counts = [{"lanes": 0, "chunks": 0, "elapsed_s": 0.0} for _ in specs]
        errors: List[BaseException] = []
        error_lock = threading.Lock()
        reference = backends.get_backend("cpu-reference")
        fallback_reasons: Dict[int, Dict[str, Any]] = {}

        def retry_on_reference(lanes: Sequence[int], reasons: Mapping[int, Mapping[str, Any]]) -> None:
            indices = np.asarray(lanes, dtype=int)
            values = reference.solve_burn(batch.select(indices), _backend_options(options, EngineSpec("cpu-reference"),
                                                                                   self.default_workers)).to_results()
            if len(values) != len(indices):
                raise RuntimeError("cpu-reference returned an unexpected number of fallback results")
            for lane, result in zip(indices, values):
                lane = int(lane)
                provenance = result.setdefault("provenance", {})
                execution = dict(reference.provenance())
                execution["fallback"] = {"lane_reason": list(reasons[lane]["reasons"]), "ran_on": "cpu-reference"}
                if reasons[lane].get("error"):
                    execution["fallback"]["engine_error"] = reasons[lane]["error"]
                execution["scenario_inputs"] = {
                    "burn_rate_factor": float(batch.arrays["burn_rate_factor"][lane])
                }
                execution["requested_backend"] = requested
                provenance["execution"] = execution
                results[lane] = result

        def run_feeder(index: int) -> None:
            backend = backend_instances[index]
            spec = specs[index]
            limit = _preferred_chunk_size(
                backend, spec, lane_count, chunk_size, len(specs), self.default_workers
            )
            engine_options = _backend_options(options, spec, self.default_workers)
            try:
                while True:
                    lanes = work.take(index, limit)
                    if not len(lanes):
                        return
                    started = time.perf_counter()
                    reasons: Dict[int, Dict[str, Any]] = {}
                    try:
                        outcome = backend.solve_burn(batch.select(lanes), engine_options)
                        values = outcome.to_results()
                        if len(values) != len(lanes):
                            raise RuntimeError(
                                f"backend {spec.name!r} returned {len(values)} results for {len(lanes)} lanes"
                            )
                    except Exception as exc:
                        values = [None] * len(lanes)
                        for lane in lanes:
                            reasons[int(lane)] = {
                                "reasons": ["engine_error"],
                                "error": {"type": type(exc).__name__, "message": str(exc)},
                            }
                    for lane, result in zip(lanes, values):
                        lane = int(lane)
                        if lane in reasons:
                            continue
                        if result is None:
                            reasons[lane] = {"reasons": ["integration_failed"]}
                        elif result.get("provenance", {}).get("execution", {}).get("step_overflow"):
                            reasons[lane] = {"reasons": ["step_overflow"]}
                        elif fallback_only[index]:
                            lane_features = sorted({
                                feature
                                for mapping in refused
                                for feature in mapping.get(lane, [])
                            })
                            detail = {"reasons": lane_features or ["no_selected_backend_supports_lane"]}
                            fallback_reasons[lane] = detail
                            execution = dict(backend.provenance())
                            execution["fallback"] = {
                                "lane_reason": list(detail["reasons"]), "ran_on": "cpu-reference"
                            }
                            execution["scenario_inputs"] = {
                                "burn_rate_factor": float(batch.arrays["burn_rate_factor"][lane])
                            }
                            execution["requested_backend"] = requested
                            result.setdefault("provenance", {})["execution"] = execution
                            results[lane] = result
                        else:
                            execution = _execution(result)
                            if spec.name == "cpu-reference":
                                execution.update(backend.provenance())
                            execution["requested_backend"] = requested
                            results[lane] = result
                    if reasons:
                        fallback_reasons.update(reasons)
                        retry_on_reference(sorted(reasons), reasons)
                    counts[index]["lanes"] += len(lanes)
                    counts[index]["chunks"] += 1
                    counts[index]["elapsed_s"] += time.perf_counter() - started
            except BaseException as exc:
                with error_lock:
                    errors.append(exc)

        with ThreadPoolExecutor(max_workers=len(specs), thread_name_prefix="solidpy-feeder") as pool:
            feeders = [pool.submit(run_feeder, i) for i in range(len(specs))]
            for feeder in feeders:
                feeder.result()
        if errors:
            raise errors[0]
        if any(result is None for result in results):
            missing = [i for i, result in enumerate(results) if result is None]
            raise RuntimeError(f"heterogeneous scheduler left lane(s) without a result: {missing}")

        engine_summary = []
        for spec, backend, count in zip(specs, backend_instances, counts):
            if count["chunks"]:
                engine_summary.append({
                    "backend": spec.name,
                    "device": getattr(backend, "device", spec.device or "cpu"),
                    "workers": spec.workers if spec.workers is not None else (
                        self.default_workers if spec.name == "cpu-reference" else None
                    ),
                    **count,
                })
        summary = {
            "requested_backend": requested,
            "effective_backend": "heterogeneous",
            "lanes": lane_count,
            "engines": engine_summary,
            "fallback_lanes": sorted(fallback_reasons),
            "fallback_reasons": {lane: detail["reasons"] for lane, detail in sorted(fallback_reasons.items())},
            "chunks": sum(engine["chunks"] for engine in engine_summary),
            "schedule": "shared_dynamic_queue",
        }
        return BatchResult(results, "heterogeneous", summary)


def execute_burn(batch, specs, options: SolveOptions, **kwargs) -> BatchResult:
    """Convenience function for one heterogeneous batch solve."""
    return HeterogeneousExecutor(specs, default_workers=options.workers).solve_burn(
        batch, options, requested=kwargs.pop("requested", specs), **kwargs
    )


__all__ = ["EngineSpec", "HeterogeneousExecutor", "execute_burn", "parse_engine_specs"]
