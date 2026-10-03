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

import numbers
from typing import Any, Dict, List, Optional

import numpy as np

from . import backends
from .backends import SolveOptions, UnsupportedLane
from .backends._protocol import refused_lanes, unsupported_lane_error
from .batch import ProblemBatch
from .batch.kernels.tables import evaluate as evaluate_table
from .batch.result import BatchResult

__all__ = ["ProblemBatch", "SolveOptions", "UnsupportedLane", "lane_cost", "simulate_burn"]

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
        rate = np.where(a["burn_rate_mode"] == 1.0, tabulated, rate)
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
