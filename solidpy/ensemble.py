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

from typing import Any, Dict, List, Optional

import numpy as np

from . import backends
from .backends import SolveOptions, UnsupportedLane
from .batch import ProblemBatch
from .batch.result import BatchResult

__all__ = ["ProblemBatch", "SolveOptions", "UnsupportedLane", "lane_cost", "simulate_burn"]

#: Backends ``backend="auto"`` may pick, best first, and the smallest batch worth sending to one.
AUTO_ACCELERATORS = ("jax",)
AUTO_MIN_LANES = 2048


def lane_cost(batch: ProblemBatch) -> np.ndarray:
    """A cheap estimate of the work of each lane, used only to group lanes of similar cost.

    Lockstep batches run as many iterations as their slowest lane, so lanes with a similar number of steps
    belong together. The estimate is the burn time (deepest grain over the equilibrium burn rate) in units of
    the largest step, plus a term per grain for the restart at each burnout. Lanes the kernels cannot
    describe (NaN burn rate) sort last.
    """
    a = batch.arrays
    cstar = np.sqrt(a["gas_constant"] * a["source_temperature"] / a["gamma"]) * (
        (a["gamma"] + 1.0) / 2.0
    ) ** ((a["gamma"] + 1.0) / (2.0 * (a["gamma"] - 1.0)))
    burning_area = (np.where(a["grain_valid"], np.pi * (a["outer_radius"] ** 2 - a["inner_radius0"] ** 2), 0.0)).sum(axis=1)
    kn = np.maximum(burning_area, 1e-12) / a["throat_area"]
    n = a["burn_rate_n"]
    with np.errstate(all="ignore"):
        pressure = (a["density"] * a["burn_rate_a"] * (1e-6) ** n * kn * cstar / (1000.0 * a["discharge_coefficient"])) ** (
            1.0 / (1.0 - n)
        )
        rate = a["burn_rate_a"] * np.maximum(pressure * 1e-6, 1e-12) ** n / 1000.0
        depth = np.where(a["grain_valid"], a["burnout_depth"], 0.0).max(axis=1)
        cost = depth / rate / a["max_step_size"] + 40.0 * a["n_valid_grains"]
    return np.where(np.isfinite(cost), cost, np.finfo(float).max)


def _auto_backend(batch: ProblemBatch) -> str:
    """An accelerator when one is usable and the batch is large enough to amortise it, else the reference."""
    if len(batch) >= AUTO_MIN_LANES:
        status = backends.available()
        for name in AUTO_ACCELERATORS:
            if status.get(name) == "ok":
                try:
                    if any(not d.startswith("cpu") for d in backends.get_backend(name).devices()):
                        return name
                except ImportError:
                    continue
    return "cpu-reference"


def _chunks(order: np.ndarray, chunk_size: Optional[int]) -> List[np.ndarray]:
    if not chunk_size or chunk_size >= len(order):
        return [order]
    return [order[i : i + chunk_size] for i in range(0, len(order), chunk_size)]


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
    cost into the same chunk, which matters when lanes need very different numbers of steps. ``tiers`` are the
    iteration caps a batched backend runs before its uncapped tier (``None``: ``batch.tiers.DEFAULT_TIERS``,
    ``()``: one uncapped solve); see ``solidpy.batch.tiers``.
    """
    if not isinstance(batch, ProblemBatch):
        raise TypeError("simulate_burn needs a ProblemBatch; build one with ProblemBatch.from_objects")
    if backend == "auto":
        backend = _auto_backend(batch)
    if backend is None:
        backend, selected_device = backends.current_backend()
        device = device if device is not None else selected_device
    chosen = backends.get_backend(backend, device)
    options = SolveOptions(history=history, workers=workers, max_steps=max_steps, tiers=tiers)

    missing = batch.unsupported(chosen.capabilities())
    refused = {lane: features for lane, features in enumerate(missing) if features}
    if refused and strict:
        raise UnsupportedLane(
            f"backend {backend!r} cannot run lane(s) "
            + "; ".join(f"{lane}: {', '.join(features)}" for lane, features in refused.items())
        )

    results: List[Optional[Dict[str, Any]]] = [None] * len(batch)
    supported = np.asarray([lane for lane in range(len(batch)) if lane not in refused], dtype=int)
    chunks: List[np.ndarray] = []
    tier_log: List[Any] = []
    if len(supported):
        order = supported[np.argsort(lane_cost(batch.select(supported)), kind="stable")] if sort else supported
        chunks = _chunks(order, chunk_size)
        for chunk in chunks:
            outcome = chosen.solve_burn(batch.select(chunk), options)
            tier_log.append(outcome.execution.get("tiers"))
            for lane, result in zip(chunk, outcome.to_results()):
                results[int(lane)] = result
    if refused:
        reference = backends.get_backend("cpu-reference")
        lanes = np.asarray(sorted(refused), dtype=int)
        solved = reference.solve_burn(batch.select(lanes), options).to_results()
        for lane, result in zip(lanes, solved):
            execution = dict(reference.provenance())
            execution["fallback"] = {"lane_reason": list(refused[int(lane)]), "ran_on": "cpu-reference"}
            result["provenance"]["execution"] = execution
            results[int(lane)] = result
    if backend != "cpu-reference":
        for result in results:
            result["provenance"]["execution"]["requested_backend"] = backend

    summary = {"requested_backend": backend, "lanes": len(batch), "fallback_lanes": sorted(int(i) for i in refused),
               "chunks": len(chunks), "tiers": tier_log}
    return BatchResult(results, backend, summary)
