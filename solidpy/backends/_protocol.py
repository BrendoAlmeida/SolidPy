# -*- coding: utf-8 -*-
"""Backend protocol, capability description and the errors shared by every backend."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Mapping, Optional, Protocol, Tuple

#: Bumped on breaking changes of the :class:`Backend` protocol.
BACKEND_API_VERSION = 1

SUPPORTED = "supported"
PARTIAL = "partial"
UNSUPPORTED = "unsupported"
_LEVELS = (SUPPORTED, PARTIAL, UNSUPPORTED)
HISTORY_POLICY_TEMPLATES = ("metrics", "full", "decimated:N", "uniform:N")


def parse_history_policy(history: str) -> Tuple[str, Optional[int]]:
    """Return the policy kind and point count for a canonical history policy string."""
    if isinstance(history, str) and history in ("metrics", "full"):
        return history, None
    if isinstance(history, str):
        kind, separator, raw_count = history.partition(":")
        if separator and kind in ("decimated", "uniform") and raw_count.isdecimal():
            count = int(raw_count)
            if count >= 2:
                return kind, count
    raise ValueError(
        "history must be 'metrics', 'full', 'decimated:N' or 'uniform:N' with N an integer of at least 2, "
        f"got {history!r}"
    )


class UnsupportedLane(ValueError):
    """A lane needs features the chosen backend does not support (raised with ``strict=True``)."""


class BackendUnavailable(ImportError):
    """A backend was requested but cannot run here (library missing, no device, unknown name).

    It subclasses ``ImportError`` so callers can treat it like any other missing optional dependency;
    the message always says how to fix the situation.
    """


@dataclass(frozen=True)
class Capabilities:
    """What a backend can run for one lane.

    ``features`` maps a lane feature name to ``"supported"``, ``"partial"`` or ``"unsupported"``. A feature
    that is not listed is unsupported: the router never assumes support (design principle 3). ``services`` lists
    the optional Tier 1 services the backend implements besides ``solve_burn`` (currently ``"thermal_ablation"``,
    ``"structural_response"`` and ``"advanced_physics_proxies"``). ``history_policies`` may use
    ``"decimated:N"`` and ``"uniform:N"`` templates
    to advertise policies whose point count is chosen per solve.
    """

    features: Mapping[str, str] = field(default_factory=dict)
    dtypes: Tuple[str, ...] = ("float64",)
    history_policies: Tuple[str, ...] = ("metrics",)
    services: Tuple[str, ...] = ()

    def __post_init__(self):
        for name, level in self.features.items():
            if level not in _LEVELS:
                raise ValueError(f"capability {name!r} must be one of {_LEVELS}, got {level!r}")

    def level(self, feature: str) -> str:
        return self.features.get(feature, UNSUPPORTED)

    def supports(self, feature: str) -> bool:
        return self.level(feature) == SUPPORTED

    def provides(self, service: str) -> bool:
        return service in self.services

    def missing(self, features) -> List[str]:
        """Return the requested features this backend does not fully support, in request order."""
        return [name for name in features if not self.supports(name)]


def refused_lanes(batch, capabilities) -> Dict[int, List[str]]:
    """``{lane: [features the backend does not fully support]}`` for the lanes it cannot run."""
    return {lane: missing for lane, missing in enumerate(batch.unsupported(capabilities)) if missing}


def unsupported_lane_error(backend: str, refused: Mapping[int, List[str]]) -> UnsupportedLane:
    return UnsupportedLane(
        f"backend {backend!r} cannot run lane(s) "
        + "; ".join(f"{lane}: {', '.join(features)}" for lane, features in refused.items())
    )


@dataclass(frozen=True)
class SolveOptions:
    """Options of ``Backend.solve_burn`` that are not part of the problem itself.

    ``history`` is the history policy (``"metrics"``, ``"full"``, ``"decimated:N"`` or ``"uniform:N"``),
    ``workers`` the number of processes a CPU backend may use and ``max_steps`` the accepted points a batched lane may store before it is failed
    (``None``: the backend's default) and ``tiers`` the iteration caps of the capped tiers a batched backend runs
    before its uncapped one (``None``: ``batch.tiers.DEFAULT_TIERS``; ``()``: a single uncapped solve). A backend
    ignores an option that does not apply to it and says so in its docstring. CPU process pools use ``spawn``; scripts
    that call an API with ``workers > 1`` must do so under ``if __name__ == "__main__":``.
    """

    history: str = "metrics"
    workers: Optional[int] = None
    max_steps: Optional[int] = None
    tiers: Optional[Tuple[int, ...]] = None

    def __post_init__(self):
        if self.tiers is not None and (
            not isinstance(self.tiers, tuple)
            or any(isinstance(t, bool) or not isinstance(t, int) or t < 1 for t in self.tiers)
            or list(self.tiers) != sorted(set(self.tiers))
        ):
            raise ValueError(f"tiers must be a tuple of increasing positive integers or None, got {self.tiers!r}")
        if self.max_steps is not None and (
            isinstance(self.max_steps, bool) or not isinstance(self.max_steps, int) or self.max_steps < 2
        ):
            raise ValueError(f"max_steps must be an integer of at least 2 or None, got {self.max_steps!r}")
        parse_history_policy(self.history)
        if self.workers is not None and (
            isinstance(self.workers, bool) or not isinstance(self.workers, int) or self.workers < 1
        ):
            raise ValueError(f"workers must be a positive integer or None, got {self.workers!r}")


class Backend(Protocol):
    """An execution engine for batches of independent motors."""

    name: str
    api_version: int

    def capabilities(self) -> Capabilities: ...

    def devices(self) -> List[str]: ...

    def solve_burn(self, batch: Any, options: Any) -> Any: ...

    # Optional Tier 1 services, advertised in ``Capabilities.services``:
    #   def thermal_ablation(self, batch: ThermalBatch, options: SolveOptions) -> BatchResult
    #   def structural_response(self, geometry, chamber_pressure_pa, casing_material, ...) -> Mapping[str, Any]
    #   def advanced_physics_proxies(self, batch: AdvancedPhysicsBatch, options: SolveOptions) -> BatchResult

    def provenance(self) -> Dict[str, Any]: ...
