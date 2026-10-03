# -*- coding: utf-8 -*-
"""Backend protocol, capability description and the errors shared by every backend."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Mapping, Protocol, Tuple

#: Bumped on breaking changes of the :class:`Backend` protocol.
BACKEND_API_VERSION = 1

SUPPORTED = "supported"
PARTIAL = "partial"
UNSUPPORTED = "unsupported"
_LEVELS = (SUPPORTED, PARTIAL, UNSUPPORTED)


class BackendUnavailable(ImportError):
    """A backend was requested but cannot run here (library missing, no device, unknown name).

    It subclasses ``ImportError`` so callers can treat it like any other missing optional dependency;
    the message always says how to fix the situation.
    """


@dataclass(frozen=True)
class Capabilities:
    """What a backend can run for one lane.

    ``features`` maps a lane feature name to ``"supported"``, ``"partial"`` or ``"unsupported"``. A feature
    that is not listed is unsupported: the router never assumes support (design principle 3).
    """

    features: Mapping[str, str] = field(default_factory=dict)
    dtypes: Tuple[str, ...] = ("float64",)
    history_policies: Tuple[str, ...] = ("metrics",)

    def __post_init__(self):
        for name, level in self.features.items():
            if level not in _LEVELS:
                raise ValueError(f"capability {name!r} must be one of {_LEVELS}, got {level!r}")

    def level(self, feature: str) -> str:
        return self.features.get(feature, UNSUPPORTED)

    def supports(self, feature: str) -> bool:
        return self.level(feature) == SUPPORTED

    def missing(self, features) -> List[str]:
        """Return the requested features this backend does not fully support, in request order."""
        return [name for name in features if not self.supports(name)]


class Backend(Protocol):
    """An execution engine for batches of independent motors."""

    name: str
    api_version: int

    def capabilities(self) -> Capabilities: ...

    def devices(self) -> List[str]: ...

    def solve_burn(self, batch: Any, options: Any) -> Any: ...

    def provenance(self) -> Dict[str, Any]: ...
