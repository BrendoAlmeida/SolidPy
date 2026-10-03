# -*- coding: utf-8 -*-
"""Result of solving a batch: one canonical result mapping per lane, in lane order."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List


@dataclass
class BatchResult:
    """What ``Backend.solve_burn`` returns.

    ``to_results()`` gives the same canonical mappings (``history``, ``metrics``, ``status``,
    ``efficiencies``, ``provenance``) that ``BurnSimulation.result`` holds, one per lane.
    """

    results: List[Dict[str, Any]]
    backend: str
    execution: Dict[str, Any] = field(default_factory=dict)

    def __len__(self) -> int:
        return len(self.results)

    def to_results(self) -> List[Dict[str, Any]]:
        return list(self.results)
