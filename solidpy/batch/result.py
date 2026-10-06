# -*- coding: utf-8 -*-
"""Result of solving a batch: one canonical result mapping per lane, in lane order."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Iterator, List, Tuple


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


@dataclass
class DeviceBatchResult:
    """A bounded stream of solved device chunks.

    Device arrays stay on their backend until a caller consumes a chunk. ``iter_device_batches``
    is for downstream accelerator kernels; ``iter_results`` materializes canonical host records
    a chunk at a time, and ``to_results`` is the explicit all-host compatibility boundary.
    Iteration is single pass so consumed device buffers can be released promptly.
    """

    lane_count: int
    backend: str
    execution: Dict[str, Any]
    _device_batches: Iterator[Tuple[Any, Dict[str, Any]]]
    _materialize: Callable[[Any, Dict[str, Any], str | None], Iterator[Dict[str, Any]]]

    def __len__(self) -> int:
        return self.lane_count

    def iter_device_batches(self) -> Iterator[Tuple[Any, Dict[str, Any]]]:
        """Yield ``(input_batch, device_outputs)`` once, without copying histories to the host."""
        yield from self._device_batches

    def materialize_device_batch(
        self, batch: Any, output: Dict[str, Any], *, history: str | None = None
    ) -> Iterator[Dict[str, Any]]:
        """Materialize one yielded device chunk; ``history`` may select a smaller host payload."""
        yield from self._materialize(batch, output, history)

    def iter_results(self, *, history: str | None = None) -> Iterator[Dict[str, Any]]:
        """Yield canonical lane mappings while materializing only the current sub-batch."""
        for batch, output in self._device_batches:
            yield from self._materialize(batch, output, history)

    def to_results(self) -> List[Dict[str, Any]]:
        """Materialize all canonical host mappings (potentially memory intensive)."""
        return list(self.iter_results())
