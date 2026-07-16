"""
Pipeline timing helpers for TileVision AI.

Provides structured stage timing logs for indexing and search profiling.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Dict

logger = logging.getLogger("tilevision.timing")


@dataclass
class StageTimings:
    """Accumulated stage durations in seconds."""

    stages: Dict[str, float] = field(default_factory=dict)

    def record(self, stage: str, elapsed: float) -> None:
        self.stages[stage] = self.stages.get(stage, 0.0) + elapsed

    def total(self) -> float:
        return sum(self.stages.values())


class PipelineTimer:
    """Context manager / helper for measuring pipeline stages."""

    def __init__(self, label: str) -> None:
        self._label = label
        self._timings = StageTimings()
        self._wall_start = time.perf_counter()

    @property
    def timings(self) -> StageTimings:
        return self._timings

    def measure(self, stage: str):
        return _StageMeasure(self, stage)

    def log_summary(
        self,
        extra_stages: Dict[str, float] | None = None,
        log: logging.Logger | None = None,
    ) -> None:
        log = log or logger
        lines = [self._label, "-" * len(self._label)]
        for stage, elapsed in self._timings.stages.items():
            lines.append(f"{stage}={elapsed:.3f}s")
        if extra_stages:
            for stage, elapsed in extra_stages.items():
                lines.append(f"{stage}={elapsed:.3f}s")
        wall_total = time.perf_counter() - self._wall_start
        lines.append(f"total={wall_total:.3f}s")
        log.info("\n".join(lines))


class _StageMeasure:
    def __init__(self, timer: PipelineTimer, stage: str) -> None:
        self._timer = timer
        self._stage = stage
        self._start = 0.0

    def __enter__(self):
        self._start = time.perf_counter()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        elapsed = time.perf_counter() - self._start
        self._timer.timings.record(self._stage, elapsed)
        return False
