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
        stages = dict(self._timings.stages)
        if extra_stages:
            for stage, elapsed in extra_stages.items():
                stages[stage] = stages.get(stage, 0.0) + elapsed

        # Stable, console-friendly profile block (milliseconds).
        # Matches the required Search pipeline report layout.
        display_order = [
            ("image_load", "Image Load"),
            ("crop", "Crop"),
            ("preprocessing", "Crop"),  # alias when crop not split out
            ("embedding", "Embedding"),
            ("dinov2", "Embedding"),
            ("descriptors", "Descriptors"),
            ("feature_extract", "Feature Extract"),
            ("faiss", "FAISS"),
            ("metadata", "SQLite"),
            ("database", "SQLite"),
            ("rerank", "Rerank"),
            ("reranking", "Rerank"),
            ("thumbnail", "Thumbnail"),
        ]
        lines = [f"=== {self._label} ==="]
        printed: set[str] = set()
        seen_labels: set[str] = set()
        for key, label in display_order:
            if key not in stages:
                continue
            if label in seen_labels:
                # Prefer the first matching key for a display label.
                printed.add(key)
                continue
            ms = stages[key] * 1000.0
            lines.append(f"{label:.<22}{ms:>8.1f} ms")
            printed.add(key)
            seen_labels.add(label)
        for key, elapsed in stages.items():
            if key in printed:
                continue
            lines.append(f"{key:.<22}{elapsed * 1000.0:>8.1f} ms")
        wall_total = time.perf_counter() - self._wall_start
        lines.append(f"{'TOTAL':.<22}{wall_total * 1000.0:>8.1f} ms")
        block = "\n".join(lines)
        log.info("\n%s", block)
        # Also print so vendors see profiling without digging logs.
        print(block, flush=True)


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
