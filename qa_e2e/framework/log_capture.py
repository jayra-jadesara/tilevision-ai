"""Capture TileVision log records for stage assertions and HTML reports."""

from __future__ import annotations

import logging
import threading
import time
from dataclasses import dataclass, field
from typing import Iterable, List, Optional


@dataclass
class CapturedRecord:
    created: float
    level: str
    logger_name: str
    message: str


class LogCapture(logging.Handler):
    """Thread-safe in-memory log handler attached to the tilevision logger tree."""

    def __init__(self, level: int = logging.DEBUG) -> None:
        super().__init__(level=level)
        self._lock = threading.RLock()
        self.records: List[CapturedRecord] = []
        self.setFormatter(logging.Formatter("%(message)s"))

    def emit(self, record: logging.LogRecord) -> None:
        try:
            msg = self.format(record)
        except Exception:
            msg = record.getMessage()
        item = CapturedRecord(
            created=record.created,
            level=record.levelname,
            logger_name=record.name,
            message=msg,
        )
        with self._lock:
            self.records.append(item)

    def clear(self) -> None:
        with self._lock:
            self.records.clear()

    def messages(self, *, since: Optional[float] = None) -> List[str]:
        with self._lock:
            rows = list(self.records)
        if since is not None:
            rows = [r for r in rows if r.created >= since]
        return [r.message for r in rows]

    def contains(self, needle: str, *, since: Optional[float] = None) -> bool:
        return any(needle in m for m in self.messages(since=since))

    def wait_for(
        self,
        needle: str,
        *,
        timeout: float = 120.0,
        since: Optional[float] = None,
        poll: float = 0.2,
    ) -> bool:
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            if self.contains(needle, since=since):
                return True
            time.sleep(poll)
        return False

    def wait_for_all(
        self,
        needles: Iterable[str],
        *,
        timeout: float = 180.0,
        since: Optional[float] = None,
    ) -> List[str]:
        """Return list of needles that were NOT observed within timeout."""
        remaining = set(needles)
        deadline = time.monotonic() + timeout
        while remaining and time.monotonic() < deadline:
            msgs = self.messages(since=since)
            done = {n for n in remaining if any(n in m for m in msgs)}
            remaining -= done
            if remaining:
                time.sleep(0.25)
        return sorted(remaining)

    def attach(self, logger_name: str = "tilevision") -> None:
        logging.getLogger(logger_name).addHandler(self)

    def detach(self, logger_name: str = "tilevision") -> None:
        logging.getLogger(logger_name).removeHandler(self)


@dataclass
class StageTimeline:
    """Ordered search-stage observations for a single customer action."""

    started_at: float = field(default_factory=time.time)
    stages: List[str] = field(default_factory=list)

    def note(self, stage: str) -> None:
        self.stages.append(stage)
