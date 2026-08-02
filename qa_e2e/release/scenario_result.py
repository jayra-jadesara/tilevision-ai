"""Typed result for one release scenario."""

from __future__ import annotations

import time
import traceback
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class ScenarioResult:
    id: str
    name: str
    ok: bool
    started_at: float
    ended_at: float = 0.0
    detail: str = ""
    error: str = ""
    stacktrace: str = ""
    screenshot: str = ""
    metrics: Dict[str, Any] = field(default_factory=dict)
    checks: Dict[str, bool] = field(default_factory=dict)

    @property
    def duration_s(self) -> float:
        end = self.ended_at or time.time()
        return max(0.0, end - self.started_at)

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d["duration_s"] = self.duration_s
        return d


def fail_result(
    scenario_id: str,
    name: str,
    started_at: float,
    exc: BaseException,
    *,
    screenshot: str = "",
    metrics: Optional[Dict[str, Any]] = None,
) -> ScenarioResult:
    return ScenarioResult(
        id=scenario_id,
        name=name,
        ok=False,
        started_at=started_at,
        ended_at=time.time(),
        detail=str(exc),
        error=f"{exc.__class__.__name__}: {exc}",
        stacktrace=traceback.format_exc(),
        screenshot=screenshot,
        metrics=metrics or {},
    )
