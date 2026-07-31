"""Artifacts: screenshots, timings, memory/CPU, diagnostics snapshots."""

from __future__ import annotations

import json
import os
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

from PySide6.QtWidgets import QWidget

try:
    import psutil
except ImportError:  # pragma: no cover
    psutil = None  # type: ignore


@dataclass
class ActionRecord:
    name: str
    started_at: float
    ended_at: float = 0.0
    ok: bool = True
    detail: str = ""
    screenshot: str = ""
    metrics: Dict[str, Any] = field(default_factory=dict)

    @property
    def duration_s(self) -> float:
        end = self.ended_at or time.time()
        return max(0.0, end - self.started_at)


@dataclass
class ResourceSample:
    ts: float
    rss_mb: float
    cpu_percent: float


class ArtifactCollector:
    """Collects customer-session evidence for the HTML report."""

    def __init__(self, out_dir: Path) -> None:
        self.out_dir = Path(out_dir)
        self.shot_dir = self.out_dir / "screenshots"
        self.shot_dir.mkdir(parents=True, exist_ok=True)
        self.actions: List[ActionRecord] = []
        self.resources: List[ResourceSample] = []
        self.failures: List[Dict[str, Any]] = []
        self.notes: List[str] = []
        self._proc = psutil.Process(os.getpid()) if psutil else None
        if self._proc is not None:
            self._proc.cpu_percent(interval=None)  # prime

    def begin(self, name: str, detail: str = "") -> ActionRecord:
        action = ActionRecord(name=name, started_at=time.time(), detail=detail)
        self.actions.append(action)
        self.sample_resources()
        return action

    def end(
        self,
        action: ActionRecord,
        *,
        ok: bool = True,
        detail: str = "",
        screenshot_widget: Optional[QWidget] = None,
        metrics: Optional[Dict[str, Any]] = None,
    ) -> ActionRecord:
        action.ended_at = time.time()
        action.ok = ok
        if detail:
            action.detail = detail
        if metrics:
            action.metrics.update(metrics)
        if screenshot_widget is not None:
            action.screenshot = self.screenshot(screenshot_widget, name=action.name)
        self.sample_resources()
        if not ok:
            self.failures.append(
                {
                    "action": action.name,
                    "detail": action.detail,
                    "duration_s": action.duration_s,
                    "screenshot": action.screenshot,
                }
            )
        return action

    def screenshot(self, widget: QWidget, *, name: str) -> str:
        safe = "".join(c if c.isalnum() or c in "-_" else "_" for c in name)[:80]
        stamp = time.strftime("%H%M%S")
        path = self.shot_dir / f"{stamp}_{safe}.png"
        pix = widget.grab()
        pix.save(str(path), "PNG")
        return str(path.relative_to(self.out_dir))

    def sample_resources(self) -> None:
        if self._proc is None:
            return
        try:
            rss = self._proc.memory_info().rss / (1024 * 1024)
            cpu = self._proc.cpu_percent(interval=None)
            self.resources.append(
                ResourceSample(ts=time.time(), rss_mb=round(rss, 2), cpu_percent=cpu)
            )
        except Exception as exc:  # pragma: no cover
            self.notes.append(f"resource sample failed: {exc}")

    def note(self, text: str) -> None:
        self.notes.append(text)

    def dump_json(self) -> Path:
        payload = {
            "actions": [asdict(a) for a in self.actions],
            "failures": self.failures,
            "resources": [asdict(r) for r in self.resources],
            "notes": self.notes,
        }
        path = self.out_dir / "session.json"
        path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        return path
