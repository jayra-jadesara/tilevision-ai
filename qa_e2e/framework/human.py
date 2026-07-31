"""Simulate realistic customer timing and pointer behaviour."""

from __future__ import annotations

import random
import time
from typing import Optional, Tuple

from PySide6.QtCore import QPoint, Qt
from PySide6.QtGui import QCursor
from PySide6.QtTest import QTest
from PySide6.QtWidgets import QApplication, QWidget


class HumanSimulator:
    """
    Injects human-like pauses, jittered clicks, and pointer wander.

    All waits process the Qt event loop so the UI stays responsive.
    """

    def __init__(
        self,
        *,
        seed: Optional[int] = None,
        speed: float = 1.0,
        min_delay: float = 0.15,
        max_delay: float = 0.85,
    ) -> None:
        self._rng = random.Random(seed)
        self.speed = max(0.05, float(speed))
        self.min_delay = min_delay
        self.max_delay = max_delay

    def think(self, lo: Optional[float] = None, hi: Optional[float] = None) -> float:
        """Pause like a person reading the UI."""
        a = self.min_delay if lo is None else lo
        b = self.max_delay if hi is None else hi
        if b < a:
            a, b = b, a
        seconds = self._rng.uniform(a, b) / self.speed
        self._pump_wait(seconds)
        return seconds

    def wait(self, seconds: float) -> None:
        self._pump_wait(max(0.0, seconds) / self.speed)

    def wander(self, widget: QWidget, steps: int = 4) -> None:
        """Move the cursor around a widget before interacting."""
        if not widget.isVisible():
            return
        rect = widget.rect()
        for _ in range(max(1, steps)):
            x = self._rng.randint(2, max(3, rect.width() - 2))
            y = self._rng.randint(2, max(3, rect.height() - 2))
            global_pos = widget.mapToGlobal(QPoint(x, y))
            QCursor.setPos(global_pos)
            self._pump_wait(self._rng.uniform(0.03, 0.12) / self.speed)

    def click(self, widget: QWidget, *, button=Qt.MouseButton.LeftButton) -> None:
        """Jittered click with press/release timing."""
        widget.setFocus(Qt.FocusReason.MouseFocusReason)
        self.wander(widget, steps=self._rng.randint(2, 5))
        rect = widget.rect()
        ox = self._rng.randint(-3, 3)
        oy = self._rng.randint(-3, 3)
        x = max(1, min(rect.width() - 1, rect.center().x() + ox))
        y = max(1, min(rect.height() - 1, rect.center().y() + oy))
        # QTest.mouseClick keeps Qt input state consistent across PySide versions.
        QTest.mouseClick(widget, button, Qt.KeyboardModifier.NoModifier, QPoint(x, y))
        QApplication.processEvents()
        self.think(0.1, 0.45)

    def scroll(self, widget: QWidget, delta: Optional[int] = None) -> None:
        amount = delta if delta is not None else self._rng.choice([-180, -120, 120, 180])
        QTest.mouseClick(widget, Qt.MouseButton.LeftButton)
        # Wheel via QTest is limited; processEvents after a think approximates scroll pauses.
        self.think(0.2, 0.5)
        _ = amount  # reserved for platform-specific wheel injection

    def random_point_in(self, widget: QWidget) -> Tuple[int, int]:
        rect = widget.rect()
        return (
            self._rng.randint(1, max(2, rect.width() - 1)),
            self._rng.randint(1, max(2, rect.height() - 1)),
        )

    @staticmethod
    def _pump_wait(seconds: float) -> None:
        if seconds <= 0:
            return
        # QTest.qWait keeps the event loop alive (critical for workers/UI).
        QTest.qWait(int(seconds * 1000))
