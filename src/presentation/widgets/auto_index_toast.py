"""Transient top-right toast for auto-index activity (MainWindow-level)."""

from __future__ import annotations

from typing import Optional

from PySide6.QtCore import (
    QEasingCurve,
    QPropertyAnimation,
    QTimer,
    Qt,
)
from PySide6.QtWidgets import QGraphicsOpacityEffect, QLabel, QWidget

from src.theme.theme_manager import get_palette

_TOAST_MS = 4000
_FADE_MS = 500
_MARGIN = 16


class AutoIndexToast(QLabel):
    """
    Non-blocking, auto-dismissing toast anchored to the top-right of its parent.

    Styled via ``get_palette``; does not steal focus or block input.
    """

    def __init__(self, parent: QWidget, *, theme: str = "dark") -> None:
        super().__init__(parent)
        self._theme = theme
        self._fade_anim: Optional[QPropertyAnimation] = None
        self._dismiss_timer = QTimer(self)
        self._dismiss_timer.setSingleShot(True)
        self._dismiss_timer.timeout.connect(self._start_fade)

        self.setObjectName("AutoIndexToast")
        self.setWordWrap(True)
        self.setAlignment(Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter)
        self.setAttribute(Qt.WidgetAttribute.WA_TransparentForMouseEvents, True)
        self.setAttribute(Qt.WidgetAttribute.WA_ShowWithoutActivating, True)
        self.hide()
        self._apply_styles()

    def set_theme(self, theme: str) -> None:
        self._theme = theme
        self._apply_styles()

    def show_message(self, message: str) -> None:
        """Show ``message`` at top-right; auto-dismiss after a few seconds."""
        if self._fade_anim is not None:
            self._fade_anim.stop()
            self._fade_anim.deleteLater()
            self._fade_anim = None
        self._dismiss_timer.stop()

        self.setText(message)
        self.adjustSize()
        # Cap width so long filenames wrap instead of spanning the window.
        max_w = min(360, max(200, (self.parentWidget().width() // 3) if self.parentWidget() else 360))
        self.setFixedWidth(max_w)
        self.adjustSize()
        self._reposition()
        self.setGraphicsEffect(None)
        self.show()
        self.raise_()
        self._dismiss_timer.start(_TOAST_MS)

    def _reposition(self) -> None:
        parent = self.parentWidget()
        if parent is None:
            return
        x = parent.width() - self.width() - _MARGIN
        y = _MARGIN
        self.move(max(_MARGIN, x), y)

    def reposition_if_visible(self) -> None:
        if self.isVisible():
            self._reposition()

    def _start_fade(self) -> None:
        effect = QGraphicsOpacityEffect(self)
        self.setGraphicsEffect(effect)
        effect.setOpacity(1.0)
        anim = QPropertyAnimation(effect, b"opacity", self)
        anim.setDuration(_FADE_MS)
        anim.setStartValue(1.0)
        anim.setEndValue(0.0)
        anim.setEasingCurve(QEasingCurve.Type.OutCubic)

        def _hide() -> None:
            self.hide()
            self.setGraphicsEffect(None)
            self._fade_anim = None

        anim.finished.connect(_hide)
        self._fade_anim = anim
        anim.start()

    def _apply_styles(self) -> None:
        p = get_palette(self._theme)
        self.setStyleSheet(
            f"""
            #AutoIndexToast {{
                color: {p['text_primary']};
                background-color: {p['highlight_bg']};
                border: 1px solid {p['accent_soft']};
                border-radius: 8px;
                padding: 10px 14px;
                font-size: 12px;
                font-weight: 600;
            }}
            """
        )
