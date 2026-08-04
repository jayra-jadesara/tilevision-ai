"""
Auto-dismiss modal Qt dialogs during headless / CI runs.

Production code legitimately uses QMessageBox for customer feedback. In
automation those modals block the event loop forever unless a human clicks
OK. This helper patches QMessageBox static APIs and periodically closes any
stray modal widgets — without changing production sources.
"""

from __future__ import annotations

import logging
from typing import Any, Callable, List, Optional

from PySide6.QtCore import QTimer
from PySide6.QtWidgets import QApplication, QDialog, QMessageBox, QWidget

logger = logging.getLogger("tilevision.qa_e2e.dialogs")

_PATCHED = False
_SEEN: List[str] = []


def seen_dialogs() -> List[str]:
    return list(_SEEN)


def install_dialog_auto_dismiss(app: QApplication) -> None:
    """Patch QMessageBox + poll for open modal dialogs."""
    global _PATCHED
    if _PATCHED:
        return
    _PATCHED = True

    def _record(kind: str, title: Any, text: Any) -> None:
        entry = f"{kind}: {title} — {text}"
        _SEEN.append(entry)
        logger.info("[QA] Auto-dismissed dialog: %s", entry)

    def _wrap(kind: str, default_ret: Any) -> Callable:
        def _fn(parent: Optional[QWidget], title: str, text: str, *args, **kwargs):
            _record(kind, title, text)
            return default_ret

        return _fn

    QMessageBox.information = staticmethod(  # type: ignore[method-assign]
        _wrap("information", QMessageBox.StandardButton.Ok)
    )
    QMessageBox.warning = staticmethod(  # type: ignore[method-assign]
        _wrap("warning", QMessageBox.StandardButton.Ok)
    )
    QMessageBox.critical = staticmethod(  # type: ignore[method-assign]
        _wrap("critical", QMessageBox.StandardButton.Ok)
    )
    QMessageBox.question = staticmethod(  # type: ignore[method-assign]
        _wrap("question", QMessageBox.StandardButton.Yes)
    )

    # Production uses themed message_box helpers (not native QMessageBox).
    try:
        from src.presentation.dialogs import message_box as themed_mb

        themed_mb.information = _wrap("information", QMessageBox.StandardButton.Ok)  # type: ignore[method-assign]
        themed_mb.warning = _wrap("warning", QMessageBox.StandardButton.Ok)  # type: ignore[method-assign]
        themed_mb.critical = _wrap("critical", QMessageBox.StandardButton.Ok)  # type: ignore[method-assign]
        themed_mb.question = _wrap("question", QMessageBox.StandardButton.Yes)  # type: ignore[method-assign]
    except Exception as exc:
        logger.warning("Could not patch themed message_box for QA: %s", exc)

    def _close_stray_modals() -> None:
        for widget in list(app.topLevelWidgets()):
            if not isinstance(widget, QDialog):
                continue
            if not widget.isModal() or not widget.isVisible():
                continue
            title = widget.windowTitle() or widget.__class__.__name__
            _SEEN.append(f"close-modal: {title}")
            logger.info("[QA] Closing stray modal: %s", title)
            try:
                widget.done(QDialog.DialogCode.Accepted)
            except Exception:
                try:
                    widget.reject()
                except Exception:
                    widget.close()
        QApplication.processEvents()

    timer = QTimer(app)
    timer.setInterval(400)
    timer.timeout.connect(_close_stray_modals)
    timer.start()
    # Keep a reference on the app so GC does not stop the timer.
    app._qa_dialog_timer = timer  # type: ignore[attr-defined]
