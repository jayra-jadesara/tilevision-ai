"""
Application-level UI activity logging for TileVision AI.

Installs a ``QApplication`` event filter that writes INFO lines to logger
``tilevision.ui`` for meaningful user actions (clicks, menu opens, tab
switches, dialogs, shortcuts). Sensitive fields (licence / password) are
never logged as typed text.

Meaningful actions can also be forwarded to the in-app Activity page via
an optional ``activity_recorder(activity_type, message)`` callback.
"""

from __future__ import annotations

import logging
import re
import time
from typing import Callable, Optional

from PySide6.QtCore import QEvent, QObject, Qt
from PySide6.QtGui import QAction, QKeySequence
from PySide6.QtWidgets import (
    QAbstractButton,
    QApplication,
    QCheckBox,
    QComboBox,
    QDialog,
    QLineEdit,
    QMenu,
    QRadioButton,
    QTabWidget,
    QTableWidget,
    QToolButton,
    QWidget,
)

logger = logging.getLogger("tilevision.ui")

ActivityRecorder = Optional[Callable[[str, str], None]]

# Collapse identical events within this window (ms).
_DEDUP_MS = 300

# Object names / accessible names that must never leak typed text.
_SENSITIVE_NAME_RE = re.compile(
    r"(license|licence|password|passwd|secret|api[_-]?key|token)",
    re.IGNORECASE,
)

_MEANINGFUL_BUTTON_HINTS = (
    "export",
    "rebuild",
    "update",
    "index",
    "search",
    "browse",
    "start",
    "cancel",
    "pause",
    "resume",
    "crop",
    "auto",
    "precise",
    "manual",
    "duplicate",
    "help",
    "settings",
    "theme",
    "preview",
    "reveal",
    "copy",
    "activate",
    "check for update",
)


class UiActivityFilter(QObject):
    """QApplication event filter — logs UI actions without pixel coordinates."""

    def __init__(
        self,
        *,
        activity_recorder: ActivityRecorder = None,
        parent: Optional[QObject] = None,
    ) -> None:
        super().__init__(parent)
        self._recorder = activity_recorder
        self._last_key: Optional[str] = None
        self._last_mono: float = 0.0

    def set_activity_recorder(self, recorder: ActivityRecorder) -> None:
        self._recorder = recorder

    def eventFilter(self, obj: QObject, event: QEvent) -> bool:  # noqa: N802
        try:
            self._handle(obj, event)
        except Exception:
            # Never break the UI for logging.
            logger.debug("ui activity filter error", exc_info=True)
        return False

    def _handle(self, obj: QObject, event: QEvent) -> None:
        et = event.type()

        if et == QEvent.Type.MouseButtonRelease and isinstance(obj, QAbstractButton):
            if event.button() != Qt.MouseButton.LeftButton:  # type: ignore[attr-defined]
                return
            self._log_click(obj)
            return

        if et == QEvent.Type.Triggered and isinstance(obj, QAction):
            self._log_action(obj)
            return

        if et == QEvent.Type.Show and isinstance(obj, QMenu):
            title = (obj.title() or obj.objectName() or "menu").strip() or "menu"
            self._emit_log(f"[UI] menu open '{title}' (QMenu) in {self._owner_name(obj)}")
            return

        if et == QEvent.Type.CurrentChanged and isinstance(obj, QTabWidget):
            idx = obj.currentIndex()
            label = obj.tabText(idx) if idx >= 0 else "?"
            self._emit_log(
                f"[UI] tab '{label}' (QTabWidget) in {self._owner_name(obj)}"
            )
            self._maybe_record("navigation", f"Switched to tab '{label}'")
            return

        if et in (QEvent.Type.FocusIn,) and isinstance(obj, QComboBox):
            # Combo activation is noisy; log on activated via signal hookups
            # is better — here log selection change via QEvent.Type if needed.
            return

        if et == QEvent.Type.MouseButtonDblClick and isinstance(obj, QTableWidget):
            self._emit_log(
                f"[UI] double-click row (QTableWidget) in {self._owner_name(obj)}"
            )
            return

        if et == QEvent.Type.ContextMenu and isinstance(obj, QWidget):
            self._emit_log(
                f"[UI] context-menu ({obj.__class__.__name__}) in {self._owner_name(obj)}"
            )
            return

        if et == QEvent.Type.Show and isinstance(obj, QDialog):
            title = obj.windowTitle() or obj.objectName() or obj.__class__.__name__
            self._emit_log(f"[UI] dialog open '{title}' ({obj.__class__.__name__})")
            return

        if et == QEvent.Type.Close and isinstance(obj, QDialog):
            title = obj.windowTitle() or obj.objectName() or obj.__class__.__name__
            self._emit_log(f"[UI] dialog close '{title}' ({obj.__class__.__name__})")
            return

        if et == QEvent.Type.WindowStateChange and isinstance(obj, QWidget):
            if obj.isWindow():
                state = "minimised" if obj.isMinimized() else "restored"
                self._emit_log(f"[UI] window {state} ({obj.__class__.__name__})")
            return

        if et == QEvent.Type.Close and isinstance(obj, QWidget) and obj.isWindow():
            self._emit_log(f"[UI] window close ({obj.__class__.__name__})")
            return

        if et == QEvent.Type.KeyPress:
            self._log_shortcut(obj, event)
            return

        if et == QEvent.Type.Drop and isinstance(obj, QWidget):
            self._emit_log(f"[UI] drop onto ({obj.__class__.__name__}) in {self._owner_name(obj)}")
            self._maybe_record("search", "Dropped image onto search drop zone")
            return

        # Checkbox / radio toggles
        if et == QEvent.Type.MouseButtonRelease and isinstance(
            obj, (QCheckBox, QRadioButton)
        ):
            # Also handled as QAbstractButton above — skip duplicate.
            return

    def _log_click(self, button: QAbstractButton) -> None:
        if isinstance(button, QCheckBox):
            state = "checked" if button.isChecked() else "unchecked"
            label = self._widget_label(button)
            self._emit_log(
                f"[UI] checkbox '{label}' {state} (QCheckBox) in {self._owner_name(button)}"
            )
            self._maybe_record_filter_or_setting(label)
            return
        if isinstance(button, QRadioButton):
            label = self._widget_label(button)
            self._emit_log(
                f"[UI] radio '{label}' (QRadioButton) in {self._owner_name(button)}"
            )
            return

        label = self._widget_label(button)
        cls = button.__class__.__name__
        owner = self._owner_name(button)
        self._emit_log(f"[UI] click '{label}' ({cls}) in {owner}")
        self._maybe_record_click(label, owner)

    def _log_action(self, action: QAction) -> None:
        label = (action.text() or action.objectName() or "action").replace("&", "").strip()
        self._emit_log(f"[UI] action '{label}' (QAction)")
        self._maybe_record_click(label, "QAction")

    def _log_shortcut(self, obj: QObject, event: QEvent) -> None:
        # Never log text from sensitive line edits.
        if isinstance(obj, QLineEdit) and self._is_sensitive_widget(obj):
            return
        try:
            key = event.key()  # type: ignore[attr-defined]
            mods = event.modifiers()  # type: ignore[attr-defined]
        except Exception:
            return
        # Only log chord shortcuts (Ctrl/Alt/Meta), not typing.
        if not (
            mods
            & (
                Qt.KeyboardModifier.ControlModifier
                | Qt.KeyboardModifier.AltModifier
                | Qt.KeyboardModifier.MetaModifier
            )
        ):
            return
        seq = QKeySequence(mods | Qt.Key(key)).toString()
        if not seq:
            return
        self._emit_log(f"[UI] shortcut '{seq}' in {self._owner_name(obj)}")

    def _widget_label(self, widget: QWidget) -> str:
        for attr in ("text", "accessibleName", "toolTip", "objectName"):
            getter = getattr(widget, attr, None)
            if callable(getter):
                try:
                    value = getter()
                except Exception:
                    value = None
                if value and str(value).strip():
                    text = str(value).replace("&", "").strip()
                    if self._is_sensitive_widget(widget):
                        return "(masked)"
                    return text[:80]
        return widget.__class__.__name__

    def _is_sensitive_widget(self, widget: QWidget) -> bool:
        echo = getattr(widget, "echoMode", None)
        if callable(echo):
            try:
                if echo() == QLineEdit.EchoMode.Password:
                    return True
            except Exception:
                pass
        for name in (
            widget.objectName() or "",
            widget.accessibleName() if hasattr(widget, "accessibleName") else "",
            getattr(widget, "placeholderText", lambda: "")(),
        ):
            if name and _SENSITIVE_NAME_RE.search(str(name)):
                return True
        return False

    def _owner_name(self, obj: QObject) -> str:
        w = obj if isinstance(obj, QWidget) else None
        cur = w
        while cur is not None:
            name = cur.objectName() or ""
            if name and name not in {"qt_scrollarea_viewport", "qt_scrollarea_hcontainer"}:
                # Prefer known view objectNames
                if name.endswith("View") or name in {
                    "MainWindow",
                    "NavButton",
                    "PrimaryButton",
                    "ResultsTable",
                }:
                    return name
            cls = cur.__class__.__name__
            if cls.endswith("View") or cls in {"MainWindow", "SettingsView"}:
                return cls
            cur = cur.parentWidget() if hasattr(cur, "parentWidget") else None
        return obj.__class__.__name__

    def _emit_log(self, message: str) -> None:
        now = time.monotonic()
        key = message
        if key == self._last_key and (now - self._last_mono) * 1000.0 < _DEDUP_MS:
            return
        self._last_key = key
        self._last_mono = now
        logger.info(message)

    def _maybe_record(self, activity_type: str, message: str) -> None:
        if self._recorder is None:
            return
        try:
            self._recorder(activity_type, message)
        except Exception:
            logger.debug("activity recorder failed", exc_info=True)

    def _maybe_record_click(self, label: str, owner: str) -> None:
        low = label.casefold()
        if any(h in low for h in ("index", "search", "settings", "help", "activity", "history", "catalog", "pricing")):
            if "NavButton" in owner or owner == "MainWindow" or "nav" in low:
                self._maybe_record("navigation", f"Opened {label}")
                return
        if "export" in low:
            self._maybe_record("export", f"Clicked '{label}'")
            return
        if "help" in low:
            self._maybe_record("navigation", "Opened Help")
            return
        if any(h in low for h in ("rebuild", "update search", "full rebuild", "start indexing")):
            self._maybe_record("index", f"Clicked '{label}'")
            return
        if any(h in low for h in ("auto crop", "precise", "manual crop", "crop")):
            self._maybe_record("search", f"Crop action '{label}'")
            return
        if "duplicate" in low:
            self._maybe_record("duplicate_scan", f"Opened '{label}'")
            return
        if "theme" in low or "dark" in low or "light" in low:
            self._maybe_record("theme", f"Theme control '{label}'")
            return
        if "preview" in low or "reveal" in low or "copy" in low:
            self._maybe_record("search", f"Result action '{label}'")
            return
        if "licence" in low or "license" in low or "activate" in low:
            self._maybe_record("license", f"Clicked '{label}'")
            return

    def _maybe_record_filter_or_setting(self, label: str) -> None:
        low = label.casefold()
        if "filter" in low or any(
            x in low for x in ("brand", "category", "color", "size")
        ):
            self._maybe_record("search", f"Filter '{label}' toggled")
        elif "theme" in low:
            self._maybe_record("theme", f"Theme '{label}'")
        else:
            self._maybe_record("settings", f"Setting '{label}' changed")


_installed_filter: Optional[UiActivityFilter] = None


def install_ui_activity_logging(
    app: QApplication,
    *,
    activity_recorder: ActivityRecorder = None,
) -> UiActivityFilter:
    """Install (or update) the process-wide UI activity event filter."""
    global _installed_filter
    if _installed_filter is None:
        _installed_filter = UiActivityFilter(activity_recorder=activity_recorder, parent=app)
        app.installEventFilter(_installed_filter)
        logger.info("UI activity logging installed (logger=tilevision.ui)")
    else:
        _installed_filter.set_activity_recorder(activity_recorder)
    return _installed_filter


def mask_if_sensitive(field_name: str, value: str) -> str:
    """Return ``***`` when ``field_name`` looks like a licence/password field."""
    if field_name and _SENSITIVE_NAME_RE.search(field_name):
        return "***"
    return value
