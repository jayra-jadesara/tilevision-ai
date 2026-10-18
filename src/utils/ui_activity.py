"""
Application-level UI activity logging for TileVision AI.

Installs a ``QApplication`` event filter for event types that exist in
PySide6 (Show / Close / ContextMenu / Drop / KeyPress / WindowStateChange /
MouseButtonRelease) plus signal hooks for actions, tabs, combos, and
stacked widgets. Sensitive fields (licence / password) are never logged as
typed text.

Meaningful actions can also be forwarded to the in-app Activity page via
an optional ``activity_recorder(activity_type, message)`` callback.
Navigation ("Opened Index" etc.) is owned by ``MainWindow._navigate`` —
this filter does not double-record it.
"""

from __future__ import annotations

import logging
import re
import time
import weakref
from typing import Callable, Optional, Set

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
    QStackedWidget,
    QTabBar,
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


class UiActivityFilter(QObject):
    """
    QApplication event filter + signal hooks for UI activity logging.

    PySide6 has no ``QEvent.Type.Triggered`` / ``CurrentChanged`` — those
    used to raise on every non-button event and were swallowed at DEBUG,
    leaving only button clicks logged. Actions / tabs / combos / stacks
    are wired via real Qt signals instead.
    """

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
        self._hooked_ids: Set[int] = set()

    def set_activity_recorder(self, recorder: ActivityRecorder) -> None:
        self._recorder = recorder

    def eventFilter(self, obj: QObject, event: QEvent) -> bool:  # noqa: N802
        try:
            self._handle(obj, event)
        except Exception:
            # Never break the UI for logging — one bad branch must not
            # kill the others (log at DEBUG only).
            logger.debug("ui activity filter error", exc_info=True)
        return False

    def _handle(self, obj: QObject, event: QEvent) -> None:
        et = event.type()

        # Install signal hooks when widgets appear / children are added.
        if et in (QEvent.Type.Show, QEvent.Type.ChildAdded, QEvent.Type.Polish):
            self._maybe_hook_signals(obj)
            if et == QEvent.Type.ChildAdded:
                child = event.child()  # type: ignore[attr-defined]
                if isinstance(child, QObject):
                    self._maybe_hook_signals(child)

        if et == QEvent.Type.MouseButtonRelease and isinstance(obj, QAbstractButton):
            if event.button() != Qt.MouseButton.LeftButton:  # type: ignore[attr-defined]
                return
            self._log_click(obj)
            return

        if et == QEvent.Type.Show and isinstance(obj, QMenu):
            title = (obj.title() or obj.objectName() or "menu").strip() or "menu"
            self._emit_log(f"[UI] menu open '{title}' (QMenu) in {self._owner_name(obj)}")
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
            self._emit_log(
                f"[UI] drop onto ({obj.__class__.__name__}) in {self._owner_name(obj)}"
            )
            self._maybe_record("search", "Dropped image onto search drop zone")
            return

    def _maybe_hook_signals(self, obj: QObject) -> None:
        """Connect once to signals that replace non-existent QEvent types."""
        oid = id(obj)
        if oid in self._hooked_ids:
            return

        hooked = False
        if isinstance(obj, QAction):
            obj.triggered.connect(self._on_action_triggered)
            hooked = True
        elif isinstance(obj, QTabWidget):
            obj.currentChanged.connect(
                lambda idx, w=weakref.ref(obj): self._on_tab_changed(w, idx)
            )
            hooked = True
        elif isinstance(obj, QTabBar):
            obj.currentChanged.connect(
                lambda idx, w=weakref.ref(obj): self._on_tab_bar_changed(w, idx)
            )
            hooked = True
        elif isinstance(obj, QComboBox):
            obj.activated.connect(
                lambda idx, w=weakref.ref(obj): self._on_combo_activated(w, idx)
            )
            hooked = True
        elif isinstance(obj, QStackedWidget):
            obj.currentChanged.connect(
                lambda idx, w=weakref.ref(obj): self._on_stack_changed(w, idx)
            )
            hooked = True

        if hooked:
            self._hooked_ids.add(oid)

        # Also walk immediate children so late-created actions are covered.
        if isinstance(obj, QWidget):
            for child in obj.findChildren(QAction):
                cid = id(child)
                if cid not in self._hooked_ids:
                    child.triggered.connect(self._on_action_triggered)
                    self._hooked_ids.add(cid)
            for child in obj.findChildren(QTabWidget):
                self._maybe_hook_signals(child)
            for child in obj.findChildren(QComboBox):
                self._maybe_hook_signals(child)
            for child in obj.findChildren(QStackedWidget):
                self._maybe_hook_signals(child)
            for child in obj.findChildren(QTabBar):
                self._maybe_hook_signals(child)

    def _on_action_triggered(self, checked: bool = False) -> None:  # noqa: ARG002
        action = self.sender()
        if isinstance(action, QAction):
            self._log_action(action)

    def _on_tab_changed(self, widget_ref: weakref.ref, index: int) -> None:
        widget = widget_ref()
        if not isinstance(widget, QTabWidget):
            return
        label = widget.tabText(index) if index >= 0 else "?"
        self._emit_log(
            f"[UI] tab '{label}' (QTabWidget) in {self._owner_name(widget)}"
        )
        self._maybe_record("navigation", f"Switched to tab '{label}'")

    def _on_tab_bar_changed(self, widget_ref: weakref.ref, index: int) -> None:
        widget = widget_ref()
        if not isinstance(widget, QTabBar):
            return
        label = widget.tabText(index) if index >= 0 else "?"
        self._emit_log(
            f"[UI] tab '{label}' (QTabBar) in {self._owner_name(widget)}"
        )

    def _on_combo_activated(self, widget_ref: weakref.ref, index: int) -> None:
        widget = widget_ref()
        if not isinstance(widget, QComboBox):
            return
        label = widget.itemText(index) if index >= 0 else widget.currentText()
        name = self._widget_label(widget)
        self._emit_log(
            f"[UI] combo '{name}' -> '{label}' (QComboBox) in {self._owner_name(widget)}"
        )
        self._maybe_record("search", f"Filter '{name}' set to '{label}'")

    def _on_stack_changed(self, widget_ref: weakref.ref, index: int) -> None:
        widget = widget_ref()
        if not isinstance(widget, QStackedWidget):
            return
        page = widget.widget(index)
        name = (
            (page.objectName() or page.__class__.__name__)
            if page is not None
            else str(index)
        )
        self._emit_log(
            f"[UI] stack page '{name}' (QStackedWidget) in {self._owner_name(widget)}"
        )
        # Do NOT record navigation here — MainWindow._navigate owns Activity
        # "Opened …" lines so they appear once.

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
        for attr in ("text", "accessibleName", "toolTip", "objectName", "currentText"):
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
        # Navigation ("Opened Index/Search/…") is recorded once by
        # MainWindow._navigate — skip here to avoid Activity page duplicates.
        if any(
            h in low
            for h in (
                "index",
                "search",
                "settings",
                "help",
                "activity",
                "history",
                "catalog",
                "pricing",
            )
        ):
            if "NavButton" in owner or owner == "MainWindow" or "nav" in low:
                return
        if "export" in low:
            self._maybe_record("export", f"Clicked '{label}'")
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
