"""Thread-safe bridge from watchdog auto-index events to the Qt UI."""

from __future__ import annotations

from PySide6.QtCore import QObject, Signal


class AutoIndexNotifier(QObject):
    """Emit auto-index results on the Qt main thread."""

    catalog_updated = Signal(str, str)  # file_path, action: indexed|removed|failed

    def notify(self, file_path: str, action: str, success: bool) -> None:
        """Thread-safe: emit so MainWindow slots run on the Qt main thread."""
        if action == "skipped":
            return
        # Failures arrive with success=False — still surface them in the UI.
        if action == "failed":
            self.catalog_updated.emit(file_path, action)
            return
        if success and action in {"indexed", "removed"}:
            self.catalog_updated.emit(file_path, action)
