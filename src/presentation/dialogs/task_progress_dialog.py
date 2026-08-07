"""Themed, screen-centered progress dialog for long-running tasks."""

from __future__ import annotations

from PySide6.QtCore import Qt
from PySide6.QtGui import QGuiApplication
from PySide6.QtWidgets import QDialog, QLabel, QProgressBar, QVBoxLayout

from src.theme.theme_manager import get_dialog_qss


def center_on_screen(widget) -> None:
    """Place *widget* in the center of the primary screen's available area."""
    screen = QGuiApplication.primaryScreen()
    if screen is None:
        return
    frame = widget.frameGeometry()
    frame.moveCenter(screen.availableGeometry().center())
    widget.move(frame.topLeft())


class TaskProgressDialog(QDialog):
    """Modal progress window styled like other TileVision dialogs."""

    def __init__(
        self,
        *,
        title: str,
        message: str,
        theme: str = "light",
        parent=None,
    ) -> None:
        super().__init__(parent)
        self._theme = theme if theme in ("light", "dark") else "light"

        self.setWindowTitle(title)
        self.setWindowModality(Qt.WindowModality.ApplicationModal)
        self.setWindowFlag(Qt.WindowType.WindowCloseButtonHint, False)
        self.setMinimumWidth(460)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(20, 18, 20, 18)
        layout.setSpacing(12)

        self._label = QLabel(message)
        self._label.setObjectName("DialogStatus")
        self._label.setWordWrap(True)
        layout.addWidget(self._label)

        self._progress = QProgressBar()
        self._progress.setObjectName("DialogProgressBar")
        self._progress.setRange(0, 1)
        self._progress.setValue(0)
        self._progress.setTextVisible(True)
        self._progress.setFormat("%p%")
        self._progress.setFixedHeight(22)
        layout.addWidget(self._progress)

        self.setStyleSheet(get_dialog_qss(self._theme))

    def set_message(self, message: str) -> None:
        self._label.setText(message)

    def set_progress(self, value: int, maximum: int) -> None:
        if maximum > 0 and self._progress.maximum() != maximum:
            self._progress.setRange(0, maximum)
        self._progress.setValue(min(value, max(maximum, 1)))

    @property
    def maximum(self) -> int:
        return self._progress.maximum()

    def show_centered(self) -> None:
        self.show()
        self.raise_()
        self.activateWindow()
        center_on_screen(self)

    def set_theme(self, theme: str) -> None:
        self._theme = theme if theme in ("light", "dark") else self._theme
        self.setStyleSheet(get_dialog_qss(self._theme))
