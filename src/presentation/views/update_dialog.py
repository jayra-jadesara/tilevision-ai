"""Update available dialog — opens the platform download link in the browser."""

from __future__ import annotations

from typing import Optional

from PySide6.QtCore import QUrl, Qt
from PySide6.QtGui import QDesktopServices
from PySide6.QtWidgets import (
    QDialog,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QTextEdit,
    QVBoxLayout,
)

from src.utils.update_check import UpdateInfo


class UpdateAvailableDialog(QDialog):
    """Notify the customer that a new TileVision AI build is ready."""

    def __init__(
        self,
        info: UpdateInfo,
        *,
        theme: str = "light",
        parent=None,
    ) -> None:
        super().__init__(parent)
        self._info = info
        self._theme = theme

        self.setWindowTitle("Update Available")
        self.setModal(True)
        self.setMinimumWidth(460)

        layout = QVBoxLayout(self)
        layout.setSpacing(12)

        headline = QLabel(
            f"<b>TileVision AI {info.latest_version}</b> is available "
            f"(you have {info.current_version})."
        )
        headline.setWordWrap(True)
        layout.addWidget(headline)

        hint = QLabel(
            "Download the installer for your computer, run it, then reopen TileVision AI. "
            "Your license key and tile catalogue stay on this PC."
        )
        hint.setWordWrap(True)
        layout.addWidget(hint)

        if info.release_notes:
            notes = QTextEdit()
            notes.setReadOnly(True)
            notes.setPlainText(info.release_notes)
            notes.setMaximumHeight(120)
            layout.addWidget(notes)

        buttons = QHBoxLayout()
        buttons.addStretch()

        later_btn = QPushButton("Remind Me Later")
        later_btn.clicked.connect(self.reject)
        buttons.addWidget(later_btn)

        skip_btn = QPushButton("Skip This Version")
        skip_btn.clicked.connect(self._on_skip)
        buttons.addWidget(skip_btn)

        download_btn = QPushButton("Download Update")
        download_btn.setDefault(True)
        download_btn.clicked.connect(self._on_download)
        buttons.addWidget(download_btn)

        layout.addLayout(buttons)

    def _on_download(self) -> None:
        QDesktopServices.openUrl(QUrl(self._info.download_url))
        self.accept()

    def _on_skip(self) -> None:
        self.done(2)

    @staticmethod
    def skipped_version_result(result: int) -> Optional[str]:
        return "skip" if result == 2 else None
