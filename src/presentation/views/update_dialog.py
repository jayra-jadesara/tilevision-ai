"""Update available dialog — fast in-app installer download with progress."""

from __future__ import annotations

import logging
import subprocess
import sys
from pathlib import Path
from typing import Optional

from PySide6.QtCore import QTimer, QUrl, Qt
from PySide6.QtGui import QDesktopServices
from PySide6.QtWidgets import (
    QDialog,
    QHBoxLayout,
    QLabel,
    QMessageBox,
    QProgressBar,
    QPushButton,
    QTextEdit,
    QVBoxLayout,
)

from src.presentation.workers.update_download_worker import UpdateDownloadWorker
from src.utils.update_check import UpdateInfo, platform_download_label
from src.utils.update_downloader import (
    DEFAULT_CONNECTIONS,
    eta_seconds,
    format_bytes,
    format_eta,
    format_speed,
)

logger = logging.getLogger("tilevision.presentation.views.update_dialog")


class UpdateAvailableDialog(QDialog):
    """Notify the customer that a new TileVision AI build is ready."""

    def __init__(
        self,
        info: UpdateInfo,
        *,
        theme: str = "light",
        parent=None,
        auto_start_download: bool = True,
    ) -> None:
        super().__init__(parent)
        self._info = info
        self._theme = theme
        self._worker: Optional[UpdateDownloadWorker] = None
        self._downloaded_path: Optional[Path] = None

        self.setWindowTitle("Update Available")
        self.setModal(True)
        self.setMinimumWidth(520)

        layout = QVBoxLayout(self)
        layout.setSpacing(12)

        headline = QLabel(
            f"<b>TileVision AI {info.latest_version}</b> is available "
            f"(you have {info.current_version})."
        )
        headline.setWordWrap(True)
        layout.addWidget(headline)

        hint = QLabel(
            f"Downloading <b>{platform_download_label()}</b> inside TileVision "
            f"with a fast multi-connection transfer (not the browser). "
            "When it finishes, open the installer. "
            "Your license key and tile catalogue stay on this computer."
        )
        hint.setWordWrap(True)
        hint.setTextFormat(Qt.TextFormat.RichText)
        layout.addWidget(hint)

        if info.release_notes:
            notes = QTextEdit()
            notes.setReadOnly(True)
            notes.setPlainText(info.release_notes)
            notes.setMaximumHeight(120)
            layout.addWidget(notes)

        self._status = QLabel("Preparing in-app download…")
        self._status.setWordWrap(True)
        layout.addWidget(self._status)

        self._progress = QProgressBar()
        self._progress.setRange(0, 1000)
        self._progress.setValue(0)
        layout.addWidget(self._progress)

        buttons = QHBoxLayout()
        buttons.addStretch()

        self._later_btn = QPushButton("Remind Me Later")
        self._later_btn.clicked.connect(self.reject)
        buttons.addWidget(self._later_btn)

        self._skip_btn = QPushButton("Skip This Version")
        self._skip_btn.clicked.connect(self._on_skip)
        buttons.addWidget(self._skip_btn)

        self._cancel_btn = QPushButton("Cancel Download")
        self._cancel_btn.hide()
        self._cancel_btn.clicked.connect(self._on_cancel_download)
        buttons.addWidget(self._cancel_btn)

        self._open_btn = QPushButton("Open Installer")
        self._open_btn.hide()
        self._open_btn.setDefault(True)
        self._open_btn.clicked.connect(self._on_open_installer)
        buttons.addWidget(self._open_btn)

        self._download_btn = QPushButton("Download in App")
        self._download_btn.setDefault(True)
        self._download_btn.clicked.connect(self._on_download)
        buttons.addWidget(self._download_btn)

        layout.addLayout(buttons)

        # Browser is a last-resort fallback only — not the primary path.
        self._browser_btn = QPushButton("Slow browser download…")
        self._browser_btn.setFlat(True)
        self._browser_btn.setToolTip(
            "Fallback only. Browser downloads from GitHub are often much slower."
        )
        self._browser_btn.clicked.connect(self._on_open_browser)
        layout.addWidget(self._browser_btn)

        if auto_start_download:
            QTimer.singleShot(0, self._on_download)

    def closeEvent(self, event) -> None:  # noqa: N802
        if self._worker is not None and self._worker.isRunning():
            self._worker.cancel()
            self._worker.wait(2000)
        super().closeEvent(event)

    def _on_download(self) -> None:
        if self._worker is not None and self._worker.isRunning():
            return

        self._downloaded_path = None
        self._status.show()
        self._status.setText(
            f"Downloading in app ({DEFAULT_CONNECTIONS} parallel connections)…"
        )
        self._progress.show()
        self._progress.setRange(0, 1000)
        self._progress.setValue(0)
        self._download_btn.setEnabled(False)
        self._later_btn.setEnabled(False)
        self._skip_btn.setEnabled(False)
        self._browser_btn.setEnabled(False)
        self._cancel_btn.show()
        self._open_btn.hide()

        self._worker = UpdateDownloadWorker(
            self._info.download_url,
            connections=DEFAULT_CONNECTIONS,
            parent=self,
        )
        self._worker.progress.connect(self._on_progress)
        self._worker.finished_ok.connect(self._on_download_ok)
        self._worker.finished_error.connect(self._on_download_error)
        self._worker.finished_cancelled.connect(self._on_download_cancelled)
        self._worker.finished.connect(self._worker.deleteLater)
        self._worker.start()

    def _on_progress(self, received: int, total: int, speed: float) -> None:
        if total > 0:
            self._progress.setValue(int(received * 1000 / total))
            eta = format_eta(eta_seconds(received, total, speed))
            self._status.setText(
                f"Downloading… {format_bytes(received)} / {format_bytes(total)} "
                f"({format_speed(speed)}) — {eta}"
            )
        else:
            self._progress.setRange(0, 0)
            self._status.setText(
                f"Downloading… {format_bytes(received)} ({format_speed(speed)}) — calculating…"
            )

    def _on_download_ok(self, path: str) -> None:
        self._downloaded_path = Path(path)
        self._progress.setRange(0, 1000)
        self._progress.setValue(1000)
        self._status.setText(
            f"Download complete:\n{path}\n\n"
            "Click Open Installer, then reopen TileVision AI when setup finishes."
        )
        self._cancel_btn.hide()
        self._open_btn.show()
        self._download_btn.setEnabled(True)
        self._download_btn.setText("Download in App")
        self._later_btn.setEnabled(True)
        self._skip_btn.setEnabled(True)
        self._browser_btn.setEnabled(True)

    def _on_download_error(self, message: str) -> None:
        self._reset_idle_buttons()
        self._status.show()
        self._status.setText(f"Download failed: {message}")
        self._progress.hide()
        QMessageBox.warning(
            self,
            "Download Failed",
            "Could not download the update in the app.\n\n"
            f"{message}\n\n"
            "You can try again, or use Open in Browser / Google Drive if your "
            "vendor shared a mirror link.",
        )

    def _on_download_cancelled(self) -> None:
        self._reset_idle_buttons()
        self._status.show()
        self._status.setText("Download cancelled.")
        self._progress.hide()

    def _reset_idle_buttons(self) -> None:
        self._cancel_btn.hide()
        self._open_btn.hide()
        self._download_btn.setEnabled(True)
        self._later_btn.setEnabled(True)
        self._skip_btn.setEnabled(True)
        self._browser_btn.setEnabled(True)

    def _on_cancel_download(self) -> None:
        if self._worker is not None and self._worker.isRunning():
            self._status.setText("Cancelling…")
            self._worker.cancel()

    def _on_open_browser(self) -> None:
        QDesktopServices.openUrl(QUrl(self._info.download_url))

    def _on_open_installer(self) -> None:
        if self._downloaded_path is None or not self._downloaded_path.exists():
            QMessageBox.warning(self, "File Missing", "Downloaded installer not found.")
            return
        path = self._downloaded_path
        try:
            if sys.platform == "win32":
                os_start = getattr(__import__("os"), "startfile", None)
                if os_start is not None:
                    os_start(str(path))  # type: ignore[misc]
                else:
                    QDesktopServices.openUrl(QUrl.fromLocalFile(str(path)))
            elif sys.platform == "darwin":
                subprocess.Popen(["open", str(path)], start_new_session=True)
            else:
                QDesktopServices.openUrl(QUrl.fromLocalFile(str(path)))
        except Exception as exc:
            logger.error("Failed to open installer: %s", exc)
            QMessageBox.information(
                self,
                "Open Manually",
                f"Could not launch the installer automatically.\n\n"
                f"Open this file yourself:\n{path}",
            )

    def _on_skip(self) -> None:
        if self._worker is not None and self._worker.isRunning():
            self._worker.cancel()
            self._worker.wait(1500)
        self.done(2)

    @staticmethod
    def skipped_version_result(result: int) -> Optional[str]:
        return "skip" if result == 2 else None
