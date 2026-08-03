"""Update available dialog — fast in-app download + install & restart."""

from __future__ import annotations

import logging
import subprocess
import sys
from pathlib import Path
from typing import Optional

from PySide6.QtCore import QTimer, QUrl, Qt
from PySide6.QtGui import QDesktopServices
from PySide6.QtWidgets import (
    QApplication,
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
from src.utils.update_installer import UpdateInstallError, launch_update_installer

logger = logging.getLogger("tilevision.presentation.views.update_dialog")


class UpdateAvailableDialog(QDialog):
    """Download the new build in-app, then install and restart TileVision AI."""

    def __init__(
        self,
        info: UpdateInfo,
        *,
        theme: str = "light",
        parent=None,
        auto_start_download: bool = True,
        auto_install_after_download: bool = True,
    ) -> None:
        super().__init__(parent)
        self._info = info
        self._theme = theme
        self._auto_install_after_download = auto_install_after_download
        self._worker: Optional[UpdateDownloadWorker] = None
        self._downloaded_path: Optional[Path] = None
        self._installing = False

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
            f"TileVision downloads <b>{platform_download_label()}</b> "
            f"with {DEFAULT_CONNECTIONS} parallel connections (fast — not the browser), "
            "then installs and restarts itself. "
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

        self._open_btn = QPushButton("Open File…")
        self._open_btn.hide()
        self._open_btn.setToolTip("Fallback: open the downloaded installer manually")
        self._open_btn.clicked.connect(self._on_open_installer)
        buttons.addWidget(self._open_btn)

        self._install_btn = QPushButton("Install & Restart")
        self._install_btn.hide()
        self._install_btn.setDefault(True)
        self._install_btn.clicked.connect(self._on_install_and_restart)
        buttons.addWidget(self._install_btn)

        self._download_btn = QPushButton("Download & Install")
        self._download_btn.setDefault(True)
        self._download_btn.clicked.connect(self._on_download)
        buttons.addWidget(self._download_btn)

        layout.addLayout(buttons)

        # Browser is a last-resort fallback only — not the primary path.
        self._browser_btn = QPushButton("Very slow browser download (not recommended)…")
        self._browser_btn.setFlat(True)
        self._browser_btn.setToolTip(
            "Fallback only. Prefer Download & Install inside TileVision."
        )
        self._browser_btn.clicked.connect(self._on_open_browser)
        layout.addWidget(self._browser_btn)

        if auto_start_download:
            QTimer.singleShot(0, self._on_download)

    def closeEvent(self, event) -> None:  # noqa: N802
        if self._installing:
            event.ignore()
            return
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
            f"Fast download ({DEFAULT_CONNECTIONS} parallel CDN connections)…"
        )
        self._progress.show()
        self._progress.setRange(0, 1000)
        self._progress.setValue(0)
        # Hide idle actions — do not leave greyed-out duplicate buttons visible.
        self._download_btn.hide()
        self._later_btn.hide()
        self._skip_btn.hide()
        self._browser_btn.hide()
        self._open_btn.hide()
        self._install_btn.hide()
        self._cancel_btn.show()

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
        self._cancel_btn.hide()
        self._later_btn.show()
        self._later_btn.setEnabled(True)
        self._skip_btn.show()
        self._skip_btn.setEnabled(True)
        self._browser_btn.hide()
        self._download_btn.hide()
        self._install_btn.show()
        self._install_btn.setEnabled(True)
        self._open_btn.show()
        self._open_btn.setEnabled(True)

        if self._auto_install_after_download:
            self._status.setText(
                "Download complete.\nInstalling update and restarting TileVision AI…"
            )
            QTimer.singleShot(250, self._on_install_and_restart)
        else:
            self._status.setText(
                f"Download complete:\n{path}\n\n"
                "Click Install & Restart to apply the update."
            )

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
            "You can try Download & Install again.",
        )

    def _on_download_cancelled(self) -> None:
        self._reset_idle_buttons()
        self._status.show()
        self._status.setText("Download cancelled.")
        self._progress.hide()

    def _reset_idle_buttons(self) -> None:
        self._cancel_btn.hide()
        self._open_btn.hide()
        self._install_btn.hide()
        self._download_btn.show()
        self._download_btn.setEnabled(True)
        self._download_btn.setText("Download & Install")
        self._later_btn.show()
        self._later_btn.setEnabled(True)
        self._skip_btn.show()
        self._skip_btn.setEnabled(True)
        self._browser_btn.show()
        self._browser_btn.setEnabled(True)

    def _on_cancel_download(self) -> None:
        if self._worker is not None and self._worker.isRunning():
            self._status.setText("Cancelling…")
            self._worker.cancel()

    def _on_open_browser(self) -> None:
        reply = QMessageBox.warning(
            self,
            "Browser download is very slow",
            "Browser downloads from GitHub are often limited to "
            "~100–200 KB/s.\n\n"
            "Use Download & Install inside TileVision instead.\n\n"
            "Open the slow browser link anyway?",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            QMessageBox.StandardButton.No,
        )
        if reply != QMessageBox.StandardButton.Yes:
            return
        QDesktopServices.openUrl(QUrl(self._info.download_url))

    def _on_install_and_restart(self) -> None:
        if self._installing:
            return
        if self._downloaded_path is None or not self._downloaded_path.exists():
            QMessageBox.warning(self, "File Missing", "Downloaded installer not found.")
            return

        self._installing = True
        self._install_btn.hide()
        self._open_btn.hide()
        self._download_btn.hide()
        self._later_btn.hide()
        self._skip_btn.hide()
        self._browser_btn.hide()
        self._cancel_btn.hide()
        self._status.setText("Installing update… TileVision AI will restart.")

        try:
            launch_update_installer(self._downloaded_path)
        except UpdateInstallError as exc:
            self._installing = False
            self._later_btn.show()
            self._later_btn.setEnabled(True)
            self._skip_btn.show()
            self._skip_btn.setEnabled(True)
            self._install_btn.show()
            self._install_btn.setEnabled(True)
            self._open_btn.show()
            self._open_btn.setEnabled(True)
            logger.error("In-app install failed: %s", exc)
            QMessageBox.warning(
                self,
                "Install Failed",
                f"{exc}\n\nYou can use Open File… to install manually.",
            )
            return
        except Exception as exc:
            self._installing = False
            self._later_btn.show()
            self._later_btn.setEnabled(True)
            self._skip_btn.show()
            self._skip_btn.setEnabled(True)
            self._install_btn.show()
            self._install_btn.setEnabled(True)
            self._open_btn.show()
            self._open_btn.setEnabled(True)
            logger.exception("Unexpected install failure")
            QMessageBox.warning(
                self,
                "Install Failed",
                f"Could not start the installer:\n{exc}\n\n"
                "Use Open File… to install manually.",
            )
            return

        # Quit so Windows can overwrite Program Files / Mac helper can replace .app.
        QTimer.singleShot(400, self._quit_for_install)

    def _quit_for_install(self) -> None:
        logger.info("Quitting so the update installer can replace this build.")
        app = QApplication.instance()
        if app is not None:
            app.quit()
        else:
            self.accept()

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
