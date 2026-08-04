"""Update available dialog — fast in-app download + install & restart."""

from __future__ import annotations

import logging
import os
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
    QProgressBar,
    QPushButton,
    QTextEdit,
    QVBoxLayout,
)

from src.config.settings import AppSettings
from src.presentation.dialogs import message_box
from src.presentation.workers.update_download_worker import UpdateDownloadWorker
from src.theme.theme_manager import get_dialog_qss
from src.utils.update_check import UpdateInfo, platform_download_label
from src.utils.update_downloader import (
    DEFAULT_CONNECTIONS,
    eta_seconds,
    format_bytes,
    format_eta,
    format_speed,
    resolve_cached_installer,
)
from src.utils.update_installer import (
    UpdateInstallError,
    begin_force_quit_for_update,
    is_force_quit_for_update,
    launch_update_installer,
)

logger = logging.getLogger("tilevision.presentation.views.update_dialog")

# If quit stalls (indexing confirm / non-daemon threads), unlock the UI so the
# customer is not trapped on "Installing…" forever.
_INSTALL_WATCHDOG_MS = 90_000
# Hard process exit so the Windows/macOS helper can replace files even when
# Qt quit is blocked by a modal closeEvent or a non-daemon worker thread.
# Keep this short: Windows ShellExecuteEx already waited for UAC consent.
_HARD_EXIT_AFTER_QUIT_MS = 400


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
        settings: AppSettings | None = None,
    ) -> None:
        super().__init__(parent)
        self._info = info
        self._theme = theme if theme in ("light", "dark") else "light"
        self._auto_install_after_download = auto_install_after_download
        self._settings = settings
        self._worker: Optional[UpdateDownloadWorker] = None
        self._downloaded_path: Optional[Path] = None
        self._installing = False
        self._install_watchdog: Optional[QTimer] = None

        self.setWindowTitle("Update Available")
        self.setObjectName("UpdateAvailableDialog")
        self.setModal(True)
        self.setMinimumWidth(520)

        layout = QVBoxLayout(self)
        layout.setSpacing(12)
        layout.setContentsMargins(20, 18, 20, 16)

        headline = QLabel(
            f"TileVision AI {info.latest_version} is available "
            f"(you have {info.current_version})."
        )
        headline.setObjectName("DialogTitle")
        headline.setWordWrap(True)
        layout.addWidget(headline)

        hint = QLabel(
            f"TileVision downloads <b>{platform_download_label()}</b> "
            f"with {DEFAULT_CONNECTIONS} parallel connections (fast — not the browser), "
            "then installs and restarts itself. "
            "Your license key and tile catalogue stay on this computer."
        )
        hint.setObjectName("DialogHint")
        hint.setWordWrap(True)
        hint.setTextFormat(Qt.TextFormat.RichText)
        layout.addWidget(hint)

        if info.release_notes:
            notes = QTextEdit()
            notes.setObjectName("DialogNotes")
            notes.setReadOnly(True)
            notes.setPlainText(info.release_notes)
            notes.setMaximumHeight(120)
            layout.addWidget(notes)

        self._status = QLabel("Preparing in-app download…")
        self._status.setObjectName("DialogStatus")
        self._status.setWordWrap(True)
        layout.addWidget(self._status)

        self._progress = QProgressBar()
        self._progress.setObjectName("DialogProgressBar")
        self._progress.setRange(0, 1000)
        self._progress.setValue(0)
        self._progress.setTextVisible(False)
        self._progress.setFixedHeight(18)
        layout.addWidget(self._progress)

        buttons = QHBoxLayout()
        buttons.setSpacing(8)
        buttons.addStretch()

        self._later_btn = QPushButton("Remind Me Later")
        self._later_btn.setObjectName("SecondaryButton")
        self._later_btn.clicked.connect(self.reject)
        buttons.addWidget(self._later_btn)

        self._skip_btn = QPushButton("Skip This Version")
        self._skip_btn.setObjectName("SecondaryButton")
        self._skip_btn.clicked.connect(self._on_skip)
        buttons.addWidget(self._skip_btn)

        self._cancel_btn = QPushButton("Cancel Download")
        self._cancel_btn.setObjectName("SecondaryButton")
        self._cancel_btn.hide()
        self._cancel_btn.clicked.connect(self._on_cancel_download)
        buttons.addWidget(self._cancel_btn)

        self._open_btn = QPushButton("Open File…")
        self._open_btn.setObjectName("SecondaryButton")
        self._open_btn.hide()
        self._open_btn.setToolTip("Fallback: open the downloaded installer manually")
        self._open_btn.clicked.connect(self._on_open_installer)
        buttons.addWidget(self._open_btn)

        self._install_btn = QPushButton("Install & Restart")
        self._install_btn.setObjectName("PrimaryButton")
        self._install_btn.hide()
        self._install_btn.setDefault(True)
        self._install_btn.clicked.connect(self._on_install_and_restart)
        buttons.addWidget(self._install_btn)

        self._download_btn = QPushButton("Download & Install")
        self._download_btn.setObjectName("PrimaryButton")
        self._download_btn.setDefault(True)
        self._download_btn.clicked.connect(self._on_download)
        buttons.addWidget(self._download_btn)

        layout.addLayout(buttons)

        # Browser is a last-resort fallback only — not the primary path.
        self._browser_btn = QPushButton("Very slow browser download (not recommended)…")
        self._browser_btn.setObjectName("LinkButton")
        self._browser_btn.setFlat(True)
        self._browser_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        self._browser_btn.setToolTip(
            "Fallback only. Prefer Download & Install inside TileVision."
        )
        self._browser_btn.clicked.connect(self._on_open_browser)
        layout.addWidget(self._browser_btn)

        self._apply_styles()

        if auto_start_download:
            QTimer.singleShot(0, self._start_download_or_reuse_cache)

    def _apply_styles(self) -> None:
        """Match TileVision light/dark theme (not classic Windows chrome)."""
        self.setStyleSheet(get_dialog_qss(self._theme))

    def closeEvent(self, event) -> None:  # noqa: N802
        # Never block process exit once the silent installer has been scheduled.
        if is_force_quit_for_update():
            event.accept()
            return
        if self._installing:
            # Brief protect while helper starts; watchdog unlocks if quit stalls.
            event.ignore()
            return
        if self._worker is not None and self._worker.isRunning():
            self._worker.cancel()
            self._worker.wait(2000)
        super().closeEvent(event)

    def _preferred_cached_path(self) -> Optional[Path]:
        if self._settings is None:
            return None
        if self._settings.pending_update_version != self._info.latest_version:
            return None
        raw = self._settings.pending_update_installer_path
        return Path(raw) if raw else None

    def _remember_downloaded_installer(self, path: Path) -> None:
        if self._settings is None:
            return
        try:
            self._settings.set_pending_update(self._info.latest_version, str(path))
        except Exception:
            logger.exception("Failed to remember pending update installer path")

    def _start_download_or_reuse_cache(self) -> None:
        """Reuse a completed installer when present; otherwise start download."""
        preferred = self._preferred_cached_path()
        cached: Optional[Path] = None
        try:
            cached = resolve_cached_installer(
                self._info.download_url,
                preferred_path=preferred,
            )
        except Exception:
            logger.exception("Cache reuse probe failed; downloading again")

        if cached is not None:
            self._status.setText(
                "Installer already downloaded.\n"
                "Installing update and restarting TileVision AI…"
            )
            self._progress.setRange(0, 1000)
            self._progress.setValue(1000)
            self._on_download_ok(str(cached))
            return

        self._on_download()

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
        self._remember_downloaded_installer(self._downloaded_path)
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
        message_box.warning(
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
        reply = message_box.warning(
            self,
            "Browser download is very slow",
            "Browser downloads from GitHub are often limited to "
            "~100–200 KB/s.\n\n"
            "Use Download & Install inside TileVision instead.\n\n"
            "Open the slow browser link anyway?",
            message_box.StandardButton.Yes | message_box.StandardButton.No,
            message_box.StandardButton.No,
        )
        if reply != message_box.StandardButton.Yes:
            return
        QDesktopServices.openUrl(QUrl(self._info.download_url))

    def _cancel_indexing_quietly(self) -> None:
        """Stop folder indexing so quit is not blocked by a confirm dialog."""
        parent = self.parent()
        vm = getattr(parent, "_indexing_viewmodel", None) if parent is not None else None
        if vm is None:
            return
        try:
            cancel = getattr(vm, "cancel_indexing", None)
            if callable(cancel):
                logger.info("Cancelling indexing before update install/restart.")
                cancel()
        except Exception:
            logger.exception("Failed to cancel indexing before update quit")

    def _unlock_install_ui(self, message: str) -> None:
        """Recover from a stalled install so the customer is not trapped."""
        self._stop_install_watchdog()
        self._installing = False
        self._later_btn.show()
        self._later_btn.setEnabled(True)
        self._skip_btn.show()
        self._skip_btn.setEnabled(True)
        self._install_btn.show()
        self._install_btn.setEnabled(True)
        self._open_btn.show()
        self._open_btn.setEnabled(True)
        self._status.setText(message)

    def _stop_install_watchdog(self) -> None:
        if self._install_watchdog is not None:
            self._install_watchdog.stop()
            self._install_watchdog.deleteLater()
            self._install_watchdog = None

    def _on_install_watchdog(self) -> None:
        if not self._installing:
            return
        logger.error(
            "Update install/restart stalled; unlocking UI. installer=%s",
            self._downloaded_path,
        )
        self._unlock_install_ui(
            "Install did not finish restarting TileVision.\n\n"
            "The installer is still saved on this computer — use "
            "Install & Restart again, or Open File… to install manually.\n"
            "You will not need to download again."
        )
        message_box.warning(
            self,
            "Restart Taking Too Long",
            "TileVision could not finish installing and restarting automatically.\n\n"
            "Click Install & Restart to try again, or Open File… to run the "
            "installer yourself.\n\n"
            "The file is already downloaded — no second download is required.",
        )

    def _on_install_and_restart(self) -> None:
        if self._installing:
            return
        if self._downloaded_path is None or not self._downloaded_path.exists():
            message_box.warning(self, "File Missing", "Downloaded installer not found.")
            return

        self._installing = True
        self._install_btn.hide()
        self._open_btn.hide()
        self._download_btn.hide()
        self._later_btn.hide()
        self._skip_btn.hide()
        self._browser_btn.hide()
        self._cancel_btn.hide()
        self._status.setText(
            "Starting installer…\n"
            "Windows: click Yes on the permission (UAC) prompt if shown.\n"
            "Mac: TileVision will quit, then reopen on the new version."
        )

        try:
            # Cancel indexing BEFORE spawning the helper so MainWindow.closeEvent
            # does not show a confirm dialog that blocks quit behind this modal.
            self._cancel_indexing_quietly()
            begin_force_quit_for_update()
            # Windows: ShellExecuteEx(runas) blocks until UAC Yes/No, then Inno
            # force-closes us and [Run] relaunches. Mac: helper waits/kills PID.
            launch_update_installer(self._downloaded_path)
            self._status.setText(
                "Installer started.\nTileVision AI is closing to finish the update…"
            )
        except UpdateInstallError as exc:
            self._unlock_install_ui(f"Install failed: {exc}")
            logger.error("In-app install failed: %s", exc)
            message_box.warning(
                self,
                "Install Failed",
                f"{exc}\n\nYou can use Open File… to install manually.",
            )
            return
        except Exception as exc:
            self._unlock_install_ui(f"Install failed: {exc}")
            logger.exception("Unexpected install failure")
            message_box.warning(
                self,
                "Install Failed",
                f"Could not start the installer:\n{exc}\n\n"
                "Use Open File… to install manually.",
            )
            return

        self._stop_install_watchdog()
        self._install_watchdog = QTimer(self)
        self._install_watchdog.setSingleShot(True)
        self._install_watchdog.timeout.connect(self._on_install_watchdog)
        self._install_watchdog.start(_INSTALL_WATCHDOG_MS)

        # Quit so Windows can overwrite Program Files / Mac helper can replace .app.
        QTimer.singleShot(400, self._quit_for_install)

    def _quit_for_install(self) -> None:
        logger.info("Quitting so the update installer can replace this build.")
        begin_force_quit_for_update()
        self._cancel_indexing_quietly()
        # Allow the modal to dismiss so nested event loops do not trap quit.
        self._installing = False
        self.hide()
        app = QApplication.instance()
        if app is not None:
            # Guarantee the PID dies even if Qt quit is blocked by threads/modals.
            QTimer.singleShot(_HARD_EXIT_AFTER_QUIT_MS, self._hard_exit_for_install)
            app.quit()
        else:
            QTimer.singleShot(_HARD_EXIT_AFTER_QUIT_MS, self._hard_exit_for_install)
        self.accept()

    @staticmethod
    def _hard_exit_for_install() -> None:
        logger.info("Hard-exiting process for update installer.")
        os._exit(0)

    def _on_open_installer(self) -> None:
        if self._downloaded_path is None or not self._downloaded_path.exists():
            message_box.warning(self, "File Missing", "Downloaded installer not found.")
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
            message_box.information(
                self,
                "Open Manually",
                f"Could not launch the installer automatically.\n\n"
                f"Open this file yourself:\n{path}",
            )

    def _on_skip(self) -> None:
        if self._worker is not None and self._worker.isRunning():
            self._worker.cancel()
            self._worker.wait(1500)
        if self._settings is not None:
            try:
                self._settings.clear_pending_update()
            except Exception:
                logger.exception("Failed to clear pending update on skip")
        self.done(2)

    @staticmethod
    def skipped_version_result(result: int) -> Optional[str]:
        return "skip" if result == 2 else None
