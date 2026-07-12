"""
Settings View for TileVision AI (Task D: Settings).

Lets the user configure:
  - Theme (dark/light), thumbnail size, number of search results.
  - Watched folders for auto-indexing (Feature 7) — add/remove.
  - Language (placeholder — English only for now).
  - Database path, Backup Database.
  - Rebuild FAISS Index (force re-embed everything).
  - Clear thumbnail Cache.
  - Export Logs.
"""

import logging
import shutil
from pathlib import Path
from typing import Callable, List, Optional

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QWidget,
    QVBoxLayout,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QListWidget,
    QListWidgetItem,
    QFileDialog,
    QSpinBox,
    QComboBox,
    QGroupBox,
    QFormLayout,
    QMessageBox,
    QProgressDialog,
)

from src.config.settings import AppSettings
from src.core.use_cases.index_images import IndexImagesUseCase
from src.presentation.workers.rebuild_index_worker import RebuildIndexWorker
from src.utils.logger import get_log_file_path

logger = logging.getLogger("tilevision.presentation.views.settings_view")


class SettingsView(QWidget):
    """Settings page widget."""

    def __init__(
        self,
        settings: AppSettings,
        license_details: Optional[dict] = None,
        catalog_count_provider: Optional[Callable[[], int]] = None,
        on_theme_changed: Optional[Callable[[str], None]] = None,
        db_path_provider: Optional[Callable[[], Path]] = None,
        indexing_use_case: Optional[IndexImagesUseCase] = None,
        indexed_folders_provider: Optional[Callable[[], List[str]]] = None,
        parent: Optional[QWidget] = None,
    ) -> None:
        """
        Args:
            settings: The shared AppSettings instance (read/write).
            license_details: Current license/trial info for display.
            catalog_count_provider: Callable returning the current number
                of indexed tiles, for the "Catalog" stat.
            on_theme_changed: Callback invoked with the new theme name
                ("dark"/"light") when the user changes the theme dropdown.
            db_path_provider: Callable returning the SQLite database's
                absolute Path, for the Database Path display and Backup
                Database action. If omitted, those controls are disabled.
            indexing_use_case: The shared IndexImagesUseCase, needed for
                the "Rebuild FAISS Index" action (Task D). If omitted,
                that button is disabled.
            indexed_folders_provider: Callable returning every folder
                that's been indexed at least once — the set Rebuild FAISS
                Index operates over. If omitted, Rebuild is disabled.
            parent: Optional Qt parent widget.
        """
        super().__init__(parent)
        self._settings = settings
        self._license_details = license_details or {}
        self._catalog_count_provider = catalog_count_provider
        self._on_theme_changed = on_theme_changed
        self._db_path_provider = db_path_provider
        self._indexing_use_case = indexing_use_case
        self._indexed_folders_provider = indexed_folders_provider
        self._rebuild_worker: Optional[RebuildIndexWorker] = None
        self._rebuild_progress_dialog: Optional[QProgressDialog] = None
        self._setup_ui()
        self._apply_styles()

    def _setup_ui(self) -> None:
        self.setObjectName("SettingsView")
        layout = QVBoxLayout(self)
        layout.setContentsMargins(24, 20, 24, 20)
        layout.setSpacing(16)

        title = QLabel("Settings")
        title.setObjectName("PageTitle")
        layout.addWidget(title)

        layout.addWidget(self._build_overview_section())
        layout.addWidget(self._build_watched_folders_section())
        layout.addWidget(self._build_preferences_section())
        layout.addWidget(self._build_maintenance_section())
        layout.addStretch()

    def _build_overview_section(self) -> QGroupBox:
        box = QGroupBox("Overview")
        form = QFormLayout(box)

        catalog_count = self._catalog_count_provider() if self._catalog_count_provider else "—"
        form.addRow("Indexed Tiles:", QLabel(str(catalog_count)))

        if self._license_details.get("is_trial"):
            days = self._license_details.get("days_remaining", 0)
            license_text = f"🕐 Trial — {days} day(s) remaining"
        elif self._license_details:
            license_text = f"🔐 {self._license_details.get('license_type', 'Licensed')} — {self._license_details.get('customer_name', '')}"
        else:
            license_text = "🔓 Unlicensed"
        form.addRow("License:", QLabel(license_text))

        form.addRow("Thumbnail Cache:", QLabel(self._settings.thumbnail_dir))

        db_path_text = str(self._db_path_provider()) if self._db_path_provider else "—"
        form.addRow("Database Path:", QLabel(db_path_text))

        return box

    def _build_watched_folders_section(self) -> QGroupBox:
        box = QGroupBox("Auto Folder Monitoring (Feature 7)")
        layout = QVBoxLayout(box)

        note = QLabel(
            "Folders listed here are watched automatically — new or changed images are "
            "indexed in the background without a manual scan. Changes here take effect "
            "the next time you start TileVision AI."
        )
        note.setObjectName("SectionNote")
        note.setWordWrap(True)
        layout.addWidget(note)

        self._folders_list = QListWidget()
        self._folders_list.setObjectName("FoldersList")
        for folder in self._settings.watch_folders:
            self._folders_list.addItem(QListWidgetItem(folder))
        layout.addWidget(self._folders_list)

        button_row = QHBoxLayout()
        add_button = QPushButton("➕  Add Folder")
        add_button.clicked.connect(self._on_add_folder)
        button_row.addWidget(add_button)

        remove_button = QPushButton("➖  Remove Selected")
        remove_button.clicked.connect(self._on_remove_folder)
        button_row.addWidget(remove_button)
        button_row.addStretch()
        layout.addLayout(button_row)

        return box

    def _build_preferences_section(self) -> QGroupBox:
        box = QGroupBox("Preferences")
        form = QFormLayout(box)

        self._top_k_spinbox = QSpinBox()
        self._top_k_spinbox.setRange(1, 100)
        self._top_k_spinbox.setValue(self._settings.top_k)
        self._top_k_spinbox.valueChanged.connect(self._on_top_k_changed)
        form.addRow("Search Results Shown:", self._top_k_spinbox)

        self._thumbnail_size_spinbox = QSpinBox()
        self._thumbnail_size_spinbox.setRange(64, 512)
        self._thumbnail_size_spinbox.setSingleStep(32)
        self._thumbnail_size_spinbox.setValue(getattr(self._settings, "thumbnail_size", 200))
        self._thumbnail_size_spinbox.setSuffix(" px")
        self._thumbnail_size_spinbox.valueChanged.connect(self._on_thumbnail_size_changed)
        form.addRow("Thumbnail Size:", self._thumbnail_size_spinbox)

        self._theme_combo = QComboBox()
        self._theme_combo.addItems(["dark", "light"])
        current_theme = getattr(self._settings, "theme", "dark")
        idx = self._theme_combo.findText(current_theme)
        self._theme_combo.setCurrentIndex(idx if idx >= 0 else 0)
        self._theme_combo.currentTextChanged.connect(self._on_theme_selected)
        form.addRow("Theme:", self._theme_combo)

        self._language_combo = QComboBox()
        self._language_combo.addItem("English")
        self._language_combo.setEnabled(False)
        self._language_combo.setToolTip("Additional languages coming in a future release.")
        form.addRow("Language:", self._language_combo)

        return box

    def _build_maintenance_section(self) -> QGroupBox:
        box = QGroupBox("Maintenance")
        layout = QVBoxLayout(box)

        note = QLabel(
            "These actions affect your whole catalog. Backup Database and Export Logs "
            "are always safe; Rebuild FAISS Index and Clear Cache are safe but will take "
            "time to regenerate on next use."
        )
        note.setObjectName("SectionNote")
        note.setWordWrap(True)
        layout.addWidget(note)

        row1 = QHBoxLayout()
        self._backup_button = QPushButton("💾  Backup Database")
        self._backup_button.clicked.connect(self._on_backup_database)
        self._backup_button.setEnabled(self._db_path_provider is not None)
        row1.addWidget(self._backup_button)

        self._export_logs_button = QPushButton("📤  Export Logs")
        self._export_logs_button.clicked.connect(self._on_export_logs)
        row1.addWidget(self._export_logs_button)
        row1.addStretch()
        layout.addLayout(row1)

        row2 = QHBoxLayout()
        self._rebuild_button = QPushButton("🔄  Rebuild FAISS Index")
        self._rebuild_button.clicked.connect(self._on_rebuild_faiss)
        self._rebuild_button.setEnabled(
            self._indexing_use_case is not None and self._indexed_folders_provider is not None
        )
        row2.addWidget(self._rebuild_button)

        self._clear_cache_button = QPushButton("🧹  Clear Cache")
        self._clear_cache_button.clicked.connect(self._on_clear_cache)
        row2.addWidget(self._clear_cache_button)
        row2.addStretch()
        layout.addLayout(row2)

        return box

    # ── Handlers: Watched Folders ────────────────────────────────────────

    def _on_add_folder(self) -> None:
        folder = QFileDialog.getExistingDirectory(self, "Select Folder to Watch")
        if not folder:
            return

        current = self._settings.watch_folders
        resolved = str(Path(folder).resolve())
        if resolved in current:
            QMessageBox.information(self, "Already Added", "This folder is already being watched.")
            return

        current.append(resolved)
        self._settings.watch_folders = current
        self._folders_list.addItem(QListWidgetItem(resolved))
        logger.info(f"Added watched folder: {resolved}")

    def _on_remove_folder(self) -> None:
        selected = self._folders_list.currentItem()
        if not selected:
            return

        folder_path = selected.text()
        current = [f for f in self._settings.watch_folders if f != folder_path]
        self._settings.watch_folders = current

        self._folders_list.takeItem(self._folders_list.row(selected))
        logger.info(f"Removed watched folder: {folder_path}")

    # ── Handlers: Preferences ────────────────────────────────────────────

    def _on_top_k_changed(self, value: int) -> None:
        self._settings.top_k = value

    def _on_thumbnail_size_changed(self, value: int) -> None:
        self._settings.thumbnail_size = value

    def _on_theme_selected(self, theme: str) -> None:
        self._settings.theme = theme
        if self._on_theme_changed:
            self._on_theme_changed(theme)

    # ── Handlers: Maintenance ────────────────────────────────────────────

    def _on_backup_database(self) -> None:
        if self._db_path_provider is None:
            return
        db_path = self._db_path_provider()
        if not db_path.exists():
            QMessageBox.warning(self, "No Database Found", f"Database file not found:\n{db_path}")
            return

        default_name = f"tilevision_backup_{db_path.stem}.db"
        dest_str, _ = QFileDialog.getSaveFileName(
            self, "Backup Database", default_name, "SQLite Database (*.db)"
        )
        if not dest_str:
            return

        try:
            shutil.copy2(db_path, dest_str)
            QMessageBox.information(self, "Backup Complete", f"Database backed up to:\n{dest_str}")
            logger.info(f"Database backed up to {dest_str}")
        except OSError as e:
            QMessageBox.critical(self, "Backup Failed", f"Could not back up database:\n{e}")
            logger.error(f"Database backup failed: {e}")

    def _on_export_logs(self) -> None:
        log_path = get_log_file_path()
        if not log_path.exists():
            QMessageBox.information(self, "No Logs Found", f"No log file found yet at:\n{log_path}")
            return

        dest_str, _ = QFileDialog.getSaveFileName(
            self, "Export Logs", log_path.name, "Log Files (*.log);;All Files (*)"
        )
        if not dest_str:
            return

        try:
            shutil.copy2(log_path, dest_str)
            QMessageBox.information(self, "Logs Exported", f"Logs exported to:\n{dest_str}")
            logger.info(f"Logs exported to {dest_str}")
        except OSError as e:
            QMessageBox.critical(self, "Export Failed", f"Could not export logs:\n{e}")
            logger.error(f"Log export failed: {e}")

    def _on_clear_cache(self) -> None:
        thumb_dir = Path(self._settings.thumbnail_dir)
        if not thumb_dir.exists():
            QMessageBox.information(self, "Nothing to Clear", "No thumbnail cache found.")
            return

        confirm = QMessageBox.question(
            self,
            "Clear Cache",
            "Delete all cached thumbnails? They'll be regenerated automatically the "
            "next time you view search results or the catalog.",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            QMessageBox.StandardButton.No,
        )
        if confirm != QMessageBox.StandardButton.Yes:
            return

        deleted = 0
        errors = 0
        for file_path in thumb_dir.glob("*"):
            if file_path.is_file():
                try:
                    file_path.unlink()
                    deleted += 1
                except OSError:
                    errors += 1

        message = f"Cleared {deleted} cached thumbnail(s)."
        if errors:
            message += f" ({errors} could not be deleted.)"
        QMessageBox.information(self, "Cache Cleared", message)
        logger.info(f"Cleared thumbnail cache: {deleted} deleted, {errors} errors")

    def _on_rebuild_faiss(self) -> None:
        if self._indexing_use_case is None or self._indexed_folders_provider is None:
            return

        folders = self._indexed_folders_provider()
        if not folders:
            QMessageBox.information(self, "Nothing to Rebuild", "No indexed folders found.")
            return

        confirm = QMessageBox.question(
            self,
            "Rebuild FAISS Index",
            f"This will re-analyze every image in {len(folders)} indexed folder(s) and "
            "rebuild the search index from scratch. This can take a while for large "
            "catalogs and cannot be cancelled once started.\n\nContinue?",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            QMessageBox.StandardButton.No,
        )
        if confirm != QMessageBox.StandardButton.Yes:
            return

        self._rebuild_button.setEnabled(False)
        self._rebuild_progress_dialog = QProgressDialog(
            "Rebuilding FAISS index...", None, 0, len(folders), self
        )
        self._rebuild_progress_dialog.setWindowTitle("Rebuild FAISS Index")
        self._rebuild_progress_dialog.setWindowModality(Qt.WindowModality.WindowModal)
        self._rebuild_progress_dialog.setMinimumDuration(0)
        self._rebuild_progress_dialog.setCancelButton(None)
        self._rebuild_progress_dialog.show()

        self._rebuild_worker = RebuildIndexWorker(self._indexing_use_case, folders)
        self._rebuild_worker.progress_updated.connect(self._on_rebuild_progress)
        self._rebuild_worker.rebuild_finished.connect(self._on_rebuild_finished)
        self._rebuild_worker.rebuild_failed.connect(self._on_rebuild_failed)
        self._rebuild_worker.finished.connect(self._rebuild_worker.deleteLater)
        self._rebuild_worker.start()

    def _on_rebuild_progress(self, processed: int, total: int, current_folder: str) -> None:
        if self._rebuild_progress_dialog is not None:
            self._rebuild_progress_dialog.setValue(processed)
            self._rebuild_progress_dialog.setLabelText(f"Rebuilding: {current_folder} ({processed}/{total})")

    def _on_rebuild_finished(self, total_reembedded: int, total_failed: int) -> None:
        self._rebuild_button.setEnabled(True)
        if self._rebuild_progress_dialog is not None:
            self._rebuild_progress_dialog.close()
            self._rebuild_progress_dialog = None
        self._rebuild_worker = None

        message = f"Rebuild complete. {total_reembedded} image(s) re-indexed."
        if total_failed:
            message += f" {total_failed} failed."
        QMessageBox.information(self, "Rebuild Complete", message)

    def _on_rebuild_failed(self, error_message: str) -> None:
        self._rebuild_button.setEnabled(True)
        if self._rebuild_progress_dialog is not None:
            self._rebuild_progress_dialog.close()
            self._rebuild_progress_dialog = None
        self._rebuild_worker = None
        QMessageBox.critical(self, "Rebuild Failed", error_message)

    def _apply_styles(self) -> None:
        self.setStyleSheet(
            """
            #PageTitle { font-size: 20px; font-weight: 700; color: #E8EAF6; }
            QGroupBox {
                color: #E8EAF6; border: 1px solid #2D3250; border-radius: 8px;
                margin-top: 12px; padding-top: 12px; font-weight: 600;
            }
            QGroupBox::title { subcontrol-origin: margin; left: 10px; padding: 0 6px; color: #7C83D3; }
            QLabel { color: #C7CAD9; }
            #SectionNote { color: #8A8FA3; font-size: 11px; }
            #FoldersList {
                background-color: #1E212C; border: 1px solid #2E3243; border-radius: 6px;
                color: #D6D9E8; min-height: 100px;
            }
            QPushButton {
                background-color: #2A2E3D; border: 1px solid #3A3F52; border-radius: 6px;
                padding: 6px 12px; color: #C7CAD9;
            }
            QPushButton:hover:enabled { background-color: #333852; }
            QPushButton:disabled { color: #55596B; }
            QSpinBox, QComboBox {
                background-color: #232634; border: 1px solid #3A3F52; border-radius: 6px;
                padding: 4px 8px; color: #E8EAF6;
            }
            """
        )
