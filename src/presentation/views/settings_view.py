"""
Settings View for TileVision AI.

Lets the user configure:
  - Watched folders for auto-indexing (Feature 7) — add/remove; takes
    effect on next app restart, since FolderMonitorController currently
    starts its watchdog observers once at startup rather than supporting
    live add/remove.
  - Number of search results returned (top_k).
  - Theme (dark/light).
  - View current license/trial status and catalog size at a glance.
"""

import logging
from pathlib import Path
from typing import Callable, Optional

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
)

from src.config.settings import AppSettings

logger = logging.getLogger("tilevision.presentation.views.settings_view")


class SettingsView(QWidget):
    """Settings page widget."""

    def __init__(
        self,
        settings: AppSettings,
        license_details: Optional[dict] = None,
        catalog_count_provider: Optional[Callable[[], int]] = None,
        on_theme_changed: Optional[Callable[[str], None]] = None,
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
            parent: Optional Qt parent widget.
        """
        super().__init__(parent)
        self._settings = settings
        self._license_details = license_details or {}
        self._catalog_count_provider = catalog_count_provider
        self._on_theme_changed = on_theme_changed
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

        self._theme_combo = QComboBox()
        self._theme_combo.addItems(["dark", "light"])
        current_theme = getattr(self._settings, "theme", "dark")
        idx = self._theme_combo.findText(current_theme)
        self._theme_combo.setCurrentIndex(idx if idx >= 0 else 0)
        self._theme_combo.currentTextChanged.connect(self._on_theme_selected)
        form.addRow("Theme:", self._theme_combo)

        return box

    # ── Handlers ─────────────────────────────────────────────────────────

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
        self._settings.save()
        self._folders_list.addItem(QListWidgetItem(resolved))
        logger.info(f"Added watched folder: {resolved}")

    def _on_remove_folder(self) -> None:
        selected = self._folders_list.currentItem()
        if not selected:
            return

        folder_path = selected.text()
        current = [f for f in self._settings.watch_folders if f != folder_path]
        self._settings.watch_folders = current
        self._settings.save()

        self._folders_list.takeItem(self._folders_list.row(selected))
        logger.info(f"Removed watched folder: {folder_path}")

    def _on_top_k_changed(self, value: int) -> None:
        self._settings.top_k = value
        self._settings.save()

    def _on_theme_selected(self, theme: str) -> None:
        self._settings.theme = theme
        self._settings.save()
        if self._on_theme_changed:
            self._on_theme_changed(theme)

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
            QPushButton:hover { background-color: #333852; }
            QSpinBox, QComboBox {
                background-color: #232634; border: 1px solid #3A3F52; border-radius: 6px;
                padding: 4px 8px; color: #E8EAF6;
            }
            """
        )
