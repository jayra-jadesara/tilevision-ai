"""
Duplicates View for TileVision AI (Feature 5 / Task B: Duplicate Detection UI).

A modal dialog that scans the indexed catalog for exact and near-duplicate
tiles and displays them grouped, each tile card showing:
  - Thumbnail
  - Duplicate % (100% for exact matches; a Hamming-distance-derived
    percentage for near-duplicates)
  - Exact / Near Duplicate badge
  - Delete (removes the file from disk + FAISS + SQLite, after confirmation)
  - Open Image / Open Containing Folder (Feature 6, for consistency)
"""

import logging
from pathlib import Path
from typing import List

from PySide6.QtCore import Qt, QUrl, Signal
from PySide6.QtGui import QDesktopServices, QPixmap, QIcon
from PySide6.QtWidgets import (
    QDialog,
    QVBoxLayout,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QScrollArea,
    QWidget,
    QFrame,
    QRadioButton,
    QButtonGroup,
    QProgressBar,
    QMessageBox,
)

from src.core.models import TileImage
from src.core.use_cases.find_duplicates import FindDuplicatesUseCase
from src.presentation.workers.duplicates_worker import DuplicatesWorker
from src.theme.theme_manager import get_palette, get_shared_view_qss
from src.utils.brand_assets import APP_ICON_PATH

logger = logging.getLogger("tilevision.presentation.views.duplicates_view")


class _DuplicateGroupWidget(QFrame):
    """A single duplicate group: a header + a horizontal row of tile cards."""

    # Emitted whenever a tile in this group is deleted, so the parent
    # dialog can update its overall status line / remove empty groups.
    tile_deleted = Signal()

    def __init__(
        self, group: List[TileImage], is_exact: bool, use_case: FindDuplicatesUseCase, parent=None
    ) -> None:
        super().__init__(parent)
        self._use_case = use_case
        self._is_exact = is_exact
        self._tiles = list(group)
        self.setObjectName("DuplicateGroupFrame")

        self._layout = QVBoxLayout(self)
        self._layout.setSpacing(6)

        kind_label = "Exact Duplicates" if is_exact else "Near Duplicates"
        self._header = QLabel(f"{kind_label} — {len(self._tiles)} files")
        self._header.setObjectName("GroupHeader")
        self._layout.addWidget(self._header)

        self._row_layout = QHBoxLayout()
        self._row_layout.setSpacing(10)
        self._reference_hash = self._tiles[0].perceptual_hash if self._tiles else None

        for tile in self._tiles:
            self._row_layout.addWidget(self._build_tile_card(tile))
        self._row_layout.addStretch()
        self._layout.addLayout(self._row_layout)

    def _compute_duplicate_percent(self, tile: TileImage) -> float:
        if self._is_exact:
            return 100.0
        if not self._reference_hash or not tile.perceptual_hash:
            return 0.0
        return self._use_case.similarity_percent(self._reference_hash, tile.perceptual_hash)

    def _build_tile_card(self, tile: TileImage) -> QWidget:
        card = QFrame()
        card.setObjectName("TileCard")
        card_layout = QVBoxLayout(card)
        card_layout.setSpacing(4)

        thumb_label = QLabel()
        pixmap = QPixmap(tile.file_path)
        if not pixmap.isNull():
            thumb_label.setPixmap(
                pixmap.scaled(
                    100, 100, Qt.AspectRatioMode.KeepAspectRatio, Qt.TransformationMode.SmoothTransformation
                )
            )
        thumb_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        thumb_label.setFixedSize(100, 100)
        card_layout.addWidget(thumb_label)

        # Duplicate % + Exact/Near badge (Task B)
        pct = self._compute_duplicate_percent(tile)
        badge_text = f"{'Exact' if self._is_exact else 'Near'} · {pct:.0f}%"
        badge_label = QLabel(badge_text)
        badge_label.setObjectName("DuplicateBadgeExact" if self._is_exact else "DuplicateBadgeNear")
        badge_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        card_layout.addWidget(badge_label)

        name_label = QLabel(tile.file_name)
        name_label.setObjectName("TileCardName")
        name_label.setWordWrap(True)
        name_label.setFixedWidth(100)
        name_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        card_layout.addWidget(name_label)

        button_row = QHBoxLayout()
        open_button = QPushButton("Open")
        open_button.setToolTip("Open Image")
        open_button.setFixedWidth(28)
        open_button.clicked.connect(lambda: self._open_image(tile))
        button_row.addWidget(open_button)

        folder_button = QPushButton("📂")
        folder_button.setToolTip("Open Containing Folder")
        folder_button.setFixedWidth(28)
        folder_button.clicked.connect(lambda: self._open_folder(tile))
        button_row.addWidget(folder_button)

        delete_button = QPushButton("Delete")
        delete_button.setObjectName("DeleteButton")
        delete_button.setToolTip("Delete this duplicate")
        delete_button.setFixedWidth(28)
        delete_button.clicked.connect(lambda: self._on_delete_clicked(tile, card))
        button_row.addWidget(delete_button)

        card_layout.addLayout(button_row)

        return card

    def _on_delete_clicked(self, tile: TileImage, card: QWidget) -> None:
        confirm = QMessageBox.question(
            self,
            "Delete Duplicate",
            f"Permanently delete this file?\n\n{tile.file_path}\n\n"
            "This removes it from disk and from the search index. This cannot be undone.",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            QMessageBox.StandardButton.No,
        )
        if confirm != QMessageBox.StandardButton.Yes:
            return

        try:
            self._use_case.delete_duplicate(tile)
        except RuntimeError as e:
            QMessageBox.critical(self, "Delete Failed", str(e))
            return

        self._tiles = [t for t in self._tiles if t.id != tile.id]
        self._row_layout.removeWidget(card)
        card.deleteLater()
        self._header.setText(
            f"{'Exact Duplicates' if self._is_exact else 'Near Duplicates'} — {len(self._tiles)} files"
        )
        self.tile_deleted.emit()

        if len(self._tiles) <= 1:
            # Fewer than 2 files left means this is no longer a duplicate
            # group at all — remove the whole group card.
            self.setVisible(False)
            self.deleteLater()

    def _open_image(self, tile: TileImage) -> None:
        path = Path(tile.file_path)
        if path.exists():
            QDesktopServices.openUrl(QUrl.fromLocalFile(str(path)))
        else:
            QMessageBox.warning(self, "File Not Found", f"This image no longer exists:\n{path}")

    def _open_folder(self, tile: TileImage) -> None:
        folder = Path(tile.file_path).parent
        if folder.exists():
            QDesktopServices.openUrl(QUrl.fromLocalFile(str(folder)))
        else:
            QMessageBox.warning(self, "Folder Not Found", f"This folder no longer exists:\n{folder}")


class DuplicatesView(QDialog):
    """Modal dialog for scanning and reviewing duplicate/near-duplicate tiles."""

    def __init__(self, use_case: FindDuplicatesUseCase, parent=None, theme: str = "dark") -> None:
        super().__init__(parent)
        self._use_case = use_case
        self._worker: DuplicatesWorker = None
        self._theme = theme

        self.setWindowTitle("Duplicate Detection")
        if APP_ICON_PATH.exists():
            self.setWindowIcon(QIcon(str(APP_ICON_PATH)))
        self.resize(800, 620)
        self._setup_ui()
        self._apply_styles()

    def _setup_ui(self) -> None:
        layout = QVBoxLayout(self)
        layout.setContentsMargins(20, 16, 20, 16)
        layout.setSpacing(12)

        title = QLabel("Duplicate and Near-Duplicate Tile Detection")
        title.setObjectName("PageTitle")
        layout.addWidget(title)

        options_row = QHBoxLayout()
        self._include_near_radio = QRadioButton("Exact + Near Duplicates (thorough)")
        self._exact_only_radio = QRadioButton("Exact Duplicates Only (fast)")
        self._include_near_radio.setChecked(True)
        group = QButtonGroup(self)
        group.addButton(self._include_near_radio)
        group.addButton(self._exact_only_radio)
        options_row.addWidget(self._include_near_radio)
        options_row.addWidget(self._exact_only_radio)
        options_row.addStretch()

        self._scan_button = QPushButton("Scan for Duplicates")
        self._scan_button.setObjectName("ScanButton")
        self._scan_button.clicked.connect(self._on_scan_clicked)
        options_row.addWidget(self._scan_button)
        layout.addLayout(options_row)

        self._progress_bar = QProgressBar()
        self._progress_bar.setRange(0, 0)
        self._progress_bar.setTextVisible(False)
        self._progress_bar.setFixedHeight(4)
        self._progress_bar.setVisible(False)
        layout.addWidget(self._progress_bar)

        self._status_label = QLabel("Click \"Scan for Duplicates\" to begin.")
        self._status_label.setObjectName("StatusLabel")
        layout.addWidget(self._status_label)

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        self._results_container = QWidget()
        self._results_layout = QVBoxLayout(self._results_container)
        self._results_layout.addStretch()
        scroll.setWidget(self._results_container)
        layout.addWidget(scroll, stretch=1)

    def _on_scan_clicked(self) -> None:
        self._scan_button.setEnabled(False)
        self._progress_bar.setVisible(True)
        self._status_label.setText("Scanning catalog for duplicates...")
        self._clear_results()

        include_near = self._include_near_radio.isChecked()
        self._worker = DuplicatesWorker(self._use_case, include_near_duplicates=include_near)
        self._worker.scan_completed.connect(self._on_scan_completed)
        self._worker.scan_failed.connect(self._on_scan_failed)
        self._worker.finished.connect(self._worker.deleteLater)
        self._worker.start()

    def _on_scan_completed(self, exact_groups: list, near_groups: list) -> None:
        self._scan_button.setEnabled(True)
        self._progress_bar.setVisible(False)

        total_groups = len(exact_groups) + len(near_groups)
        if total_groups == 0:
            self._status_label.setText("No duplicates found in your catalog.")
        else:
            self._status_label.setText(
                f"Found {len(exact_groups)} exact duplicate group(s) and "
                f"{len(near_groups)} near-duplicate group(s)."
            )

        for group in exact_groups:
            widget = _DuplicateGroupWidget(group, is_exact=True, use_case=self._use_case)
            widget.tile_deleted.connect(self._on_tile_deleted)
            self._results_layout.insertWidget(self._results_layout.count() - 1, widget)
        for group in near_groups:
            widget = _DuplicateGroupWidget(group, is_exact=False, use_case=self._use_case)
            widget.tile_deleted.connect(self._on_tile_deleted)
            self._results_layout.insertWidget(self._results_layout.count() - 1, widget)

    def _on_tile_deleted(self) -> None:
        self._status_label.setText("Duplicate deleted. Re-scan to refresh remaining groups.")

    def _on_scan_failed(self, message: str) -> None:
        self._scan_button.setEnabled(True)
        self._progress_bar.setVisible(False)
        self._status_label.setText(f"Scan failed: {message}")
        QMessageBox.warning(self, "Scan Failed", message)

    def _clear_results(self) -> None:
        while self._results_layout.count() > 1:
            item = self._results_layout.takeAt(0)
            if item.widget():
                item.widget().deleteLater()

    def _apply_styles(self) -> None:
        p = get_palette(self._theme)
        self.setStyleSheet(
            get_shared_view_qss(self._theme)
            + f"""
            QDialog {{ background-color: {p['bg_app']}; }}
            QWidget {{ color: {p['text_primary']}; }}
            #StatusLabel {{ color: {p['text_muted']}; font-size: 12px; }}
            #DuplicateGroupFrame {{
                background-color: {p['bg_panel']}; border: 1px solid {p['border']}; border-radius: 8px;
                padding: 10px; margin-bottom: 8px;
            }}
            #GroupHeader {{ font-weight: 600; color: {p['text_secondary']}; font-size: 12px; }}
            #TileCard {{ background-color: {p['bg_panel_alt']}; border-radius: 6px; padding: 6px; }}
            #TileCardName {{ font-size: 10px; color: {p['text_muted']}; }}
            #DuplicateBadgeExact {{
                background-color: {p['success_bg']}; color: {p['success_text']}; border-radius: 4px;
                font-size: 10px; font-weight: 600; padding: 2px;
            }}
            #DuplicateBadgeNear {{
                background-color: {p['warning_bg']}; color: {p['warning_text']}; border-radius: 4px;
                font-size: 10px; font-weight: 600; padding: 2px;
            }}
            #DeleteButton {{ background-color: {p['danger_bg']}; }}
            #DeleteButton:hover {{ background-color: {p['danger_hover']}; }}
            QRadioButton {{ font-size: 12px; }}
            """
        )
