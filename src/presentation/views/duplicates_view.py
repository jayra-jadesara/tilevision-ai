"""
Duplicates View for TileVision AI (Feature 5: Duplicate Detection).

A modal dialog that scans the indexed catalog for exact and near-duplicate
tiles and displays them grouped, with quick actions to open each file or
its containing folder — mirroring the Search view's Open Image / Open
Folder actions (Feature 6) for consistency.
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

logger = logging.getLogger("tilevision.presentation.views.duplicates_view")


class _DuplicateGroupWidget(QFrame):
    """A single duplicate group: a header + a horizontal row of tile cards."""

    def __init__(self, group: List[TileImage], group_kind: str, parent=None) -> None:
        super().__init__(parent)
        self.setObjectName("DuplicateGroupFrame")
        layout = QVBoxLayout(self)
        layout.setSpacing(6)

        header = QLabel(f"{group_kind} — {len(group)} files")
        header.setObjectName("GroupHeader")
        layout.addWidget(header)

        row = QHBoxLayout()
        row.setSpacing(10)
        for tile in group:
            row.addWidget(self._build_tile_card(tile))
        row.addStretch()
        layout.addLayout(row)

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

        name_label = QLabel(tile.file_name)
        name_label.setObjectName("TileCardName")
        name_label.setWordWrap(True)
        name_label.setFixedWidth(100)
        name_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        card_layout.addWidget(name_label)

        button_row = QHBoxLayout()
        open_button = QPushButton("🖼️")
        open_button.setToolTip("Open Image")
        open_button.setFixedWidth(30)
        open_button.clicked.connect(lambda: self._open_image(tile))
        button_row.addWidget(open_button)

        folder_button = QPushButton("📂")
        folder_button.setToolTip("Open Containing Folder")
        folder_button.setFixedWidth(30)
        folder_button.clicked.connect(lambda: self._open_folder(tile))
        button_row.addWidget(folder_button)
        card_layout.addLayout(button_row)

        return card

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

    def __init__(self, use_case: FindDuplicatesUseCase, parent=None) -> None:
        super().__init__(parent)
        self._use_case = use_case
        self._worker: DuplicatesWorker = None

        self.setWindowTitle("Duplicate Detection")
        self.resize(760, 600)
        self._setup_ui()
        self._apply_styles()

    def _setup_ui(self) -> None:
        layout = QVBoxLayout(self)
        layout.setContentsMargins(20, 16, 20, 16)
        layout.setSpacing(12)

        title = QLabel("🔍  Duplicate & Near-Duplicate Tile Detection")
        title.setObjectName("Title")
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

        self._scan_button = QPushButton("▶  Scan for Duplicates")
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
            self._status_label.setText("✅ No duplicates found in your catalog.")
        else:
            self._status_label.setText(
                f"Found {len(exact_groups)} exact duplicate group(s) and "
                f"{len(near_groups)} near-duplicate group(s)."
            )

        for group in exact_groups:
            self._results_layout.insertWidget(
                self._results_layout.count() - 1,
                _DuplicateGroupWidget(group, "Exact Duplicates"),
            )
        for group in near_groups:
            self._results_layout.insertWidget(
                self._results_layout.count() - 1,
                _DuplicateGroupWidget(group, "Near Duplicates"),
            )

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
        self.setStyleSheet(
            """
            QDialog { background-color: #1A1D26; }
            QWidget { color: #E8EAF6; }
            #Title { font-size: 16px; font-weight: 700; }
            #StatusLabel { color: #8A8FA3; font-size: 12px; }
            #ScanButton { background-color: #3949AB; border-radius: 6px; padding: 8px 16px; font-weight: 600; }
            #ScanButton:hover:enabled { background-color: #5C6BC0; }
            #ScanButton:disabled { background-color: #2A2E3D; color: #55596B; }
            #DuplicateGroupFrame {
                background-color: #232634; border: 1px solid #2E3243; border-radius: 8px;
                padding: 10px; margin-bottom: 8px;
            }
            #GroupHeader { font-weight: 600; color: #ACB0C4; font-size: 12px; }
            #TileCard { background-color: #1E212C; border-radius: 6px; padding: 6px; }
            #TileCardName { font-size: 10px; color: #8A8FA3; }
            QRadioButton { font-size: 12px; }
            """
        )
