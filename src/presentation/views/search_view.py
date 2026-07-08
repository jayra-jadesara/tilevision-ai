"""
Search View for TileVision AI (Feature 2: AI Tile Search).

Provides a fully implemented PySide6 QWidget panel that lets a user find
visually similar tiles by dragging an image in or browsing for one. This
view is purely presentation: it delegates all logic to SearchViewModel.

UI Sections:
    1. Query Panel — drag-and-drop zone + Browse button + query preview.
    2. Results Table — Top-K matches: thumbnail, similarity %, product code,
       brand, and image path. Double-click a row (or right-click for a
       context menu) to open the image or its containing folder.
    3. Status line — live search status ("Searching...", result counts, errors).

Design Decision:
    Uses QSS (Qt Style Sheets) for dark-themed styling consistent with the
    rest of the app. The view is completely decoupled from business logic
    via the ViewModel pattern — it only ever talks to SearchViewModel.
"""

import logging
from pathlib import Path
from typing import List, Optional

from PySide6.QtCore import Qt, Slot, QUrl, QSize
from PySide6.QtGui import QDragEnterEvent, QDropEvent, QDesktopServices, QPixmap, QIcon
from PySide6.QtWidgets import (
    QWidget,
    QVBoxLayout,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QFileDialog,
    QFrame,
    QSizePolicy,
    QTableWidget,
    QTableWidgetItem,
    QHeaderView,
    QAbstractItemView,
    QMenu,
    QMessageBox,
    QProgressBar,
)

from src.core.models import SearchResult
from src.presentation.viewmodels.search_viewmodel import SearchViewModel, SearchState

logger = logging.getLogger("tilevision.presentation.views.search_view")

# Query image formats accepted for drag-and-drop / browse. Broader than the
# indexing pipeline's supported set on purpose — a customer's WhatsApp photo
# or phone screenshot might arrive as HEIC/BMP/GIF, and PIL can still read
# most of these to extract a query embedding even though we never index
# non jpg/jpeg/png/webp files into the catalog itself.
_QUERY_IMAGE_EXTENSIONS = {
    ".jpg", ".jpeg", ".png", ".webp", ".bmp", ".gif", ".tiff", ".tif"
}

_TABLE_COLUMNS = ["Thumbnail", "Similarity", "Product Code", "Brand", "Category", "Image Path"]


class DropZone(QFrame):
    """
    A clickable, drag-and-drop capable frame for selecting a query image.

    Emits image_selected(path) either when a valid image file is dropped
    onto it, or when the user clicks it and picks one via a file dialog.
    """

    def __init__(self, on_image_selected, parent: Optional[QWidget] = None) -> None:
        """
        Args:
            on_image_selected: Callable[[str], None] invoked with the
                absolute path of the chosen image.
            parent: Optional Qt parent widget.
        """
        super().__init__(parent)
        self._on_image_selected = on_image_selected
        self.setObjectName("DropZone")
        self.setAcceptDrops(True)
        self.setCursor(Qt.CursorShape.PointingHandCursor)
        self.setMinimumHeight(180)

        layout = QVBoxLayout(self)
        layout.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.setSpacing(8)

        self._icon_label = QLabel("🖼️")
        self._icon_label.setObjectName("DropZoneIcon")
        self._icon_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(self._icon_label)

        self._preview_label = QLabel()
        self._preview_label.setObjectName("DropZonePreview")
        self._preview_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._preview_label.setVisible(False)
        layout.addWidget(self._preview_label)

        self._title_label = QLabel("Drag a tile photo here")
        self._title_label.setObjectName("DropZoneTitle")
        self._title_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(self._title_label)

        self._subtitle_label = QLabel("or click to browse for an image")
        self._subtitle_label.setObjectName("DropZoneSubtitle")
        self._subtitle_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(self._subtitle_label)

    def set_busy(self, busy: bool) -> None:
        """Disable interaction while a search is in progress."""
        self.setEnabled(not busy)

    def show_preview(self, image_path: str) -> None:
        """Display a small preview thumbnail of the selected query image."""
        pixmap = QPixmap(image_path)
        if not pixmap.isNull():
            scaled = pixmap.scaled(
                96, 96, Qt.AspectRatioMode.KeepAspectRatio, Qt.TransformationMode.SmoothTransformation
            )
            self._preview_label.setPixmap(scaled)
            self._preview_label.setVisible(True)
            self._icon_label.setVisible(False)
        self._title_label.setText(Path(image_path).name)
        self._subtitle_label.setText("Click to search a different image")

    def reset(self) -> None:
        """Reset the drop zone back to its empty prompt state."""
        self._preview_label.clear()
        self._preview_label.setVisible(False)
        self._icon_label.setVisible(True)
        self._title_label.setText("Drag a tile photo here")
        self._subtitle_label.setText("or click to browse for an image")

    # ── Drag & Drop ──────────────────────────────────────────────────────

    def dragEnterEvent(self, event: QDragEnterEvent) -> None:
        if event.mimeData().hasUrls():
            for url in event.mimeData().urls():
                path = Path(url.toLocalFile())
                if path.suffix.lower() in _QUERY_IMAGE_EXTENSIONS:
                    event.acceptProposedAction()
                    self.setProperty("dragActive", True)
                    self.style().unpolish(self)
                    self.style().polish(self)
                    return
        event.ignore()

    def dragLeaveEvent(self, event) -> None:
        self.setProperty("dragActive", False)
        self.style().unpolish(self)
        self.style().polish(self)

    def dropEvent(self, event: QDropEvent) -> None:
        self.setProperty("dragActive", False)
        self.style().unpolish(self)
        self.style().polish(self)

        for url in event.mimeData().urls():
            path = Path(url.toLocalFile())
            if path.suffix.lower() in _QUERY_IMAGE_EXTENSIONS and path.is_file():
                self._on_image_selected(str(path))
                event.acceptProposedAction()
                return
        event.ignore()

    def mousePressEvent(self, event) -> None:
        if event.button() == Qt.MouseButton.LeftButton and self.isEnabled():
            self._browse()
        super().mousePressEvent(event)

    def _browse(self) -> None:
        extensions = " ".join(f"*{ext}" for ext in sorted(_QUERY_IMAGE_EXTENSIONS))
        file_path, _ = QFileDialog.getOpenFileName(
            self, "Select a tile photo to search", "", f"Image Files ({extensions})"
        )
        if file_path:
            self._on_image_selected(file_path)


class SearchView(QWidget):
    """
    Full-featured Visual Similarity Search panel widget.

    Connects to a SearchViewModel instance to drive all state and logic.
    """

    def __init__(self, viewmodel: SearchViewModel, parent: Optional[QWidget] = None) -> None:
        """
        Initialize the SearchView.

        Args:
            viewmodel: The bound SearchViewModel instance.
            parent: Optional Qt parent widget.
        """
        super().__init__(parent)
        self._viewmodel = viewmodel
        self._current_results: List[SearchResult] = []
        self._setup_ui()
        self._connect_signals()
        self._apply_styles()
        logger.debug("SearchView initialized.")

    # ── UI Construction ─────────────────────────────────────────────────

    def _setup_ui(self) -> None:
        self.setObjectName("SearchView")

        root_layout = QVBoxLayout(self)
        root_layout.setContentsMargins(24, 20, 24, 20)
        root_layout.setSpacing(16)

        root_layout.addWidget(self._build_header())
        root_layout.addWidget(self._build_query_panel())
        root_layout.addWidget(self._build_progress_bar())
        root_layout.addWidget(self._build_results_table(), stretch=1)
        root_layout.addWidget(self._build_status_line())

    def _build_header(self) -> QWidget:
        header = QWidget()
        layout = QVBoxLayout(header)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(2)

        title = QLabel("Visual Tile Search")
        title.setObjectName("PageTitle")
        layout.addWidget(title)

        subtitle = QLabel("Drag a tile photo in, or browse for one, to find visually similar tiles.")
        subtitle.setObjectName("PageSubtitle")
        layout.addWidget(subtitle)

        return header

    def _build_query_panel(self) -> QWidget:
        panel = QWidget()
        layout = QHBoxLayout(panel)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(12)

        self._drop_zone = DropZone(on_image_selected=self._on_image_chosen)
        layout.addWidget(self._drop_zone, stretch=1)

        button_col = QVBoxLayout()
        button_col.setSpacing(8)
        button_col.addStretch()

        self._clear_button = QPushButton("✕  Clear")
        self._clear_button.setObjectName("SecondaryButton")
        self._clear_button.clicked.connect(self._on_clear_clicked)
        self._clear_button.setEnabled(False)
        button_col.addWidget(self._clear_button)

        button_col.addStretch()
        layout.addLayout(button_col)

        return panel

    def _build_progress_bar(self) -> QWidget:
        self._progress_bar = QProgressBar()
        self._progress_bar.setObjectName("SearchProgressBar")
        self._progress_bar.setRange(0, 0)  # indeterminate — search duration is sub-second
        self._progress_bar.setTextVisible(False)
        self._progress_bar.setFixedHeight(4)
        self._progress_bar.setVisible(False)
        return self._progress_bar

    def _build_results_table(self) -> QWidget:
        self._results_table = QTableWidget(0, len(_TABLE_COLUMNS))
        self._results_table.setObjectName("ResultsTable")
        self._results_table.setHorizontalHeaderLabels(_TABLE_COLUMNS)
        self._results_table.verticalHeader().setVisible(False)
        self._results_table.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        self._results_table.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
        self._results_table.setSelectionMode(QAbstractItemView.SelectionMode.SingleSelection)
        self._results_table.setIconSize(QSize(72, 72))
        self._results_table.verticalHeader().setDefaultSectionSize(80)
        self._results_table.setAlternatingRowColors(True)
        self._results_table.setContextMenuPolicy(Qt.ContextMenuPolicy.CustomContextMenu)

        header = self._results_table.horizontalHeader()
        header.setSectionResizeMode(0, QHeaderView.ResizeMode.Fixed)
        self._results_table.setColumnWidth(0, 88)
        header.setSectionResizeMode(1, QHeaderView.ResizeMode.ResizeToContents)
        header.setSectionResizeMode(2, QHeaderView.ResizeMode.ResizeToContents)
        header.setSectionResizeMode(3, QHeaderView.ResizeMode.ResizeToContents)
        header.setSectionResizeMode(4, QHeaderView.ResizeMode.ResizeToContents)
        header.setSectionResizeMode(5, QHeaderView.ResizeMode.Stretch)

        self._results_table.cellDoubleClicked.connect(self._on_row_double_clicked)
        self._results_table.customContextMenuRequested.connect(self._on_results_context_menu)

        self._empty_state_label = QLabel(
            "No results yet.\nDrag or browse a tile photo above to start searching."
        )
        self._empty_state_label.setObjectName("EmptyStateLabel")
        self._empty_state_label.setAlignment(Qt.AlignmentFlag.AlignCenter)

        container = QWidget()
        container_layout = QVBoxLayout(container)
        container_layout.setContentsMargins(0, 0, 0, 0)
        container_layout.addWidget(self._empty_state_label)
        container_layout.addWidget(self._results_table)
        self._results_table.setVisible(False)
        self._results_container = container
        return container

    def _build_status_line(self) -> QWidget:
        self._status_label = QLabel("Ready. Drag an image or click Browse to search.")
        self._status_label.setObjectName("SearchStatusLabel")
        return self._status_label

    # ── Signal Wiring ────────────────────────────────────────────────────

    def _connect_signals(self) -> None:
        self._viewmodel.state_changed.connect(self._on_state_changed)
        self._viewmodel.results_ready.connect(self._on_results_ready)
        self._viewmodel.status_message.connect(self._status_label.setText)
        self._viewmodel.search_error.connect(self._on_search_error)
        self._viewmodel.query_image_selected.connect(self._drop_zone.show_preview)

    # ── Event Handlers ───────────────────────────────────────────────────

    def _on_image_chosen(self, image_path: str) -> None:
        logger.info(f"Query image selected: {image_path}")
        self._viewmodel.search_by_image(image_path)

    def _on_clear_clicked(self) -> None:
        self._drop_zone.reset()
        self._viewmodel.clear_results()

    @Slot(str)
    def _on_state_changed(self, state: str) -> None:
        is_searching = state == SearchState.SEARCHING
        self._drop_zone.set_busy(is_searching)
        self._progress_bar.setVisible(is_searching)
        self._clear_button.setEnabled(state != SearchState.IDLE)

    @Slot(list)
    def _on_results_ready(self, results: List[SearchResult]) -> None:
        self._current_results = results
        self._populate_table(results)

    @Slot(str)
    def _on_search_error(self, message: str) -> None:
        QMessageBox.warning(self, "Search Failed", message)

    def _populate_table(self, results: List[SearchResult]) -> None:
        self._results_table.setRowCount(0)

        if not results:
            self._results_table.setVisible(False)
            self._empty_state_label.setVisible(True)
            return

        self._empty_state_label.setVisible(False)
        self._results_table.setVisible(True)
        self._results_table.setRowCount(len(results))

        for row, result in enumerate(results):
            tile = result.tile

            thumb_item = QTableWidgetItem()
            pixmap = QPixmap(result.thumbnail_path)
            if not pixmap.isNull():
                thumb_item.setIcon(QIcon(pixmap))
            thumb_item.setFlags(thumb_item.flags() & ~Qt.ItemFlag.ItemIsEditable)
            self._results_table.setItem(row, 0, thumb_item)

            similarity_item = QTableWidgetItem(f"{result.similarity_score:.1f}%")
            similarity_item.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
            self._results_table.setItem(row, 1, similarity_item)

            self._results_table.setItem(row, 2, QTableWidgetItem(tile.product_code or "—"))
            self._results_table.setItem(row, 3, QTableWidgetItem(tile.brand or "—"))
            self._results_table.setItem(row, 4, QTableWidgetItem(tile.category or "—"))

            path_item = QTableWidgetItem(tile.file_path)
            path_item.setToolTip(tile.file_path)
            self._results_table.setItem(row, 5, path_item)

    def _on_row_double_clicked(self, row: int, _column: int) -> None:
        self._open_image_at_row(row)

    def _on_results_context_menu(self, position) -> None:
        row = self._results_table.rowAt(position.y())
        if row < 0 or row >= len(self._current_results):
            return

        menu = QMenu(self)
        open_image_action = menu.addAction("🖼️  Open Image")
        open_folder_action = menu.addAction("📂  Open Containing Folder")

        chosen = menu.exec(self._results_table.viewport().mapToGlobal(position))
        if chosen == open_image_action:
            self._open_image_at_row(row)
        elif chosen == open_folder_action:
            self._open_folder_at_row(row)

    def _open_image_at_row(self, row: int) -> None:
        if row < 0 or row >= len(self._current_results):
            return
        path = Path(self._current_results[row].tile.file_path)
        if not path.exists():
            QMessageBox.warning(self, "File Not Found", f"This image no longer exists:\n{path}")
            return
        QDesktopServices.openUrl(QUrl.fromLocalFile(str(path)))

    def _open_folder_at_row(self, row: int) -> None:
        if row < 0 or row >= len(self._current_results):
            return
        path = Path(self._current_results[row].tile.file_path)
        folder = path.parent
        if not folder.exists():
            QMessageBox.warning(self, "Folder Not Found", f"This folder no longer exists:\n{folder}")
            return
        QDesktopServices.openUrl(QUrl.fromLocalFile(str(folder)))

    # ── Styling ──────────────────────────────────────────────────────────

    def _apply_styles(self) -> None:
        self.setStyleSheet(
            """
            #PageTitle { font-size: 20px; font-weight: 700; color: #E8EAF6; }
            #PageSubtitle { font-size: 12px; color: #8A8FA3; }

            #DropZone {
                background-color: #232634;
                border: 2px dashed #3A3F52;
                border-radius: 10px;
            }
            #DropZone[dragActive="true"] {
                border: 2px dashed #5C6BC0;
                background-color: #262B3D;
            }
            #DropZoneIcon { font-size: 36px; }
            #DropZoneTitle { font-size: 14px; font-weight: 600; color: #E8EAF6; }
            #DropZoneSubtitle { font-size: 11px; color: #8A8FA3; }

            #SecondaryButton {
                background-color: #2A2E3D;
                color: #C7CAD9;
                border: 1px solid #3A3F52;
                border-radius: 6px;
                padding: 8px 14px;
                font-size: 12px;
            }
            #SecondaryButton:hover:enabled { background-color: #333852; }
            #SecondaryButton:disabled { color: #55596B; }

            #ResultsTable {
                background-color: #1E212C;
                alternate-background-color: #232634;
                gridline-color: #2E3243;
                color: #D6D9E8;
                border: 1px solid #2E3243;
                border-radius: 8px;
            }
            #ResultsTable::item:selected { background-color: #3B4270; }
            QHeaderView::section {
                background-color: #262B3D;
                color: #ACB0C4;
                padding: 6px;
                border: none;
                border-bottom: 1px solid #2E3243;
                font-weight: 600;
                font-size: 11px;
            }

            #EmptyStateLabel { color: #6C7086; font-size: 13px; padding: 40px; }
            #SearchStatusLabel { color: #8A8FA3; font-size: 12px; }
            #SearchProgressBar {
                background-color: #232634;
                border-radius: 2px;
            }
            #SearchProgressBar::chunk { background-color: #5C6BC0; }
            """
        )
