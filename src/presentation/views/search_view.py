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
from typing import Callable, List, Optional

from PySide6.QtCore import Qt, Slot, QUrl, QSize, QTimer
from PySide6.QtGui import QDragEnterEvent, QDropEvent, QDesktopServices, QPixmap, QIcon, QColor
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
    QProgressBar,
    QComboBox,
    QApplication,
)

from src.core.models import SearchResult
from src.presentation.viewmodels.search_viewmodel import SearchViewModel, SearchState
from src.presentation.views.crop_dialog import CropDialog
from src.presentation.workers.tile_crop_worker import TileCropWorker
from src.theme.theme_manager import get_palette, get_shared_view_qss
from src.utils.query_image_hints import confidence_message, should_suggest_crop
from src.utils.image_utils import validate_image
from src.presentation.dialogs import message_box

logger = logging.getLogger("tilevision.presentation.views.search_view")

from src.utils.image_formats import query_image_extensions

# Query formats include HEIC on Mac when pillow-heif is installed.
_QUERY_IMAGE_EXTENSIONS = query_image_extensions()

_TABLE_COLUMNS = [
    "Thumbnail",
    "Similarity",
    "Product Code",
    "Brand",
    "Category",
    "Preview",
    "Image Path"
]

class DropZone(QFrame):
    """
    A clickable, drag-and-drop capable frame for selecting a query image.

    Emits image_selected(path) either when a valid image file is dropped
    onto it, or when the user clicks it and picks one via a file dialog.
    """

    def __init__(
        self,
        on_image_selected,
        parent: Optional[QWidget] = None,
        on_drop_rejected: Optional[Callable[[str], None]] = None,
    ) -> None:
        """
        Args:
            on_image_selected: Callable[[str], None] invoked with the
                absolute path of the chosen image.
            parent: Optional Qt parent widget.
            on_drop_rejected: Optional Callable[[str], None] with a
                user-visible reason when a drop is ignored.
        """
        super().__init__(parent)
        self._on_image_selected = on_image_selected
        self._on_drop_rejected = on_drop_rejected
        self.setObjectName("DropZone")
        self.setAcceptDrops(True)
        self.setCursor(Qt.CursorShape.PointingHandCursor)
        self.setMinimumHeight(180)

        layout = QVBoxLayout(self)
        layout.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.setSpacing(8)

        self._icon_label = QLabel("Tile Image")
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

        if not event.mimeData().hasUrls():
            reason = "Drop ignored — no file path found. Save the image locally, then drop the file."
            logger.warning("[SEARCH] %s", reason)
            if self._on_drop_rejected is not None:
                self._on_drop_rejected(reason)
            event.ignore()
            return

        for url in event.mimeData().urls():
            local = url.toLocalFile()
            if not local:
                continue
            path = Path(local)
            suffix = path.suffix.lower()
            if suffix not in _QUERY_IMAGE_EXTENSIONS:
                reason = (
                    f"Unsupported image type '{suffix or '(none)'}'. "
                    "Use JPG, JPEG, PNG, WEBP, BMP, or TIFF."
                )
                logger.warning("[SEARCH] Drop rejected: %s (%s)", reason, path)
                if self._on_drop_rejected is not None:
                    self._on_drop_rejected(reason)
                event.ignore()
                return
            if not path.is_file():
                reason = f"Dropped path is not a readable file: {path.name}"
                logger.warning("[SEARCH] %s", reason)
                if self._on_drop_rejected is not None:
                    self._on_drop_rejected(reason)
                event.ignore()
                return
            if not validate_image(path):
                reason = f"Selected file is not a valid, readable image: {path.name}"
                logger.warning("[SEARCH] Drop rejected: %s", reason)
                if self._on_drop_rejected is not None:
                    self._on_drop_rejected(reason)
                event.ignore()
                return
            logger.info("[SEARCH] Drop accepted: %s", path)
            self._on_image_selected(str(path))
            event.acceptProposedAction()
            return

        reason = (
            "Drop ignored — could not read a local image file. "
            "If this came from a browser or cloud app, save it to disk first."
        )
        logger.warning("[SEARCH] %s", reason)
        if self._on_drop_rejected is not None:
            self._on_drop_rejected(reason)
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


class _ResultPreviewPanel(QWidget):
    """
    Lightweight, reusable "large preview" popup for Task C: Search UX.
    A single click on a results row calls show_result() on one shared
    instance, which updates its content and pops to the front — cheaper
    than constructing a new dialog per click.
    """

    def __init__(self, parent=None, theme: str = "dark") -> None:
        super().__init__(parent, Qt.WindowType.Tool)
        self.setWindowTitle("Tile Preview")
        self.setMinimumSize(360, 420)
        self._theme = theme
        self._setup_ui()
        self._apply_styles()

    def _setup_ui(self) -> None:
        layout = QVBoxLayout(self)
        layout.setContentsMargins(16, 16, 16, 16)
        layout.setSpacing(10)

        self._image_label = QLabel()
        self._image_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._image_label.setFixedSize(320, 320)
        self._image_label.setObjectName("PreviewImage")
        layout.addWidget(self._image_label)

        self._details_label = QLabel()
        self._details_label.setObjectName("PreviewDetails")
        self._details_label.setWordWrap(True)
        layout.addWidget(self._details_label)
        layout.addStretch()

    def show_result(self, result: SearchResult) -> None:
        p = get_palette(self._theme)
        tile = result.tile
        pixmap = QPixmap(tile.file_path)
        if pixmap.isNull():
            pixmap = QPixmap(result.thumbnail_path)
        if not pixmap.isNull():
            scaled = pixmap.scaled(
                320, 320, Qt.AspectRatioMode.KeepAspectRatio, Qt.TransformationMode.SmoothTransformation
            )
            self._image_label.setPixmap(scaled)

        self._details_label.setText(
            f"<b>{tile.file_name}</b><br>"
            f"Similarity: {result.similarity_score:.1f}%<br>"
            f"Product Code: {tile.product_code or '—'}<br>"
            f"Brand: {tile.brand or '—'}<br>"
            f"Category: {tile.category or '—'}<br>"
            f"<span style='color:{p['text_muted']}; font-size:10px;'>{tile.file_path}</span>"
        )
        self.show()
        self.raise_()
        self.activateWindow()

    def set_theme(self, theme: str) -> None:
        """Re-skin this panel for a newly-selected theme."""
        self._theme = theme
        self._apply_styles()

    def _apply_styles(self) -> None:
        p = get_palette(self._theme)
        self.setStyleSheet(
            f"""
            QWidget {{ background-color: {p['bg_panel_alt']}; color: {p['text_primary']}; }}
            #PreviewImage {{ background-color: {p['bg_sidebar']}; border-radius: 8px; }}
            #PreviewDetails {{ font-size: 12px; line-height: 1.5; }}
            """
        )


class SearchView(QWidget):
    """
    Full-featured Visual Similarity Search panel widget.

    Connects to a SearchViewModel instance to drive all state and logic.
    """

    def __init__(
        self, viewmodel: SearchViewModel, theme: str = "dark", parent: Optional[QWidget] = None,
        on_open_profiles_settings: Optional[Callable[[], None]] = None,
        catalogue_master_service=None,
    ) -> None:
        """
        Initialize the SearchView.

        Args:
            viewmodel: The bound SearchViewModel instance.
            theme: Initial theme ("dark"/"light") to render with.
            parent: Optional Qt parent widget.
        """
        super().__init__(parent)
        self._theme = theme
        self._viewmodel = viewmodel
        self._on_open_profiles_settings = on_open_profiles_settings
        self._catalogue_master_service = catalogue_master_service
        self._current_results: List[SearchResult] = []
        self._results_generation = 0
        self._current_query_image_path: Optional[str] = None
        self._crop_worker: Optional[TileCropWorker] = None
        self._crop_busy = False
        self._preview_panel = _ResultPreviewPanel(self, theme=theme)
        self._search_animation_timer = QTimer(self)
        self._search_animation_timer.setInterval(400)
        self._search_animation_timer.timeout.connect(self._tick_searching_animation)
        self._search_animation_dots = 0
        self._search_animation_base_text = "Searching"
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
        root_layout.addWidget(self._build_filter_bar())
        root_layout.addWidget(self._build_progress_bar())
        self._confidence_banner = QLabel("")
        self._confidence_banner.setObjectName("ConfidenceBanner")
        self._confidence_banner.setWordWrap(True)
        self._confidence_banner.setVisible(False)
        root_layout.addWidget(self._confidence_banner)
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

        self._drop_zone = DropZone(
            on_image_selected=self._on_image_chosen,
            on_drop_rejected=self._on_drop_rejected,
        )
        layout.addWidget(self._drop_zone, stretch=1)

        button_col = QVBoxLayout()
        button_col.setSpacing(8)
        button_col.addStretch()

        self._export_button = QPushButton("Export Catalogue")
        self._export_button.setObjectName("SecondaryButton")
        self._export_button.clicked.connect(self._on_export_clicked)
        button_col.addWidget(self._export_button)

        self._crop_button = QPushButton("Crop and Search")
        self._crop_button.setObjectName("SecondaryButton")
        self._crop_button.clicked.connect(self._on_crop_clicked)
        self._crop_button.setEnabled(False)
        button_col.addWidget(self._crop_button)

        self._auto_crop_button = QPushButton("Auto Crop & Search")
        self._auto_crop_button.setObjectName("SecondaryButton")
        self._auto_crop_button.setToolTip(
            "Fast tile-region crop for room photos (OpenCV, no SAM). "
            "Use Crop and Search for a manual selection."
        )
        self._auto_crop_button.clicked.connect(self._on_auto_crop_clicked)
        self._auto_crop_button.setEnabled(False)
        button_col.addWidget(self._auto_crop_button)

        self._precise_crop_button = QPushButton("Precise Crop & Search")
        self._precise_crop_button.setObjectName("SecondaryButton")
        self._precise_crop_button.setToolTip(
            "Precise tile crop for Windows, Mac Intel, and Mac Apple Silicon. "
            "Uses SAM 2 when enabled in Settings → Experimental (or TILEVISION_ENABLE_SAM2=1); "
            "otherwise GrabCut. Not used by default search."
        )
        self._precise_crop_button.clicked.connect(self._on_precise_crop_clicked)
        self._precise_crop_button.setEnabled(False)
        button_col.addWidget(self._precise_crop_button)

        self._clear_button = QPushButton("Clear")
        self._clear_button.setObjectName("SecondaryButton")
        self._clear_button.clicked.connect(self._on_clear_clicked)
        self._clear_button.setEnabled(False)
        button_col.addWidget(self._clear_button)

        self._history_button = QPushButton("Recent Searches")
        self._history_button.setObjectName("SecondaryButton")
        self._history_button.clicked.connect(self._on_history_clicked)
        # Only open the popup when the user clicks Recent Searches.
        # Search completion also refreshes history via the same signal — that
        # must NOT pop a blocking QMenu (it freezes offscreen/CI and surprises customers).
        self._history_menu_pending = False
        button_col.addWidget(self._history_button)

        button_col.addStretch()
        layout.addLayout(button_col)

        return panel

    def _build_filter_bar(self) -> QWidget:
        bar = QWidget()
        layout = QHBoxLayout(bar)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(10)

        filter_label = QLabel("Filters:")
        filter_label.setObjectName("FilterLabel")
        layout.addWidget(filter_label)

        self._filter_combos: dict = {}
        for field, display_name in [
            ("brand", "Brand"), ("category", "Category"), ("color", "Color"), ("size", "Size")
        ]:
            combo = QComboBox()
            combo.setObjectName("FilterCombo")
            combo.addItem(f"Any {display_name}")
            combo.currentTextChanged.connect(
                lambda value, f=field: self._on_filter_changed(f, value)
            )
            self._filter_combos[field] = combo
            layout.addWidget(combo)

        layout.addStretch()
        return bar

    def _on_filter_changed(self, field: str, value: str) -> None:
        # Ignore the placeholder "Any X" item text as a real filter value.
        is_placeholder = value.startswith("Any ")
        self._viewmodel.set_filter(field, "" if is_placeholder else value)

    def _on_export_clicked(self) -> None:
        import traceback
        try:
            from src.presentation.dialogs.export_catalog_dialog import ExportCatalogDialog
            from src.services.pdf_export_service import PDFExportService
        except Exception as exc:
            traceback.print_exc()
            raise
            # message_box.critical(self, "Export Error", f"Missing export modules:\n{exc}")
            # return

        if not self._current_results:
            message_box.information(self, "Export Catalogue", "No results to export yet.")
            return

        dialog = ExportCatalogDialog(
            self,
            theme=self._theme,
            on_open_profiles_settings=self._on_open_profiles_settings,
            catalogue_master_service=self._catalogue_master_service,
        )
        if dialog.exec() != dialog.DialogCode.Accepted:
            return

        output_path = dialog.output_path()
        if not output_path:
            return

        options = dialog.options()
        selected_indices = None

        if options.include_selected_only:
            selected_indices = sorted({index.row() for index in self._results_table.selectionModel().selectedRows()})
            if not selected_indices:
                message_box.information(self, "Export Catalogue", "Select one or more results first.")
                return

        service = PDFExportService()
        try:
            created = service.export_catalogue(
                output_file=output_path,
                query_image_path=self._current_query_image_path,
                results=self._current_results,
                options=options,
                selected_indices=selected_indices,
            )
        except Exception as exc:
            message_box.critical(self, "Export Failed", str(exc))
            return

        message_box.information(self, "Export Complete", f"Catalogue saved to:\n{created}")
        
    @Slot(dict)
    def _on_filters_available(self, options: dict) -> None:
        """
        Refresh filter dropdowns after catalog changes (including auto-index).

        Must not clear displayed search results and must not auto-re-search.
        A no-op set_filter → search_by_image path previously wiped results when
        a concurrent auto-index made the feature-version health check look empty.
        """
        for field, combo in self._filter_combos.items():
            values = options.get(field, [])
            current = combo.currentText()
            placeholder = f"Any {field.capitalize()}"
            combo.blockSignals(True)
            combo.clear()
            combo.addItem(placeholder)
            combo.addItems(values)
            idx = combo.findText(current)
            if idx >= 0:
                combo.setCurrentIndex(idx)
            else:
                combo.setCurrentIndex(0)
                # Stale brand/category/etc. — drop quietly; keep current results.
                self._viewmodel.drop_filter_quietly(field)
            combo.blockSignals(False)

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
        header.setSectionResizeMode(5, QHeaderView.ResizeMode.ResizeToContents)
        header.setSectionResizeMode(6, QHeaderView.ResizeMode.ResizeToContents)
        header.setSectionResizeMode(5, QHeaderView.ResizeMode.Stretch)

        self._results_table.cellDoubleClicked.connect(self._on_row_double_clicked)
        self._results_table.cellClicked.connect(self._on_table_clicked)
        self._results_table.customContextMenuRequested.connect(self._on_results_context_menu)

        self._empty_state_icon = QLabel("")
        self._empty_state_icon.setObjectName("EmptyStateIcon")
        self._empty_state_icon.setAlignment(Qt.AlignmentFlag.AlignCenter)

        self._empty_state_label = QLabel(
            "No results yet.\nDrag or browse a tile photo above to start searching."
        )
        self._empty_state_label.setObjectName("EmptyStateLabel")
        self._empty_state_label.setAlignment(Qt.AlignmentFlag.AlignCenter)

        empty_state_container = QVBoxLayout()
        empty_state_container.addStretch()
        empty_state_container.addWidget(self._empty_state_icon)
        empty_state_container.addWidget(self._empty_state_label)
        empty_state_container.addStretch()
        self._empty_state_widget = QWidget()
        self._empty_state_widget.setLayout(empty_state_container)

        container = QWidget()
        container_layout = QVBoxLayout(container)
        container_layout.setContentsMargins(0, 0, 0, 0)
        container_layout.addWidget(self._empty_state_widget)
        container_layout.addWidget(self._results_table)
        self._results_table.setVisible(False)
        self._results_container = container
        return container

    def _build_status_line(self) -> QWidget:
        container = QWidget()
        layout = QHBoxLayout(container)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(12)

        self._status_label = QLabel("Ready. Drag an image or click Browse to search.")
        self._status_label.setObjectName("SearchStatusLabel")
        layout.addWidget(self._status_label, stretch=1)

        self._stats_label = QLabel("")
        self._stats_label.setObjectName("SearchStatsLabel")
        self._stats_label.setVisible(False)
        layout.addWidget(self._stats_label)

        return container

    # ── Signal Wiring ────────────────────────────────────────────────────

    def _connect_signals(self) -> None:
        self._viewmodel.state_changed.connect(self._on_state_changed)
        self._viewmodel.results_ready.connect(self._on_results_ready)
        self._viewmodel.status_message.connect(self._status_label.setText)
        self._viewmodel.search_error.connect(self._on_search_error)
        self._viewmodel.query_image_selected.connect(self._drop_zone.show_preview)
        self._viewmodel.filters_available.connect(self._on_filters_available)
        self._viewmodel.search_history_updated.connect(self._show_history_menu)
        self._viewmodel.search_stats_ready.connect(self._on_search_stats_ready)
        self._viewmodel.load_filter_options()

    def _start_searching_animation(self) -> None:
        """Animate the status label with cycling dots while a search runs
        (Task C: 'Searching animation')."""
        self._search_animation_dots = 0
        self._search_animation_timer.start()

    def _stop_searching_animation(self) -> None:
        self._search_animation_timer.stop()

    def _tick_searching_animation(self) -> None:
        self._search_animation_dots = (self._search_animation_dots + 1) % 4
        dots = "." * self._search_animation_dots
        name = Path(self._current_query_image_path).name if self._current_query_image_path else "image"
        self._status_label.setText(f"Searching for tiles similar to '{name}'{dots}")

    # ── Event Handlers ───────────────────────────────────────────────────

    def _on_image_chosen(self, image_path: str) -> None:
        logger.info(f"Query image selected: {image_path}")
        self._current_query_image_path = image_path
        self._crop_button.setEnabled(True)
        self._auto_crop_button.setEnabled(True)
        self._precise_crop_button.setEnabled(True)
        # Customer can Clear as soon as a photo is loaded — do not wait for
        # SEARCHING (state can lag one event-loop tick after drop).
        self._clear_button.setEnabled(True)
        self._confidence_banner.setVisible(False)

        suggest_crop, crop_reason = should_suggest_crop(image_path)
        if suggest_crop:
            # Non-blocking: drop-search isolates the tile (SAM2/OpenCV),
            # straightens perspective, and multi-crop FAISS searches.
            self._status_label.setText(
                f"{crop_reason} Isolating tile region for a better match…"
            )
        self._viewmodel.search_by_image(image_path)

    def _on_drop_rejected(self, reason: str) -> None:
        self._status_label.setText(reason)
        self._empty_state_label.setText(reason)
        self._empty_state_widget.setVisible(True)
        message_box.information(self, "Cannot Search This File", reason)

    def _on_crop_clicked(self) -> None:
        if not self._current_query_image_path:
            return
        dialog = CropDialog(self._current_query_image_path, parent=self, theme=self._theme)
        if dialog.exec() == dialog.DialogCode.Accepted and dialog.cropped_image_path:
            logger.info(f"Searching with cropped region: {dialog.cropped_image_path}")
            self._viewmodel.search_by_image(
                dialog.cropped_image_path, query_origin="crop_tool"
            )

    def _on_auto_crop_clicked(self) -> None:
        if not self._current_query_image_path:
            return
        self._start_background_crop("auto")

    def _on_precise_crop_clicked(self) -> None:
        if not self._current_query_image_path:
            return
        self._start_background_crop("precise")

    def _start_background_crop(self, mode: str) -> None:
        """Run Auto/Precise crop off the UI thread (critical on Mac Intel)."""
        if self._crop_busy or self._viewmodel.is_searching:
            return
        if not self._current_query_image_path:
            return

        label = "Precise Crop" if mode == "precise" else "Auto Crop"
        self._crop_busy = True
        self._set_crop_controls_enabled(False)
        self._status_label.setText(f"{label} running… UI stays responsive.")
        self._progress_bar.setVisible(True)

        worker = TileCropWorker(self._current_query_image_path, mode)  # type: ignore[arg-type]
        self._crop_worker = worker
        worker.crop_finished.connect(
            lambda path, crop, m=mode: self._on_background_crop_finished(m, path, crop)
        )
        worker.crop_failed.connect(
            lambda message, m=mode: self._on_background_crop_failed(m, message)
        )
        worker.finished.connect(worker.deleteLater)
        worker.start()

    def _on_background_crop_finished(self, mode: str, crop_path: str, crop) -> None:
        self._crop_busy = False
        self._crop_worker = None
        self._progress_bar.setVisible(self._viewmodel.is_searching)
        method = getattr(crop, "method", "crop")
        confidence = float(getattr(crop, "confidence", 0.0) or 0.0)
        if mode == "precise":
            self._status_label.setText(
                f"Precise crop ({method}, {confidence:.0%}) — searching…"
            )
            logger.info(
                "Searching with precise crop: %s method=%s detail=%s",
                crop_path,
                method,
                getattr(crop, "detail", ""),
            )
        else:
            self._status_label.setText(
                f"Auto-cropped tile region ({method}, {confidence:.0%} confidence) "
                "— searching…"
            )
            logger.info("Searching with auto-cropped region: %s", crop_path)

        self._drop_zone.show_preview(str(crop_path))
        self._viewmodel.search_by_image(str(crop_path), query_origin="crop_tool")

    def _on_background_crop_failed(self, mode: str, message: str) -> None:
        self._crop_busy = False
        self._crop_worker = None
        self._progress_bar.setVisible(self._viewmodel.is_searching)
        self._refresh_crop_controls()
        logger.error("%s crop failed: %s", mode, message)
        if mode == "precise":
            message_box.warning(
                self,
                "Precise Crop Failed",
                "Could not run precise crop on this photo.\n\n"
                "Try Auto Crop & Search or manual Crop and Search.",
            )
        else:
            message_box.warning(
                self,
                "Auto Crop Failed",
                "Could not auto-crop this photo.\n\n"
                "Try Crop and Search to select the tile area manually.",
            )

    def _set_crop_controls_enabled(self, enabled: bool) -> None:
        has_query = self._current_query_image_path is not None
        allow = enabled and has_query and not self._crop_busy
        self._crop_button.setEnabled(allow)
        self._auto_crop_button.setEnabled(allow)
        self._precise_crop_button.setEnabled(allow)

    def _refresh_crop_controls(self) -> None:
        searching = self._viewmodel.is_searching
        self._set_crop_controls_enabled(not searching and not self._crop_busy)

    def _on_clear_clicked(self) -> None:
        if self._crop_worker is not None and self._crop_worker.isRunning():
            try:
                self._crop_worker.requestInterruption()
            except Exception:
                pass
        self._crop_busy = False
        self._crop_worker = None
        self._drop_zone.reset()
        self._current_query_image_path = None
        self._crop_button.setEnabled(False)
        self._auto_crop_button.setEnabled(False)
        self._precise_crop_button.setEnabled(False)
        self._confidence_banner.setVisible(False)
        self._progress_bar.setVisible(False)
        self._viewmodel.clear_results()
        
    def _on_history_clicked(self) -> None:
        """Show a popup menu of recent searches (Task C: Search History)."""
        self._history_menu_pending = True
        self._viewmodel.load_search_history()
        # _show_history_menu runs when search_history_updated fires.

    def _show_history_menu(self, entries: List) -> None:
        # Ignore silent history refreshes after search completion / recording.
        if not self._history_menu_pending:
            return
        self._history_menu_pending = False
        menu = QMenu(self)
        if not entries:
            no_history_action = menu.addAction("No recent searches yet")
            no_history_action.setEnabled(False)
        else:
            for entry in entries:
                name = Path(entry.query_image_path).name
                when = entry.searched_at.strftime("%b %d, %I:%M %p") if entry.searched_at else ""
                label = f"{name} — {entry.result_count} result(s)  ({when})"
                action = menu.addAction(label)
                action.triggered.connect(
                    lambda checked=False, path=entry.query_image_path: self._viewmodel.repeat_search(path)
                )
        menu.exec(self._history_button.mapToGlobal(self._history_button.rect().bottomLeft()))

    @Slot(str)
    def _on_state_changed(self, state: str) -> None:
        is_searching = state == SearchState.SEARCHING
        self._drop_zone.set_busy(is_searching or self._crop_busy)
        self._progress_bar.setVisible(is_searching or self._crop_busy)
        has_query = self._current_query_image_path is not None
        self._clear_button.setEnabled((state != SearchState.IDLE or has_query) and not self._crop_busy)
        self._refresh_crop_controls()
        for combo in self._filter_combos.values():
            combo.setEnabled(not is_searching and not self._crop_busy)

        if is_searching:
            self._stats_label.setVisible(False)
            self._confidence_banner.setVisible(False)
            self._results_table.setVisible(False)
            self._empty_state_widget.setVisible(True)
            self._empty_state_label.setText(
                "Searching for similar tiles…\nPlease wait — this can take a few seconds on first search."
            )
            self._start_searching_animation()
        else:
            self._stop_searching_animation()
            if state == SearchState.NO_RESULTS:
                self._empty_state_label.setText(
                    "No similar tiles found.\nTry another photo, Auto Crop, or confirm your catalogue is indexed."
                )
            elif state == SearchState.IDLE:
                self._empty_state_label.setText(
                    "No results yet.\nDrag or browse a tile photo above to start searching."
                )
            elif state == SearchState.ERROR:
                self._empty_state_label.setText(
                    "Search failed.\nCheck the message, then try again or use Auto Crop & Search."
                )
            elif state == SearchState.RESULTS:
                # Belt-and-suspenders: never leave "Searching…" visible after RESULTS.
                if self._results_table.rowCount() > 0:
                    self._empty_state_widget.setVisible(False)
                    self._results_table.setVisible(True)

    @Slot(list)
    def _on_results_ready(self, results: List[SearchResult]) -> None:
        self._current_results = results
        self._results_generation += 1
        self._populate_table(results)
        self._show_confidence_banner(results)
        if results:
            self._empty_state_widget.setVisible(False)
            self._results_table.setVisible(True)
            logger.info("[SEARCH] UI displaying %d result row(s)", len(results))
        else:
            logger.warning(
                "[SEARCH] UI received empty result list (vm_state=%s has_query=%s)",
                self._viewmodel.state,
                bool(self._current_query_image_path),
            )

    def _show_confidence_banner(self, results: List[SearchResult]) -> None:
        # Clear / idle must not show "No similar tiles found" after wiping the table.
        if self._viewmodel.state == SearchState.IDLE:
            self._confidence_banner.setVisible(False)
            return
        message = confidence_message(results)
        if message:
            self._confidence_banner.setText(f"Warning: {message}")
            self._confidence_banner.setVisible(True)
        else:
            self._confidence_banner.setVisible(False)

    @Slot(str)
    def _on_search_error(self, message: str) -> None:
        message_box.warning(self, "Search Failed", message)

    @Slot(int, float)
    def _on_search_stats_ready(self, result_count: int, elapsed_seconds: float) -> None:
        """Show elapsed search time + results count (Task C: Search UX)."""
        self._stats_label.setText(
            f"{result_count} result(s) in {elapsed_seconds * 1000:.0f} ms"
            if elapsed_seconds < 1
            else f"{result_count} result(s) in {elapsed_seconds:.2f}s"
        )
        self._stats_label.setVisible(True)

    def _populate_table(self, results: List[SearchResult]) -> None:
        self._results_table.setRowCount(0)

        if not results:
            self._results_table.setVisible(False)
            self._empty_state_widget.setVisible(True)
            # Release cached icons when the results panel is cleared.
            try:
                from src.presentation.thumbnail_cache import THUMBNAIL_PIXMAP_CACHE

                THUMBNAIL_PIXMAP_CACHE.clear()
            except Exception:
                pass
            return

        self._empty_state_widget.setVisible(False)
        self._results_table.setVisible(True)
        self._results_table.setRowCount(len(results))

        # Feature 3: distinguish the single top match ("Best Match") from
        # the rest ("Similar Alternatives") rather than presenting all
        # top-K results identically — matters most for the customer-photo
        # search scenario (WhatsApp photo, phone snapshot) where the user
        # usually wants "which exact tile is this" first, with the rest as
        # fallback options if the best match isn't quite right.
        best_match_bg = QColor(get_palette(self._theme)["highlight_bg"])
        best_match_text = QColor(get_palette(self._theme)["text_primary"])

        for row, result in enumerate(results):
            tile = result.tile
            is_best_match = row == 0

            # Placeholder only — QPixmap decode is deferred so the table
            # paints immediately after search returns.
            thumb_item = QTableWidgetItem()
            thumb_item.setFlags(thumb_item.flags() & ~Qt.ItemFlag.ItemIsEditable)
            self._results_table.setItem(row, 0, thumb_item)

            similarity_text = f"★ {result.similarity_score:.1f}%" if is_best_match else f"{result.similarity_score:.1f}%"
            similarity_item = QTableWidgetItem(similarity_text)
            similarity_item.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
            if is_best_match:
                similarity_item.setToolTip("Best Match")
                bold_font = similarity_item.font()
                bold_font.setBold(True)
                similarity_item.setFont(bold_font)
            self._results_table.setItem(row, 1, similarity_item)

            self._results_table.setItem(row, 2, QTableWidgetItem(tile.product_code or "—"))
            self._results_table.setItem(row, 3, QTableWidgetItem(tile.brand or "—"))
            self._results_table.setItem(row, 4, QTableWidgetItem(tile.category or "—"))

            preview_item = QTableWidgetItem("View")
            preview_item.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
            preview_item.setToolTip("Preview Tile")
            self._results_table.setItem(row, 5, preview_item)

            path_item = QTableWidgetItem(tile.file_path)
            path_item.setToolTip(tile.file_path)
            self._results_table.setItem(row, 6, path_item)

            if is_best_match:
                for col in range(self._results_table.columnCount()):
                    item = self._results_table.item(row, col)
                    if item is not None:
                        item.setBackground(best_match_bg)
                        item.setForeground(best_match_text)
                        font = item.font()
                        font.setBold(True)
                        item.setFont(font)

        # Defer pixmap I/O until after the table is visible.
        generation = self._results_generation
        QTimer.singleShot(0, lambda gen=generation: self._load_result_thumbnails(gen))

    def _load_result_thumbnails(self, generation: int | None = None) -> None:
        """Lazy-load result icons after search returns (keeps UI responsive)."""
        if generation is not None and generation != self._results_generation:
            return
        from src.presentation.thumbnail_cache import THUMBNAIL_PIXMAP_CACHE

        results = list(self._current_results)
        for row, result in enumerate(results):
            if generation is not None and generation != self._results_generation:
                return
            if row >= self._results_table.rowCount():
                break
            item = self._results_table.item(row, 0)
            if item is None:
                continue
            pixmap = THUMBNAIL_PIXMAP_CACHE.get(
                result.thumbnail_path,
                fallback=result.tile.file_path,
            )
            if not pixmap.isNull():
                item.setIcon(QIcon(pixmap))
            # Yield to the event loop every few icons.
            if row > 0 and row % 4 == 0:
                QApplication.processEvents()
                if generation is not None and generation != self._results_generation:
                    return

    def _on_row_double_clicked(self, row: int, _column: int) -> None:
        self._open_image_at_row(row)

    def _on_table_clicked(self, row: int, column: int) -> None:
        # Preview column
        if column != 5:
            return

        if row < 0 or row >= len(self._current_results):
            return

        self._preview_panel.show_result(self._current_results[row])

    def _on_results_context_menu(self, position) -> None:
        row = self._results_table.rowAt(position.y())
        if row < 0 or row >= len(self._current_results):
            return

        menu = QMenu(self)
        open_image_action = menu.addAction("Open Image")
        open_folder_action = menu.addAction("📂  Open Containing Folder")
        copy_path_action = menu.addAction("Copy Path")

        chosen = menu.exec(self._results_table.viewport().mapToGlobal(position))
        if chosen == open_image_action:
            self._open_image_at_row(row)
        elif chosen == open_folder_action:
            self._open_folder_at_row(row)
        elif chosen == copy_path_action:
            self._copy_path_at_row(row)

    def _copy_path_at_row(self, row: int) -> None:
        if row < 0 or row >= len(self._current_results):
            return
        path = self._current_results[row].tile.file_path
        QApplication.clipboard().setText(path)
        self._status_label.setText(f"Copied path to clipboard: {path}")

    def _open_image_at_row(self, row: int) -> None:
        if row < 0 or row >= len(self._current_results):
            return
        path = Path(self._current_results[row].tile.file_path)
        if not path.exists():
            message_box.warning(self, "File Not Found", f"This image no longer exists:\n{path}")
            return
        QDesktopServices.openUrl(QUrl.fromLocalFile(str(path)))

    def _open_folder_at_row(self, row: int) -> None:
        if row < 0 or row >= len(self._current_results):
            return
        path = Path(self._current_results[row].tile.file_path)
        folder = path.parent
        if not folder.exists():
            message_box.warning(self, "Folder Not Found", f"This folder no longer exists:\n{folder}")
            return
        QDesktopServices.openUrl(QUrl.fromLocalFile(str(folder)))

    # ── Styling ──────────────────────────────────────────────────────────

    def set_theme(self, theme: str) -> None:
        """Re-skin this view (and its preview panel) for a newly-selected theme."""
        self._theme = theme
        self._preview_panel.set_theme(theme)
        self._apply_styles()

    def _apply_styles(self) -> None:
        p = get_palette(self._theme)
        self.setStyleSheet(
            get_shared_view_qss(self._theme)
            + f"""
            #FilterLabel {{ color: {p['text_muted']}; font-size: 12px; font-weight: 600; }}
            #FilterCombo {{
                background-color: {p['bg_input']};
                color: {p['text_secondary']};
                border: 1px solid {p['border_strong']};
                border-radius: 6px;
                padding: 4px 8px;
                min-width: 110px;
                font-size: 12px;
            }}
            #FilterCombo:hover {{ border-color: {p['accent_hover']}; }}

            #DropZone {{
                background-color: {p['bg_panel']};
                border: 2px dashed {p['border_strong']};
                border-radius: 10px;
            }}
            #DropZone[dragActive="true"] {{
                border: 2px dashed {p['accent_hover']};
                background-color: {p['row_alt']};
            }}
            #DropZoneIcon {{ font-size: 36px; }}
            #DropZoneTitle {{ font-size: 14px; font-weight: 600; color: {p['text_primary']}; }}
            #DropZoneSubtitle {{ font-size: 11px; color: {p['text_muted']}; }}

            #ResultsTable {{
                background-color: {p['bg_panel_alt']};
                alternate-background-color: {p['bg_panel']};
                gridline-color: {p['border']};
                color: {p['text_secondary']};
                border: 1px solid {p['border']};
                border-radius: 8px;
            }}
            #ResultsTable::item:selected {{ background-color: {p['highlight_bg']}; color: {p['text_primary']}; }}
            QHeaderView::section {{
                background-color: {p['row_alt']};
                color: {p['text_secondary']};
                padding: 6px;
                border: none;
                border-bottom: 1px solid {p['border']};
                font-weight: 600;
                font-size: 11px;
            }}

            #EmptyStateIcon {{ font-size: 48px; padding-bottom: 8px; }}
            #EmptyStateLabel {{ color: {p['text_faint']}; font-size: 13px; padding: 8px 40px 40px 40px; }}
            #SearchStatusLabel {{ color: {p['text_muted']}; font-size: 12px; }}
            #SearchStatsLabel {{ color: {p['accent_text']}; font-size: 12px; font-weight: 600; }}
            #SearchProgressBar {{
                background-color: {p['bg_panel']};
                border-radius: 2px;
            }}
            #SearchProgressBar::chunk {{ background-color: {p['accent_soft']}; }}
            #ConfidenceBanner {{
                background-color: {p['warning_bg']};
                color: {p['warning_text']};
                border: 1px solid {p['border_strong']};
                border-radius: 8px;
                padding: 10px 14px;
                font-size: 12px;
            }}
            """
        )
