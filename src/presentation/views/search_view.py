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

from PySide6.QtCore import (
    Qt,
    Slot,
    QUrl,
    QSize,
    QTimer,
    QPoint,
    QPropertyAnimation,
    QEasingCurve,
)
from PySide6.QtGui import QDragEnterEvent, QDropEvent, QDesktopServices, QPixmap, QIcon, QColor
from PySide6.QtWidgets import (
    QWidget,
    QVBoxLayout,
    QHBoxLayout,
    QGridLayout,
    QLabel,
    QPushButton,
    QToolButton,
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
    QLineEdit,
    QApplication,
    QGraphicsOpacityEffect,
    QCheckBox,
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

# Query formats include HEIC on Windows/Mac when pillow-heif is installed.
_QUERY_IMAGE_EXTENSIONS = query_image_extensions()

_TABLE_COLUMNS = [
    "Thumbnail",
    "Similarity",
    "Product Code",
    "Brand",
    "Category",
    "Image Path",
    "Actions",
]

_PATH_COL = 5
_ACTIONS_COL = 6
# Preview + Copy + Reveal (Why is omitted unless ORB data exists).
_ACTIONS_COL_WIDTH = 104
_ACTIONS_COL_WIDTH_WITH_WHY = 136


class _ResultActionsWidget(QWidget):
    """
    Compact themed line-icon actions: Preview, optional Why, Copy, Reveal.

    Copied feedback is a floating overlay (not in the layout) so it never
    squeezes the Image Path column or causes double-text overlap.
    """

    def __init__(
        self,
        path: str,
        *,
        theme: str,
        on_preview,
        on_copy,
        on_reveal,
        on_why=None,
        why_enabled: bool = False,
        parent: Optional[QWidget] = None,
    ) -> None:
        super().__init__(parent)
        self._path = path
        self._theme = theme
        self._on_preview = on_preview
        self._on_copy = on_copy
        self._on_reveal = on_reveal
        self._on_why = on_why
        self._why_enabled = bool(why_enabled and on_why is not None)
        self._fade_anim: Optional[QPropertyAnimation] = None
        self._copied_toast: Optional[QLabel] = None
        self._why_btn: Optional[QToolButton] = None

        layout = QHBoxLayout(self)
        layout.setContentsMargins(4, 2, 4, 2)
        layout.setSpacing(4)
        layout.setAlignment(Qt.AlignmentFlag.AlignCenter)

        from src.utils.reveal_file import reveal_action_label

        reveal_tip = reveal_action_label().lstrip("📍 ").strip()

        self._preview_btn = self._make_action_btn(
            "Preview tile", "ResultPreviewButton"
        )
        self._preview_btn.clicked.connect(lambda: self._on_preview(self._path))

        # Only render Why when ORB geometric data exists (never for description
        # search). Hidden — not merely disabled — so path gets the column space.
        if self._why_enabled:
            self._why_btn = self._make_action_btn(
                "Why this match", "ResultWhyButton"
            )
            self._why_btn.clicked.connect(lambda: self._on_why(self._path))

        self._copy_btn = self._make_action_btn("Copy path", "ResultCopyButton")
        self._copy_btn.clicked.connect(self._handle_copy)

        self._reveal_btn = self._make_action_btn(reveal_tip, "ResultRevealButton")
        self._reveal_btn.clicked.connect(lambda: self._on_reveal(self._path))

        layout.addWidget(self._preview_btn)
        if self._why_btn is not None:
            layout.addWidget(self._why_btn)
        layout.addWidget(self._copy_btn)
        layout.addWidget(self._reveal_btn)
        self._apply_styles()

    @staticmethod
    def _make_action_btn(tooltip: str, object_name: str) -> QToolButton:
        btn = QToolButton()
        btn.setObjectName(object_name)
        btn.setToolTip(tooltip)
        btn.setCursor(Qt.CursorShape.PointingHandCursor)
        btn.setFixedSize(28, 28)
        btn.setAutoRaise(True)
        btn.setFocusPolicy(Qt.FocusPolicy.StrongFocus)
        # Kill the default Windows blue focus rectangle; themed ring is in QSS.
        btn.setAttribute(Qt.WidgetAttribute.WA_MacShowFocusRect, False)
        btn.setStyle(btn.style())
        return btn

    def _handle_copy(self) -> None:
        self._on_copy(self._path)
        self._show_copied_toast()

    def _show_copied_toast(self) -> None:
        """Floating tooltip-style 'Copied' badge over the copy icon (no layout shift)."""
        if self._fade_anim is not None:
            self._fade_anim.stop()
            self._fade_anim.deleteLater()
            self._fade_anim = None
        if self._copied_toast is not None:
            self._copied_toast.hide()
            self._copied_toast.deleteLater()
            self._copied_toast = None

        p = get_palette(self._theme)
        toast = QLabel("Copied")
        toast.setObjectName("PathCopiedToast")
        toast.setAlignment(Qt.AlignmentFlag.AlignCenter)
        toast.setWindowFlags(
            Qt.WindowType.ToolTip
            | Qt.WindowType.FramelessWindowHint
            | Qt.WindowType.WindowStaysOnTopHint
        )
        toast.setAttribute(Qt.WidgetAttribute.WA_ShowWithoutActivating, True)
        toast.setAttribute(Qt.WidgetAttribute.WA_TransparentForMouseEvents, True)
        toast.setStyleSheet(
            f"""
            #PathCopiedToast {{
                color: {p['accent_text']};
                background-color: {p['highlight_bg']};
                border: 1px solid {p['accent_soft']};
                border-radius: 5px;
                padding: 3px 8px;
                font-size: 10px;
                font-weight: 700;
            }}
            """
        )
        toast.adjustSize()
        # Center horizontally on the copy button; sit just above it.
        anchor = self._copy_btn.mapToGlobal(
            QPoint(self._copy_btn.width() // 2, 0)
        )
        toast.move(anchor.x() - toast.width() // 2, anchor.y() - toast.height() - 6)
        toast.show()
        self._copied_toast = toast

        effect = QGraphicsOpacityEffect(toast)
        toast.setGraphicsEffect(effect)
        effect.setOpacity(1.0)
        anim = QPropertyAnimation(effect, b"opacity", self)
        anim.setDuration(1200)
        anim.setStartValue(1.0)
        anim.setKeyValueAt(0.55, 1.0)
        anim.setEndValue(0.0)
        anim.setEasingCurve(QEasingCurve.Type.OutCubic)

        def _hide() -> None:
            if self._copied_toast is not None:
                self._copied_toast.hide()
                self._copied_toast.deleteLater()
                self._copied_toast = None
            self._fade_anim = None

        anim.finished.connect(_hide)
        self._fade_anim = anim
        anim.start()

    def set_theme(self, theme: str) -> None:
        self._theme = theme
        from src.utils.reveal_file import reveal_action_label

        self._reveal_btn.setToolTip(reveal_action_label().lstrip("📍 ").strip())
        self._apply_styles()

    def _apply_styles(self) -> None:
        from src.presentation.icons.result_action_icons import (
            icon_copy,
            icon_preview,
            icon_reveal,
            icon_why,
        )

        p = get_palette(self._theme)
        normal = p["text_secondary"]
        accent = p["accent_text"]
        disabled = p.get("text_faint", p["text_secondary"])

        self._preview_btn.setIcon(icon_preview(normal))
        self._preview_btn.setIconSize(QSize(18, 18))
        self._copy_btn.setIcon(icon_copy(normal))
        self._copy_btn.setIconSize(QSize(18, 18))
        self._reveal_btn.setIcon(icon_reveal(normal))
        self._reveal_btn.setIconSize(QSize(18, 18))
        if self._why_btn is not None:
            self._why_btn.setIcon(icon_why(normal))
            self._why_btn.setIconSize(QSize(18, 18))

        # Store accent icons for hover via property (QSS cannot swap QIcon).
        self._preview_btn.setProperty("icon_normal", icon_preview(normal))
        self._preview_btn.setProperty("icon_hover", icon_preview(accent))
        self._copy_btn.setProperty("icon_normal", icon_copy(normal))
        self._copy_btn.setProperty("icon_hover", icon_copy(accent))
        self._reveal_btn.setProperty("icon_normal", icon_reveal(normal))
        self._reveal_btn.setProperty("icon_hover", icon_reveal(accent))
        if self._why_btn is not None:
            self._why_btn.setProperty("icon_normal", icon_why(normal))
            self._why_btn.setProperty("icon_hover", icon_why(accent))

        for btn in (
            self._preview_btn,
            self._copy_btn,
            self._reveal_btn,
            self._why_btn,
        ):
            if btn is None:
                continue
            btn.enterEvent = self._make_hover_enter(btn)  # type: ignore[method-assign]
            btn.leaveEvent = self._make_hover_leave(btn)  # type: ignore[method-assign]

        self.setStyleSheet(
            f"""
            QToolButton {{
                background-color: {p['bg_panel']};
                color: {p['text_secondary']};
                border: 1px solid {p['border_strong']};
                border-radius: 5px;
                padding: 0;
                outline: none;
            }}
            QToolButton:hover {{
                background-color: {p['button_hover']};
                border-color: {p['accent_soft']};
            }}
            QToolButton:pressed {{
                background-color: {p['accent_deep']};
                border-color: {p['accent']};
            }}
            QToolButton:disabled {{
                background-color: {p['bg_panel']};
                border-color: {p['border_strong']};
                color: {disabled};
            }}
            QToolButton:focus {{
                border: 2px solid {p['accent']};
                outline: none;
            }}
            """
        )

    @staticmethod
    def _make_hover_enter(btn: QToolButton):
        def _enter(event) -> None:
            hover = btn.property("icon_hover")
            if hover is not None:
                btn.setIcon(hover)
            QToolButton.enterEvent(btn, event)

        return _enter

    @staticmethod
    def _make_hover_leave(btn: QToolButton):
        def _leave(event) -> None:
            normal = btn.property("icon_normal")
            if normal is not None:
                btn.setIcon(normal)
            QToolButton.leaveEvent(btn, event)

        return _leave


class DropZone(QFrame):
    """
    A clickable, drag-and-drop capable frame for selecting a query image.

    Emits via ``on_images_selected(paths)`` (1 path by default; up to 3 when
    multi-photo mode is enabled). Default single-photo browse/drop is unchanged.
    """

    _MAX_MULTI = 3

    def __init__(
        self,
        on_image_selected=None,
        parent: Optional[QWidget] = None,
        on_drop_rejected: Optional[Callable[[str], None]] = None,
        on_images_selected: Optional[Callable[[list], None]] = None,
    ) -> None:
        """
        Args:
            on_image_selected: Legacy Callable[[str], None] for a single path.
            parent: Optional Qt parent widget.
            on_drop_rejected: Optional Callable[[str], None] with a
                user-visible reason when a drop is ignored.
            on_images_selected: Callable[[list[str]], None] preferred for
                multi-photo; falls back to wrapping ``on_image_selected``.
        """
        super().__init__(parent)
        self._on_image_selected = on_image_selected
        self._on_images_selected = on_images_selected
        self._on_drop_rejected = on_drop_rejected
        self._multi_photo_enabled = False
        self._on_thumbnail_selected: Optional[Callable[[int], None]] = None
        self._selected_thumb_index = 0
        self.setObjectName("DropZone")
        self.setAcceptDrops(True)
        self.setCursor(Qt.CursorShape.PointingHandCursor)
        self.setMinimumHeight(110)
        self.setMaximumHeight(150)

        layout = QVBoxLayout(self)
        layout.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.setSpacing(6)
        layout.setContentsMargins(8, 8, 8, 8)

        self._icon_label = QLabel("Tile Image")
        self._icon_label.setObjectName("DropZoneIcon")
        self._icon_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(self._icon_label)

        self._preview_label = QLabel()
        self._preview_label.setObjectName("DropZonePreview")
        self._preview_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._preview_label.setVisible(False)
        layout.addWidget(self._preview_label)

        # Multi-photo: row of small thumbnails (hidden in single-photo mode).
        self._multi_preview_row = QWidget()
        self._multi_preview_row.setObjectName("DropZoneMultiPreviewRow")
        self._multi_preview_row.setVisible(False)
        multi_layout = QHBoxLayout(self._multi_preview_row)
        multi_layout.setContentsMargins(0, 0, 0, 0)
        multi_layout.setSpacing(6)
        multi_layout.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._multi_preview_labels: list[QLabel] = []
        for i in range(self._MAX_MULTI):
            lbl = QLabel()
            lbl.setObjectName("DropZoneMultiPreview")
            lbl.setFixedSize(72, 72)
            lbl.setAlignment(Qt.AlignmentFlag.AlignCenter)
            lbl.setVisible(False)
            lbl.setCursor(Qt.CursorShape.PointingHandCursor)
            lbl.mousePressEvent = (  # type: ignore[method-assign]
                lambda event, idx=i: self._on_multi_thumb_pressed(event, idx)
            )
            multi_layout.addWidget(lbl)
            self._multi_preview_labels.append(lbl)
        layout.addWidget(self._multi_preview_row)

        self._title_label = QLabel("Drag a tile photo here")
        self._title_label.setObjectName("DropZoneTitle")
        self._title_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(self._title_label)

        self._subtitle_label = QLabel("or click to browse for an image")
        self._subtitle_label.setObjectName("DropZoneSubtitle")
        self._subtitle_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(self._subtitle_label)

    def set_multi_photo_enabled(self, enabled: bool) -> None:
        """Opt-in: accept 2–3 photos. Default off keeps the fast single-photo path."""
        self._multi_photo_enabled = bool(enabled)
        if self._multi_photo_enabled:
            self._title_label.setText("Drag 2–3 tile photos here")
            self._subtitle_label.setText("or click to browse for multiple images")
        else:
            self._title_label.setText("Drag a tile photo here")
            self._subtitle_label.setText("or click to browse for an image")

    def set_thumbnail_selected_callback(
        self, callback: Optional[Callable[[int], None]]
    ) -> None:
        """Notify SearchView when the user clicks a multi-photo thumbnail."""
        self._on_thumbnail_selected = callback

    def set_selected_thumb_index(self, index: int) -> None:
        """Highlight which photo Manual Crop will operate on."""
        self._selected_thumb_index = max(0, int(index))
        self._refresh_thumb_selection_style()

    def _on_multi_thumb_pressed(self, event, index: int) -> None:
        if event.button() != Qt.MouseButton.LeftButton:
            return
        self._selected_thumb_index = index
        self._refresh_thumb_selection_style()
        if self._on_thumbnail_selected is not None:
            self._on_thumbnail_selected(index)

    def _refresh_thumb_selection_style(self) -> None:
        for i, lbl in enumerate(self._multi_preview_labels):
            if not lbl.isVisible():
                continue
            selected = i == self._selected_thumb_index
            lbl.setStyleSheet(
                "border: 2px solid #3B82F6; border-radius: 4px;"
                if selected
                else "border: 2px solid transparent; border-radius: 4px;"
            )

    def set_busy(self, busy: bool) -> None:
        """Disable interaction while a search is in progress."""
        self.setEnabled(not busy)

    def show_preview(
        self,
        image_path: str,
        *,
        extra_count: int = 0,
        image_paths: Optional[list] = None,
    ) -> None:
        """
        Display preview thumbnail(s) of the selected query image(s).

        Single-photo: one 96×96 thumbnail (unchanged look).
        Multi-photo: pass ``image_paths`` (2–3) to show a thumbnail for each.
        """
        paths = [str(p) for p in (image_paths or []) if str(p).strip()]
        if not paths:
            paths = [image_path]
            if extra_count > 0:
                # Legacy call sites that only pass extra_count — still show one.
                pass

        self._icon_label.setVisible(False)
        if len(paths) <= 1:
            self._multi_preview_row.setVisible(False)
            for lbl in self._multi_preview_labels:
                lbl.clear()
                lbl.setVisible(False)
            pixmap = QPixmap(paths[0])
            if not pixmap.isNull():
                scaled = pixmap.scaled(
                    96,
                    96,
                    Qt.AspectRatioMode.KeepAspectRatio,
                    Qt.TransformationMode.SmoothTransformation,
                )
                self._preview_label.setPixmap(scaled)
                self._preview_label.setVisible(True)
            name = Path(paths[0]).name
            self._title_label.setText(name)
            self._subtitle_label.setText("Click to search a different image")
            return

        # Multi-photo: hide the single large preview; show a row of thumbs.
        self._preview_label.clear()
        self._preview_label.setVisible(False)
        self._multi_preview_row.setVisible(True)
        for i, lbl in enumerate(self._multi_preview_labels):
            if i < len(paths):
                pix = QPixmap(paths[i])
                if not pix.isNull():
                    scaled = pix.scaled(
                        72,
                        72,
                        Qt.AspectRatioMode.KeepAspectRatio,
                        Qt.TransformationMode.SmoothTransformation,
                    )
                    lbl.setPixmap(scaled)
                lbl.setVisible(True)
            else:
                lbl.clear()
                lbl.setVisible(False)
        names = ", ".join(Path(p).name for p in paths[:3])
        self._title_label.setText(f"{len(paths)} photos selected")
        self._subtitle_label.setText(
            f"{names} — click a thumbnail to select for Crop, or click zone to replace"
        )
        if self._selected_thumb_index >= len(paths):
            self._selected_thumb_index = 0
        self._refresh_thumb_selection_style()

    def reset(self) -> None:
        """Reset the drop zone back to its empty prompt state."""
        self._preview_label.clear()
        self._preview_label.setVisible(False)
        self._multi_preview_row.setVisible(False)
        for lbl in self._multi_preview_labels:
            lbl.clear()
            lbl.setVisible(False)
        self._icon_label.setVisible(True)
        self.set_multi_photo_enabled(self._multi_photo_enabled)

    def _emit_paths(self, paths: list) -> None:
        if self._on_images_selected is not None:
            self._on_images_selected(list(paths))
        elif self._on_image_selected is not None and paths:
            self._on_image_selected(paths[0])

    def _validate_path(self, path: Path) -> Optional[str]:
        """Return a rejection reason, or None if the path is usable."""
        suffix = path.suffix.lower()
        if suffix not in _QUERY_IMAGE_EXTENSIONS:
            return (
                f"Unsupported image type '{suffix or '(none)'}'. "
                "Use JPG, JPEG, PNG, WEBP, BMP, or TIFF."
            )
        if not path.is_file():
            return f"Dropped path is not a readable file: {path.name}"
        if not validate_image(path):
            return f"Selected file is not a valid, readable image: {path.name}"
        return None

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

        accepted: list[str] = []
        for url in event.mimeData().urls():
            local = url.toLocalFile()
            if not local:
                continue
            path = Path(local)
            reason = self._validate_path(path)
            if reason is not None:
                if not self._multi_photo_enabled:
                    logger.warning("[SEARCH] Drop rejected: %s (%s)", reason, path)
                    if self._on_drop_rejected is not None:
                        self._on_drop_rejected(reason)
                    event.ignore()
                    return
                # In multi mode, skip bad files and keep collecting.
                logger.warning("[SEARCH] Skipping invalid drop: %s", reason)
                continue
            accepted.append(str(path))
            if not self._multi_photo_enabled:
                break
            if len(accepted) >= self._MAX_MULTI:
                break

        if not accepted:
            reason = (
                "Drop ignored — could not read a local image file. "
                "If this came from a browser or cloud app, save it to disk first."
            )
            logger.warning("[SEARCH] %s", reason)
            if self._on_drop_rejected is not None:
                self._on_drop_rejected(reason)
            event.ignore()
            return

        if self._multi_photo_enabled and len(accepted) < 2:
            reason = (
                "Multi-photo search needs 2–3 images. "
                "Drop at least two tile photos, or turn multi-photo off."
            )
            logger.warning("[SEARCH] %s", reason)
            if self._on_drop_rejected is not None:
                self._on_drop_rejected(reason)
            event.ignore()
            return

        logger.info("[SEARCH] Drop accepted (%d): %s", len(accepted), accepted)
        self._emit_paths(accepted)
        event.acceptProposedAction()

    def mousePressEvent(self, event) -> None:
        if event.button() == Qt.MouseButton.LeftButton and self.isEnabled():
            self._browse()
        super().mousePressEvent(event)

    def _browse(self) -> None:
        extensions = " ".join(f"*{ext}" for ext in sorted(_QUERY_IMAGE_EXTENSIONS))
        filter_str = f"Image Files ({extensions})"
        if self._multi_photo_enabled:
            file_paths, _ = QFileDialog.getOpenFileNames(
                self,
                "Select 2–3 tile photos to search",
                "",
                filter_str,
            )
            if not file_paths:
                return
            paths = list(file_paths)[: self._MAX_MULTI]
            if len(paths) < 2:
                if self._on_drop_rejected is not None:
                    self._on_drop_rejected(
                        "Multi-photo search needs 2–3 images. Select at least two."
                    )
                return
            self._emit_paths(paths)
            return

        file_path, _ = QFileDialog.getOpenFileName(
            self, "Select a tile photo to search", "", filter_str
        )
        if file_path:
            self._emit_paths([file_path])


class _ResultPreviewPanel(QWidget):
    """
    Lightweight, reusable "large preview" popup for Task C: Search UX.
    A single click on a results row calls show_result() on one shared
    instance, which updates its content and pops to the front — cheaper
    than constructing a new dialog per click.

    Note: ``Qt.WindowType.Tool`` keeps the OS native title bar/frame —
    that chrome is intentionally left alone (not themeable from Qt).
    """

    def __init__(self, parent=None, theme: str = "dark") -> None:
        super().__init__(parent, Qt.WindowType.Tool)
        self.setWindowTitle("Tile Preview")
        self.setMinimumSize(360, 420)
        self._theme = theme
        self._current_path: Optional[str] = None
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
        self._details_label.setTextInteractionFlags(
            Qt.TextInteractionFlag.TextBrowserInteraction
        )
        self._details_label.setOpenExternalLinks(False)
        self._details_label.linkActivated.connect(self._on_path_link_activated)
        layout.addWidget(self._details_label)

        self._status_label = QLabel("")
        self._status_label.setObjectName("PreviewStatus")
        self._status_label.setWordWrap(True)
        self._status_label.setVisible(False)
        layout.addWidget(self._status_label)
        layout.addStretch()

    def show_result(self, result: SearchResult) -> None:
        from src.utils.reveal_file import reveal_action_label

        p = get_palette(self._theme)
        tile = result.tile
        self._current_path = tile.file_path
        self._status_label.setVisible(False)
        self._status_label.clear()
        pixmap = QPixmap(tile.file_path)
        if pixmap.isNull():
            pixmap = QPixmap(result.thumbnail_path)
        if not pixmap.isNull():
            scaled = pixmap.scaled(
                320, 320, Qt.AspectRatioMode.KeepAspectRatio, Qt.TransformationMode.SmoothTransformation
            )
            self._image_label.setPixmap(scaled)

        # Escape path for HTML attribute/body; link triggers reveal_in_file_manager.
        safe_path = (
            tile.file_path.replace("&", "&amp;")
            .replace("<", "&lt;")
            .replace(">", "&gt;")
            .replace('"', "&quot;")
        )
        reveal_hint = reveal_action_label().lstrip("📍 ").strip()
        self._details_label.setText(
            f"<b>{tile.file_name}</b><br>"
            f"Similarity: {result.similarity_score:.1f}%<br>"
            f"Product Code: {tile.product_code or '—'}<br>"
            f"Brand: {tile.brand or '—'}<br>"
            f"Category: {tile.category or '—'}<br>"
            f"<a href='reveal' style='color:{p['accent_text']}; font-size:10px;' "
            f"title='{reveal_hint}'>{safe_path}</a>"
        )
        self.show()
        self.raise_()
        self.activateWindow()

    def _on_path_link_activated(self, _href: str) -> None:
        if not self._current_path:
            return
        # Defer so the click returns immediately (Explorer/Finder launch
        # can take a moment even when non-blocking).
        path = self._current_path
        QTimer.singleShot(0, lambda p=path: self._reveal_path_now(p))

    def _reveal_path_now(self, path: str) -> None:
        from src.utils.reveal_file import reveal_in_file_manager

        ok, error = reveal_in_file_manager(path)
        if ok:
            self._status_label.setVisible(False)
            self._status_label.clear()
            return
        # Inline status avoids stacking a message box on top of this Tool popup.
        p = get_palette(self._theme)
        self._status_label.setStyleSheet(f"color: {p['danger_text']}; font-size: 11px;")
        self._status_label.setText(error)
        self._status_label.setVisible(True)

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
            #PreviewDetails a {{ color: {p['accent_text']}; text-decoration: underline; }}
            #PreviewStatus {{ color: {p['danger_text']}; font-size: 11px; }}
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
        self._current_query_paths: list[str] = []
        # Original user photos — crops never overwrite these (precise-after-auto
        # must start from the photo, not from a previous crop output).
        self._original_query_paths: list[str] = []
        self._selected_query_index: int = 0
        self._crop_worker: Optional[TileCropWorker] = None
        self._crop_busy = False
        self._multi_crop_queue: list[tuple[int, str]] = []
        self._multi_crop_results: dict[int, str] = {}
        self._multi_crop_mode: Optional[str] = None
        self._multi_crop_total: int = 0
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
        # Tight chrome so ≥5 result rows fit at 1366×768 inside a real window.
        root_layout.setContentsMargins(16, 10, 16, 10)
        root_layout.setSpacing(8)

        root_layout.addWidget(self._build_header())
        query_panel = self._build_query_panel()
        # Tall enough for the 2-col grid (incl. Recent) without clipping;
        # keep ≤~220 so ≥5 result rows remain at 1366×768.
        query_panel.setMaximumHeight(220)
        root_layout.addWidget(query_panel, stretch=0)
        # Description + filters share one row (was two stacked bars).
        root_layout.addWidget(self._build_description_and_filters_row(), stretch=0)
        root_layout.addWidget(self._build_progress_bar(), stretch=0)
        self._confidence_banner = QLabel("")
        self._confidence_banner.setObjectName("ConfidenceBanner")
        self._confidence_banner.setWordWrap(True)
        self._confidence_banner.setVisible(False)
        root_layout.addWidget(self._confidence_banner, stretch=0)
        root_layout.addWidget(self._build_results_table(), stretch=1)
        root_layout.addWidget(self._build_status_line(), stretch=0)

    def _build_header(self) -> QWidget:
        header = QWidget()
        layout = QVBoxLayout(header)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(2)

        title = QLabel("Visual Tile Search")
        title.setObjectName("PageTitle")
        layout.addWidget(title)

        subtitle = QLabel(
            "Drag a tile photo in, browse for one, or type a short description "
            "to find matching tiles."
        )
        subtitle.setObjectName("PageSubtitle")
        layout.addWidget(subtitle)

        return header

    def _build_query_panel(self) -> QWidget:
        panel = QWidget()
        layout = QHBoxLayout(panel)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(12)

        drop_column = QWidget()
        drop_col_layout = QVBoxLayout(drop_column)
        drop_col_layout.setContentsMargins(0, 0, 0, 0)
        drop_col_layout.setSpacing(6)

        self._drop_zone = DropZone(
            # Keep legacy single-path callback populated for headless QA /
            # tools that still call DropZone._on_image_selected(path).
            on_image_selected=self._on_image_chosen,
            on_images_selected=self._on_images_chosen,
            on_drop_rejected=self._on_drop_rejected,
        )
        self._drop_zone.set_thumbnail_selected_callback(self._on_query_thumbnail_selected)
        drop_col_layout.addWidget(self._drop_zone, stretch=0)

        # Opt-in only — default off preserves the fast single-photo CPU path.
        self._multi_photo_checkbox = QCheckBox("Search with multiple photos (2–3)")
        self._multi_photo_checkbox.setObjectName("MultiPhotoCheckbox")
        self._multi_photo_checkbox.setChecked(False)
        self._multi_photo_checkbox.setToolTip(
            "Optional: fuse 2–3 separate photos into one search. "
            "Leave unchecked for the fastest single-photo match."
        )
        self._multi_photo_checkbox.toggled.connect(self._on_multi_photo_toggled)
        drop_col_layout.addWidget(self._multi_photo_checkbox)

        layout.addWidget(drop_column, stretch=1)

        # 2-column action grid — shorter than a tall single column, no clipped text.
        button_panel = QWidget()
        button_panel.setObjectName("SearchActionColumn")
        button_panel.setMinimumWidth(320)
        button_panel.setMaximumWidth(360)
        button_panel.setSizePolicy(
            QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Maximum
        )
        grid = QGridLayout(button_panel)
        grid.setContentsMargins(0, 0, 0, 0)
        grid.setHorizontalSpacing(8)
        grid.setVerticalSpacing(6)

        def _action_btn(text: str) -> QPushButton:
            # Dedicated object name so shared #SecondaryButton padding
            # (6px 14px) cannot inflate min-height past max-height.
            btn = QPushButton(text)
            btn.setObjectName("SearchActionButton")
            btn.setMinimumHeight(28)
            btn.setMaximumHeight(32)
            btn.setSizePolicy(
                QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed
            )
            return btn

        self._export_button = _action_btn("Export Catalogue")
        self._export_button.clicked.connect(self._on_export_clicked)
        self._export_button.setEnabled(False)
        grid.addWidget(self._export_button, 0, 0, 1, 2)

        self._crop_button = _action_btn("Crop and Search")
        self._crop_button.clicked.connect(self._on_crop_clicked)
        self._crop_button.setEnabled(False)
        grid.addWidget(self._crop_button, 1, 0)

        self._auto_crop_button = _action_btn("Auto Crop Search")
        self._auto_crop_button.setToolTip(
            "Fast tile-region crop for room photos (OpenCV, no SAM). "
            "Use Crop and Search for a manual selection."
        )
        self._auto_crop_button.clicked.connect(self._on_auto_crop_clicked)
        self._auto_crop_button.setEnabled(False)
        grid.addWidget(self._auto_crop_button, 1, 1)

        self._precise_crop_button = _action_btn("Precise Crop Search")
        self._precise_crop_button.setToolTip(
            "Precise tile crop for Windows, Mac Intel, and Mac Apple Silicon. "
            "Uses SAM 2 when enabled in Settings → Experimental (or TILEVISION_ENABLE_SAM2=1); "
            "otherwise GrabCut. Not used by default search."
        )
        self._precise_crop_button.clicked.connect(self._on_precise_crop_clicked)
        self._precise_crop_button.setEnabled(False)
        grid.addWidget(self._precise_crop_button, 2, 0)

        self._clear_button = _action_btn("Clear")
        self._clear_button.clicked.connect(self._on_clear_clicked)
        self._clear_button.setEnabled(False)
        grid.addWidget(self._clear_button, 2, 1)

        self._history_button = _action_btn("Recent Searches")
        self._history_button.clicked.connect(self._on_history_clicked)
        # Only open the popup when the user clicks Recent Searches.
        # Search completion also refreshes history via the same signal — that
        # must NOT pop a blocking QMenu (it freezes offscreen/CI and surprises customers).
        self._history_menu_pending = False
        grid.addWidget(self._history_button, 3, 0)

        self._view_all_history_button = _action_btn("View full history")
        self._view_all_history_button.setVisible(False)
        grid.addWidget(self._view_all_history_button, 3, 1)

        layout.addWidget(button_panel, stretch=0)

        return panel

    def _build_description_and_filters_row(self) -> QWidget:
        """Description search + filters on one compact row."""
        row = QWidget()
        row.setObjectName("DescriptionFilterRow")
        layout = QHBoxLayout(row)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(8)

        label = QLabel("Describe:")
        label.setObjectName("DescriptionSearchLabel")
        layout.addWidget(label)

        self._description_input = QLineEdit()
        self._description_input.setObjectName("DescriptionSearchInput")
        self._description_input.setPlaceholderText(
            "e.g. matte white subway tile with soft grey veins"
        )
        self._description_input.setClearButtonEnabled(True)
        self._description_input.setMinimumHeight(30)
        self._description_input.returnPressed.connect(self._on_description_search_clicked)
        self._description_input.textChanged.connect(
            lambda _t: self._refresh_action_buttons()
        )
        layout.addWidget(self._description_input, stretch=2)

        self._description_search_button = QPushButton("Search")
        self._description_search_button.setObjectName("PrimaryButton")
        self._description_search_button.setMinimumHeight(32)
        self._description_search_button.setToolTip(
            "Search the catalogue with a short text description "
            "(uses a separate CLIP index — not hybrid with image search)."
        )
        self._description_search_button.clicked.connect(self._on_description_search_clicked)
        layout.addWidget(self._description_search_button)

        filter_label = QLabel("Filters:")
        filter_label.setObjectName("FilterLabel")
        layout.addWidget(filter_label)

        self._filter_combos: dict = {}
        for field, display_name in [
            ("brand", "Brand"),
            ("category", "Category"),
            ("color", "Color"),
            ("size", "Size"),
        ]:
            combo = QComboBox()
            combo.setObjectName("FilterCombo")
            combo.setMinimumHeight(30)
            combo.addItem(f"Any {display_name}")
            combo.currentTextChanged.connect(
                lambda value, f=field: self._on_filter_changed(f, value)
            )
            self._filter_combos[field] = combo
            layout.addWidget(combo)

        return row

    def _on_description_search_clicked(self) -> None:
        text = self._description_input.text().strip()
        if not text:
            message_box.information(
                self,
                "Description Search",
                "Enter a short description of the tile you are looking for.",
            )
            return
        # Clear image-query chrome so results are clearly from text search.
        self._current_query_image_path = None
        self._drop_zone.reset()
        self._viewmodel.search_by_description(text)

    @staticmethod
    def _make_button_separator() -> QFrame:
        """Horizontal rule between action-button groups (legacy helper)."""
        separator = QFrame()
        separator.setFrameShape(QFrame.Shape.HLine)
        separator.setObjectName("SearchButtonSeparator")
        separator.setFixedHeight(8)
        return separator

    def _on_filter_changed(self, field: str, value: str) -> None:
        # Ignore the placeholder "Any X" item text as a real filter value.
        is_placeholder = value.startswith("Any ")
        applied = "" if is_placeholder else value
        logger.info(
            "[UI] filter '%s' → %s (QComboBox) in SearchView",
            field,
            applied or "(any)",
        )
        recorder = getattr(self._viewmodel, "_activity_repo", None)
        if recorder is not None:
            try:
                recorder.record_activity(
                    "search",
                    f"Filter {field}={applied or 'any'}",
                )
            except Exception:
                pass
        self._viewmodel.set_filter(field, applied)

    def _on_export_clicked(self) -> None:
        try:
            from src.presentation.dialogs.export_catalog_dialog import ExportCatalogDialog
            from src.presentation.workers.export_catalogue_worker import (
                ExportCatalogueWorker,
            )
        except Exception as exc:
            logger.exception("Export modules missing")
            message_box.critical(
                self,
                "Export Error",
                f"Export Catalogue is unavailable in this build:\n{exc}",
            )
            return

        if not self._current_results:
            message_box.information(self, "Export Catalogue", "No results to export yet.")
            return

        if getattr(self, "_export_worker", None) is not None and self._export_worker.isRunning():
            message_box.information(
                self,
                "Export Catalogue",
                "An export is already in progress.",
            )
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
            selected_indices = sorted(
                {index.row() for index in self._results_table.selectionModel().selectedRows()}
            )
            if not selected_indices:
                message_box.information(
                    self,
                    "Export Catalogue",
                    "Select one or more results first "
                    "(Ctrl/Shift-click to select multiple rows).",
                )
                return

        # Unique tiles only — copies listed as "Also at" under each card.
        export_results = self._unique_results_for_export(
            self._current_results, selected_indices
        )
        logger.info(
            "Export Catalogue starting: results=%s unique_tiles=%s "
            "query_image=%s output=%s",
            len(self._current_results),
            len(export_results),
            self._current_query_image_path or "(none)",
            output_path,
        )

        from PySide6.QtWidgets import QProgressDialog

        progress = QProgressDialog(
            "Preparing catalogue…", "Cancel", 0, 100, self
        )
        progress.setWindowTitle("Export Catalogue")
        progress.setWindowModality(Qt.WindowModality.WindowModal)
        progress.setMinimumDuration(0)
        progress.setValue(0)
        progress.show()

        worker = ExportCatalogueWorker(
            output_file=output_path,
            query_image_path=self._current_query_image_path,
            results=export_results,
            options=options,
            parent=self,
        )
        self._export_worker = worker
        self._export_progress = progress
        self._export_unique_count = len(export_results)

        def _on_progress(done: int, total: int, label: str) -> None:
            total = max(1, int(total))
            progress.setMaximum(total)
            progress.setValue(min(int(done), total))
            progress.setLabelText(label or "Exporting…")

        def _on_finished(created: str) -> None:
            progress.close()
            self._export_worker = None
            logger.info(
                "Export Catalogue finished: unique_tiles=%s path=%s",
                self._export_unique_count,
                created,
            )
            if hasattr(self._viewmodel, "record_catalogue_export"):
                self._viewmodel.record_catalogue_export(
                    created, self._export_unique_count
                )
            message_box.information(
                self, "Export Complete", f"Catalogue saved to:\n{created}"
            )

        def _on_failed(message: str) -> None:
            progress.close()
            self._export_worker = None
            logger.error("Export Catalogue failed: %s", message)
            if hasattr(self._viewmodel, "record_catalogue_export_failed"):
                self._viewmodel.record_catalogue_export_failed(message)
            elif getattr(self._viewmodel, "_activity_repo", None) is not None:
                try:
                    self._viewmodel._activity_repo.record_activity(
                        "export", f"Export Catalogue failed — {message}"
                    )
                except Exception:
                    pass
            if "cancel" in (message or "").lower():
                message_box.information(self, "Export Cancelled", message)
            else:
                message_box.critical(self, "Export Failed", message)

        def _on_cancel() -> None:
            worker.cancel()
            progress.setLabelText("Cancelling…")

        progress.canceled.connect(_on_cancel)
        worker.progress.connect(_on_progress)
        worker.export_finished.connect(_on_finished)
        worker.export_failed.connect(_on_failed)
        worker.start()

   
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
        # Ctrl/Shift multi-select for "Export only selected" catalogues.
        self._results_table.setSelectionMode(
            QAbstractItemView.SelectionMode.ExtendedSelection
        )
        self._results_table.setIconSize(QSize(64, 64))
        self._results_table.verticalHeader().setDefaultSectionSize(68)
        self._results_table.setAlternatingRowColors(True)
        self._results_table.setContextMenuPolicy(Qt.ContextMenuPolicy.CustomContextMenu)

        header = self._results_table.horizontalHeader()
        header.setMinimumSectionSize(40)
        header.setSectionResizeMode(0, QHeaderView.ResizeMode.Fixed)
        self._results_table.setColumnWidth(0, 88)
        header.setSectionResizeMode(1, QHeaderView.ResizeMode.ResizeToContents)
        header.setSectionResizeMode(2, QHeaderView.ResizeMode.ResizeToContents)
        header.setSectionResizeMode(3, QHeaderView.ResizeMode.ResizeToContents)
        header.setSectionResizeMode(4, QHeaderView.ResizeMode.ResizeToContents)
        # Path is plain text (no cell widget) — stretch to fill.
        header.setSectionResizeMode(_PATH_COL, QHeaderView.ResizeMode.Stretch)
        # Actions: preview / copy / reveal icons.
        header.setSectionResizeMode(_ACTIONS_COL, QHeaderView.ResizeMode.Fixed)
        self._results_table.setColumnWidth(_ACTIONS_COL, _ACTIONS_COL_WIDTH)
        header.setStretchLastSection(False)

        self._results_table.cellDoubleClicked.connect(self._on_row_double_clicked)
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
        # History re-run / search_by_image emit this — keep path + action buttons in sync.
        self._viewmodel.query_image_selected.connect(self._on_query_image_selected)
        self._viewmodel.filters_available.connect(self._on_filters_available)
        self._viewmodel.search_history_updated.connect(self._show_history_menu)
        self._viewmodel.search_stats_ready.connect(self._on_search_stats_ready)
        self._viewmodel.load_filter_options()
        self._refresh_action_buttons()

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

    def _on_query_image_selected(self, image_path: str) -> None:
        """Keep drop-zone preview and action buttons aligned with the ViewModel query."""
        # Do not overwrite _original_query_paths — crops are display/search only.
        self._current_query_image_path = image_path
        paths = list(getattr(self, "_current_query_paths", []) or [])
        if len(paths) > 1:
            self._drop_zone.show_preview(image_path, image_paths=paths)
            self._drop_zone.set_selected_thumb_index(self._selected_query_index)
        else:
            self._drop_zone.show_preview(image_path)
        self._refresh_action_buttons()

    def _on_multi_photo_toggled(self, checked: bool) -> None:
        self._drop_zone.set_multi_photo_enabled(checked)
        if checked:
            self._status_label.setText(
                "Multi-photo mode on — drop or browse 2–3 tile photos."
            )
        else:
            self._status_label.setText("Ready — drop a tile photo to search.")

    def _on_images_chosen(self, image_paths: list) -> None:
        paths = [str(p) for p in (image_paths or []) if str(p).strip()]
        if not paths:
            return
        if len(paths) == 1:
            self._on_image_chosen(paths[0])
            return

        logger.info("Multi-photo query selected: %s", paths)
        self._original_query_paths = list(paths)
        self._current_query_paths = list(paths)
        self._current_query_image_path = paths[0]
        self._selected_query_index = 0
        self._confidence_banner.setVisible(False)
        self._drop_zone.set_selected_thumb_index(0)
        self._drop_zone.show_preview(paths[0], image_paths=paths)
        self._refresh_action_buttons()
        self._status_label.setText(
            f"Multi-photo search with {len(paths)} images…"
        )
        self._viewmodel.search_by_images(paths)

    def _on_image_chosen(self, image_path: str) -> None:
        logger.info(f"Query image selected: {image_path}")
        self._original_query_paths = [image_path]
        self._current_query_paths = [image_path]
        self._current_query_image_path = image_path
        self._selected_query_index = 0
        self._confidence_banner.setVisible(False)
        # Enable Clear / crop immediately (SEARCHING can lag one event-loop tick).
        self._refresh_action_buttons()

        suggest_crop, crop_reason = should_suggest_crop(image_path)
        if suggest_crop:
            # Non-blocking: drop-search isolates the tile (SAM2/OpenCV),
            # straightens perspective, and multi-crop FAISS searches.
            self._status_label.setText(
                f"{crop_reason} Isolating tile region for a better match…"
            )
        self._viewmodel.search_by_image(image_path)

    def _on_query_thumbnail_selected(self, index: int) -> None:
        """Manual Crop operates on the selected multi-photo thumbnail."""
        if not self._original_query_paths:
            return
        if index < 0 or index >= len(self._original_query_paths):
            return
        self._selected_query_index = index
        self._drop_zone.set_selected_thumb_index(index)
        n = len(self._original_query_paths)
        name = Path(self._original_query_paths[index]).name
        self._status_label.setText(
            f"Selected photo {index + 1} of {n} ({name}) for Crop and Search."
        )

    def _crop_source_path(self, index: Optional[int] = None) -> Optional[str]:
        """Always crop from the original photo, never from a prior crop output."""
        idx = self._selected_query_index if index is None else index
        originals = list(getattr(self, "_original_query_paths", []) or [])
        if originals and 0 <= idx < len(originals):
            return originals[idx]
        return self._current_query_image_path

    def _on_drop_rejected(self, reason: str) -> None:
        self._status_label.setText(reason)
        self._empty_state_label.setText(reason)
        self._empty_state_widget.setVisible(True)
        message_box.information(self, "Cannot Search This File", reason)

    def _on_crop_clicked(self) -> None:
        source = self._crop_source_path()
        if not source:
            return
        n = max(1, len(self._original_query_paths) or 1)
        idx = self._selected_query_index
        title = (
            f"Crop photo {idx + 1} of {n}"
            if n > 1
            else "Crop & Search — Select a Region"
        )
        dialog = CropDialog(
            source, parent=self, theme=self._theme, window_title=title
        )
        if dialog.exec() == dialog.DialogCode.Accepted and dialog.cropped_image_path:
            crop_path = str(dialog.cropped_image_path)
            logger.info("Searching with cropped region: %s", crop_path)
            if len(self._original_query_paths) >= 2:
                paths = list(self._current_query_paths or self._original_query_paths)
                if idx < len(paths):
                    paths[idx] = crop_path
                self._current_query_paths = paths
                self._current_query_image_path = paths[0]
                self._drop_zone.show_preview(paths[0], image_paths=paths)
                self._drop_zone.set_selected_thumb_index(idx)
                self._viewmodel.search_by_images(paths)
            else:
                # Single-photo: preview the crop but keep original for next crop.
                self._current_query_paths = [crop_path]
                self._current_query_image_path = crop_path
                self._drop_zone.show_preview(crop_path)
                self._viewmodel.search_by_image(crop_path, query_origin="crop_tool")

    def _on_auto_crop_clicked(self) -> None:
        if not self._crop_source_path():
            return
        if len(self._original_query_paths) >= 2:
            self._start_multi_background_crop("auto")
        else:
            self._start_background_crop("auto")

    def _on_precise_crop_clicked(self) -> None:
        if not self._crop_source_path():
            return
        if len(self._original_query_paths) >= 2:
            self._start_multi_background_crop("precise")
        else:
            self._start_background_crop("precise")

    def _start_multi_background_crop(self, mode: str) -> None:
        """Crop every original photo, then one multi-photo search."""
        if self._crop_busy or self._viewmodel.is_searching:
            return
        originals = list(self._original_query_paths)
        if len(originals) < 2:
            self._start_background_crop(mode)
            return

        self._stop_prior_crop_worker()
        self._multi_crop_mode = mode
        self._multi_crop_queue = list(enumerate(originals))
        self._multi_crop_results = {}
        self._multi_crop_total = len(originals)
        self._crop_busy = True
        self._set_crop_controls_enabled(False)
        self._progress_bar.setVisible(True)
        self._run_next_multi_crop()

    def _run_next_multi_crop(self) -> None:
        if not self._multi_crop_queue:
            self._finish_multi_crop_search()
            return
        index, source = self._multi_crop_queue.pop(0)
        mode = self._multi_crop_mode or "auto"
        label = "Precise Crop" if mode == "precise" else "Auto Crop"
        self._status_label.setText(
            f"Cropping photo {index + 1} of {self._multi_crop_total} ({label})…"
        )
        worker = TileCropWorker(source, mode)  # type: ignore[arg-type]
        self._crop_worker = worker
        worker.crop_finished.connect(
            lambda path, crop, idx=index, m=mode: self._on_multi_crop_one_finished(
                m, idx, path, crop
            ),
            Qt.ConnectionType.QueuedConnection,
        )
        worker.crop_failed.connect(
            lambda message, m=mode: self._on_background_crop_failed(m, message),
            Qt.ConnectionType.QueuedConnection,
        )
        worker.finished.connect(
            lambda w=worker: self._on_crop_worker_finished(w),
            Qt.ConnectionType.QueuedConnection,
        )
        worker.start()

    def _on_multi_crop_one_finished(
        self, mode: str, index: int, crop_path: str, crop
    ) -> None:
        self._multi_crop_results[index] = str(crop_path)
        logger.info(
            "Multi-photo %s crop %d/%d → %s",
            mode,
            index + 1,
            self._multi_crop_total,
            crop_path,
        )
        # Continue after this worker's finished signal has cleaned up.
        QTimer.singleShot(0, self._run_next_multi_crop)

    def _finish_multi_crop_search(self) -> None:
        self._crop_busy = False
        self._progress_bar.setVisible(False)
        n = self._multi_crop_total
        paths = [
            self._multi_crop_results.get(i, self._original_query_paths[i])
            for i in range(n)
        ]
        self._multi_crop_mode = None
        self._multi_crop_queue = []
        self._current_query_paths = paths
        self._current_query_image_path = paths[0] if paths else None
        self._drop_zone.show_preview(paths[0], image_paths=paths)
        self._drop_zone.set_selected_thumb_index(self._selected_query_index)
        self._status_label.setText(
            f"Cropped {n} photos — running multi-photo search…"
        )
        self._refresh_action_buttons()
        self._viewmodel.search_by_images(paths)

    def _stop_prior_crop_worker(self) -> None:
        prev = self._crop_worker
        if prev is not None:
            if prev.isRunning():
                prev.requestInterruption()
                prev.wait(5_000)
            self._crop_worker = None

    def _start_background_crop(self, mode: str) -> None:
        """Run Auto/Precise crop off the UI thread (critical on Mac Intel)."""
        if self._crop_busy or self._viewmodel.is_searching:
            return
        source = self._crop_source_path()
        if not source:
            return

        self._stop_prior_crop_worker()
        self._multi_crop_mode = None
        self._multi_crop_queue = []

        label = "Precise Crop" if mode == "precise" else "Auto Crop"
        self._crop_busy = True
        self._set_crop_controls_enabled(False)
        self._status_label.setText(f"{label} running… UI stays responsive.")
        self._progress_bar.setVisible(True)

        worker = TileCropWorker(source, mode)  # type: ignore[arg-type]
        self._crop_worker = worker
        # QueuedConnection: never run UI/search work nested inside QThread.run().
        worker.crop_finished.connect(
            lambda path, crop, m=mode: self._on_background_crop_finished(m, path, crop),
            Qt.ConnectionType.QueuedConnection,
        )
        worker.crop_failed.connect(
            lambda message, m=mode: self._on_background_crop_failed(m, message),
            Qt.ConnectionType.QueuedConnection,
        )
        worker.finished.connect(
            lambda w=worker: self._on_crop_worker_finished(w),
            Qt.ConnectionType.QueuedConnection,
        )
        worker.start()

    def _on_crop_worker_finished(self, worker: TileCropWorker) -> None:
        """Release the worker only after QThread.finished (thread fully stopped)."""
        if self._crop_worker is worker:
            self._crop_worker = None
        worker.deleteLater()

    def _on_background_crop_finished(self, mode: str, crop_path: str, crop) -> None:
        # Multi-photo queue uses _on_multi_crop_one_finished instead.
        if self._multi_crop_mode is not None and self._multi_crop_total > 1:
            return
        self._crop_busy = False
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

        # Preview the crop for search, but keep _original_query_paths intact
        # so Precise after Auto still uses the user's photo.
        self._current_query_paths = [str(crop_path)]
        self._current_query_image_path = str(crop_path)
        self._drop_zone.show_preview(str(crop_path))
        self._viewmodel.search_by_image(str(crop_path), query_origin="crop_tool")

    def _on_background_crop_failed(self, mode: str, message: str) -> None:
        self._crop_busy = False
        self._multi_crop_mode = None
        self._multi_crop_queue = []
        self._progress_bar.setVisible(self._viewmodel.is_searching)
        self._refresh_action_buttons()
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
        has_query = bool(self._current_query_image_path)
        allow = enabled and has_query and not self._crop_busy
        self._crop_button.setEnabled(allow)
        self._auto_crop_button.setEnabled(allow)
        self._precise_crop_button.setEnabled(allow)

    def _refresh_crop_controls(self) -> None:
        searching = self._viewmodel.is_searching
        self._set_crop_controls_enabled(not searching and not self._crop_busy)

    def _refresh_action_buttons(self) -> None:
        """Sync Export / Clear / crop enablement from query + results state."""
        has_query = bool(self._current_query_image_path)
        has_description = bool(self._description_input.text().strip())
        searching = self._viewmodel.is_searching
        has_results = bool(self._current_results)
        # Export needs results (and implies a query was run).
        self._export_button.setEnabled(
            has_results and not searching and not self._crop_busy
        )
        self._clear_button.setEnabled(
            (has_query or has_description or has_results) and not self._crop_busy
        )
        self._description_search_button.setEnabled(not searching and not self._crop_busy)
        self._description_input.setEnabled(not searching and not self._crop_busy)
        self._refresh_crop_controls()

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
        self._description_input.clear()
        self._current_query_paths = []
        self._original_query_paths = []
        self._selected_query_index = 0
        self._multi_crop_queue = []
        self._multi_crop_results = {}
        self._multi_crop_mode = None
        self._current_results = []
        self._confidence_banner.setVisible(False)
        self._progress_bar.setVisible(False)
        self._viewmodel.clear_results()
        self._refresh_action_buttons()

    def set_view_all_history(self, callback) -> None:
        """Wire the Search page's 'View full history' button to the History stack page."""
        self._view_all_history_button.setVisible(callback is not None)
        if callback is not None:
            try:
                self._view_all_history_button.clicked.disconnect()
            except RuntimeError:
                pass
            self._view_all_history_button.clicked.connect(callback)

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
        if self._view_all_history_button.isVisible():
            menu.addSeparator()
            view_all = menu.addAction("View full history…")
            view_all.triggered.connect(self._view_all_history_button.click)
        menu.exec(self._history_button.mapToGlobal(self._history_button.rect().bottomLeft()))

    @Slot(str)
    def _on_state_changed(self, state: str) -> None:
        is_searching = state == SearchState.SEARCHING
        self._drop_zone.set_busy(is_searching or self._crop_busy)
        self._progress_bar.setVisible(is_searching or self._crop_busy)
        self._refresh_action_buttons()
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
        self._refresh_action_buttons()
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
        # Kind-aware copy: description search must never suggest Crop & Search.
        search_kind = getattr(self._viewmodel, "_active_search_kind", "image") or "image"
        message = confidence_message(results, search_kind=search_kind)
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
        palette = get_palette(self._theme)
        best_match_bg = QColor(palette["highlight_bg"])
        best_match_text = QColor(palette["text_primary"])
        any_why = False

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

            # Plain path text + optional "+N copies" badge. Actions live in
            # the Actions column so the path never double-paints under a widget.
            extra = int(getattr(result, "extra_copy_count", 0) or 0)
            path_label = tile.file_path
            if extra > 0:
                path_label = f"{tile.file_path}  (+{extra} copies)"
            path_item = QTableWidgetItem(path_label)
            copy_paths = list(getattr(result, "copy_paths", None) or [tile.file_path])
            tip_lines = ["Primary:"] + [f"  {copy_paths[0]}"]
            if len(copy_paths) > 1:
                tip_lines.append("Other copies:")
                tip_lines.extend(f"  {p}" for p in copy_paths[1:])
                tip_lines.append("Right-click → Show all copies…")
            path_item.setToolTip("\n".join(tip_lines))
            self._results_table.setItem(row, _PATH_COL, path_item)

            # Placeholder item under the actions widget (selection/background).
            actions_item = QTableWidgetItem("")
            actions_item.setFlags(actions_item.flags() & ~Qt.ItemFlag.ItemIsEditable)
            self._results_table.setItem(row, _ACTIONS_COL, actions_item)
            # Why only when ORB data exists *and* this is not a description search
            # (no query photo to overlay). Hidden entirely otherwise.
            search_kind = getattr(self._viewmodel, "_active_search_kind", "image")
            has_orb = bool(getattr(result, "orb_match_points", None))
            show_why = bool(has_orb and search_kind != "description")
            actions = _ResultActionsWidget(
                tile.file_path,
                theme=self._theme,
                on_preview=lambda _p, r=row: self._preview_at_row(r),
                on_copy=self._copy_path_text,
                on_reveal=self._reveal_path_deferred,
                on_why=lambda _p, r=row: self._why_at_row(r),
                why_enabled=show_why,
                parent=self._results_table,
            )
            self._results_table.setCellWidget(row, _ACTIONS_COL, actions)
            if show_why:
                any_why = True

            if is_best_match:
                for col in range(self._results_table.columnCount()):
                    item = self._results_table.item(row, col)
                    if item is not None:
                        item.setBackground(best_match_bg)
                        item.setForeground(best_match_text)
                        font = item.font()
                        font.setBold(True)
                        item.setFont(font)

        width = _ACTIONS_COL_WIDTH_WITH_WHY if any_why else _ACTIONS_COL_WIDTH
        self._results_table.setColumnWidth(_ACTIONS_COL, width)

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

    def _preview_at_row(self, row: int) -> None:
        if row < 0 or row >= len(self._current_results):
            return
        self._preview_panel.show_result(self._current_results[row])

    def _why_at_row(self, row: int) -> None:
        """Show ORB inlier overlay for a result that was verified during search."""
        if row < 0 or row >= len(self._current_results):
            return
        result = self._current_results[row]
        points = getattr(result, "orb_match_points", None) or []
        if not points:
            message_box.information(
                self,
                "Why this match",
                "No ORB match data for this result.\n\n"
                "Match highlights are only available when geometric "
                "verification ran for a near-tie during search.",
            )
            return
        query_path = self._current_query_image_path
        if not query_path:
            message_box.information(
                self,
                "Why this match",
                "No query photo is available for this search.",
            )
            return
        from src.presentation.views.why_match_dialog import WhyThisMatchDialog

        dialog = WhyThisMatchDialog(
            query_path,
            result.tile.file_path,
            points,
            theme=self._theme,
            parent=self,
        )
        dialog.exec()

    def _on_results_context_menu(self, position) -> None:
        row = self._results_table.rowAt(position.y())
        if row < 0 or row >= len(self._current_results):
            return

        from src.utils.reveal_file import reveal_action_label

        result = self._current_results[row]
        copy_paths = list(getattr(result, "copy_paths", None) or [result.tile.file_path])
        extra = int(getattr(result, "extra_copy_count", 0) or 0)

        menu = QMenu(self)
        open_image_action = menu.addAction("Open Image")
        open_folder_action = menu.addAction("Open Containing Folder")
        reveal_action = menu.addAction(reveal_action_label())
        copy_path_action = menu.addAction("Copy Path")
        show_copies_action = None
        reveal_other_actions: list = []
        if extra > 0 and len(copy_paths) > 1:
            menu.addSeparator()
            show_copies_action = menu.addAction("Show all copies…")
            others = menu.addMenu("Reveal other copy")
            for other in copy_paths[1:]:
                act = others.addAction(other)
                reveal_other_actions.append((act, other))

        chosen = menu.exec(self._results_table.viewport().mapToGlobal(position))
        if chosen == open_image_action:
            self._open_image_at_row(row)
        elif chosen == open_folder_action:
            self._open_folder_at_row(row)
        elif chosen == reveal_action:
            self._reveal_file_at_row(row)
        elif chosen == copy_path_action:
            self._copy_path_at_row(row)
        elif show_copies_action is not None and chosen == show_copies_action:
            message_box.information(
                self,
                "All copies",
                "\n".join(copy_paths),
            )
        else:
            for act, other_path in reveal_other_actions:
                if chosen == act:
                    self._reveal_path_deferred(other_path)
                    break

    @staticmethod
    def _unique_results_for_export(
        results: List[SearchResult],
        selected_indices: Optional[List[int]],
    ) -> List[SearchResult]:
        """
        One SearchResult per unique tile for PDF export.

        Extra catalogue copies stay on ``copy_paths`` and render as
        "Also at: …" under the card — they must not consume ``max_results``
        slots (repeating the same tile N times in a dealer PDF is wrong).
        """
        chosen = (
            [results[i] for i in selected_indices if 0 <= i < len(results)]
            if selected_indices is not None
            else list(results)
        )
        unique: List[SearchResult] = []
        for result in chosen:
            paths = list(getattr(result, "copy_paths", None) or [])
            if not paths and getattr(result.tile, "file_path", None):
                paths = [result.tile.file_path]
            if paths and (
                not getattr(result, "copy_paths", None)
                or list(result.copy_paths or []) != paths
            ):
                from dataclasses import replace

                unique.append(replace(result, copy_paths=paths))
            else:
                unique.append(result)
        return unique

    # Back-compat alias for older tests / callers.
    _expand_results_for_export = _unique_results_for_export

    def _copy_path_text(self, path: str) -> None:
        """Copy path silently — Copied feedback is the floating toast on the icon."""
        QApplication.clipboard().setText(path)

    def _copy_path_at_row(self, row: int) -> None:
        if row < 0 or row >= len(self._current_results):
            return
        path = self._current_results[row].tile.file_path
        self._copy_path_text(path)
        cell = self._results_table.cellWidget(row, _ACTIONS_COL)
        if isinstance(cell, _ResultActionsWidget):
            cell._show_copied_toast()

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

    def _reveal_path_deferred(self, path: str) -> None:
        """Return immediately; open the file manager on the next event-loop tick."""
        QTimer.singleShot(0, lambda p=path: self._reveal_path_now(p))

    def _reveal_path_now(self, path: str) -> None:
        from src.utils.reveal_file import reveal_in_file_manager

        ok, error = reveal_in_file_manager(path)
        if not ok:
            message_box.warning(self, "File Not Found", error)

    def _reveal_file_at_row(self, row: int) -> None:
        if row < 0 or row >= len(self._current_results):
            return
        self._reveal_path_deferred(self._current_results[row].tile.file_path)

    # ── Styling ──────────────────────────────────────────────────────────

    def set_theme(self, theme: str) -> None:
        """Re-skin this view (and its preview panel) for a newly-selected theme."""
        self._theme = theme
        self._preview_panel.set_theme(theme)
        for row in range(self._results_table.rowCount()):
            cell = self._results_table.cellWidget(row, _ACTIONS_COL)
            if isinstance(cell, _ResultActionsWidget):
                cell.set_theme(theme)
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

            #DescriptionSearchLabel {{
                color: {p['text_muted']};
                font-size: 12px;
                font-weight: 600;
            }}
            #DescriptionSearchInput {{
                background-color: {p['bg_input']};
                color: {p['text_primary']};
                border: 1px solid {p['border_strong']};
                border-radius: 6px;
                padding: 8px 10px;
                font-size: 13px;
                min-height: 20px;
            }}
            #DescriptionSearchInput:focus {{
                border-color: {p['accent_hover']};
            }}
            #DescriptionSearchBar {{
                background: transparent;
            }}

            #DropZone {{
                background-color: {p['bg_panel']};
                border: 2px dashed {p['border_strong']};
                border-radius: 10px;
            }}
            #DropZone[dragActive="true"] {{
                border: 2px dashed {p['accent_hover']};
                background-color: {p['row_alt']};
            }}
            #DropZoneIcon {{ font-size: 28px; }}
            #DropZoneTitle {{ font-size: 13px; font-weight: 600; color: {p['text_primary']}; }}
            #DropZoneSubtitle {{ font-size: 11px; color: {p['text_muted']}; }}
            #DropZoneMultiPreview {{
                background-color: {p['bg_panel_alt']};
                border: 1px solid {p['border_strong']};
                border-radius: 6px;
            }}
            #MultiPhotoCheckbox {{
                color: {p['text_secondary']};
                font-size: 12px;
                spacing: 8px;
                background: transparent;
            }}

            #ResultsTable {{
                background-color: {p['bg_panel_alt']};
                alternate-background-color: {p['bg_panel']};
                gridline-color: {p['border']};
                color: {p['text_secondary']};
                border: 1px solid {p['border']};
                border-radius: 8px;
            }}
            #ResultsTable::item:selected {{ background-color: {p['highlight_bg']}; color: {p['text_primary']}; }}
            #ResultsTable::item:hover {{
                background-color: {p['hover_bg']};
                color: {p['text_primary']};
            }}
            #SearchActionColumn {{
                max-width: 360px;
                background: transparent;
            }}
            #SearchActionButton {{
                background-color: {p['button_bg']};
                color: {p['text_secondary']};
                border: 1px solid {p['border_strong']};
                border-radius: 6px;
                min-height: 28px;
                max-height: 32px;
                padding: 4px 8px;
                font-size: 11px;
                font-weight: 600;
            }}
            #SearchActionButton:hover:enabled {{
                background-color: {p['button_hover']};
                color: {p['text_primary']};
                border-color: {p['accent_soft']};
            }}
            #SearchActionButton:disabled {{
                color: {p['text_faint']};
                border-color: {p['border']};
            }}
            #SearchButtonSeparator {{
                border: none;
                border-top: 1px solid {p['border_strong']};
                margin: 6px 4px 2px 4px;
                background: transparent;
            }}
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
