"""
Crop Dialog for TileVision AI (Feature 4: Partial Image Search).

Lets the user drag a rectangular selection over their query image and
search using only that cropped region — useful when a customer's photo
shows a whole room but they only want to match the tile pattern in one
corner, or when a WhatsApp photo has multiple tile types in frame.

Manual Crop never silently overrides the user's rectangle. It *does*
pre-fill a starting box aligned with Auto Crop and shows a live coverage
hint so tight selections (which collapse edge/pattern similarity) are
obvious before Search.
"""

import logging
import tempfile
from pathlib import Path
from typing import Optional

from PIL import Image, ImageOps
from PySide6.QtCore import Qt, QRect, QPoint, Signal
from PySide6.QtGui import QPixmap, QPainter, QPen, QColor, QMouseEvent, QPaintEvent
from PySide6.QtWidgets import (
    QDialog,
    QVBoxLayout,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QWidget,
)

from src.presentation.views.manual_crop_ux import (
    format_selection_status,
    is_tight_selection,
    suggest_manual_default_box,
)
from src.theme.theme_manager import get_palette, get_shared_view_qss

logger = logging.getLogger("tilevision.presentation.views.crop_dialog")

_DISPLAY_MAX_SIZE = 640  # max width/height for the crop preview canvas
_MIN_DRAG_PX = 10


class _CropCanvas(QWidget):
    """
    Displays the query image scaled to fit and lets the user drag out a
    rectangular selection with the mouse. Emits selection_changed whenever
    the dragged rectangle updates.
    """

    selection_changed = Signal(object)  # QRect in canvas coordinates, or None

    def __init__(
        self,
        pixmap: QPixmap,
        parent=None,
        *,
        initial_selection: Optional[QRect] = None,
    ) -> None:
        super().__init__(parent)
        self._original_pixmap = pixmap
        self._display_pixmap = pixmap.scaled(
            _DISPLAY_MAX_SIZE,
            _DISPLAY_MAX_SIZE,
            Qt.AspectRatioMode.KeepAspectRatio,
            Qt.TransformationMode.SmoothTransformation,
        )
        self.setFixedSize(self._display_pixmap.size())
        self.setCursor(Qt.CursorShape.CrossCursor)

        self._drag_start: Optional[QPoint] = None
        self._selection_rect: Optional[QRect] = initial_selection
        self._pre_drag_selection: Optional[QRect] = None
        self._warn_tight = False

    @property
    def selection_rect(self) -> Optional[QRect]:
        return self._selection_rect

    @property
    def display_to_original_scale(self) -> float:
        """Ratio to convert a canvas-space coordinate back to the original image's pixels."""
        if self._display_pixmap.width() == 0:
            return 1.0
        return self._original_pixmap.width() / self._display_pixmap.width()

    def set_warn_tight(self, warn: bool) -> None:
        if self._warn_tight != warn:
            self._warn_tight = warn
            self.update()

    def clear_selection(self) -> None:
        self._selection_rect = None
        self.update()
        self.selection_changed.emit(None)

    def paintEvent(self, event: QPaintEvent) -> None:
        painter = QPainter(self)
        painter.drawPixmap(0, 0, self._display_pixmap)

        if self._selection_rect is not None:
            # Dim everything outside the selection so the crop region is obvious.
            painter.setBrush(QColor(0, 0, 0, 140))
            painter.setPen(Qt.PenStyle.NoPen)
            full_rect = self.rect()

            for dim_rect in self._regions_outside_selection(full_rect, self._selection_rect):
                painter.drawRect(dim_rect)

            # Amber border when the selection is unusually tight (hint only).
            border = QColor("#F59E0B") if self._warn_tight else QColor("#0EA5E9")
            pen = QPen(border, 2)
            painter.setPen(pen)
            painter.setBrush(Qt.BrushStyle.NoBrush)
            painter.drawRect(self._selection_rect)

    @staticmethod
    def _regions_outside_selection(full: QRect, selection: QRect):
        """Yield up to 4 rectangles covering `full` minus `selection`."""
        yield QRect(full.left(), full.top(), full.width(), selection.top() - full.top())
        yield QRect(
            full.left(),
            selection.bottom(),
            full.width(),
            full.bottom() - selection.bottom(),
        )
        yield QRect(
            full.left(),
            selection.top(),
            selection.left() - full.left(),
            selection.height(),
        )
        yield QRect(
            selection.right(),
            selection.top(),
            full.right() - selection.right(),
            selection.height(),
        )

    def mousePressEvent(self, event: QMouseEvent) -> None:
        if event.button() == Qt.MouseButton.LeftButton:
            self._drag_start = event.position().toPoint()
            # Keep prior selection until the drag is large enough so a
            # mis-click does not wipe the Auto-aligned default box.
            self._pre_drag_selection = (
                QRect(self._selection_rect) if self._selection_rect is not None else None
            )
            self._selection_rect = QRect(self._drag_start, self._drag_start)
            self.update()

    def mouseMoveEvent(self, event: QMouseEvent) -> None:
        if self._drag_start is not None:
            current = event.position().toPoint()
            self._selection_rect = QRect(self._drag_start, current).normalized()
            # Clamp to canvas bounds
            self._selection_rect = self._selection_rect.intersected(self.rect())
            self.selection_changed.emit(self._selection_rect)
            self.update()

    def mouseReleaseEvent(self, event: QMouseEvent) -> None:
        if event.button() == Qt.MouseButton.LeftButton:
            self._drag_start = None
            if self._selection_rect is not None and (
                self._selection_rect.width() < _MIN_DRAG_PX
                or self._selection_rect.height() < _MIN_DRAG_PX
            ):
                # Tiny / accidental drag — restore previous selection (or none).
                self._selection_rect = self._pre_drag_selection
            self._pre_drag_selection = None
            self.selection_changed.emit(self._selection_rect)
            self.update()


class CropDialog(QDialog):
    """
    Dialog for cropping a region out of the query image before searching.

    Usage:
        dialog = CropDialog(query_image_path)
        if dialog.exec() == QDialog.DialogCode.Accepted:
            cropped_path = dialog.cropped_image_path  # feed this into search
    """

    def __init__(self, image_path: str, parent=None, theme: str = "dark") -> None:
        super().__init__(parent)
        self._source_path = Path(image_path)
        self._cropped_image_path: Optional[str] = None
        self._theme = theme
        self._image_size = (0, 0)
        self._default_method = "none"

        self.setWindowTitle("Crop & Search — Select a Region")
        self._setup_ui()
        self._apply_styles()

    @property
    def cropped_image_path(self) -> Optional[str]:
        """Absolute path to the cropped temp image, set only after a successful accept()."""
        return self._cropped_image_path

    def _setup_ui(self) -> None:
        layout = QVBoxLayout(self)
        layout.setContentsMargins(20, 16, 20, 16)
        layout.setSpacing(12)

        instructions = QLabel(
            "Drag to adjust the search region. A starting box is pre-filled from "
            "Auto Crop (you can redraw it anytime). Expand a small selection if "
            "you meant the full tile surface."
        )
        instructions.setObjectName("Instructions")
        instructions.setWordWrap(True)
        layout.addWidget(instructions)

        pixmap = QPixmap(str(self._source_path))
        self._image_size = (pixmap.width(), pixmap.height())
        initial_canvas = self._compute_initial_canvas_rect(pixmap)
        self._canvas = _CropCanvas(pixmap, initial_selection=initial_canvas)
        self._canvas.selection_changed.connect(self._on_selection_changed)

        canvas_wrapper = QHBoxLayout()
        canvas_wrapper.addStretch()
        canvas_wrapper.addWidget(self._canvas)
        canvas_wrapper.addStretch()
        layout.addLayout(canvas_wrapper)

        self._status_label = QLabel("")
        self._status_label.setObjectName("SelectionStatus")
        self._status_label.setWordWrap(True)
        layout.addWidget(self._status_label)

        button_row = QHBoxLayout()
        clear_button = QPushButton("Clear Selection")
        clear_button.setObjectName("SecondaryButton")
        clear_button.clicked.connect(self._canvas.clear_selection)
        button_row.addWidget(clear_button)
        button_row.addStretch()

        cancel_button = QPushButton("Cancel")
        cancel_button.setObjectName("SecondaryButton")
        cancel_button.clicked.connect(self.reject)
        button_row.addWidget(cancel_button)

        self._use_selection_button = QPushButton("Search This Region")
        self._use_selection_button.setObjectName("PrimaryButton")
        self._use_selection_button.setEnabled(False)
        self._use_selection_button.clicked.connect(self._on_use_selection)
        button_row.addWidget(self._use_selection_button)

        layout.addLayout(button_row)

        # Apply initial status / enable Search for the Auto-aligned default.
        self._on_selection_changed(self._canvas.selection_rect)

    def _compute_initial_canvas_rect(self, pixmap: QPixmap) -> Optional[QRect]:
        """Map Auto Crop's suggested box into canvas coordinates."""
        try:
            with Image.open(self._source_path) as img:
                source = ImageOps.exif_transpose(img.convert("RGB"))
            suggestion = suggest_manual_default_box(source)
            self._default_method = suggestion.method
            self._image_size = source.size
            left, top, right, bottom = suggestion.box
            # Recompute the same KeepAspectRatio scale QPixmap.scaled uses.
            scale = min(
                _DISPLAY_MAX_SIZE / max(1, pixmap.width()),
                _DISPLAY_MAX_SIZE / max(1, pixmap.height()),
            )
            display_w = max(1, int(round(pixmap.width() * scale)))
            display_h = max(1, int(round(pixmap.height() * scale)))
            sx = display_w / max(1, source.size[0])
            sy = display_h / max(1, source.size[1])
            canvas = QRect(
                int(round(left * sx)),
                int(round(top * sy)),
                max(1, int(round((right - left) * sx))),
                max(1, int(round((bottom - top) * sy))),
            )
            logger.info(
                "Manual Crop default box method=%s keep=%.1f%% canvas=%sx%s",
                suggestion.method,
                100.0 * suggestion.keep_ratio,
                canvas.width(),
                canvas.height(),
            )
            return canvas
        except Exception as exc:
            logger.warning("Manual Crop default suggestion failed: %s", exc)
            return None

    def _canvas_rect_to_image_box(self, rect: QRect) -> tuple[int, int, int, int]:
        scale = self._canvas.display_to_original_scale
        left = int(rect.x() * scale)
        top = int(rect.y() * scale)
        width = int(rect.width() * scale)
        height = int(rect.height() * scale)
        return (left, top, left + width, top + height)

    def _on_selection_changed(self, rect: Optional[QRect]) -> None:
        self._use_selection_button.setEnabled(rect is not None)
        box = None if rect is None else self._canvas_rect_to_image_box(rect)
        label, warn = format_selection_status(self._image_size, box)
        self._status_label.setText(label)
        self._status_label.setProperty("warn", "true" if warn else "false")
        self._status_label.style().unpolish(self._status_label)
        self._status_label.style().polish(self._status_label)
        self._canvas.set_warn_tight(
            bool(box is not None and is_tight_selection(self._image_size, box))
        )

    def _on_use_selection(self) -> None:
        rect = self._canvas.selection_rect
        if rect is None:
            return

        try:
            scale = self._canvas.display_to_original_scale
            original_rect = QRect(
                int(rect.x() * scale),
                int(rect.y() * scale),
                int(rect.width() * scale),
                int(rect.height() * scale),
            )

            original_pixmap = QPixmap(str(self._source_path))
            cropped = original_pixmap.copy(original_rect)

            if cropped.isNull() or cropped.width() == 0 or cropped.height() == 0:
                logger.warning("Crop produced an empty image — ignoring.")
                return

            keep = (cropped.width() * cropped.height()) / max(
                1, original_pixmap.width() * original_pixmap.height()
            )
            logger.info(
                "Manual Crop accepted %dx%d keep=%.1f%% (default_method=%s)",
                cropped.width(),
                cropped.height(),
                100.0 * keep,
                self._default_method,
            )

            temp_dir = Path(tempfile.gettempdir()) / "tilevision_crops"
            temp_dir.mkdir(parents=True, exist_ok=True)
            temp_path = temp_dir / f"crop_{self._source_path.stem}_{id(self)}.jpg"
            cropped.save(str(temp_path), "JPEG", quality=95)
            try:
                import shutil

                shutil.copy2(temp_path, temp_dir / "last_manual.jpg")
            except OSError:
                pass

            self._cropped_image_path = str(temp_path)
            self.accept()
        except Exception as e:
            logger.error(f"Failed to produce cropped image: {e}")

    def _apply_styles(self) -> None:
        p = get_palette(self._theme)
        self.setStyleSheet(
            get_shared_view_qss(self._theme)
            + f"""
            QDialog {{ background-color: {p['bg_app']}; }}
            QWidget {{ color: {p['text_primary']}; }}
            #Instructions {{ color: {p['text_muted']}; font-size: 12px; }}
            #SelectionStatus {{ color: {p['text_muted']}; font-size: 12px; }}
            #SelectionStatus[warn="true"] {{ color: #F59E0B; font-weight: 600; }}
            """
        )
