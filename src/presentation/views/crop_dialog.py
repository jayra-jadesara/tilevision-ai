"""
Crop Dialog for TileVision AI (Feature 4: Partial Image Search).

Lets the user drag a rectangular selection over their query image and
search using only that cropped region — useful when a customer's photo
shows a whole room but they only want to match the tile pattern in one
corner, or when a WhatsApp photo has multiple tile types in frame.
"""

import logging
import tempfile
from pathlib import Path
from typing import Optional

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

from src.theme.theme_manager import get_palette

logger = logging.getLogger("tilevision.presentation.views.crop_dialog")

_DISPLAY_MAX_SIZE = 640  # max width/height for the crop preview canvas


class _CropCanvas(QWidget):
    """
    Displays the query image scaled to fit and lets the user drag out a
    rectangular selection with the mouse. Emits selection_changed whenever
    the dragged rectangle updates.
    """

    selection_changed = Signal(object)  # QRect in canvas coordinates, or None

    def __init__(self, pixmap: QPixmap, parent=None) -> None:
        super().__init__(parent)
        self._original_pixmap = pixmap
        self._display_pixmap = pixmap.scaled(
            _DISPLAY_MAX_SIZE, _DISPLAY_MAX_SIZE,
            Qt.AspectRatioMode.KeepAspectRatio, Qt.TransformationMode.SmoothTransformation,
        )
        self.setFixedSize(self._display_pixmap.size())
        self.setCursor(Qt.CursorShape.CrossCursor)

        self._drag_start: Optional[QPoint] = None
        self._selection_rect: Optional[QRect] = None

    @property
    def selection_rect(self) -> Optional[QRect]:
        return self._selection_rect

    @property
    def display_to_original_scale(self) -> float:
        """Ratio to convert a canvas-space coordinate back to the original image's pixels."""
        if self._display_pixmap.width() == 0:
            return 1.0
        return self._original_pixmap.width() / self._display_pixmap.width()

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

            pen = QPen(QColor("#5C6BC0"), 2)
            painter.setPen(pen)
            painter.setBrush(Qt.BrushStyle.NoBrush)
            painter.drawRect(self._selection_rect)

    @staticmethod
    def _regions_outside_selection(full: QRect, selection: QRect):
        """Yield up to 4 rectangles covering `full` minus `selection`."""
        yield QRect(full.left(), full.top(), full.width(), selection.top() - full.top())
        yield QRect(full.left(), selection.bottom(), full.width(), full.bottom() - selection.bottom())
        yield QRect(full.left(), selection.top(), selection.left() - full.left(), selection.height())
        yield QRect(selection.right(), selection.top(), full.right() - selection.right(), selection.height())

    def mousePressEvent(self, event: QMouseEvent) -> None:
        if event.button() == Qt.MouseButton.LeftButton:
            self._drag_start = event.position().toPoint()
            self._selection_rect = QRect(self._drag_start, self._drag_start)
            self.update()

    def mouseMoveEvent(self, event: QMouseEvent) -> None:
        if self._drag_start is not None:
            current = event.position().toPoint()
            self._selection_rect = QRect(self._drag_start, current).normalized()
            # Clamp to canvas bounds
            self._selection_rect = self._selection_rect.intersected(self.rect())
            self.update()

    def mouseReleaseEvent(self, event: QMouseEvent) -> None:
        if event.button() == Qt.MouseButton.LeftButton:
            self._drag_start = None
            if self._selection_rect is not None and (
                self._selection_rect.width() < 10 or self._selection_rect.height() < 10
            ):
                # Treat a tiny/accidental drag as "no selection".
                self._selection_rect = None
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
            "Drag a rectangle over the part of the photo you want to search with "
            "(e.g. just the tile pattern, not the whole room)."
        )
        instructions.setObjectName("Instructions")
        instructions.setWordWrap(True)
        layout.addWidget(instructions)

        pixmap = QPixmap(str(self._source_path))
        self._canvas = _CropCanvas(pixmap)
        self._canvas.selection_changed.connect(self._on_selection_changed)

        canvas_wrapper = QHBoxLayout()
        canvas_wrapper.addStretch()
        canvas_wrapper.addWidget(self._canvas)
        canvas_wrapper.addStretch()
        layout.addLayout(canvas_wrapper)

        button_row = QHBoxLayout()
        clear_button = QPushButton("↺  Clear Selection")
        clear_button.clicked.connect(self._canvas.clear_selection)
        button_row.addWidget(clear_button)
        button_row.addStretch()

        cancel_button = QPushButton("Cancel")
        cancel_button.clicked.connect(self.reject)
        button_row.addWidget(cancel_button)

        self._use_selection_button = QPushButton("Search This Region")
        self._use_selection_button.setObjectName("PrimaryButton")
        self._use_selection_button.setEnabled(False)
        self._use_selection_button.clicked.connect(self._on_use_selection)
        button_row.addWidget(self._use_selection_button)

        layout.addLayout(button_row)

    def _on_selection_changed(self, rect: Optional[QRect]) -> None:
        self._use_selection_button.setEnabled(rect is not None)

    def _on_use_selection(self) -> None:
        rect = self._canvas.selection_rect
        if rect is None:
            return

        try:
            scale = self._canvas.display_to_original_scale
            original_rect = QRect(
                int(rect.x() * scale), int(rect.y() * scale),
                int(rect.width() * scale), int(rect.height() * scale),
            )

            original_pixmap = QPixmap(str(self._source_path))
            cropped = original_pixmap.copy(original_rect)

            if cropped.isNull() or cropped.width() == 0 or cropped.height() == 0:
                logger.warning("Crop produced an empty image — ignoring.")
                return

            temp_dir = Path(tempfile.gettempdir()) / "tilevision_crops"
            temp_dir.mkdir(parents=True, exist_ok=True)
            temp_path = temp_dir / f"crop_{self._source_path.stem}_{id(self)}.jpg"
            cropped.save(str(temp_path), "JPEG", quality=92)

            self._cropped_image_path = str(temp_path)
            self.accept()
        except Exception as e:
            logger.error(f"Failed to produce cropped image: {e}")

    def _apply_styles(self) -> None:
        p = get_palette(self._theme)
        self.setStyleSheet(
            f"""
            QDialog {{ background-color: {p['bg_app']}; }}
            QWidget {{ color: {p['text_primary']}; }}
            #Instructions {{ color: {p['text_muted']}; font-size: 12px; }}
            QPushButton {{
                background-color: {p['button_bg']}; border: 1px solid {p['border_strong']}; border-radius: 6px;
                padding: 8px 14px; color: {p['text_secondary']};
            }}
            QPushButton:hover:enabled {{ background-color: {p['button_hover']}; }}
            #PrimaryButton {{ background-color: {p['accent']}; color: white; font-weight: 600; }}
            #PrimaryButton:hover:enabled {{ background-color: {p['accent_hover']}; }}
            #PrimaryButton:disabled {{ background-color: {p['button_bg']}; color: {p['text_faint']}; }}
            """
        )
