"""Side-by-side 'Why this match' visualization using ORB inlier points."""

from __future__ import annotations

from typing import Optional, Sequence

from PySide6.QtCore import Qt, QPointF
from PySide6.QtGui import QColor, QPainter, QPen, QPixmap
from PySide6.QtWidgets import (
    QDialog,
    QDialogButtonBox,
    QHBoxLayout,
    QLabel,
    QVBoxLayout,
    QWidget,
)

from src.theme.theme_manager import get_palette


class _MatchCanvas(QWidget):
    """Image with simple circle overlays at ORB inlier locations."""

    def __init__(
        self,
        pixmap: QPixmap,
        points: Sequence[tuple[float, float]],
        *,
        theme: str = "dark",
        parent: Optional[QWidget] = None,
    ) -> None:
        super().__init__(parent)
        self._theme = theme
        self._source = pixmap
        self._points = list(points)
        self._display = QPixmap()
        self._scale = 1.0
        self._offset_x = 0.0
        self._offset_y = 0.0
        self.setMinimumSize(280, 280)
        self._rebuild_display()

    def _rebuild_display(self) -> None:
        if self._source.isNull():
            self._display = QPixmap()
            return
        target_w = max(280, self.width() or 280)
        target_h = max(280, self.height() or 280)
        self._display = self._source.scaled(
            target_w,
            target_h,
            Qt.AspectRatioMode.KeepAspectRatio,
            Qt.TransformationMode.SmoothTransformation,
        )
        self._scale = self._display.width() / max(1, self._source.width())
        self._offset_x = (self.width() - self._display.width()) / 2
        self._offset_y = (self.height() - self._display.height()) / 2
        self.update()

    def resizeEvent(self, event) -> None:
        super().resizeEvent(event)
        self._rebuild_display()

    def paintEvent(self, event) -> None:
        painter = QPainter(self)
        painter.fillRect(self.rect(), QColor(get_palette(self._theme)["bg_panel"]))
        if self._display.isNull():
            return
        x = int((self.width() - self._display.width()) / 2)
        y = int((self.height() - self._display.height()) / 2)
        painter.drawPixmap(x, y, self._display)

        # Customer-legible accents: soft dots + a light bounding region.
        p = get_palette(self._theme)
        accent = QColor(p["accent"])
        fill = QColor(p["accent"])
        fill.setAlpha(90)

        mapped: list[QPointF] = []
        for px, py in self._points:
            mx = x + px * self._scale
            my = y + py * self._scale
            mapped.append(QPointF(mx, my))

        if len(mapped) >= 2:
            xs = [pt.x() for pt in mapped]
            ys = [pt.y() for pt in mapped]
            pad = 10
            painter.setPen(QPen(accent, 2))
            painter.setBrush(fill)
            painter.drawRoundedRect(
                int(min(xs) - pad),
                int(min(ys) - pad),
                int(max(xs) - min(xs) + 2 * pad),
                int(max(ys) - min(ys) + 2 * pad),
                8,
                8,
            )

        painter.setPen(QPen(accent, 2))
        painter.setBrush(QColor(255, 255, 255, 220))
        for pt in mapped[:40]:  # cap clutter for showroom readability
            painter.drawEllipse(pt, 5, 5)


class WhyThisMatchDialog(QDialog):
    """Show query vs result with ORB match highlights."""

    def __init__(
        self,
        query_image_path: str,
        result_image_path: str,
        match_points: Sequence[tuple[tuple[float, float], tuple[float, float]]],
        *,
        theme: str = "dark",
        parent: Optional[QWidget] = None,
    ) -> None:
        super().__init__(parent)
        self._theme = theme
        self.setWindowTitle("Why this match")
        self.setMinimumSize(720, 420)
        self.setObjectName("WhyThisMatchDialog")

        q_pts = [pair[0] for pair in match_points]
        c_pts = [pair[1] for pair in match_points]

        root = QVBoxLayout(self)
        root.setContentsMargins(16, 16, 16, 16)
        root.setSpacing(12)

        title = QLabel("Why this match")
        title.setObjectName("DialogTitle")
        subtitle = QLabel(
            "Highlighted spots show where the photos share the same tile detail."
        )
        subtitle.setObjectName("DialogSubtitle")
        subtitle.setWordWrap(True)
        root.addWidget(title)
        root.addWidget(subtitle)

        row = QHBoxLayout()
        row.setSpacing(12)

        q_col = QVBoxLayout()
        q_label = QLabel("Your photo")
        q_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        q_pix = QPixmap(query_image_path)
        q_canvas = _MatchCanvas(q_pix, q_pts, theme=theme)
        q_col.addWidget(q_label)
        q_col.addWidget(q_canvas, stretch=1)

        c_col = QVBoxLayout()
        c_label = QLabel("Catalogue match")
        c_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        c_pix = QPixmap(result_image_path)
        c_canvas = _MatchCanvas(c_pix, c_pts, theme=theme)
        c_col.addWidget(c_label)
        c_col.addWidget(c_canvas, stretch=1)

        row.addLayout(q_col, stretch=1)
        row.addLayout(c_col, stretch=1)
        root.addLayout(row, stretch=1)

        buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Close)
        buttons.rejected.connect(self.reject)
        buttons.accepted.connect(self.accept)
        close_btn = buttons.button(QDialogButtonBox.StandardButton.Close)
        if close_btn is not None:
            close_btn.clicked.connect(self.accept)
        root.addWidget(buttons)

        self._apply_styles()

    def _apply_styles(self) -> None:
        p = get_palette(self._theme)
        self.setStyleSheet(
            f"""
            #WhyThisMatchDialog {{
                background-color: {p['bg_app']};
                color: {p['text_primary']};
            }}
            #DialogTitle {{
                font-size: 18px;
                font-weight: 700;
                color: {p['text_primary']};
            }}
            #DialogSubtitle {{
                font-size: 12px;
                color: {p['text_muted']};
            }}
            QLabel {{
                color: {p['text_secondary']};
                background: transparent;
            }}
            QPushButton {{
                min-width: 88px;
                padding: 6px 14px;
                background-color: {p['button_bg']};
                color: {p['text_primary']};
                border: 1px solid {p['border']};
                border-radius: 6px;
            }}
            QPushButton:hover {{
                background-color: {p['button_hover']};
            }}
            """
        )
