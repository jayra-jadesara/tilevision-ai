"""Theme-tinted line icons for search result action buttons."""

from __future__ import annotations

from PySide6.QtCore import QPointF, QRectF, Qt
from PySide6.QtGui import QColor, QIcon, QPainter, QPen, QPixmap


def _pen(color: str, width: float = 1.6) -> QPen:
    pen = QPen(QColor(color))
    pen.setWidthF(width)
    pen.setCapStyle(Qt.PenCapStyle.RoundCap)
    pen.setJoinStyle(Qt.PenJoinStyle.RoundJoin)
    return pen


def _blank(size: int = 28) -> QPixmap:
    pix = QPixmap(size, size)
    pix.fill(Qt.GlobalColor.transparent)
    return pix


def icon_preview(color: str, size: int = 28) -> QIcon:
    """Simple eye outline."""
    pix = _blank(size)
    painter = QPainter(pix)
    painter.setRenderHint(QPainter.RenderHint.Antialiasing, True)
    painter.setPen(_pen(color))
    painter.setBrush(Qt.BrushStyle.NoBrush)
    # Eye oval
    painter.drawEllipse(QRectF(5, 9, 18, 10))
    # Pupil
    painter.drawEllipse(QRectF(11.5, 11.5, 5, 5))
    painter.end()
    return QIcon(pix)


def icon_copy(color: str, size: int = 28) -> QIcon:
    """Two overlapping rectangles."""
    pix = _blank(size)
    painter = QPainter(pix)
    painter.setRenderHint(QPainter.RenderHint.Antialiasing, True)
    painter.setPen(_pen(color))
    painter.setBrush(Qt.BrushStyle.NoBrush)
    painter.drawRoundedRect(QRectF(9, 7, 11, 13), 1.5, 1.5)
    painter.drawRoundedRect(QRectF(6, 10, 11, 13), 1.5, 1.5)
    painter.end()
    return QIcon(pix)


def icon_reveal(color: str, size: int = 28) -> QIcon:
    """Folder outline."""
    pix = _blank(size)
    painter = QPainter(pix)
    painter.setRenderHint(QPainter.RenderHint.Antialiasing, True)
    painter.setPen(_pen(color))
    painter.setBrush(Qt.BrushStyle.NoBrush)
    # Tab
    painter.drawLine(QPointF(5, 10), QPointF(5, 8))
    painter.drawLine(QPointF(5, 8), QPointF(11, 8))
    painter.drawLine(QPointF(11, 8), QPointF(13, 10))
    # Body
    painter.drawRoundedRect(QRectF(5, 10, 18, 11), 1.5, 1.5)
    painter.end()
    return QIcon(pix)


def icon_why(color: str, size: int = 28) -> QIcon:
    """Question mark in a circle (only when ORB data exists)."""
    pix = _blank(size)
    painter = QPainter(pix)
    painter.setRenderHint(QPainter.RenderHint.Antialiasing, True)
    painter.setPen(_pen(color))
    painter.setBrush(Qt.BrushStyle.NoBrush)
    painter.drawEllipse(QRectF(5, 5, 18, 18))
    font = painter.font()
    font.setPixelSize(13)
    font.setBold(True)
    painter.setFont(font)
    painter.drawText(QRectF(5, 5, 18, 18), int(Qt.AlignmentFlag.AlignCenter), "?")
    painter.end()
    return QIcon(pix)
