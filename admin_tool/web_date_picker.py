"""Web-style date picker: single field with integrated calendar dropdown."""

from __future__ import annotations

from typing import Callable, Optional

from PySide6.QtCore import QDate, QPoint, QSize, Qt, Signal
from PySide6.QtGui import QMouseEvent
from PySide6.QtWidgets import (
    QCalendarWidget,
    QFrame,
    QHBoxLayout,
    QLineEdit,
    QPushButton,
    QSizePolicy,
    QVBoxLayout,
    QWidget,
)

_FIELD_HEIGHT = 36
_CALENDAR_BTN_WIDTH = 40


class _ClickableLineEdit(QLineEdit):
    clicked = Signal()

    def mousePressEvent(self, event: QMouseEvent) -> None:
        if event.button() == Qt.MouseButton.LeftButton:
            self.clicked.emit()
        super().mousePressEvent(event)


class _CalendarPopup(QFrame):
    def __init__(self, picker: "WebDatePicker") -> None:
        super().__init__(None, Qt.WindowType.Popup | Qt.WindowType.FramelessWindowHint)
        self._picker = picker
        self.setObjectName("WebDatePopup")

        layout = QVBoxLayout(self)
        layout.setContentsMargins(8, 8, 8, 8)

        self._calendar = QCalendarWidget()
        self._calendar.setObjectName("WebDateCalendar")
        self._calendar.setGridVisible(True)
        layout.addWidget(self._calendar)

        self._calendar.clicked.connect(self._on_date_chosen)
        self._calendar.activated.connect(self._on_date_chosen)

    def sync_from_picker(self) -> None:
        self._calendar.setMinimumDate(self._picker.minimumDate())
        self._calendar.setSelectedDate(self._picker.date())

    def _on_date_chosen(self, date: QDate) -> None:
        self._picker.setDate(date)
        self.hide()


class WebDatePicker(QWidget):
    """Single-field date input with a dropdown calendar (web datepicker style)."""

    dateChanged = Signal(QDate)

    def __init__(
        self,
        parent: Optional[QWidget] = None,
        *,
        get_qss: Optional[Callable[[], str]] = None,
    ) -> None:
        super().__init__(parent)
        self._get_qss = get_qss
        self._min_date = QDate.currentDate()
        self._date = QDate.currentDate()
        self._popup: Optional[_CalendarPopup] = None

        self.setObjectName("WebDatePicker")
        self.setFixedHeight(_FIELD_HEIGHT)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)

        layout = QHBoxLayout(self)
        layout.setContentsMargins(1, 1, 1, 1)
        layout.setSpacing(0)

        inner_height = _FIELD_HEIGHT - 2

        self._edit = _ClickableLineEdit()
        self._edit.setObjectName("WebDateEdit")
        self._edit.setReadOnly(True)
        self._edit.setFixedHeight(inner_height)
        self._edit.setCursor(Qt.CursorShape.PointingHandCursor)
        self._edit.clicked.connect(self._open_calendar)
        layout.addWidget(self._edit, stretch=1)

        self._calendar_btn = QPushButton("\U0001F4C5")
        self._calendar_btn.setObjectName("WebDateButton")
        self._calendar_btn.setFixedSize(_CALENDAR_BTN_WIDTH, inner_height)
        self._calendar_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        self._calendar_btn.setToolTip("Pick a date")
        self._calendar_btn.clicked.connect(self._open_calendar)
        layout.addWidget(self._calendar_btn)

        self._refresh_display()

    def sizeHint(self) -> QSize:
        return QSize(280, _FIELD_HEIGHT)

    def minimumSizeHint(self) -> QSize:
        return QSize(200, _FIELD_HEIGHT)

    def setMinimumDate(self, date: QDate) -> None:
        self._min_date = date
        if self._date < date:
            self.setDate(date)

    def minimumDate(self) -> QDate:
        return self._min_date

    def setDate(self, date: QDate) -> None:
        if not date.isValid():
            return
        if date < self._min_date:
            date = self._min_date
        if date == self._date:
            return
        self._date = date
        self._refresh_display()
        self.dateChanged.emit(self._date)

    def date(self) -> QDate:
        return self._date

    def setDisplayFormat(self, fmt: str) -> None:
        self._display_format = fmt
        self._refresh_display()

    def _refresh_display(self) -> None:
        fmt = getattr(self, "_display_format", "yyyy-MM-dd")
        self._edit.setText(self._date.toString(fmt))

    def _open_calendar(self) -> None:
        if not self.isEnabled():
            return

        if self._popup is None:
            self._popup = _CalendarPopup(self)

        if self._get_qss is not None:
            self._popup.setStyleSheet(self._get_qss())

        self._popup.sync_from_picker()
        anchor = self.mapToGlobal(QPoint(0, self.height() + 2))
        self._popup.move(anchor)
        self._popup.setFixedWidth(max(self.width(), 320))
        self._popup.show()
        self._popup.raise_()
        self._popup._calendar.setFocus()
