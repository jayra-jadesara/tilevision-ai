"""Web-style date picker: single field with integrated calendar dropdown."""

from __future__ import annotations

from typing import Callable, Optional

from PySide6.QtCore import QDate, QPoint, QSize, Qt, Signal
from PySide6.QtGui import QGuiApplication, QMouseEvent
from PySide6.QtWidgets import (
    QCalendarWidget,
    QComboBox,
    QFrame,
    QHBoxLayout,
    QLineEdit,
    QPushButton,
    QSizePolicy,
    QVBoxLayout,
    QWidget,
)

_FIELD_HEIGHT = 36
_FIELD_WIDTH = 210
_CALENDAR_BTN_WIDTH = 40
_POPUP_WIDTH = 272
_POPUP_HEIGHT = 252
_MAX_YEAR = 2099
_COMBO_VISIBLE_ITEMS = 10


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
        self._updating_nav = False
        self.setObjectName("WebDatePopup")
        self.setFixedSize(_POPUP_WIDTH, _POPUP_HEIGHT)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(10, 10, 10, 10)
        layout.setSpacing(8)

        nav = QWidget()
        nav.setObjectName("WebDateNav")
        nav_layout = QHBoxLayout(nav)
        nav_layout.setContentsMargins(0, 0, 0, 0)
        nav_layout.setSpacing(6)

        self._prev_btn = QPushButton("\u2039")
        self._prev_btn.setObjectName("WebDateNavButton")
        self._prev_btn.setFixedSize(28, 28)
        self._prev_btn.setToolTip("Previous month")

        self._month_combo = QComboBox()
        self._month_combo.setObjectName("WebDateMonth")
        self._month_combo.setMaxVisibleItems(_COMBO_VISIBLE_ITEMS)
        for month in range(1, 13):
            self._month_combo.addItem(QDate(2000, month, 1).toString("MMMM"), month)

        self._year_combo = QComboBox()
        self._year_combo.setObjectName("WebDateYear")
        self._year_combo.setMaxVisibleItems(_COMBO_VISIBLE_ITEMS)

        self._next_btn = QPushButton("\u203a")
        self._next_btn.setObjectName("WebDateNavButton")
        self._next_btn.setFixedSize(28, 28)
        self._next_btn.setToolTip("Next month")

        nav_layout.addWidget(self._prev_btn)
        nav_layout.addWidget(self._month_combo, stretch=1)
        nav_layout.addWidget(self._year_combo)
        nav_layout.addWidget(self._next_btn)
        layout.addWidget(nav)

        self._calendar = QCalendarWidget()
        self._calendar.setObjectName("WebDateCalendar")
        self._calendar.setNavigationBarVisible(False)
        self._calendar.setGridVisible(True)
        self._calendar.setVerticalHeaderFormat(
            QCalendarWidget.VerticalHeaderFormat.NoVerticalHeader
        )
        self._calendar.setHorizontalHeaderFormat(
            QCalendarWidget.HorizontalHeaderFormat.ShortDayNames
        )
        self._calendar.setMinimumWidth(_POPUP_WIDTH - 20)
        self._calendar.setFixedHeight(192)
        layout.addWidget(self._calendar)

        self._prev_btn.clicked.connect(self._calendar.showPreviousMonth)
        self._next_btn.clicked.connect(self._calendar.showNextMonth)
        self._month_combo.currentIndexChanged.connect(self._on_nav_changed)
        self._year_combo.currentIndexChanged.connect(self._on_nav_changed)
        self._calendar.currentPageChanged.connect(self._sync_nav_from_calendar)
        self._calendar.clicked.connect(self._on_date_chosen)
        self._calendar.activated.connect(self._on_date_chosen)

    def _populate_years(self, focus_year: int) -> None:
        min_year = self._picker.minimumDate().year()
        max_year = max(_MAX_YEAR, focus_year)
        self._year_combo.blockSignals(True)
        self._year_combo.clear()
        for year in range(min_year, max_year + 1):
            self._year_combo.addItem(str(year), year)
        index = self._year_combo.findData(focus_year)
        if index >= 0:
            self._year_combo.setCurrentIndex(index)
        self._year_combo.blockSignals(False)

    def _sync_nav_controls(self, year: int, month: int) -> None:
        self._month_combo.blockSignals(True)
        self._year_combo.blockSignals(True)
        self._month_combo.setCurrentIndex(max(0, month - 1))
        year_index = self._year_combo.findData(year)
        if year_index >= 0:
            self._year_combo.setCurrentIndex(year_index)
        self._month_combo.blockSignals(False)
        self._year_combo.blockSignals(False)

    def _sync_nav_from_calendar(self, year: int, month: int) -> None:
        if self._updating_nav:
            return
        self._sync_nav_controls(year, month)

    def _on_nav_changed(self) -> None:
        month = self._month_combo.currentData()
        year = self._year_combo.currentData()
        if month is None or year is None:
            return
        self._updating_nav = True
        self._calendar.setCurrentPage(int(year), int(month))
        self._updating_nav = False

    def sync_from_picker(self) -> None:
        selected = self._picker.date()
        minimum = self._picker.minimumDate()
        self._calendar.setMinimumDate(minimum)
        self._calendar.setSelectedDate(selected)
        self._populate_years(selected.year())
        self._updating_nav = True
        self._calendar.setCurrentPage(selected.year(), selected.month())
        self._updating_nav = False
        self._sync_nav_controls(selected.year(), selected.month())

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
        self.setFixedSize(_FIELD_WIDTH, _FIELD_HEIGHT)
        self.setSizePolicy(QSizePolicy.Policy.Fixed, QSizePolicy.Policy.Fixed)

        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
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
        return QSize(_FIELD_WIDTH, _FIELD_HEIGHT)

    def minimumSizeHint(self) -> QSize:
        return QSize(_FIELD_WIDTH, _FIELD_HEIGHT)

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

        anchor = self.mapToGlobal(QPoint(0, self.height() + 4))
        x, y = anchor.x(), anchor.y()

        screen = QGuiApplication.screenAt(anchor) or QGuiApplication.primaryScreen()
        if screen is not None:
            bounds = screen.availableGeometry()
            if x + _POPUP_WIDTH > bounds.right():
                x = max(bounds.left() + 8, bounds.right() - _POPUP_WIDTH - 8)
            if y + _POPUP_HEIGHT > bounds.bottom():
                y = self.mapToGlobal(QPoint(0, 0)).y() - _POPUP_HEIGHT - 4

        self._popup.move(x, y)
        self._popup.show()
        self._popup.raise_()
        self._popup._calendar.setFocus()
