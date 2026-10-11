"""
Activity Log page — full feed from IActivityLogRepository.
"""

from __future__ import annotations

import logging
from datetime import datetime, time
from typing import Callable, List, Optional

from PySide6.QtCore import QDate
from PySide6.QtWidgets import (
    QAbstractItemView,
    QComboBox,
    QDateEdit,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QPushButton,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from src.core.models import ActivityLogEntry
from src.theme.theme_manager import get_palette, get_shared_view_qss
from src.utils.activity_date_filters import (
    ACTIVITY_DATE_PRESETS,
    resolve_activity_date_range,
)

logger = logging.getLogger("tilevision.presentation.views.activity_log_view")

_PAGE_SIZE = 50

_TYPE_LABELS = {
    "search": "Search",
    "index": "Index",
    "auto_index": "Auto-index",
    "duplicate_scan": "Duplicates",
    "export": "Export",
    "license": "License",
    "theme": "Theme",
    "settings": "Settings",
    "maintenance": "Maintenance",
    "navigation": "Navigation",
}


class ActivityLogView(QWidget):
    """Stack page listing system activity with date filters and load-more."""

    def __init__(
        self,
        activity_provider: Optional[
            Callable[..., List[ActivityLogEntry]]
        ] = None,
        theme: str = "dark",
        parent: Optional[QWidget] = None,
    ) -> None:
        super().__init__(parent)
        self._theme = theme
        self._activity_provider = activity_provider
        self._offset = 0
        self._entries: List[ActivityLogEntry] = []
        self._filter_key = "all"
        self.setObjectName("ActivityLogView")
        self._setup_ui()
        self._apply_styles()

    def set_theme(self, theme: str) -> None:
        self._theme = theme
        self._apply_styles()

    def refresh(self) -> None:
        self._offset = 0
        self._entries = []
        self._load_more(reset=True)

    def _setup_ui(self) -> None:
        layout = QVBoxLayout(self)
        layout.setContentsMargins(28, 22, 28, 18)
        layout.setSpacing(14)

        title = QLabel("Activity Log")
        title.setObjectName("PageTitle")
        layout.addWidget(title)

        subtitle = QLabel(
            "A running record of indexing, searches, exports, duplicate scans, "
            "theme changes, settings, maintenance, and license events."
        )
        subtitle.setObjectName("PageSubtitle")
        subtitle.setWordWrap(True)
        layout.addWidget(subtitle)

        filter_row = QHBoxLayout()
        filter_row.setSpacing(10)
        filter_label = QLabel("Show")
        filter_label.setObjectName("FilterLabel")
        filter_row.addWidget(filter_label)

        self._filter_combo = QComboBox()
        self._filter_combo.setObjectName("ActivityFilterCombo")
        for preset in ACTIVITY_DATE_PRESETS:
            self._filter_combo.addItem(preset.label, preset.key)
        self._filter_combo.currentIndexChanged.connect(self._on_filter_changed)
        filter_row.addWidget(self._filter_combo)

        self._custom_from = QDateEdit()
        self._custom_from.setObjectName("ActivityDateEdit")
        self._custom_from.setCalendarPopup(True)
        self._custom_from.setDisplayFormat("MMM d, yyyy")
        self._custom_from.setDate(QDate.currentDate().addDays(-7))
        self._custom_from.dateChanged.connect(self._on_custom_date_changed)

        self._custom_to = QDateEdit()
        self._custom_to.setObjectName("ActivityDateEdit")
        self._custom_to.setCalendarPopup(True)
        self._custom_to.setDisplayFormat("MMM d, yyyy")
        self._custom_to.setDate(QDate.currentDate())
        self._custom_to.dateChanged.connect(self._on_custom_date_changed)

        self._custom_from_label = QLabel("From")
        self._custom_from_label.setObjectName("FilterLabel")
        self._custom_to_label = QLabel("To")
        self._custom_to_label.setObjectName("FilterLabel")

        filter_row.addWidget(self._custom_from_label)
        filter_row.addWidget(self._custom_from)
        filter_row.addWidget(self._custom_to_label)
        filter_row.addWidget(self._custom_to)
        filter_row.addStretch()
        layout.addLayout(filter_row)
        self._set_custom_range_visible(False)

        self._table = QTableWidget(0, 3)
        self._table.setObjectName("ActivityTable")
        self._table.setHorizontalHeaderLabels(["When", "Type", "What happened"])
        self._table.verticalHeader().setVisible(False)
        self._table.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        self._table.setSelectionMode(QAbstractItemView.SelectionMode.NoSelection)
        self._table.setAlternatingRowColors(True)
        self._table.setWordWrap(True)
        header = self._table.horizontalHeader()
        header.setSectionResizeMode(0, QHeaderView.ResizeMode.ResizeToContents)
        header.setSectionResizeMode(1, QHeaderView.ResizeMode.ResizeToContents)
        header.setSectionResizeMode(2, QHeaderView.ResizeMode.Stretch)
        layout.addWidget(self._table, stretch=1)

        self._empty = QLabel("No activity yet.")
        self._empty.setObjectName("EmptyListLabel")
        layout.addWidget(self._empty)

        actions = QHBoxLayout()
        self._load_more_btn = QPushButton("Load more")
        self._load_more_btn.setObjectName("SecondaryButton")
        self._load_more_btn.clicked.connect(lambda: self._load_more(reset=False))
        actions.addWidget(self._load_more_btn)

        refresh_btn = QPushButton("Refresh")
        refresh_btn.setObjectName("SecondaryButton")
        refresh_btn.clicked.connect(self.refresh)
        actions.addWidget(refresh_btn)
        actions.addStretch()
        layout.addLayout(actions)

    def _set_custom_range_visible(self, visible: bool) -> None:
        for w in (
            self._custom_from_label,
            self._custom_from,
            self._custom_to_label,
            self._custom_to,
        ):
            w.setVisible(visible)

    def _on_filter_changed(self, _index: int = 0) -> None:
        key = self._filter_combo.currentData()
        self._filter_key = str(key or "all")
        self._set_custom_range_visible(self._filter_key == "custom")
        self.refresh()

    def _on_custom_date_changed(self, *_args) -> None:
        if self._filter_key == "custom":
            self.refresh()

    def _current_bounds(self):
        custom_since = None
        custom_until = None
        if self._filter_key == "custom":
            from datetime import date as date_cls

            q_from = self._custom_from.date()
            q_to = self._custom_to.date()
            local_tz = datetime.now().astimezone().tzinfo
            custom_since = datetime.combine(
                date_cls(q_from.year(), q_from.month(), q_from.day()),
                time.min,
                tzinfo=local_tz,
            )
            custom_until = datetime.combine(
                date_cls(q_to.year(), q_to.month(), q_to.day()),
                time.min,
                tzinfo=local_tz,
            )
        return resolve_activity_date_range(
            self._filter_key,
            custom_since=custom_since,
            custom_until=custom_until,
        )

    def _load_more(self, *, reset: bool) -> None:
        if self._activity_provider is None:
            self._empty.setVisible(True)
            self._load_more_btn.setEnabled(False)
            return
        since, until = self._current_bounds()
        try:
            batch = self._activity_provider(
                _PAGE_SIZE, self._offset, since=since, until=until
            )
        except TypeError:
            # Older provider signature (limit, offset) only.
            try:
                batch = self._activity_provider(_PAGE_SIZE, self._offset)
            except Exception as exc:
                logger.error("Activity log provider failed: %s", exc)
                batch = []
        except Exception as exc:
            logger.error("Activity log provider failed: %s", exc)
            batch = []

        if reset:
            self._entries = list(batch)
        else:
            self._entries.extend(batch)

        self._offset = len(self._entries)
        self._populate_table()
        self._load_more_btn.setEnabled(len(batch) >= _PAGE_SIZE)

    def _populate_table(self) -> None:
        self._table.setRowCount(len(self._entries))
        for row, entry in enumerate(self._entries):
            when = (
                entry.created_at.strftime("%b %d, %Y  %I:%M %p")
                if entry.created_at
                else ""
            )
            type_label = _TYPE_LABELS.get(
                entry.activity_type, entry.activity_type.replace("_", " ").title()
            )
            self._table.setItem(row, 0, QTableWidgetItem(when))
            self._table.setItem(row, 1, QTableWidgetItem(type_label))
            msg_item = QTableWidgetItem(entry.message)
            msg_item.setToolTip(entry.message)
            self._table.setItem(row, 2, msg_item)

        empty = len(self._entries) == 0
        self._empty.setText(
            "No activity in this date range." if self._filter_key != "all" else "No activity yet."
        )
        self._empty.setVisible(empty)
        self._table.setVisible(not empty)

    def _apply_styles(self) -> None:
        p = get_palette(self._theme)
        self.setStyleSheet(
            get_shared_view_qss(self._theme)
            + f"""
            #ActivityLogView {{ background-color: {p['bg_app']}; }}
            #PageTitle {{
                font-size: 24px; font-weight: 750; color: {p['text_primary']};
            }}
            #PageSubtitle {{
                font-size: 14px; color: {p['text_secondary']}; padding-bottom: 4px;
            }}
            #FilterLabel {{
                color: {p['text_muted']}; font-size: 12px; font-weight: 600;
            }}
            #ActivityFilterCombo, #ActivityDateEdit {{
                background-color: {p['bg_input']};
                color: {p['text_secondary']};
                border: 1px solid {p['border_strong']};
                border-radius: 6px;
                padding: 4px 8px;
                min-width: 120px;
                font-size: 12px;
            }}
            #ActivityFilterCombo:hover, #ActivityDateEdit:hover {{
                border-color: {p['accent_hover']};
            }}
            #EmptyListLabel {{ color: {p['text_faint']}; font-size: 12px; }}
            #ActivityTable {{
                background-color: {p['bg_panel']};
                alternate-background-color: {p['bg_panel_alt']};
                border: 1px solid {p['border']};
                border-radius: 8px;
                gridline-color: {p['border']};
                color: {p['text_primary']};
            }}
            #ActivityTable QHeaderView::section {{
                background-color: {p['bg_panel_alt']};
                color: {p['text_secondary']};
                border: none;
                border-bottom: 1px solid {p['border']};
                padding: 6px 8px;
                font-weight: 600;
            }}
            """
        )
