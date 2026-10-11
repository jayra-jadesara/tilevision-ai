"""
Activity Log page — full feed from IActivityLogRepository.
"""

from __future__ import annotations

import logging
from typing import Callable, List, Optional

from PySide6.QtWidgets import (
    QAbstractItemView,
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

logger = logging.getLogger("tilevision.presentation.views.activity_log_view")

_PAGE_SIZE = 50

_TYPE_LABELS = {
    "search": "Search",
    "index": "Index",
    "auto_index": "Auto-index",
    "duplicate_scan": "Duplicates",
    "export": "Export",
    "license": "License",
}


class ActivityLogView(QWidget):
    """Stack page listing system activity with load-more pagination."""

    def __init__(
        self,
        activity_provider: Optional[
            Callable[[int, int], List[ActivityLogEntry]]
        ] = None,
        theme: str = "dark",
        parent: Optional[QWidget] = None,
    ) -> None:
        super().__init__(parent)
        self._theme = theme
        self._activity_provider = activity_provider
        self._offset = 0
        self._entries: List[ActivityLogEntry] = []
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
            "and license events."
        )
        subtitle.setObjectName("PageSubtitle")
        subtitle.setWordWrap(True)
        layout.addWidget(subtitle)

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

    def _load_more(self, *, reset: bool) -> None:
        if self._activity_provider is None:
            self._empty.setVisible(True)
            self._load_more_btn.setEnabled(False)
            return
        try:
            batch = self._activity_provider(_PAGE_SIZE, self._offset)
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
