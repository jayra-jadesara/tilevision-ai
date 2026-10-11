"""
Image History page — full search history + per-folder indexed image counts.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Callable, List, Optional

from PySide6.QtCore import Qt
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

from src.core.models import IndexedFolderState, SearchHistoryEntry
from src.theme.theme_manager import get_palette, get_shared_view_qss

logger = logging.getLogger("tilevision.presentation.views.image_history_view")

_PAGE_SIZE = 40


class ImageHistoryView(QWidget):
    """Stack page listing search history and per-folder image counts."""

    def __init__(
        self,
        search_history_provider: Optional[
            Callable[[int, int], List[SearchHistoryEntry]]
        ] = None,
        folder_counts_provider: Optional[
            Callable[[], List[IndexedFolderState]]
        ] = None,
        on_repeat_search: Optional[Callable[[str], None]] = None,
        theme: str = "dark",
        parent: Optional[QWidget] = None,
    ) -> None:
        super().__init__(parent)
        self._theme = theme
        self._search_history_provider = search_history_provider
        self._folder_counts_provider = folder_counts_provider
        self._on_repeat_search = on_repeat_search
        self._history_offset = 0
        self._history_entries: List[SearchHistoryEntry] = []
        self.setObjectName("ImageHistoryView")
        self._setup_ui()
        self._apply_styles()

    def set_theme(self, theme: str) -> None:
        self._theme = theme
        self._apply_styles()

    def refresh(self) -> None:
        """Reload history (from the top) and folder counts."""
        self._history_offset = 0
        self._history_entries = []
        self._load_more_history(reset=True)
        self._populate_folders()

    def _setup_ui(self) -> None:
        layout = QVBoxLayout(self)
        layout.setContentsMargins(28, 22, 28, 18)
        layout.setSpacing(14)

        title = QLabel("Image History")
        title.setObjectName("PageTitle")
        layout.addWidget(title)

        subtitle = QLabel(
            "Past searches and how many images are indexed in each folder. "
            "Click a search to run it again."
        )
        subtitle.setObjectName("PageSubtitle")
        subtitle.setWordWrap(True)
        layout.addWidget(subtitle)

        hist_label = QLabel("Search history")
        hist_label.setObjectName("SectionLabel")
        layout.addWidget(hist_label)

        self._history_table = QTableWidget(0, 3)
        self._history_table.setObjectName("HistoryTable")
        self._history_table.setHorizontalHeaderLabels(
            ["Query image", "Results", "When"]
        )
        self._history_table.verticalHeader().setVisible(False)
        self._history_table.setSelectionBehavior(
            QAbstractItemView.SelectionBehavior.SelectRows
        )
        self._history_table.setSelectionMode(
            QAbstractItemView.SelectionMode.SingleSelection
        )
        self._history_table.setEditTriggers(
            QAbstractItemView.EditTrigger.NoEditTriggers
        )
        self._history_table.setAlternatingRowColors(True)
        header = self._history_table.horizontalHeader()
        header.setSectionResizeMode(0, QHeaderView.ResizeMode.Stretch)
        header.setSectionResizeMode(1, QHeaderView.ResizeMode.ResizeToContents)
        header.setSectionResizeMode(2, QHeaderView.ResizeMode.ResizeToContents)
        self._history_table.cellDoubleClicked.connect(self._on_history_activated)
        layout.addWidget(self._history_table, stretch=3)

        hist_actions = QHBoxLayout()
        self._load_more_btn = QPushButton("Load more")
        self._load_more_btn.setObjectName("SecondaryButton")
        self._load_more_btn.clicked.connect(lambda: self._load_more_history(reset=False))
        hist_actions.addWidget(self._load_more_btn)

        self._rerun_btn = QPushButton("Re-run selected")
        self._rerun_btn.setObjectName("PrimaryButton")
        self._rerun_btn.clicked.connect(self._on_rerun_clicked)
        hist_actions.addWidget(self._rerun_btn)
        hist_actions.addStretch()
        layout.addLayout(hist_actions)

        self._history_empty = QLabel("No searches yet.")
        self._history_empty.setObjectName("EmptyListLabel")
        layout.addWidget(self._history_empty)

        folders_label = QLabel("Images per indexed folder")
        folders_label.setObjectName("SectionLabel")
        layout.addWidget(folders_label)

        self._folders_table = QTableWidget(0, 3)
        self._folders_table.setObjectName("FoldersTable")
        self._folders_table.setHorizontalHeaderLabels(
            ["Folder", "Indexed images", "Last indexed"]
        )
        self._folders_table.verticalHeader().setVisible(False)
        self._folders_table.setEditTriggers(
            QAbstractItemView.EditTrigger.NoEditTriggers
        )
        self._folders_table.setSelectionMode(
            QAbstractItemView.SelectionMode.NoSelection
        )
        self._folders_table.setAlternatingRowColors(True)
        fheader = self._folders_table.horizontalHeader()
        fheader.setSectionResizeMode(0, QHeaderView.ResizeMode.Stretch)
        fheader.setSectionResizeMode(1, QHeaderView.ResizeMode.ResizeToContents)
        fheader.setSectionResizeMode(2, QHeaderView.ResizeMode.ResizeToContents)
        layout.addWidget(self._folders_table, stretch=2)

        self._folders_empty = QLabel("No folders indexed yet.")
        self._folders_empty.setObjectName("EmptyListLabel")
        layout.addWidget(self._folders_empty)

    def _load_more_history(self, *, reset: bool) -> None:
        if self._search_history_provider is None:
            self._history_empty.setVisible(True)
            self._load_more_btn.setEnabled(False)
            return
        try:
            batch = self._search_history_provider(_PAGE_SIZE, self._history_offset)
        except Exception as exc:
            logger.error("Image history provider failed: %s", exc)
            batch = []

        if reset:
            self._history_entries = list(batch)
        else:
            self._history_entries.extend(batch)

        self._history_offset = len(self._history_entries)
        self._populate_history_table()
        self._load_more_btn.setEnabled(len(batch) >= _PAGE_SIZE)

    def _populate_history_table(self) -> None:
        self._history_table.setRowCount(len(self._history_entries))
        for row, entry in enumerate(self._history_entries):
            name = Path(entry.query_image_path).name
            name_item = QTableWidgetItem(name)
            name_item.setData(Qt.ItemDataRole.UserRole, entry.query_image_path)
            name_item.setToolTip(entry.query_image_path)
            self._history_table.setItem(row, 0, name_item)
            self._history_table.setItem(
                row, 1, QTableWidgetItem(str(entry.result_count))
            )
            when = (
                entry.searched_at.strftime("%b %d, %Y  %I:%M %p")
                if entry.searched_at
                else ""
            )
            self._history_table.setItem(row, 2, QTableWidgetItem(when))

        empty = len(self._history_entries) == 0
        self._history_empty.setVisible(empty)
        self._history_table.setVisible(not empty)

    def _populate_folders(self) -> None:
        folders: List[IndexedFolderState] = []
        if self._folder_counts_provider is not None:
            try:
                folders = list(self._folder_counts_provider())
            except Exception as exc:
                logger.error("Folder counts provider failed: %s", exc)

        self._folders_table.setRowCount(len(folders))
        for row, folder in enumerate(folders):
            path_item = QTableWidgetItem(folder.folder_path)
            path_item.setToolTip(folder.folder_path)
            self._folders_table.setItem(row, 0, path_item)
            self._folders_table.setItem(
                row, 1, QTableWidgetItem(f"{folder.indexed_image_count:,}")
            )
            when = (
                folder.last_indexed_at.strftime("%b %d, %Y  %I:%M %p")
                if folder.last_indexed_at
                else ""
            )
            self._folders_table.setItem(row, 2, QTableWidgetItem(when))

        empty = len(folders) == 0
        self._folders_empty.setVisible(empty)
        self._folders_table.setVisible(not empty)

    def _selected_history_path(self) -> Optional[str]:
        row = self._history_table.currentRow()
        if row < 0:
            return None
        item = self._history_table.item(row, 0)
        if item is None:
            return None
        path = item.data(Qt.ItemDataRole.UserRole)
        return str(path) if path else None

    def _on_history_activated(self, row: int, _column: int) -> None:
        item = self._history_table.item(row, 0)
        if item is None or self._on_repeat_search is None:
            return
        path = item.data(Qt.ItemDataRole.UserRole)
        if path:
            self._on_repeat_search(str(path))

    def _on_rerun_clicked(self) -> None:
        path = self._selected_history_path()
        if path and self._on_repeat_search is not None:
            self._on_repeat_search(path)

    def _apply_styles(self) -> None:
        p = get_palette(self._theme)
        self.setStyleSheet(
            get_shared_view_qss(self._theme)
            + f"""
            #ImageHistoryView {{ background-color: {p['bg_app']}; }}
            #PageTitle {{
                font-size: 24px; font-weight: 750; color: {p['text_primary']};
            }}
            #PageSubtitle {{
                font-size: 14px; color: {p['text_secondary']}; padding-bottom: 4px;
            }}
            #SectionLabel {{
                font-size: 13px; font-weight: 700; color: {p['text_secondary']};
                padding-top: 6px;
            }}
            #EmptyListLabel {{ color: {p['text_faint']}; font-size: 12px; }}
            #HistoryTable, #FoldersTable {{
                background-color: {p['bg_panel']};
                alternate-background-color: {p['bg_panel_alt']};
                border: 1px solid {p['border']};
                border-radius: 8px;
                gridline-color: {p['border']};
                color: {p['text_primary']};
            }}
            #HistoryTable QHeaderView::section, #FoldersTable QHeaderView::section {{
                background-color: {p['bg_panel_alt']};
                color: {p['text_secondary']};
                border: none;
                border-bottom: 1px solid {p['border']};
                padding: 6px 8px;
                font-weight: 600;
            }}
            """
        )
