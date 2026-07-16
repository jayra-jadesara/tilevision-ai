"""
Dashboard View for TileVision AI (Task A: Professional Dashboard).

The landing/overview page. Shows:
  - Stat cards: Total Images, Indexed Folders, Database Size, FAISS Index
    Size, Last Search, License Status, Trial Remaining.
  - Recent Activity feed (indexing/search/duplicate-scan events).
  - Recent Searches list (click to re-run — mirrors the Search page's
    history panel).
  - Quick Actions: Search, Index Folder, Duplicate Detection, Settings.

All data is supplied via provider callables (consistent with the existing
pattern from the original Dashboard) rather than passing repository
objects directly into the View, keeping this layer decoupled from data
access per the app's Clean Architecture / MVVM structure.
"""

import logging
from pathlib import Path
from typing import Callable, List, Optional

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QWidget,
    QVBoxLayout,
    QHBoxLayout,
    QGridLayout,
    QLabel,
    QPushButton,
    QFrame,
    QScrollArea,
)

from src.theme.theme_manager import get_palette, get_shared_view_qss

logger = logging.getLogger("tilevision.presentation.views.dashboard_view")


class _StatCard(QFrame):
    """A single stat tile (e.g. '1,204 Total Images')."""

    def __init__(self, value: str, label: str, icon: str = "", parent=None) -> None:
        super().__init__(parent)
        self.setObjectName("StatCard")
        layout = QVBoxLayout(self)
        layout.setSpacing(2)

        if icon:
            icon_label = QLabel(icon)
            icon_label.setObjectName("StatIcon")
            layout.addWidget(icon_label)

        value_label = QLabel(value)
        value_label.setObjectName("StatValue")
        value_label.setWordWrap(True)
        layout.addWidget(value_label)

        caption_label = QLabel(label)
        caption_label.setObjectName("StatCaption")
        layout.addWidget(caption_label)

        self.value_label = value_label


def _format_bytes(num_bytes: int) -> str:
    """Human-readable file size, e.g. '4.2 MB'."""
    size = float(num_bytes)
    for unit in ("B", "KB", "MB", "GB"):
        if size < 1024 or unit == "GB":
            return f"{size:.1f} {unit}" if unit != "B" else f"{int(size)} {unit}"
        size /= 1024
    return f"{size:.1f} GB"


class DashboardView(QWidget):
    """Landing page widget showing catalog overview, activity, and quick actions."""

    def __init__(
        self,
        catalog_count_provider: Optional[Callable[[], int]] = None,
        watched_folder_count_provider: Optional[Callable[[], int]] = None,
        indexed_folder_count_provider: Optional[Callable[[], int]] = None,
        database_size_provider: Optional[Callable[[], int]] = None,
        faiss_size_provider: Optional[Callable[[], int]] = None,
        last_search_provider: Optional[Callable[[], object]] = None,
        recent_activity_provider: Optional[Callable[[], List[object]]] = None,
        recent_searches_provider: Optional[Callable[[], List[object]]] = None,
        license_details: Optional[dict] = None,
        on_go_to_index: Optional[Callable[[], None]] = None,
        on_go_to_search: Optional[Callable[[], None]] = None,
        on_go_to_duplicates: Optional[Callable[[], None]] = None,
        on_go_to_settings: Optional[Callable[[], None]] = None,
        on_repeat_search: Optional[Callable[[str], None]] = None,
        theme: str = "dark",
        parent: Optional[QWidget] = None,
    ) -> None:
        super().__init__(parent)
        self._theme = theme
        self._catalog_count_provider = catalog_count_provider
        self._watched_folder_count_provider = watched_folder_count_provider
        self._indexed_folder_count_provider = indexed_folder_count_provider
        self._database_size_provider = database_size_provider
        self._faiss_size_provider = faiss_size_provider
        self._last_search_provider = last_search_provider
        self._recent_activity_provider = recent_activity_provider
        self._recent_searches_provider = recent_searches_provider
        self._license_details = license_details or {}
        self._on_go_to_index = on_go_to_index
        self._on_go_to_search = on_go_to_search
        self._on_go_to_duplicates = on_go_to_duplicates
        self._on_go_to_settings = on_go_to_settings
        self._on_repeat_search = on_repeat_search
        self._setup_ui()
        self._apply_styles()
        self.refresh()

    def refresh(self) -> None:
        """Re-read all stats and lists (call after indexing/search/duplicate scans)."""
        self._tiles_card.value_label.setText(self._safe_call(self._catalog_count_provider, "0", fmt="{:,}"))
        self._folders_card.value_label.setText(self._safe_call(self._indexed_folder_count_provider, "0", fmt="{:,}"))
        self._db_size_card.value_label.setText(self._format_size(self._database_size_provider))
        self._faiss_size_card.value_label.setText(self._format_size(self._faiss_size_provider))
        self._last_search_card.value_label.setText(self._format_last_search())
        self._populate_activity_list()
        self._populate_search_list()

    @staticmethod
    def _safe_call(provider, default, fmt: Optional[str] = None) -> str:
        if provider is None:
            return default
        try:
            value = provider()
            return fmt.format(value) if fmt else str(value)
        except Exception as e:
            logger.error(f"Dashboard provider call failed: {e}")
            return default

    def _format_size(self, provider) -> str:
        if provider is None:
            return "—"
        try:
            return _format_bytes(provider())
        except Exception as e:
            logger.error(f"Dashboard size provider failed: {e}")
            return "—"

    def _format_last_search(self) -> str:
        if self._last_search_provider is None:
            return "Never"
        try:
            entry = self._last_search_provider()
        except Exception as e:
            logger.error(f"Dashboard last-search provider failed: {e}")
            return "—"
        if entry is None:
            return "Never"
        name = Path(entry.query_image_path).name
        return name

    def _setup_ui(self) -> None:
        outer_layout = QVBoxLayout(self)
        outer_layout.setContentsMargins(0, 0, 0, 0)

        scroll = QScrollArea()
        scroll.setObjectName("DashboardScroll")
        scroll.viewport().setObjectName("DashboardViewport")
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.Shape.NoFrame)
        outer_layout.addWidget(scroll)

        content = QWidget()
        content.setObjectName("DashboardContent")
        scroll.setWidget(content)
        layout = QVBoxLayout(content)
        layout.setContentsMargins(24, 20, 24, 20)
        layout.setSpacing(16)

        title = QLabel("Dashboard")
        title.setObjectName("PageTitle")
        layout.addWidget(title)

        subtitle = QLabel(self._build_license_subtitle())
        subtitle.setObjectName("PageSubtitle")
        layout.addWidget(subtitle)

        layout.addWidget(self._build_stats_grid())
        layout.addWidget(self._build_quick_actions())
        layout.addWidget(self._build_activity_and_history_row(), stretch=1)

    def _build_stats_grid(self) -> QWidget:
        grid_widget = QWidget()
        grid = QGridLayout(grid_widget)
        grid.setSpacing(12)

        tile_count = self._catalog_count_provider() if self._catalog_count_provider else 0
        self._tiles_card = _StatCard(f"{tile_count:,}", "Total Images")
        grid.addWidget(self._tiles_card, 0, 0)

        folder_count = self._indexed_folder_count_provider() if self._indexed_folder_count_provider else 0
        self._folders_card = _StatCard(f"{folder_count:,}", "Indexed Folders")
        grid.addWidget(self._folders_card, 0, 1)

        self._db_size_card = _StatCard(self._format_size(self._database_size_provider), "Database Size")
        grid.addWidget(self._db_size_card, 0, 2)

        self._faiss_size_card = _StatCard(self._format_size(self._faiss_size_provider), "FAISS Index Size")
        grid.addWidget(self._faiss_size_card, 0, 3)

        self._last_search_card = _StatCard(self._format_last_search(), "Last Search")
        grid.addWidget(self._last_search_card, 1, 0)

        license_text, _license_icon = self._build_license_card_text()
        self._license_card = _StatCard(license_text, "License Status")
        grid.addWidget(self._license_card, 1, 1)

        trial_text = self._build_trial_remaining_text()
        self._trial_card = _StatCard(trial_text, "Trial Remaining")
        grid.addWidget(self._trial_card, 1, 2)

        watched_count = self._watched_folder_count_provider() if self._watched_folder_count_provider else 0
        self._watched_card = _StatCard(f"{watched_count:,}", "Watched Folders")
        grid.addWidget(self._watched_card, 1, 3)

        for col in range(4):
            grid.setColumnStretch(col, 1)

        return grid_widget

    def _build_license_card_text(self) -> tuple:
        if self._license_details.get("is_trial"):
            return "Trial", ""
        if self._license_details:
            return self._license_details.get("license_type", "Licensed"), ""
        return "Unlicensed", ""

    def _build_trial_remaining_text(self) -> str:
        if self._license_details.get("is_trial"):
            return f"{self._license_details.get('days_remaining', 0)} days"
        return "N/A"

    def _build_quick_actions(self) -> QWidget:
        container = QWidget()
        layout = QVBoxLayout(container)
        layout.setContentsMargins(0, 4, 0, 0)
        layout.setSpacing(8)

        section_label = QLabel("Quick Actions")
        section_label.setObjectName("SectionLabel")
        layout.addWidget(section_label)

        actions_row = QHBoxLayout()
        actions_row.setSpacing(12)

        for _icon, label_text, callback in [
            ("Search", "Search", self._on_go_to_search),
            ("Index", "Index Folder", self._on_go_to_index),
            ("Duplicates", "Duplicate Detection", self._on_go_to_duplicates),
            ("Settings", "Settings", self._on_go_to_settings),
        ]:
            button = QPushButton(label_text)
            button.setObjectName("ActionButton")
            if callback:
                button.clicked.connect(callback)
            else:
                button.setEnabled(False)
            actions_row.addWidget(button)

        actions_row.addStretch()
        layout.addLayout(actions_row)
        return container

    def _build_activity_and_history_row(self) -> QWidget:
        container = QWidget()
        layout = QHBoxLayout(container)
        layout.setContentsMargins(0, 8, 0, 0)
        layout.setSpacing(16)

        layout.addWidget(self._build_activity_panel(), stretch=1)
        layout.addWidget(self._build_search_history_panel(), stretch=1)
        return container

    def _build_activity_panel(self) -> QWidget:
        panel = QFrame()
        panel.setObjectName("ListPanel")
        layout = QVBoxLayout(panel)

        label = QLabel("Recent Activity")
        label.setObjectName("SectionLabel")
        layout.addWidget(label)

        self._activity_list_layout = QVBoxLayout()
        self._activity_list_layout.setSpacing(4)
        layout.addLayout(self._activity_list_layout)
        layout.addStretch()

        self._populate_activity_list()
        return panel

    def _populate_activity_list(self) -> None:
        while self._activity_list_layout.count():
            item = self._activity_list_layout.takeAt(0)
            if item.widget():
                item.widget().deleteLater()

        entries = []
        if self._recent_activity_provider is not None:
            try:
                entries = self._recent_activity_provider()
            except Exception as e:
                logger.error(f"Dashboard activity provider failed: {e}")

        if not entries:
            empty = QLabel("No activity yet.")
            empty.setObjectName("EmptyListLabel")
            self._activity_list_layout.addWidget(empty)
            return

        for entry in entries:
            when = entry.created_at.strftime("%b %d, %I:%M %p") if entry.created_at else ""
            p = get_palette(self._theme)
            row = QLabel(
                f"• {entry.message} "
                f"<span style='color:{p['text_muted']};'>({when})</span>"
            )
            row.setObjectName("ActivityRow")
            row.setWordWrap(True)
            self._activity_list_layout.addWidget(row)

    def _build_search_history_panel(self) -> QWidget:
        panel = QFrame()
        panel.setObjectName("ListPanel")
        layout = QVBoxLayout(panel)

        label = QLabel("Recent Searches")
        label.setObjectName("SectionLabel")
        layout.addWidget(label)

        self._search_list_layout = QVBoxLayout()
        self._search_list_layout.setSpacing(4)
        layout.addLayout(self._search_list_layout)
        layout.addStretch()

        self._populate_search_list()
        return panel

    def _populate_search_list(self) -> None:
        while self._search_list_layout.count():
            item = self._search_list_layout.takeAt(0)
            if item.widget():
                item.widget().deleteLater()

        entries = []
        if self._recent_searches_provider is not None:
            try:
                entries = self._recent_searches_provider()
            except Exception as e:
                logger.error(f"Dashboard search history provider failed: {e}")

        if not entries:
            empty = QLabel("No searches yet.")
            empty.setObjectName("EmptyListLabel")
            self._search_list_layout.addWidget(empty)
            return

        for entry in entries:
            name = Path(entry.query_image_path).name
            row_button = QPushButton(f"{name} — {entry.result_count} result(s)")
            row_button.setObjectName("SearchHistoryRow")
            if self._on_repeat_search:
                row_button.clicked.connect(
                    lambda checked=False, p=entry.query_image_path: self._on_repeat_search(p)
                )
            self._search_list_layout.addWidget(row_button)

    def _build_license_subtitle(self) -> str:
        if self._license_details.get("is_trial"):
            days = self._license_details.get("days_remaining", 0)
            return f"Trial — {days} day(s) remaining"
        if self._license_details:
            return f"{self._license_details.get('license_type', 'Licensed')} — {self._license_details.get('customer_name', '')}"
        return "Unlicensed"

    def set_theme(self, theme: str) -> None:
        """Re-skin this view for a newly-selected theme (called by MainWindow)."""
        self._theme = theme
        self._apply_styles()
        self.refresh()

    def _apply_styles(self) -> None:
        p = get_palette(self._theme)
        self.setStyleSheet(
            get_shared_view_qss(self._theme)
            + f"""
            QWidget#DashboardContent {{
                background-color: {p['bg_app']};
            }}
            QScrollArea#DashboardScroll {{
                background-color: {p['bg_app']};
                border: none;
            }}
            QWidget#DashboardViewport {{
                background-color: {p['bg_app']};
            }}
            #StatCard {{
                background-color:{p['bg_panel']};
                border:1px solid {p['border']};
                border-radius:10px;
                padding:14px;
                min-width:140px;
                min-height:80px;
            }}
            #StatCard:hover {{
                border:1px solid {p['accent']};
            }}
            #StatIcon {{ font-size: 16px; }}
            #StatValue {{ font-size: 20px; font-weight: 700; color: {p['text_primary']}; }}
            #StatCaption {{ font-size: 11px; color: {p['text_muted']}; }}
            #ListPanel {{
                background-color: {p['bg_panel']};
                border:1px solid {p['border']};
                border-radius:10px;
                padding:14px;
            }}
            #ActivityRow {{ color: {p['text_secondary']}; font-size: 12px; padding: 3px 0; }}
            #EmptyListLabel {{ color: {p['text_faint']}; font-size: 12px; font-style: italic; }}
            #SearchHistoryRow {{
                text-align:left;
                background-color:{p['bg_panel_alt']};
                border:1px solid {p['border']};
                border-radius:6px;
                padding:8px 12px;
                color:{p['text_secondary']};
            }}
            #SearchHistoryRow:hover {{
                background-color:{p['button_hover']};
            }}
            #SearchHistoryRow:hover {{ background-color: {p['button_hover']}; }}
            QLabel {{
                background: transparent;
            }}
            """
        )
