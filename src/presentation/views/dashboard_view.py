"""
Dashboard View for TileVision AI.

The landing/overview page: catalog size, license/trial status, and quick
shortcuts into Index and Search.
"""

import logging
from typing import Callable, Optional

from PySide6.QtWidgets import QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton, QFrame

logger = logging.getLogger("tilevision.presentation.views.dashboard_view")


class _StatCard(QFrame):
    """A single stat tile (e.g. '1,204 Indexed Tiles')."""

    def __init__(self, value: str, label: str, parent=None) -> None:
        super().__init__(parent)
        self.setObjectName("StatCard")
        layout = QVBoxLayout(self)

        value_label = QLabel(value)
        value_label.setObjectName("StatValue")
        layout.addWidget(value_label)

        caption_label = QLabel(label)
        caption_label.setObjectName("StatCaption")
        layout.addWidget(caption_label)


class DashboardView(QWidget):
    """Landing page widget showing catalog overview and quick actions."""

    def __init__(
        self,
        catalog_count_provider: Optional[Callable[[], int]] = None,
        watched_folder_count_provider: Optional[Callable[[], int]] = None,
        license_details: Optional[dict] = None,
        on_go_to_index: Optional[Callable[[], None]] = None,
        on_go_to_search: Optional[Callable[[], None]] = None,
        parent: Optional[QWidget] = None,
    ) -> None:
        super().__init__(parent)
        self._catalog_count_provider = catalog_count_provider
        self._watched_folder_count_provider = watched_folder_count_provider
        self._license_details = license_details or {}
        self._on_go_to_index = on_go_to_index
        self._on_go_to_search = on_go_to_search
        self._setup_ui()
        self._apply_styles()

    def refresh(self) -> None:
        """Re-read catalog stats (call after indexing completes)."""
        count = self._catalog_count_provider() if self._catalog_count_provider else 0
        self._tiles_card_value.setText(f"{count:,}")

    def _setup_ui(self) -> None:
        layout = QVBoxLayout(self)
        layout.setContentsMargins(24, 20, 24, 20)
        layout.setSpacing(16)

        title = QLabel("Dashboard")
        title.setObjectName("PageTitle")
        layout.addWidget(title)

        subtitle_text = self._build_license_subtitle()
        subtitle = QLabel(subtitle_text)
        subtitle.setObjectName("PageSubtitle")
        layout.addWidget(subtitle)

        stats_row = QHBoxLayout()
        stats_row.setSpacing(12)

        tile_count = self._catalog_count_provider() if self._catalog_count_provider else 0
        tiles_card = _StatCard(f"{tile_count:,}", "Indexed Tiles")
        self._tiles_card_value = tiles_card.findChild(QLabel, "StatValue")
        stats_row.addWidget(tiles_card)

        watched_count = self._watched_folder_count_provider() if self._watched_folder_count_provider else 0
        stats_row.addWidget(_StatCard(str(watched_count), "Watched Folders"))

        stats_row.addStretch()
        layout.addLayout(stats_row)

        actions_row = QHBoxLayout()
        actions_row.setSpacing(12)

        index_button = QPushButton("📁  Go to Folder Indexing")
        index_button.setObjectName("ActionButton")
        if self._on_go_to_index:
            index_button.clicked.connect(self._on_go_to_index)
        actions_row.addWidget(index_button)

        search_button = QPushButton("🔍  Go to Visual Search")
        search_button.setObjectName("ActionButton")
        if self._on_go_to_search:
            search_button.clicked.connect(self._on_go_to_search)
        actions_row.addWidget(search_button)

        actions_row.addStretch()
        layout.addLayout(actions_row)

        layout.addStretch()

    def _build_license_subtitle(self) -> str:
        if self._license_details.get("is_trial"):
            days = self._license_details.get("days_remaining", 0)
            return f"Trial — {days} day(s) remaining"
        if self._license_details:
            return f"{self._license_details.get('license_type', 'Licensed')} — {self._license_details.get('customer_name', '')}"
        return "Unlicensed"

    def _apply_styles(self) -> None:
        self.setStyleSheet(
            """
            #PageTitle { font-size: 20px; font-weight: 700; color: #E8EAF6; }
            #PageSubtitle { font-size: 12px; color: #8A8FA3; }
            #StatCard {
                background-color: #232634; border: 1px solid #2E3243; border-radius: 10px;
                padding: 16px; min-width: 160px;
            }
            #StatValue { font-size: 28px; font-weight: 700; color: #E8EAF6; }
            #StatCaption { font-size: 11px; color: #8A8FA3; }
            #ActionButton {
                background-color: #2D3250; border: 1px solid #3D4166; border-radius: 8px;
                padding: 14px 18px; color: #E8EAF6; font-size: 13px; font-weight: 600;
            }
            #ActionButton:hover { background-color: #3D4166; }
            """
        )
