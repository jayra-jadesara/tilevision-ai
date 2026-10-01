"""Left sidebar navigation for the vendor admin tool."""

from __future__ import annotations

from PySide6.QtCore import Qt
from PySide6.QtWidgets import QPushButton

PAGE_OVERVIEW = 0
PAGE_LICENSES = 1
PAGE_PRICING = 2
PAGE_SIGNING_KEY = 3


class AdminNavButton(QPushButton):
    """Checkable sidebar navigation button."""

    def __init__(self, label: str, page_index: int, parent=None) -> None:
        super().__init__(label, parent)
        self.page_index = page_index
        self.setCheckable(True)
        self.setObjectName("AdminNavButton")
        self.setCursor(Qt.CursorShape.PointingHandCursor)
        self.setMinimumHeight(42)
