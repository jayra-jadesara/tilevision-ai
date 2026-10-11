"""Tests for Activity Log page filters and type labels."""

from __future__ import annotations

import sys
from datetime import datetime
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from PySide6.QtWidgets import QApplication

from src.core.models import ActivityLogEntry
from src.presentation.views.activity_log_view import ActivityLogView, _TYPE_LABELS


@pytest.fixture()
def qapp():
    app = QApplication.instance() or QApplication([])
    yield app


def test_type_labels_cover_theme_and_settings():
    assert _TYPE_LABELS["theme"] == "Theme"
    assert _TYPE_LABELS["settings"] == "Settings"
    assert _TYPE_LABELS["maintenance"] == "Maintenance"


def test_filter_combo_has_date_presets(qapp):
    calls = []

    def provider(limit, offset, since=None, until=None):
        calls.append({"limit": limit, "offset": offset, "since": since, "until": until})
        return [
            ActivityLogEntry(
                activity_type="theme",
                message="Theme changed from dark to light",
                id=1,
                created_at=datetime(2026, 10, 11, 15, 0),
            ),
            ActivityLogEntry(
                activity_type="search",
                message="Searched with 'tile.png' — 6 result(s)",
                id=2,
                created_at=datetime(2026, 10, 11, 14, 0),
            ),
        ]

    view = ActivityLogView(activity_provider=provider, theme="light")
    view.refresh()
    assert view._filter_combo.count() >= 7
    assert view._table.rowCount() == 2
    assert view._table.item(0, 1).text() == "Theme"
    assert view._table.item(1, 1).text() == "Search"
    # Switching to Today should re-query with a since bound.
    today_idx = next(
        i
        for i in range(view._filter_combo.count())
        if view._filter_combo.itemData(i) == "today"
    )
    view._filter_combo.setCurrentIndex(today_idx)
    assert calls[-1]["since"] is not None
    view.close()
