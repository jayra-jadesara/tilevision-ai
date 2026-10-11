"""Tests for Search page UI polish: clickable path, preview link, button groups."""

from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from PySide6.QtWidgets import QApplication, QFrame, QPushButton

from src.core.models import SearchResult, TileImage
from src.presentation.views.search_view import SearchView, _ResultPreviewPanel
from src.presentation.viewmodels.search_viewmodel import SearchViewModel
from src.theme.theme_manager import get_app_stylesheet, get_palette


@pytest.fixture()
def qapp():
    app = QApplication.instance() or QApplication([])
    yield app


def _tile(path: str, *, score: float = 90.0) -> SearchResult:
    tile = TileImage(
        file_path=path,
        file_name=Path(path).name,
        file_size=10,
        dimensions="10x10",
        product_code="PC-1",
        brand="Brand",
        category="Cat",
        is_indexed=True,
    )
    return SearchResult(tile=tile, similarity_score=score, thumbnail_path=path)


@pytest.fixture()
def search_view(qapp, tmp_path):
    use_case = MagicMock()
    use_case.execute.return_value = []
    vm = SearchViewModel(use_case=use_case, default_top_k=5)
    view = SearchView(vm, theme="dark")
    img = tmp_path / "tile.jpg"
    img.write_bytes(b"x")
    view._current_results = [_tile(str(img), score=95.0), _tile(str(tmp_path / "b.jpg"), score=80.0)]
    view._populate_table(view._current_results)
    yield view
    view.close()
    view.deleteLater()


def test_path_column_click_calls_reveal(search_view, monkeypatch):
    called = {}

    def fake_reveal(path):
        called["path"] = path
        return True, ""

    monkeypatch.setattr(
        "src.utils.reveal_file.reveal_in_file_manager", fake_reveal
    )
    search_view._on_table_clicked(0, 6)
    assert called["path"] == search_view._current_results[0].tile.file_path


def test_path_column_has_link_affordance(search_view):
    path_item = search_view._results_table.item(0, 6)
    assert path_item is not None
    assert path_item.font().underline()
    tip = path_item.toolTip()
    assert "Click to" in tip
    assert search_view._current_results[0].tile.file_path in tip


def test_preview_column_still_opens_preview(search_view, monkeypatch):
    shown = {}

    def fake_show(result):
        shown["path"] = result.tile.file_path

    monkeypatch.setattr(search_view._preview_panel, "show_result", fake_show)
    search_view._on_table_clicked(0, 5)
    assert shown["path"] == search_view._current_results[0].tile.file_path


def test_preview_panel_path_link_reveals(qapp, tmp_path, monkeypatch):
    img = tmp_path / "preview.jpg"
    img.write_bytes(b"x")
    panel = _ResultPreviewPanel(theme="light")
    panel.show_result(_tile(str(img)))
    html = panel._details_label.text()
    assert "href='reveal'" in html
    assert str(img) in html

    called = {}

    def fake_reveal(path):
        called["path"] = path
        return True, ""

    monkeypatch.setattr(
        "src.utils.reveal_file.reveal_in_file_manager", fake_reveal
    )
    panel._on_path_link_activated("reveal")
    assert called["path"] == str(img)
    panel.close()


def test_preview_panel_reveal_failure_shows_inline_status(qapp, tmp_path, monkeypatch):
    img = tmp_path / "gone.jpg"
    img.write_bytes(b"x")
    panel = _ResultPreviewPanel(theme="dark")
    panel.show_result(_tile(str(img)))

    monkeypatch.setattr(
        "src.utils.reveal_file.reveal_in_file_manager",
        lambda path: (False, "This file no longer exists:\n" + path),
    )
    panel._on_path_link_activated("reveal")
    assert panel._status_label.isVisible()
    assert "no longer exists" in panel._status_label.text()
    panel.close()


def test_action_buttons_are_grouped_with_separators(search_view):
    # Collect widgets in the query panel's button column via object names.
    separators = search_view.findChildren(QFrame, "SearchButtonSeparator")
    assert len(separators) == 3  # export|crop, crop|clear, clear|history

    clear = search_view._clear_button
    assert clear.objectName() == "ClearActionButton"
    assert search_view._export_button.objectName() == "SecondaryButton"
    assert search_view._crop_button.objectName() == "SecondaryButton"
    assert search_view._history_button.objectName() == "SecondaryButton"


def test_hover_qss_sets_text_color_in_both_themes():
    for theme in ("light", "dark"):
        qss = get_app_stylesheet(theme)
        assert "QTableWidget::item:hover" in qss
        # Hover block must include an explicit color rule.
        hover_idx = qss.index("QTableWidget::item:hover")
        chunk = qss[hover_idx : hover_idx + 200]
        assert "color:" in chunk
        p = get_palette(theme)
        assert p["hover_bg"]
        assert p["text_primary"]
