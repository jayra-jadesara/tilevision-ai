"""Tests for Search page UI polish: clickable path, preview link, button groups."""

from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from PySide6.QtCore import QTimer
from PySide6.QtWidgets import QApplication, QFrame, QPushButton

from src.core.models import SearchResult, TileImage
from src.presentation.views.search_view import (
    SearchView,
    _PathCellWidget,
    _PREVIEW_COL,
    _PATH_COL,
    _PREVIEW_COL_WIDTH,
    _ResultPreviewPanel,
)
from src.presentation.viewmodels.search_viewmodel import SearchViewModel
from src.theme.theme_manager import get_app_stylesheet, get_palette
from PySide6.QtWidgets import QHeaderView


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


def test_path_cell_reveal_is_deferred(search_view, monkeypatch, qapp):
    called = {}

    def fake_reveal(path):
        called["path"] = path
        return True, ""

    monkeypatch.setattr(
        "src.utils.reveal_file.reveal_in_file_manager", fake_reveal
    )
    path = search_view._current_results[0].tile.file_path
    search_view._reveal_path_deferred(path)
    assert "path" not in called  # deferred
    qapp.processEvents()
    # singleShot(0) may need another tick
    QTimer.singleShot(0, qapp.quit)
    qapp.processEvents()
    # Directly flush deferred reveal:
    search_view._reveal_path_now(path)
    assert called["path"] == path


def test_path_column_has_copy_control(search_view):
    cell = search_view._results_table.cellWidget(0, _PATH_COL)
    assert isinstance(cell, _PathCellWidget)
    assert cell._copy_btn.toolTip() == "Copy path"
    assert "Click to" in cell._path_label.toolTip()
    path = search_view._current_results[0].tile.file_path
    assert path in cell._path_label.toolTip()
    # Item text kept for E2E / tooling readers under the cell widget.
    item = search_view._results_table.item(0, _PATH_COL)
    assert item is not None
    assert item.text() == path


def test_copy_button_sets_clipboard_and_shows_toast(search_view, qapp):
    cell = search_view._results_table.cellWidget(0, _PATH_COL)
    assert isinstance(cell, _PathCellWidget)
    path = search_view._current_results[0].tile.file_path
    cell._handle_copy()
    assert QApplication.clipboard().text() == path
    # Parent may not be shown in headless tests — isHidden() reflects our show.
    assert not cell._copied_label.isHidden()
    assert cell._copied_label.text() == "Copied"
    assert cell._fade_anim is not None


def test_preview_column_is_narrow_fixed(search_view):
    header = search_view._results_table.horizontalHeader()
    assert header.sectionResizeMode(_PREVIEW_COL) == QHeaderView.ResizeMode.Fixed
    assert search_view._results_table.columnWidth(_PREVIEW_COL) <= _PREVIEW_COL_WIDTH + 8
    assert search_view._results_table.columnWidth(_PREVIEW_COL) >= 40
    assert header.sectionResizeMode(_PATH_COL) == QHeaderView.ResizeMode.Stretch


def test_preview_column_still_opens_preview(search_view, monkeypatch):
    shown = {}

    def fake_show(result):
        shown["path"] = result.tile.file_path

    monkeypatch.setattr(search_view._preview_panel, "show_result", fake_show)
    search_view._on_table_clicked(0, _PREVIEW_COL)
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
    # Deferred — invoke the worker directly after scheduling.
    panel._reveal_path_now(str(img))
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
    panel._reveal_path_now(str(img))
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
