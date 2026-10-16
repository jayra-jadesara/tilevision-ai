"""Tests for Search page UI polish: actions column, path text, button groups."""

from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import MagicMock

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from PySide6.QtWidgets import QApplication, QFrame, QHeaderView, QWidget

from src.core.models import SearchResult, TileImage
from src.presentation.views.search_view import (
    SearchView,
    _ResultActionsWidget,
    _ACTIONS_COL,
    _PATH_COL,
    _ACTIONS_COL_WIDTH,
    _ACTIONS_COL_WIDTH_WITH_WHY,
    _ResultPreviewPanel,
    _TABLE_COLUMNS,
)
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
    view._current_results = [
        _tile(str(img), score=95.0),
        _tile(str(tmp_path / "b.jpg"), score=80.0),
    ]
    view._populate_table(view._current_results)
    yield view
    view.close()
    view.deleteLater()


def test_no_preview_column_has_actions_instead():
    assert "Preview" not in _TABLE_COLUMNS
    assert _TABLE_COLUMNS[_PATH_COL] == "Image Path"
    assert _TABLE_COLUMNS[_ACTIONS_COL] == "Actions"


def test_path_column_is_plain_text_no_cell_widget(search_view):
    path = search_view._current_results[0].tile.file_path
    assert search_view._results_table.cellWidget(0, _PATH_COL) is None
    item = search_view._results_table.item(0, _PATH_COL)
    assert item is not None
    assert item.text() == path


def test_actions_column_hides_why_without_orb(search_view):
    cell = search_view._results_table.cellWidget(0, _ACTIONS_COL)
    assert isinstance(cell, _ResultActionsWidget)
    assert cell._preview_btn.toolTip() == "Preview tile"
    assert cell._copy_btn.toolTip() == "Copy path"
    assert cell._reveal_btn.toolTip()  # platform-specific reveal label
    # Why is not rendered at all when ORB data is absent.
    assert cell._why_btn is None
    assert search_view._results_table.columnWidth(_ACTIONS_COL) == _ACTIONS_COL_WIDTH


def test_actions_column_shows_why_when_orb_present(qapp, tmp_path):
    use_case = MagicMock()
    use_case.execute.return_value = []
    vm = SearchViewModel(use_case=use_case, default_top_k=5)
    vm._active_search_kind = "image"
    view = SearchView(vm, theme="light")
    img = tmp_path / "q.jpg"
    img.write_bytes(b"x")
    result = _tile(str(img))
    result.orb_match_points = [((1.0, 2.0), (3.0, 4.0))]
    view._populate_table([result])
    cell = view._results_table.cellWidget(0, _ACTIONS_COL)
    assert isinstance(cell, _ResultActionsWidget)
    assert cell._why_btn is not None
    assert cell._why_btn.isEnabled()
    assert view._results_table.columnWidth(_ACTIONS_COL) == _ACTIONS_COL_WIDTH_WITH_WHY
    view.close()
    view.deleteLater()


def test_description_search_never_shows_why(qapp, tmp_path):
    use_case = MagicMock()
    use_case.execute.return_value = []
    vm = SearchViewModel(use_case=use_case, default_top_k=5)
    vm._active_search_kind = "description"
    view = SearchView(vm, theme="light")
    img = tmp_path / "t.jpg"
    img.write_bytes(b"x")
    result = _tile(str(img))
    result.orb_match_points = [((1.0, 2.0), (3.0, 4.0))]
    view._populate_table([result])
    cell = view._results_table.cellWidget(0, _ACTIONS_COL)
    assert isinstance(cell, _ResultActionsWidget)
    assert cell._why_btn is None
    view.close()
    view.deleteLater()


def test_copy_shows_floating_toast_not_status_bar(search_view, qapp):
    cell = search_view._results_table.cellWidget(0, _ACTIONS_COL)
    assert isinstance(cell, _ResultActionsWidget)
    path = search_view._current_results[0].tile.file_path
    before_status = search_view._status_label.text()
    cell._handle_copy()
    assert QApplication.clipboard().text() == path
    # No status-bar "Copied path to clipboard…" spam.
    assert "Copied path to clipboard" not in search_view._status_label.text()
    assert search_view._status_label.text() == before_status or "clipboard" not in (
        search_view._status_label.text() or ""
    ).lower()
    assert cell._copied_toast is not None
    assert cell._copied_toast.text() == "Copied"
    assert cell._fade_anim is not None


def test_actions_column_fixed_path_stretches(search_view):
    header = search_view._results_table.horizontalHeader()
    assert header.sectionResizeMode(_ACTIONS_COL) == QHeaderView.ResizeMode.Fixed
    assert search_view._results_table.columnWidth(_ACTIONS_COL) == _ACTIONS_COL_WIDTH
    assert header.sectionResizeMode(_PATH_COL) == QHeaderView.ResizeMode.Stretch


def test_preview_action_opens_preview_panel(search_view, monkeypatch):
    shown = {}

    def fake_show(result):
        shown["path"] = result.tile.file_path

    monkeypatch.setattr(search_view._preview_panel, "show_result", fake_show)
    cell = search_view._results_table.cellWidget(0, _ACTIONS_COL)
    assert isinstance(cell, _ResultActionsWidget)
    cell._preview_btn.click()
    assert shown["path"] == search_view._current_results[0].tile.file_path


def test_reveal_action_is_deferred(search_view, monkeypatch, qapp):
    called = {}

    def fake_reveal(path):
        called["path"] = path
        return True, ""

    monkeypatch.setattr(
        "src.utils.reveal_file.reveal_in_file_manager", fake_reveal
    )
    cell = search_view._results_table.cellWidget(0, _ACTIONS_COL)
    assert isinstance(cell, _ResultActionsWidget)
    cell._reveal_btn.click()
    # Deferred via QTimer — flush by calling the worker.
    path = search_view._current_results[0].tile.file_path
    search_view._reveal_path_now(path)
    assert called["path"] == path


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


def test_action_buttons_use_grid_without_separators(search_view):
    # 2-column grid replaces the old tall separator stack.
    separators = search_view.findChildren(QFrame, "SearchButtonSeparator")
    assert len(separators) == 0

    clear = search_view._clear_button
    assert clear.objectName() == "SearchActionButton"
    assert clear.maximumHeight() <= 48  # Qt may expand slightly for font/padding
    assert search_view._export_button.objectName() == "SearchActionButton"
    assert search_view._crop_button.objectName() == "SearchActionButton"
    assert search_view._auto_crop_button.text() == "Auto Crop Search"
    assert search_view._precise_crop_button.text() == "Precise Crop Search"
    assert search_view._history_button.objectName() == "SearchActionButton"


def test_action_column_is_two_column_grid(search_view):
    col = search_view.findChild(QWidget, "SearchActionColumn")
    assert col is not None
    assert col.maximumWidth() <= 360
    assert col.minimumWidth() >= 320
    from PySide6.QtWidgets import QGridLayout

    assert isinstance(col.layout(), QGridLayout)


def test_export_disabled_without_query_or_results(search_view):
    search_view._current_query_image_path = None
    search_view._current_results = []
    search_view._refresh_action_buttons()
    assert not search_view._export_button.isEnabled()
    assert not search_view._clear_button.isEnabled()
    assert not search_view._crop_button.isEnabled()


def test_history_rerun_enables_crop_and_clear(search_view, tmp_path):
    img = tmp_path / "history.jpg"
    img.write_bytes(b"x")
    search_view._current_results = []
    search_view._refresh_action_buttons()
    # Simulate ViewModel.repeat_search → query_image_selected
    search_view._on_query_image_selected(str(img))
    assert search_view._current_query_image_path == str(img)
    assert search_view._clear_button.isEnabled()
    assert search_view._crop_button.isEnabled()
    assert search_view._auto_crop_button.isEnabled()
    assert search_view._precise_crop_button.isEnabled()
    # Export still needs results
    assert not search_view._export_button.isEnabled()
    search_view._current_results = [_tile(str(img))]
    search_view._refresh_action_buttons()
    assert search_view._export_button.isEnabled()


def test_hover_qss_sets_text_color_in_both_themes():
    for theme in ("light", "dark"):
        qss = get_app_stylesheet(theme)
        assert "QTableWidget::item:hover" in qss
        hover_idx = qss.index("QTableWidget::item:hover")
        chunk = qss[hover_idx : hover_idx + 200]
        assert "color:" in chunk
        p = get_palette(theme)
        assert p["hover_bg"]
        assert p["text_primary"]
