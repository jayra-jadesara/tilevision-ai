"""closeEvent confirms when a search is running (indexing already covered)."""

from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from PySide6.QtGui import QCloseEvent
from PySide6.QtWidgets import QApplication

from src.config.settings import AppSettings
from src.presentation.dialogs import message_box
from src.presentation.viewmodels.indexing_viewmodel import IndexingViewModel, IndexingState
from src.presentation.viewmodels.search_viewmodel import SearchState, SearchViewModel
from src.presentation.views.main_window import MainWindow


class _FakeIndexUseCase:
    pass


@pytest.fixture(scope="module")
def qapp():
    return QApplication.instance() or QApplication(sys.argv)


def _make_window(tmp_path: Path, catalogue_master_service) -> MainWindow:
    settings = AppSettings(config_dir=tmp_path / "cfg")
    settings.minimize_to_tray_on_close = False
    indexing_vm = IndexingViewModel(use_case=_FakeIndexUseCase())
    search_vm = SearchViewModel(use_case=MagicMock(), default_top_k=10)
    window = MainWindow(
        indexing_viewmodel=indexing_vm,
        search_viewmodel=search_vm,
        settings=settings,
        catalogue_master_service=catalogue_master_service,
    )
    return window


def test_close_while_searching_prompts_and_accepts_yes(
    qapp, tmp_path: Path, catalogue_master_service
):
    window = _make_window(tmp_path, catalogue_master_service)
    window._search_viewmodel._state = SearchState.SEARCHING
    assert window._search_viewmodel.is_searching is True

    with patch.object(
        message_box, "question", return_value=message_box.StandardButton.Yes
    ) as mock_q:
        event = QCloseEvent()
        window.closeEvent(event)

    mock_q.assert_called_once()
    assert "Search" in mock_q.call_args[0][1]
    assert event.isAccepted() is True
    window.deleteLater()
    qapp.processEvents()


def test_close_while_searching_ignores_on_no(
    qapp, tmp_path: Path, catalogue_master_service
):
    window = _make_window(tmp_path, catalogue_master_service)
    window._search_viewmodel._state = SearchState.SEARCHING

    with patch.object(
        message_box, "question", return_value=message_box.StandardButton.No
    ):
        event = QCloseEvent()
        window.closeEvent(event)

    assert event.isAccepted() is False
    window.deleteLater()
    qapp.processEvents()


def test_close_while_idle_accepts_without_prompt(
    qapp, tmp_path: Path, catalogue_master_service
):
    window = _make_window(tmp_path, catalogue_master_service)
    assert window._indexing_viewmodel.state == IndexingState.IDLE
    assert window._search_viewmodel.is_searching is False

    with patch.object(message_box, "question") as mock_q:
        event = QCloseEvent()
        window.closeEvent(event)

    mock_q.assert_not_called()
    assert event.isAccepted() is True
    window.deleteLater()
    qapp.processEvents()


def test_auto_index_toast_wired_on_main_window(
    qapp, tmp_path: Path, catalogue_master_service
):
    window = _make_window(tmp_path, catalogue_master_service)
    window.show()
    qapp.processEvents()
    assert hasattr(window, "_auto_index_toast")
    assert window._auto_index_toast is not None
    window.handle_auto_index_event("/tmp/sample_tile.jpg", "indexed")
    qapp.processEvents()
    assert window._auto_index_toast.isVisible()
    assert "sample_tile.jpg" in window._auto_index_toast.text()
    window.deleteLater()
    qapp.processEvents()
