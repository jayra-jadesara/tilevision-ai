"""
Tests for MainWindow's theme propagation to live child views.

Covers the actual bug report: switching themes previously only re-skinned
the app chrome, leaving every view's own panels/cards dark regardless of
the selected theme. These tests construct a real MainWindow (headless,
QT_QPA_PLATFORM=offscreen) and verify that switching themes measurably
changes each live view's own stylesheet, not just the app-level one.
"""

import sys
import types
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def _install_fake_ai_deps():
    if "torch" not in sys.modules:
        fake_torch = types.ModuleType("torch")

        class _FakeCuda:
            @staticmethod
            def is_available():
                return False

        fake_torch.cuda = _FakeCuda()
        sys.modules["torch"] = fake_torch
    if "open_clip" not in sys.modules:
        sys.modules["open_clip"] = types.ModuleType("open_clip")


_install_fake_ai_deps()

PySide6 = pytest.importorskip("PySide6")

from PySide6.QtWidgets import QApplication

from src.config.settings import AppSettings
from src.presentation.viewmodels.indexing_viewmodel import IndexingViewModel
from src.presentation.views.main_window import MainWindow
from src.theme.theme_manager import get_palette


class _FakeIndexUseCase:
    pass


@pytest.fixture(scope="module")
def qapp():
    app = QApplication.instance() or QApplication(sys.argv)
    yield app


@pytest.fixture()
def main_window(qapp, tmp_path):
    settings = AppSettings(config_dir=tmp_path)
    ivm = IndexingViewModel(use_case=_FakeIndexUseCase())
    window = MainWindow(indexing_viewmodel=ivm, settings=settings, catalog_count_provider=lambda: 0)
    yield window
    window.deleteLater()


def test_starts_in_dark_theme_by_default(main_window):
    assert main_window._current_theme == "dark"
    assert get_palette("dark")["bg_app"] in main_window._indexing_view.styleSheet()


def test_switching_to_light_updates_indexing_view(main_window):
    main_window._on_theme_changed_request("light")
    light_bg = get_palette("light")["bg_app"]
    dark_bg = get_palette("dark")["bg_app"]

    qss = main_window._indexing_view.styleSheet()
    assert light_bg in qss
    assert dark_bg not in qss


def test_switching_to_light_updates_app_level_stylesheet(main_window):
    main_window._on_theme_changed_request("light")
    app_qss = QApplication.instance().styleSheet()
    assert get_palette("light")["bg_app"] in app_qss


def test_switching_back_to_dark_restores_dark_colors(main_window):
    main_window._on_theme_changed_request("light")
    main_window._on_theme_changed_request("dark")

    qss = main_window._indexing_view.styleSheet()
    assert get_palette("dark")["bg_app"] in qss
    assert get_palette("light")["bg_app"] not in qss


def test_current_theme_tracked_correctly(main_window):
    assert main_window._current_theme == "dark"
    main_window._on_theme_changed_request("light")
    assert main_window._current_theme == "light"
