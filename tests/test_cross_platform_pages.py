"""
Cross-platform page-by-page UI tests (Windows and macOS simulated).

Walks every customer-facing screen to ensure navigation and dialogs work
identically on both platforms.
"""

from __future__ import annotations

import platform
import sys
import types
from pathlib import Path
from unittest.mock import MagicMock

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def _install_fake_ai_deps() -> None:
    if "torch" not in sys.modules:
        fake_torch = types.ModuleType("torch")

        class _FakeCuda:
            @staticmethod
            def is_available() -> bool:
                return False

        fake_torch.cuda = _FakeCuda()
        fake_torch.backends = types.SimpleNamespace(
            mps=types.SimpleNamespace(is_available=lambda: False)
        )
        sys.modules["torch"] = fake_torch
    if "open_clip" not in sys.modules:
        sys.modules["open_clip"] = types.ModuleType("open_clip")


_install_fake_ai_deps()

PySide6 = pytest.importorskip("PySide6")

from PySide6.QtWidgets import QApplication

from src.config.settings import AppSettings
from src.core.use_cases.find_duplicates import FindDuplicatesUseCase
from src.presentation.viewmodels.indexing_viewmodel import IndexingViewModel
from src.presentation.viewmodels.search_viewmodel import SearchViewModel
from src.presentation.views.duplicates_view import DuplicatesView
from src.presentation.views.help_view import HelpView
from src.presentation.views.license_view import LicenseView
from src.presentation.views.main_window import DashboardDataProviders, MainWindow
from src.presentation.views.setup_wizard import SetupWizardDialog
from src.presentation.views.update_dialog import UpdateAvailableDialog
from src.utils.platform_info import (
    app_icon_path,
    default_ui_font_family,
    is_macos,
    is_windows,
)
from src.utils.update_check import UpdateInfo, platform_download_key
from src.version import APP_VERSION


class _FakeIndexUseCase:
    pass


PLATFORMS = ("win32", "darwin")


def _simulate_platform(monkeypatch, platform: str) -> None:
    from tests.conftest import simulate_platform

    simulate_platform(monkeypatch, platform)


@pytest.fixture(scope="module")
def qapp():
    app = QApplication.instance() or QApplication(sys.argv)
    yield app


@pytest.fixture(params=PLATFORMS)
def platform_name(request):
    return request.param


@pytest.fixture()
def full_main_window(qapp, tmp_path, catalogue_master_service, monkeypatch, platform_name):
    _simulate_platform(monkeypatch, platform_name)

    settings = AppSettings(config_dir=tmp_path)
    settings.theme = "dark"
    settings.setup_wizard_completed = True

    search_use_case = MagicMock()
    search_use_case.execute.return_value = []
    search_vm = SearchViewModel(use_case=search_use_case, default_top_k=10)

    repo = MagicMock()
    repo.get_all.return_value = []
    duplicates_uc = FindDuplicatesUseCase(repo)

    dashboard = DashboardDataProviders(
        indexed_folder_count=lambda: 2,
        database_size=lambda: 1024,
        faiss_size=lambda: 2048,
        last_search=lambda: None,
        recent_activity=lambda: [],
        recent_searches=lambda: [],
    )

    window = MainWindow(
        indexing_viewmodel=IndexingViewModel(use_case=_FakeIndexUseCase()),
        search_viewmodel=search_vm,
        find_duplicates_use_case=duplicates_uc,
        settings=settings,
        catalog_count_provider=lambda: 5,
        catalogue_master_service=catalogue_master_service,
        dashboard_providers=dashboard,
        license_details={
            "license_type": "1-Year",
            "customer_name": "Test Showroom",
            "is_trial": False,
        },
        on_check_updates=lambda: None,
    )
    window.show()
    yield window
    window.close()
    window.deleteLater()


def test_platform_font_and_icon(platform_name, monkeypatch):
    _simulate_platform(monkeypatch, platform_name)
    font = default_ui_font_family()
    icon = app_icon_path()
    assert icon is not None and icon.exists()

    if platform_name == "win32":
        assert font == "Segoe UI"
        assert is_windows()
    else:
        assert font == ".AppleSystemUIFont"
        assert is_macos()


def test_update_download_key_matches_platform(platform_name, monkeypatch):
    _simulate_platform(monkeypatch, platform_name)
    if platform_name == "darwin":
        monkeypatch.setattr(platform, "machine", lambda: "arm64")
    key = platform_download_key()
    if platform_name == "win32":
        assert key == "windows"
    else:
        assert key == "macos_arm64"


def test_index_page_loads(full_main_window):
    window = full_main_window
    window._navigate(0)
    assert window._content_stack.currentIndex() == 0
    assert window._content_stack.currentWidget() is window._indexing_view
    assert window._nav_index_button.isChecked()


def test_search_page_loads(full_main_window):
    window = full_main_window
    window._navigate(1)
    assert window._content_stack.currentIndex() == 1
    assert window._content_stack.currentWidget() is window._search_view
    assert window._nav_search_button.isChecked()


def test_dashboard_page_loads(full_main_window):
    window = full_main_window
    window._navigate(2)
    assert window._content_stack.currentIndex() == 2
    assert window._content_stack.currentWidget() is window._dashboard_view
    assert window._nav_catalog_button.isChecked()


def test_settings_page_loads(full_main_window):
    window = full_main_window
    window._navigate(3)
    assert window._content_stack.currentIndex() == 3
    assert window._content_stack.currentWidget() is window._settings_view
    assert window._settings_view._check_updates_button.isEnabled()
    assert window._nav_settings_button.isChecked()


def test_settings_export_profiles_tab(full_main_window):
    window = full_main_window
    window._navigate(3)
    window._settings_view.show_export_profiles_tab()
    assert window._settings_view._tabs.currentIndex() == 1


def test_all_pages_cycle_without_error(full_main_window):
    window = full_main_window
    for index in (0, 1, 2, 3, 0, 1):
        window._navigate(index)
        assert window._content_stack.currentIndex() == index


def test_help_dialog_opens(full_main_window):
    window = full_main_window
    dialog = HelpView(parent=window, theme=window._current_theme)
    dialog.show()
    assert dialog.windowTitle()
    dialog.close()


def test_duplicates_dialog_opens(full_main_window):
    window = full_main_window
    dialog = DuplicatesView(
        window._find_duplicates_use_case,
        parent=window,
        theme=window._current_theme,
    )
    dialog.show()
    assert dialog.isVisible()
    dialog.close()


def test_license_view_shows_machine_id(qapp, tmp_path, monkeypatch, platform_name):
    _simulate_platform(monkeypatch, platform_name)

    use_case = MagicMock()
    use_case.get_hardware_fingerprint.return_value = "a" * 64

    dialog = LicenseView(validate_use_case=use_case, theme="dark")
    dialog.show()
    assert len(dialog._hw_id_edit.text()) == 64
    dialog.close()


def test_setup_wizard_constructs(qapp, tmp_path, platform_name):
    settings = AppSettings(config_dir=tmp_path)
    wizard = SetupWizardDialog(settings, theme="dark")
    assert wizard.windowTitle()
    wizard.close()


def test_update_dialog_constructs(qapp, platform_name):
    info = UpdateInfo(
        current_version=APP_VERSION,
        latest_version="9.9.9",
        release_notes="Test release",
        download_url=f"https://example.com/{platform_name}",
    )
    dialog = UpdateAvailableDialog(info, theme="dark")
    assert "9.9.9" in dialog.windowTitle() or dialog.isModal()
    dialog.close()


def test_theme_switch_on_all_pages(full_main_window):
    window = full_main_window
    for theme in ("light", "dark"):
        window._on_theme_changed_request(theme)
        assert window._current_theme == theme
        for index in range(4):
            window._navigate(index)
            view = window._content_stack.currentWidget()
            assert view is not None
            assert view.styleSheet()
