"""
Deep Mac client flow tests.

Simulates macOS (darwin) on any dev machine. GitHub Actions runs the same tests
on a real macos-latest runner — that is the closest thing to a "Mac simulator"
when you do not have a Mac locally.
"""

from __future__ import annotations

import sys
import types
from pathlib import Path
from unittest.mock import MagicMock

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

pytestmark = pytest.mark.mac_client

if "torch" not in sys.modules:
    fake_torch = types.ModuleType("torch")
    fake_torch.cuda = types.SimpleNamespace(is_available=lambda: False)
    fake_torch.backends = types.SimpleNamespace(
        mps=types.SimpleNamespace(is_available=lambda: True)
    )
    sys.modules["torch"] = fake_torch

PySide6 = pytest.importorskip("PySide6")

from PySide6.QtWidgets import QApplication

from src.config.settings import AppSettings
from src.core.use_cases.find_duplicates import FindDuplicatesUseCase
from src.core.use_cases.validate_license import ValidateLicenseUseCase
from src.licensing.crypto_store import _get_storage_dir
from src.licensing.hardware import get_machine_fingerprint
from src.presentation.viewmodels.indexing_viewmodel import IndexingViewModel
from src.presentation.viewmodels.search_viewmodel import SearchViewModel
from src.presentation.views.duplicates_view import DuplicatesView
from src.presentation.views.help_view import HelpView
from src.presentation.views.license_view import LicenseView
from src.presentation.views.main_window import DashboardDataProviders, MainWindow
from src.presentation.views.setup_wizard import SetupWizardDialog
from src.presentation.views.update_dialog import UpdateAvailableDialog
from src.utils.platform_info import app_icon_path, default_ui_font_family, is_macos
from src.utils.update_check import platform_download_key
from src.version import APP_VERSION
import src.utils.image_formats as image_formats
import src.ai.gpu_info as gpu_info


class _FakeIndexUseCase:
    pass


@pytest.fixture(scope="module")
def qapp():
    app = QApplication.instance() or QApplication(sys.argv)
    yield app


@pytest.fixture()
def mac_settings(tmp_path, mac_platform):
    settings = AppSettings(config_dir=tmp_path / "mac_config")
    settings.theme = "dark"
    settings.setup_wizard_completed = True
    settings.language = "English"
    settings.top_k = 10
    return settings


@pytest.fixture()
def mac_main_window(qapp, mac_settings, catalogue_master_service, mac_platform):
    search_use_case = MagicMock()
    search_use_case.execute.return_value = []
    search_use_case.get_index_health.return_value = MagicMock(
        is_compatible=True, stale_count=0, indexed_count=0
    )
    search_vm = SearchViewModel(use_case=search_use_case, default_top_k=10)

    repo = MagicMock()
    repo.get_all.return_value = []
    duplicates_uc = FindDuplicatesUseCase(repo)

    dashboard = DashboardDataProviders(
        indexed_folder_count=lambda: 1,
        database_size=lambda: 512,
        faiss_size=lambda: 1024,
        last_search=lambda: None,
        recent_activity=lambda: [],
        recent_searches=lambda: [],
    )

    window = MainWindow(
        indexing_viewmodel=IndexingViewModel(use_case=_FakeIndexUseCase()),
        search_viewmodel=search_vm,
        find_duplicates_use_case=duplicates_uc,
        settings=mac_settings,
        catalog_count_provider=lambda: 3,
        catalogue_master_service=catalogue_master_service,
        dashboard_providers=dashboard,
        license_details={
            "license_type": "1-Year",
            "customer_name": "Mac Showroom",
            "is_trial": False,
        },
        db_path_provider=lambda: mac_settings._config_dir / "tiles.db",
        on_check_updates=lambda: None,
    )
    window.show()
    yield window
    window.close()
    window.deleteLater()


def test_mac_license_storage_path(mac_platform, monkeypatch):
    monkeypatch.setattr(sys, "platform", "darwin")
    path = _get_storage_dir()
    assert "Library" in str(path)
    assert "Application Support" in str(path)
    assert path.name == ".lic"


def test_mac_user_config_path(mac_settings, mac_platform):
    assert mac_settings._config_dir.name == "mac_config"
    assert mac_settings._config_dir.exists()


def test_mac_machine_id_is_64_hex(mac_platform, monkeypatch):
    monkeypatch.setattr(
        "src.licensing.hardware.get_macos_platform_uuid",
        lambda: "MAC-UUID-TEST",
    )
    monkeypatch.setattr(
        "src.licensing.hardware.get_macos_serial_number",
        lambda: "SERIAL-TEST",
    )
    monkeypatch.setattr(sys, "platform", "darwin")
    fp = get_machine_fingerprint()
    assert len(fp) == 64
    assert all(c in "0123456789abcdef" for c in fp)


def test_mac_font_icon_and_update_key(mac_platform):
    assert is_macos()
    assert default_ui_font_family() == ".AppleSystemUIFont"
    icon = app_icon_path()
    assert icon is not None and icon.suffix.lower() in {".png", ".ico"}
    assert platform_download_key() in ("macos_intel", "macos_arm64")


def test_mac_heic_supported_for_showroom_photos(mac_platform, monkeypatch):
    monkeypatch.setattr(image_formats, "_heif_registered", True)
    monkeypatch.setattr(image_formats, "register_optional_image_formats", lambda: None)
    exts = image_formats.query_image_extensions()
    assert ".heic" in exts
    assert ".heif" in exts


def test_mac_apple_silicon_mps_gpu(mac_platform, monkeypatch):
    fake_torch = types.SimpleNamespace(
        __version__="2.5.1",
        cuda=types.SimpleNamespace(is_available=lambda: False, device_count=lambda: 0),
        version=types.SimpleNamespace(cuda=None),
        backends=types.SimpleNamespace(mps=types.SimpleNamespace(is_available=lambda: True)),
    )
    monkeypatch.setattr(gpu_info, "torch", fake_torch)
    monkeypatch.setattr(gpu_info, "detect_display_adapters", lambda: ["Apple M2"])
    monkeypatch.setattr(gpu_info, "has_nvidia_gpu", lambda: False)

    info = gpu_info.detect_gpu_runtime(preference="auto")
    assert info.active_device == "mps"
    assert info.using_gpu


def test_mac_activation_screen(mac_platform, qapp):
    use_case = MagicMock(spec=ValidateLicenseUseCase)
    use_case.get_hardware_fingerprint.return_value = "b" * 64

    dialog = LicenseView(validate_use_case=use_case, theme="dark")
    dialog.show()
    assert len(dialog._hw_id_edit.text()) == 64
    assert dialog._license_key_edit.isEnabled()
    dialog.close()


def test_mac_setup_wizard_opens(mac_settings, mac_platform):
    wizard = SetupWizardDialog(mac_settings, theme="dark")
    assert wizard.windowTitle()
    assert wizard._step_title.text()
    wizard.close()


def test_mac_index_page_controls(mac_main_window):
    window = mac_main_window
    window._navigate(0)
    view = window._indexing_view
    assert view._browse_button.isEnabled()
    assert view._start_button.isEnabled()
    assert view._folder_path_edit is not None


def test_mac_search_page_controls(mac_main_window):
    window = mac_main_window
    window._navigate(1)
    view = window._search_view
    assert view._drop_zone is not None
    assert view._export_button.isEnabled()
    assert view._crop_button is not None  # enabled after user picks an image
    assert view._results_table is not None


def test_mac_dashboard_page(mac_main_window):
    window = mac_main_window
    window._navigate(2)
    assert window._content_stack.currentWidget() is window._dashboard_view


def test_mac_settings_general_tab(mac_main_window):
    window = mac_main_window
    window._navigate(3)
    settings = window._settings_view
    assert settings._tabs.currentIndex() == 0
    assert settings._machine_id_edit.isReadOnly()
    assert settings._check_updates_button.isEnabled()
    assert settings._backup_button.isEnabled()
    assert settings._theme_combo.count() >= 2


def test_mac_settings_export_profiles_tab(mac_main_window):
    window = mac_main_window
    window._navigate(3)
    window._settings_view.show_export_profiles_tab()
    assert window._settings_view._tabs.currentIndex() == 1


def test_mac_help_from_sidebar(mac_main_window):
    window = mac_main_window
    dialog = HelpView(parent=window, theme=window._current_theme)
    dialog.show()
    assert dialog.windowTitle()
    dialog.close()


def test_mac_duplicates_from_sidebar(mac_main_window):
    window = mac_main_window
    dialog = DuplicatesView(
        window._find_duplicates_use_case,
        parent=window,
        theme=window._current_theme,
    )
    dialog.show()
    assert dialog.isVisible()
    dialog.close()


def test_mac_all_pages_and_themes(mac_main_window):
    window = mac_main_window
    for theme in ("light", "dark"):
        window._on_theme_changed_request(theme)
        for index in range(4):
            window._navigate(index)
            view = window._content_stack.currentWidget()
            assert view is not None
            assert view.styleSheet()


def test_mac_update_dialog(mac_platform, qapp):
    from src.utils.update_check import UpdateInfo

    info = UpdateInfo(
        current_version=APP_VERSION,
        latest_version="1.0.1",
        release_notes="Mac bug fixes",
        download_url="https://example.com/TileVisionAI-macOS.dmg",
    )
    dialog = UpdateAvailableDialog(info, theme="dark")
    assert dialog.isModal()
    dialog.close()


def test_mac_app_startup_imports(mac_platform):
    from src.app import build_application

    assert callable(build_application)
