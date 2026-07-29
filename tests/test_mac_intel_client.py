"""Mac Intel client: CPU search path + update notifications."""

from __future__ import annotations

import json
import sys
import types
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

pytestmark = pytest.mark.mac_client

from PySide6.QtWidgets import QApplication

import src.ai.gpu_info as gpu_info
import src.utils.platform_info as platform_info
import src.utils.update_check as update_check
from src.config.settings import AppSettings
from src.presentation.update_controller import UpdateController
from src.presentation.views.update_dialog import UpdateAvailableDialog
from src.utils.update_check import UpdateInfo, platform_download_key, platform_download_label


@pytest.fixture(scope="module")
def qapp():
    app = QApplication.instance() or QApplication(sys.argv)
    yield app


def test_mac_intel_never_selects_mps(mac_intel_platform, monkeypatch):
    """Intel Macs must stay on CPU even if torch claims MPS is available."""
    fake_torch = types.SimpleNamespace(
        __version__="2.2.2",
        cuda=types.SimpleNamespace(is_available=lambda: False, device_count=lambda: 0),
        version=types.SimpleNamespace(cuda=None),
        backends=types.SimpleNamespace(mps=types.SimpleNamespace(is_available=lambda: True)),
    )
    monkeypatch.setattr(gpu_info, "torch", fake_torch)
    monkeypatch.setattr(gpu_info, "detect_display_adapters", lambda: ["Intel UHD Graphics 630"])
    monkeypatch.setattr(gpu_info, "has_nvidia_gpu", lambda: False)

    info = gpu_info.detect_gpu_runtime(preference="auto")
    assert platform_info.is_mac_intel()
    assert not platform_info.is_apple_silicon()
    assert info.active_device == "cpu"
    assert not info.using_gpu
    assert "Intel Mac" in info.cpu_fallback_reason


def test_mac_intel_update_key_and_label(mac_intel_platform):
    assert platform_download_key() == "macos_intel"
    assert "Intel" in platform_download_label()


def test_mac_intel_check_for_updates_uses_intel_dmg(mac_intel_platform):
    manifest = {
        "version": "1.0.11",
        "release_notes": "Mac Intel search + update fixes",
        "downloads": {
            "windows": "https://example.com/setup.exe",
            "macos_intel": "https://example.com/TileVisionAI-macOS-Intel-1.0.11.dmg",
            "macos_arm64": "https://example.com/TileVisionAI-macOS-AppleSilicon-1.0.11.dmg",
        },
    }
    payload = json.dumps(manifest).encode("utf-8")

    class _Response:
        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

        def read(self):
            return payload

    with patch.object(update_check.urllib.request, "urlopen", return_value=_Response()):
        info = update_check.check_for_updates(current_version="1.0.10")

    assert info is not None
    assert info.latest_version == "1.0.11"
    assert "macOS-Intel" in info.download_url
    assert "AppleSilicon" not in info.download_url


def test_mac_intel_update_dialog_shows_intel_installer(mac_intel_platform, qapp):
    info = UpdateInfo(
        current_version="1.0.10",
        latest_version="1.0.11",
        release_notes="Intel Mac fixes",
        download_url="https://example.com/TileVisionAI-macOS-Intel-1.0.11.dmg",
    )
    dialog = UpdateAvailableDialog(info, theme="light", auto_start_download=False)
    assert "Mac Intel" in platform_download_label()
    dialog.close()


def test_mac_intel_frozen_startup_schedules_update_check(
    mac_intel_platform, tmp_path, monkeypatch, qapp
):
    settings = AppSettings(config_dir=tmp_path / "cfg")
    settings.check_for_updates = True
    settings.last_update_check_at = ""
    controller = UpdateController(settings, theme="light")

    monkeypatch.setattr(sys, "frozen", True, raising=False)
    parent = MagicMock()
    scheduled = {}

    def _capture(delay, callback):
        scheduled["delay"] = delay
        scheduled["callback"] = callback

    monkeypatch.setattr(
        "src.presentation.update_controller.QTimer.singleShot",
        _capture,
    )
    controller.schedule_startup_check(parent)
    assert "callback" in scheduled
    assert scheduled["delay"] == 4000
