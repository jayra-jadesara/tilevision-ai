"""Tests for online update checks."""

import json
import platform
import sys
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import src.utils.update_check as update_check


def test_compare_versions():
    assert update_check.compare_versions("1.0.1", "1.0.0") > 0
    assert update_check.compare_versions("1.0.0", "1.0.1") < 0
    assert update_check.compare_versions("1.0.0", "1.0.0") == 0


def test_check_for_updates_mac_arm64_uses_arch_specific_url():
    manifest = {
        "version": "1.2.0",
        "downloads": {
            "macos_intel": "https://example.com/intel.dmg",
            "macos_arm64": "https://example.com/arm.dmg",
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

    with patch.object(update_check.sys, "platform", "darwin"):
        with patch.object(platform, "machine", lambda: "arm64"):
            with patch.object(update_check.urllib.request, "urlopen", return_value=_Response()):
                info = update_check.check_for_updates(current_version="1.0.0")

    assert info is not None
    assert info.download_url.endswith("arm.dmg")


def test_platform_download_key_mac_intel():
    with patch.object(update_check.sys, "platform", "darwin"):
        with patch.object(platform, "machine", lambda: "x86_64"):
            assert update_check.platform_download_key() == "macos_intel"


def test_check_for_updates_returns_info_when_newer():
    manifest = {
        "version": "1.2.0",
        "release_notes": "Improvements",
        "downloads": {
            "windows": "https://example.com/setup.exe",
            "macos": "https://example.com/app.dmg",
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

    with patch.object(update_check.sys, "platform", "win32"):
        with patch.object(update_check.urllib.request, "urlopen", return_value=_Response()):
            info = update_check.check_for_updates(current_version="1.0.0")

    assert info is not None
    assert info.latest_version == "1.2.0"
    assert info.download_url.endswith("setup.exe")


def test_check_for_updates_returns_none_when_current():
    manifest = {"version": "1.0.0", "downloads": {"windows": "https://example.com/setup.exe"}}
    payload = json.dumps(manifest).encode("utf-8")

    class _Response:
        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

        def read(self):
            return payload

    with patch.object(update_check.urllib.request, "urlopen", return_value=_Response()):
        info = update_check.check_for_updates(current_version="1.0.0")

    assert info is None
