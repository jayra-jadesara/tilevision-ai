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

    with patch.object(update_check, "is_macos", lambda: True):
        with patch.object(update_check, "is_windows", lambda: False):
            with patch(
                "src.utils.platform_info.is_apple_silicon",
                lambda: True,
            ):
                with patch.object(
                    update_check.urllib.request, "urlopen", return_value=_Response()
                ):
                    info = update_check.check_for_updates(current_version="1.0.0")

    assert info is not None
    assert info.download_url.endswith("arm.dmg")


def test_platform_download_key_mac_intel():
    with patch.object(update_check, "is_macos", lambda: True):
        with patch.object(update_check, "is_windows", lambda: False):
            with patch(
                "src.utils.platform_info.is_apple_silicon",
                lambda: False,
            ):
                assert update_check.platform_download_key() == "macos_intel"


def test_check_for_updates_mac_intel_uses_intel_url():
    manifest = {
        "version": "1.0.11",
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

    with patch.object(update_check, "is_macos", lambda: True):
        with patch.object(update_check, "is_windows", lambda: False):
            with patch(
                "src.utils.platform_info.is_apple_silicon",
                lambda: False,
            ):
                with patch.object(
                    update_check.urllib.request, "urlopen", return_value=_Response()
                ):
                    info = update_check.check_for_updates(current_version="1.0.10")

    assert info is not None
    assert info.download_url.endswith("intel.dmg")


def test_mac_arm64_does_not_fallback_to_intel_url():
    downloads = {
        "macos_intel": "https://example.com/intel.dmg",
        "macos_arm64": "https://example.com/arm.dmg",
    }
    assert update_check.resolve_download_url(downloads, "macos_arm64").endswith("arm.dmg")
    assert update_check.resolve_download_url(downloads, "macos_intel").endswith("intel.dmg")


def test_mac_arm64_missing_url_does_not_use_intel_build():
    downloads = {"macos_intel": "https://example.com/intel.dmg"}
    assert update_check.resolve_download_url(downloads, "macos_arm64") == ""


def test_platform_download_label_windows():
    assert update_check.platform_download_label("windows") == "Windows installer (.exe)"


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


def test_fetch_update_manifest_falls_back_to_api_on_latest_404():
    """Empty latest tag (no assets) → /latest/download/... 404; recover via API."""
    manifest = {
        "version": "1.2.4",
        "release_notes": "fix",
        "downloads": {"windows": "https://example.com/setup.exe"},
    }
    api_payload = json.dumps(
        [
            {
                "tag_name": "v1.2.4-empty",
                "draft": False,
                "prerelease": False,
                "assets": [],
            },
            {
                "tag_name": "v1.2.4",
                "draft": False,
                "prerelease": False,
                "assets": [
                    {
                        "name": "update_manifest.json",
                        "browser_download_url": "https://example.com/update_manifest.json",
                    }
                ],
            },
        ]
    ).encode("utf-8")
    manifest_bytes = json.dumps(manifest).encode("utf-8")

    calls = {"n": 0}

    class _Response:
        def __init__(self, body: bytes):
            self._body = body

        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

        def read(self):
            return self._body

    def _urlopen(request, timeout=12.0, context=None):
        calls["n"] += 1
        url = request.full_url if hasattr(request, "full_url") else str(request)
        if url.rstrip("/") == update_check.DEFAULT_MANIFEST_URL.rstrip("/"):
            raise update_check.urllib.error.HTTPError(
                url, 404, "Not Found", hdrs=None, fp=None
            )
        if url == update_check._GITHUB_RELEASES_API:
            return _Response(api_payload)
        if url == "https://example.com/update_manifest.json":
            return _Response(manifest_bytes)
        raise AssertionError(f"Unexpected URL: {url}")

    with patch.object(update_check.urllib.request, "urlopen", side_effect=_urlopen):
        data = update_check.fetch_update_manifest(update_check.DEFAULT_MANIFEST_URL)

    assert data["version"] == "1.2.4"
    assert calls["n"] == 3
