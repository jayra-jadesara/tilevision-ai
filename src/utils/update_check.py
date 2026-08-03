"""
Online update check for TileVision AI.

Fetches a JSON manifest from GitHub Releases (or a vendor URL) and compares
semver versions. Requires a brief internet connection — the app stays offline
for all other features.

Manifest format (update_manifest.json):

    {
      "version": "1.0.1",
      "release_notes": "Bug fixes and improvements.",
      "downloads": {
        "windows": "https://.../TileVisionAI-Setup-1.0.1.exe",
        "macos_intel": "https://.../TileVisionAI-macOS-Intel-1.0.1.dmg",
        "macos_arm64": "https://.../TileVisionAI-macOS-AppleSilicon-1.0.1.dmg"
      }
    }
"""

from __future__ import annotations

import json
import logging
import re
import ssl
import sys
import urllib.error
import urllib.request
from dataclasses import dataclass
from typing import Optional

from src.utils.platform_info import is_macos, is_windows
from src.version import APP_VERSION

logger = logging.getLogger("tilevision.update_check")

DEFAULT_MANIFEST_URL = (
    "https://github.com/jayra-jadesara/tilevision-ai/releases/latest/download/update_manifest.json"
)
# Used when ``/releases/latest/download/update_manifest.json`` 404s — e.g. an
# empty non-prerelease tag briefly became "latest" (client log: HTTP 404).
_GITHUB_RELEASES_API = (
    "https://api.github.com/repos/jayra-jadesara/tilevision-ai/releases"
)
_MANIFEST_ASSET_NAME = "update_manifest.json"

_VERSION_RE = re.compile(r"^v?(?P<major>\d+)\.(?P<minor>\d+)\.(?P<patch>\d+)")


def _build_ssl_context() -> ssl.SSLContext:
    """Use certifi roots when bundled (helps frozen Mac/Windows HTTPS)."""
    try:
        import certifi

        return ssl.create_default_context(cafile=certifi.where())
    except Exception:
        return ssl.create_default_context()


def _http_get_bytes(url: str, *, timeout: float) -> bytes:
    request = urllib.request.Request(
        url,
        headers={
            "User-Agent": f"TileVisionAI/{APP_VERSION}",
            "Accept": "application/octet-stream, application/json",
        },
    )
    context = _build_ssl_context()
    with urllib.request.urlopen(request, timeout=timeout, context=context) as response:
        return response.read()


@dataclass(frozen=True, slots=True)
class UpdateInfo:
    """Result when a newer release is available."""

    current_version: str
    latest_version: str
    release_notes: str
    download_url: str

    @property
    def is_newer(self) -> bool:
        return compare_versions(self.latest_version, self.current_version) > 0


def parse_version(version: str) -> tuple[int, int, int]:
    match = _VERSION_RE.match(version.strip())
    if not match:
        return (0, 0, 0)
    return (
        int(match.group("major")),
        int(match.group("minor")),
        int(match.group("patch")),
    )


def compare_versions(left: str, right: str) -> int:
    """Return positive if left > right."""
    a = parse_version(left)
    b = parse_version(right)
    if a > b:
        return 1
    if a < b:
        return -1
    return 0


def platform_download_key() -> str:
    if is_windows():
        return "windows"
    if is_macos():
        from src.utils.platform_info import is_apple_silicon

        if is_apple_silicon():
            return "macos_arm64"
        return "macos_intel"
    return "linux"


def platform_download_label(platform_key: str | None = None) -> str:
    """Human-readable installer type for the current (or given) platform."""
    key = platform_key or platform_download_key()
    labels = {
        "windows": "Windows installer (.exe)",
        "macos_intel": "Mac Intel installer (.dmg)",
        "macos_arm64": "Mac Apple Silicon installer (.dmg)",
        "linux": "Linux package",
    }
    return labels.get(key, "installer")


def resolve_download_url(downloads: dict, platform_key: str) -> str:
    """
    Return the download URL for this platform only.

    Mac Intel and Apple Silicon builds are never cross-selected.
    """
    direct = str(downloads.get(platform_key) or "").strip()
    if direct:
        return direct

    if platform_key.startswith("macos"):
        return str(downloads.get("macos") or "").strip()
    if platform_key == "linux":
        return str(downloads.get("linux") or downloads.get("url") or "").strip()
    return ""


def _parse_manifest_payload(payload: str) -> dict:
    data = json.loads(payload)
    if not isinstance(data, dict):
        raise ValueError("Update manifest must be a JSON object.")
    return data


def _manifest_url_from_release(release: dict) -> str:
    for asset in release.get("assets") or []:
        if not isinstance(asset, dict):
            continue
        if str(asset.get("name") or "") != _MANIFEST_ASSET_NAME:
            continue
        url = str(asset.get("browser_download_url") or "").strip()
        if url:
            return url
    return ""


def fetch_update_manifest_via_github_api(*, timeout: float = 12.0) -> dict:
    """
    Find the newest non-draft release that publishes ``update_manifest.json``.

    Skips empty tags / incomplete releases so a bad "latest" redirect does not
    permanently break in-app update checks.
    """
    raw = _http_get_bytes(_GITHUB_RELEASES_API, timeout=timeout)
    releases = json.loads(raw.decode("utf-8"))
    if not isinstance(releases, list):
        raise ValueError("GitHub releases API returned a non-list payload.")

    # Prefer non-prerelease, then fall back to any release with a manifest.
    candidates: list[dict] = []
    for release in releases:
        if not isinstance(release, dict) or release.get("draft"):
            continue
        if _manifest_url_from_release(release):
            candidates.append(release)

    if not candidates:
        raise FileNotFoundError(
            f"No GitHub release publishes {_MANIFEST_ASSET_NAME}."
        )

    preferred = [r for r in candidates if not r.get("prerelease")] or candidates
    manifest_url = _manifest_url_from_release(preferred[0])
    logger.info(
        "Update manifest resolved via GitHub API from release %s",
        preferred[0].get("tag_name") or preferred[0].get("name") or "?",
    )
    payload = _http_get_bytes(manifest_url, timeout=timeout).decode("utf-8")
    return _parse_manifest_payload(payload)


def fetch_update_manifest(
    manifest_url: str,
    *,
    timeout: float = 12.0,
) -> dict:
    try:
        payload = _http_get_bytes(manifest_url, timeout=timeout).decode("utf-8")
        return _parse_manifest_payload(payload)
    except urllib.error.HTTPError as exc:
        # Empty / incomplete "latest" tags return 404 for the convenience URL.
        if (
            exc.code == 404
            and manifest_url.rstrip("/") == DEFAULT_MANIFEST_URL.rstrip("/")
        ):
            logger.warning(
                "Update manifest latest/download returned HTTP 404 — "
                "falling back to GitHub Releases API."
            )
            return fetch_update_manifest_via_github_api(timeout=timeout)
        raise


def check_for_updates(
    *,
    current_version: str = APP_VERSION,
    manifest_url: str = DEFAULT_MANIFEST_URL,
    timeout: float = 12.0,
) -> Optional[UpdateInfo]:
    """
    Return UpdateInfo when a newer version is published, else None.

    Raises on network/parse errors so callers can show a manual-check message.
    """
    data = fetch_update_manifest(manifest_url, timeout=timeout)
    latest = str(data.get("version", "")).strip()
    if not latest:
        raise ValueError("Update manifest missing 'version'.")

    if compare_versions(latest, current_version) <= 0:
        return None

    downloads = data.get("downloads") or {}
    if not isinstance(downloads, dict):
        downloads = {}

    platform_key = platform_download_key()
    download_url = resolve_download_url(downloads, platform_key)
    if not download_url:
        raise ValueError(f"Update manifest has no download URL for '{platform_key}'.")

    notes = str(data.get("release_notes") or data.get("notes") or "").strip()
    return UpdateInfo(
        current_version=current_version,
        latest_version=latest,
        release_notes=notes,
        download_url=download_url,
    )


def fetch_update_manifest_with_error(
    *,
    current_version: str = APP_VERSION,
    manifest_url: str = DEFAULT_MANIFEST_URL,
    timeout: float = 12.0,
) -> tuple[Optional[UpdateInfo], Optional[str]]:
    """
    Like check_for_updates but never raises — returns (info, error_message).
    """
    try:
        return check_for_updates(
            current_version=current_version,
            manifest_url=manifest_url,
            timeout=timeout,
        ), None
    except Exception as exc:
        logger.warning("Update check failed: %s", exc)
        return None, str(exc)
