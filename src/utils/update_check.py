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
        "macos": "https://.../TileVisionAI-macOS-1.0.1.dmg"
      }
    }
"""

from __future__ import annotations

import json
import logging
import re
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

_VERSION_RE = re.compile(r"^v?(?P<major>\d+)\.(?P<minor>\d+)\.(?P<patch>\d+)")


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
        return "macos"
    return "linux"


def fetch_update_manifest(
    manifest_url: str,
    *,
    timeout: float = 12.0,
) -> dict:
    request = urllib.request.Request(
        manifest_url,
        headers={"User-Agent": f"TileVisionAI/{APP_VERSION}"},
    )
    with urllib.request.urlopen(request, timeout=timeout) as response:
        payload = response.read().decode("utf-8")
    data = json.loads(payload)
    if not isinstance(data, dict):
        raise ValueError("Update manifest must be a JSON object.")
    return data


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
    download_url = str(downloads.get(platform_key) or downloads.get("url") or "").strip()
    if not download_url:
        raise ValueError(f"Update manifest has no download URL for '{platform_key}'.")

    notes = str(data.get("release_notes") or data.get("notes") or "").strip()
    return UpdateInfo(
        current_version=current_version,
        latest_version=latest,
        release_notes=notes,
        download_url=download_url,
    )
