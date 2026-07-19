"""Automatic vendor data backup to cloud-synced folders."""

from __future__ import annotations

import json
import sys
import zipfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

_VENDOR_DIR = Path.home() / ".tilevision_ai_vendor"
_SETTINGS_PATH = _VENDOR_DIR / "admin_settings.json"
_BACKUP_FOLDER_NAME = "TileVision-Vendor-Backup"
_MAX_BACKUPS = 10


def cloud_backup_roots() -> list[Path]:
    """Folders that are often synced to the cloud across operating systems."""
    home = Path.home()
    candidates = [
        home / "OneDrive",
        home / "OneDrive - Personal",
        home / "Documents",
        home / "Dropbox",
        home
        / "Library"
        / "Mobile Documents"
        / "com~apple~CloudDocs",
    ]
    if sys.platform.startswith("linux"):
        candidates.extend(
            [
                home / "Nextcloud",
                home / "ownCloud",
            ]
        )
    return [path for path in candidates if path.is_dir()]


def resolve_backup_dir() -> Optional[Path]:
    """Pick the first available cloud-friendly backup directory."""
    for root in cloud_backup_roots():
        backup_dir = root / _BACKUP_FOLDER_NAME
        try:
            backup_dir.mkdir(parents=True, exist_ok=True)
            return backup_dir
        except OSError:
            continue
    return None


def _save_backup_metadata(backup_dir: Path, archive_path: Path) -> None:
    _VENDOR_DIR.mkdir(parents=True, exist_ok=True)
    data: dict = {}
    if _SETTINGS_PATH.exists():
        try:
            data = json.loads(_SETTINGS_PATH.read_text(encoding="utf-8"))
        except Exception:
            data = {}
    data["last_backup_at"] = datetime.now(timezone.utc).isoformat()
    data["last_backup_path"] = str(archive_path)
    data["last_backup_dir"] = str(backup_dir)
    _SETTINGS_PATH.write_text(json.dumps(data, indent=2), encoding="utf-8")


def get_last_backup_summary() -> str:
    if not _SETTINGS_PATH.exists():
        return "No automatic backup yet."
    try:
        data = json.loads(_SETTINGS_PATH.read_text(encoding="utf-8"))
        when = data.get("last_backup_at", "")
        path = data.get("last_backup_path", "")
        if not when or not path:
            return "No automatic backup yet."
        stamp = when.replace("T", " ").split("+")[0][:16]
        return f"Last backup: {stamp} UTC → {path}"
    except Exception:
        return "No automatic backup yet."


def _prune_old_backups(backup_dir: Path) -> None:
    archives = sorted(
        backup_dir.glob("tilevision_vendor_*.zip"),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    for old in archives[_MAX_BACKUPS:]:
        try:
            old.unlink()
        except OSError:
            pass


def run_vendor_backup() -> tuple[bool, str]:
    """
    Zip the vendor folder to a cloud-synced backup location.

    Returns:
        (success, user-facing message)
    """
    if not _VENDOR_DIR.exists():
        return False, "Nothing to back up yet."

    backup_dir = resolve_backup_dir()
    if backup_dir is None:
        return (
            False,
            "Could not find a cloud-synced folder (OneDrive, iCloud, Dropbox, or Documents).",
        )

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    archive = backup_dir / f"tilevision_vendor_{stamp}.zip"
    latest = backup_dir / "tilevision_vendor_latest.zip"

    try:
        for target in (archive, latest):
            with zipfile.ZipFile(target, "w", zipfile.ZIP_DEFLATED) as zf:
                for file_path in _VENDOR_DIR.rglob("*"):
                    if file_path.is_file():
                        zf.write(file_path, file_path.relative_to(_VENDOR_DIR.parent))
        _prune_old_backups(backup_dir)
        _save_backup_metadata(backup_dir, archive)
        return True, f"Backup saved to:\n{archive}\n\nAlso updated: {latest}"
    except OSError as exc:
        return False, f"Backup failed: {exc}"
