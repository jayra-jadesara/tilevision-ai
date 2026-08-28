"""Shared vendor admin settings (password hash, theme, GitHub token, etc.)."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

_VENDOR_DIR = Path.home() / ".tilevision_ai_vendor"
_SETTINGS_PATH = _VENDOR_DIR / "admin_settings.json"

DEFAULT_GITHUB_REPO = "jayra-jadesara/tilevision-ai"
DEFAULT_GITHUB_BRANCH = "main"


def vendor_settings_path() -> Path:
    return _SETTINGS_PATH


def load_vendor_settings() -> dict[str, Any]:
    if not _SETTINGS_PATH.is_file():
        return {}
    try:
        data = json.loads(_SETTINGS_PATH.read_text(encoding="utf-8"))
        return data if isinstance(data, dict) else {}
    except Exception:
        return {}


def save_vendor_settings(**updates: Any) -> None:
    _VENDOR_DIR.mkdir(parents=True, exist_ok=True)
    data = load_vendor_settings()
    data.update(updates)
    _SETTINGS_PATH.write_text(
        json.dumps(data, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


def get_github_token() -> str:
    return str(load_vendor_settings().get("github_token", "")).strip()


def get_github_repo() -> str:
    return str(load_vendor_settings().get("github_repo", DEFAULT_GITHUB_REPO)).strip()


def get_github_branch() -> str:
    return str(load_vendor_settings().get("github_branch", DEFAULT_GITHUB_BRANCH)).strip()
