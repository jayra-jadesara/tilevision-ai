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


def ensure_github_defaults() -> None:
    """Persist default repo/branch so publish works without manual setup."""
    data = load_vendor_settings()
    updates: dict[str, str] = {}
    if not str(data.get("github_repo", "")).strip():
        updates["github_repo"] = DEFAULT_GITHUB_REPO
    if not str(data.get("github_branch", "")).strip():
        updates["github_branch"] = DEFAULT_GITHUB_BRANCH
    if updates:
        save_vendor_settings(**updates)


_DEFAULT_PRICING_DROPDOWNS: dict[str, list[str]] = {
    "plan_labels": ["1 Year", "2 Year", "3 Year", "4 Year", "Lifetime"],
    "per_year": ["38000", "34200", "32300", "30400", "One-time"],
    "discount_notes": ["-", "5% off", "10% off", "15% off", "20% off", "Best for long-term"],
    "badges": ["", "Best value", "Popular", "Limited offer"],
}


def get_pricing_dropdown_options() -> dict[str, list[str]]:
    data = load_vendor_settings()
    stored = data.get("pricing_dropdowns")
    merged: dict[str, list[str]] = {}
    for key, defaults in _DEFAULT_PRICING_DROPDOWNS.items():
        extra = stored.get(key) if isinstance(stored, dict) else None
        values: list[str] = list(defaults)
        if isinstance(extra, list):
            for item in extra:
                text = str(item).strip()
                if text and text not in values:
                    values.append(text)
        merged[key] = values
    return merged


def remember_pricing_dropdown_value(category: str, value: str) -> None:
    text = str(value).strip()
    if not text:
        return
    options = get_pricing_dropdown_options()
    current = list(options.get(category, []))
    if text in current:
        return
    current.append(text)
    data = load_vendor_settings()
    dropdowns = dict(data.get("pricing_dropdowns") or {})
    dropdowns[category] = current
    save_vendor_settings(pricing_dropdowns=dropdowns)
