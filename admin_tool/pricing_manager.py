"""Load, edit, validate, and persist pricing quote JSON for the vendor admin tool."""

from __future__ import annotations

import json
from datetime import date
from pathlib import Path
from typing import Any, Mapping

from src.services.pricing_quote_service import (
    DEFAULT_PRICES_URL,
    PricingQuoteError,
    bundled_prices_path,
    load_prices_file,
    load_prices_json_text,
    validate_prices_data,
)

_VENDOR_DIR = Path.home() / ".tilevision_ai_vendor"
_DRAFT_PATH = _VENDOR_DIR / "prices_draft.json"
_BACKUP_DIR = _VENDOR_DIR / "pricing_backups"

# Repo-relative paths updated on publish (live + bundled fallback).
PUBLISH_PATHS = (
    "pricing/prices.json",
    "src/resources/pricing/prices.json",
)


def draft_path() -> Path:
    return _DRAFT_PATH


def backup_dir() -> Path:
    return _BACKUP_DIR


def default_template_path() -> Path:
    repo = Path(__file__).resolve().parent.parent / "pricing" / "prices.json"
    if repo.is_file():
        return repo
    return bundled_prices_path()


def load_template() -> dict[str, Any]:
    """Load the best available baseline (draft → repo → bundled)."""
    if _DRAFT_PATH.is_file():
        try:
            return load_prices_file(_DRAFT_PATH)
        except PricingQuoteError:
            pass
    path = default_template_path()
    return load_prices_file(path)


def save_draft(data: Mapping[str, Any]) -> Path:
    validated = validate_prices_data(data)
    _VENDOR_DIR.mkdir(parents=True, exist_ok=True)
    text = serialize_prices_json(validated)
    _DRAFT_PATH.write_text(text, encoding="utf-8")
    return _DRAFT_PATH


def backup_current(data: Mapping[str, Any]) -> Path:
    _BACKUP_DIR.mkdir(parents=True, exist_ok=True)
    stamp = date.today().isoformat()
    path = _BACKUP_DIR / f"prices_{stamp}.json"
    counter = 1
    while path.exists():
        counter += 1
        path = _BACKUP_DIR / f"prices_{stamp}_{counter}.json"
    path.write_text(serialize_prices_json(data), encoding="utf-8")
    return path


def serialize_prices_json(data: Mapping[str, Any]) -> str:
    return json.dumps(dict(data), indent=2, ensure_ascii=False) + "\n"


def fetch_live_prices(*, timeout: float = 12.0) -> dict[str, Any]:
    from src.services.pricing_quote_service import fetch_remote_prices

    return fetch_remote_prices(DEFAULT_PRICES_URL, timeout=timeout)


def apply_editable_fields(
    base: Mapping[str, Any],
    *,
    location: str,
    hero_title: str,
    hero_body: str,
    pricing_heading: str,
    taxes_line: str,
    vendor_name: str,
    vendor_email: str,
    vendor_phone: str,
    vendor_phone_display: str,
    plans: list[dict[str, Any]],
) -> dict[str, Any]:
    """Merge form edits into a full prices.json payload (keeps features, etc.)."""
    payload = dict(base)
    payload["location"] = location.strip()
    payload["hero_title"] = hero_title.strip()
    payload["hero_body"] = hero_body.strip()
    payload["pricing_heading"] = pricing_heading.strip()
    payload["updated_at"] = date.today().isoformat()
    payload["version"] = int(payload.get("version") or 0) + 1
    payload["plans"] = plans

    vendor = dict(payload.get("vendor") or {})
    vendor["name"] = vendor_name.strip()
    vendor["email"] = vendor_email.strip()
    vendor["phone"] = vendor_phone.strip()
    vendor["phone_display"] = vendor_phone_display.strip()
    payload["vendor"] = vendor

    footer = dict(payload.get("footer") or {})
    footer["taxes"] = taxes_line.strip()
    payload["footer"] = footer

    payload["notes"] = (
        "Updated via TileVision Admin. Customers refresh Pricing in the app "
        "to see new rates — no installer rebuild required."
    )
    return validate_prices_data(payload)


def plan_row_from_dict(plan: Mapping[str, Any]) -> dict[str, Any]:
    """Normalize a plan object for the admin table."""
    row = {
        "id": str(plan.get("id", "")),
        "label": str(plan.get("label", "")),
        "price": int(round(float(plan.get("price", 0)))),
        "effective_per_year": plan.get("effective_per_year"),
        "effective_label": plan.get("effective_label"),
        "discount_note": str(plan.get("discount_note") or "-"),
        "badge": plan.get("badge"),
    }
    if row["badge"] is not None:
        row["badge"] = str(row["badge"]).strip() or None
    return row


def plans_to_publish_rows(plans: list[Mapping[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for plan in plans:
        row = plan_row_from_dict(plan)
        cleaned: dict[str, Any] = {
            "id": row["id"],
            "label": row["label"],
            "price": row["price"],
            "discount_note": row["discount_note"],
            "badge": row["badge"],
        }
        if row.get("effective_label"):
            cleaned["effective_label"] = row["effective_label"]
            cleaned["effective_per_year"] = None
        elif row.get("effective_per_year") is not None:
            cleaned["effective_per_year"] = int(row["effective_per_year"])
        rows.append(cleaned)
    return rows


def copy_to_repo_paths(data: Mapping[str, Any], repo_root: Path | None = None) -> list[Path]:
    """Write JSON to local repo copies (dev convenience after publish)."""
    root = repo_root or Path(__file__).resolve().parent.parent
    text = serialize_prices_json(data)
    written: list[Path] = []
    for rel in PUBLISH_PATHS:
        target = root / rel
        if not target.parent.is_dir():
            continue
        target.write_text(text, encoding="utf-8")
        written.append(target)
    return written


def restore_backup(backup_file: Path, draft: bool = True) -> dict[str, Any]:
    data = load_prices_file(backup_file)
    if draft:
        save_draft(data)
    return data
