"""Tests for vendor admin pricing editor and GitHub publish."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "admin_tool"))

from pricing_manager import (  # noqa: E402
    apply_editable_fields,
    backup_current,
    load_template,
    plan_row_from_dict,
    save_draft,
    serialize_prices_json,
)
from github_pricing_publish import (  # noqa: E402
    GitHubPublishError,
    publish_prices_to_github,
    verify_github_token,
)
from vendor_settings import (  # noqa: E402
    DEFAULT_GITHUB_REPO,
    load_vendor_settings,
    save_vendor_settings,
)


@pytest.fixture()
def vendor_dir(tmp_path, monkeypatch):
    vendor = tmp_path / ".tilevision_ai_vendor"
    vendor.mkdir()
    monkeypatch.setattr("vendor_settings._VENDOR_DIR", vendor)
    monkeypatch.setattr("vendor_settings._SETTINGS_PATH", vendor / "admin_settings.json")
    monkeypatch.setattr("pricing_manager._VENDOR_DIR", vendor)
    monkeypatch.setattr("pricing_manager._DRAFT_PATH", vendor / "prices_draft.json")
    monkeypatch.setattr("pricing_manager._BACKUP_DIR", vendor / "pricing_backups")
    return vendor


def _base_payload() -> dict:
    from src.services.pricing_quote_service import bundled_prices_path, load_prices_file

    return load_prices_file(bundled_prices_path())


def test_apply_editable_fields_updates_vendor_and_plans():
    base = _base_payload()
    plans = [
        {
            "id": "1y",
            "label": "1 Year",
            "price": 40000,
            "effective_per_year": 40000,
            "discount_note": "-",
            "badge": None,
        }
    ]
    updated = apply_editable_fields(
        base,
        location="Morbi, Gujarat",
        hero_title="New hero",
        hero_body="New body",
        pricing_heading="License pricing (INR)",
        taxes_line="GST extra",
        vendor_name="JD Software",
        vendor_email="test@example.com",
        vendor_phone="+91 99999 99999",
        vendor_phone_display="WhatsApp: +91 99999 99999",
        plans=plans,
    )
    assert updated["location"] == "Morbi, Gujarat"
    assert updated["vendor"]["email"] == "test@example.com"
    assert updated["footer"]["taxes"] == "GST extra"
    assert updated["plans"][0]["price"] == 40000
    assert int(updated["version"]) >= int(base.get("version", 0))


def test_save_draft_and_backup(vendor_dir):
    data = _base_payload()
    path = save_draft(data)
    assert path.is_file()
    backup = backup_current(data)
    assert backup.is_file()
    loaded = json.loads(path.read_text(encoding="utf-8"))
    assert loaded["product_name"] == data["product_name"]


def test_vendor_settings_store_github_token(vendor_dir):
    save_vendor_settings(
        github_token="ghp_test_token",
        github_repo=DEFAULT_GITHUB_REPO,
        github_branch="main",
    )
    settings = load_vendor_settings()
    assert settings["github_token"] == "ghp_test_token"
    assert settings["github_repo"] == DEFAULT_GITHUB_REPO


def test_plan_row_from_dict_lifetime():
    row = plan_row_from_dict(
        {
            "id": "lifetime",
            "label": "Lifetime",
            "price": 200000,
            "effective_label": "One-time",
            "discount_note": "Best",
            "badge": None,
        }
    )
    assert row["effective_label"] == "One-time"


def test_verify_github_token_success(monkeypatch):
  def fake_api(method, url, token, **kwargs):
      if url.endswith("/user"):
          return {"login": "jayra-jadesara"}
      if "/repos/" in url:
          return {"full_name": DEFAULT_GITHUB_REPO}
      raise AssertionError(url)

  monkeypatch.setattr(
      "github_pricing_publish._api_request",
      fake_api,
  )
  login = verify_github_token(token="ghp_x", repo=DEFAULT_GITHUB_REPO)
  assert login == "jayra-jadesara"


def test_publish_prices_to_github_updates_both_paths(monkeypatch):
    calls: list[str] = []

    def fake_update(repo_path, content_text, **kwargs):
        calls.append(repo_path)
        return {"content": {"path": repo_path}}

    monkeypatch.setattr("github_pricing_publish.update_repo_file", fake_update)
    data = _base_payload()
    result = publish_prices_to_github(
        data,
        token="ghp_x",
        repo=DEFAULT_GITHUB_REPO,
        branch="main",
    )
    assert len(result.updated_paths) == 2
    assert "pricing/prices.json" in calls
    assert "src/resources/pricing/prices.json" in calls


def test_publish_requires_token():
    data = _base_payload()
    with pytest.raises(GitHubPublishError):
        publish_prices_to_github(data, token="")
