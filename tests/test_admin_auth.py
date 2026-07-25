"""Tests for vendor admin password gate."""

import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "admin_tool"))

import admin_auth as auth


@pytest.fixture()
def vendor_settings(tmp_path, monkeypatch):
    vendor_dir = tmp_path / ".tilevision_ai_vendor"
    vendor_dir.mkdir()
    settings_path = vendor_dir / "admin_settings.json"
    monkeypatch.setattr(auth, "_VENDOR_DIR", vendor_dir)
    monkeypatch.setattr(auth, "_SETTINGS_PATH", settings_path)
    return settings_path


def test_default_password_accepts(vendor_settings):
    assert auth.verify_admin_password("raj!RAJ!") is True


def test_wrong_password_rejected(vendor_settings):
    assert auth.verify_admin_password("wrong-password") is False


def test_password_hash_stored_not_plaintext(vendor_settings):
    auth.verify_admin_password("raj!RAJ!")
    data = json.loads(vendor_settings.read_text(encoding="utf-8"))
    assert "access_password_hash" in data
    assert "access_password_salt" in data
    assert "raj!RAJ!" not in json.dumps(data)
