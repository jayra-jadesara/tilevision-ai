"""Copy Diagnostics clipboard text: required fields, no secrets."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.utils.diagnostics import (
    _UNKNOWN_SEAL_MESSAGE,
    format_copy_diagnostics_text,
)
from src.version import APP_VERSION


def test_copy_diagnostics_contains_required_fields(tmp_path, monkeypatch):
    log_dir = tmp_path / "logs"
    log_dir.mkdir()
    log_path = log_dir / "tilevision.log"
    log_path.write_text(
        "INFO something\nINFO Catalogue database encrypted at rest.\n",
        encoding="utf-8",
    )
    db_dir = tmp_path / "database"
    db_dir.mkdir()
    plain = db_dir / "tiles.db"
    enc = db_dir / "tiles.db.enc"
    enc.write_bytes(b"enc")
    # Working plain DB while app is open — expected, not a seal failure.
    plain.write_bytes(b"plain")

    monkeypatch.setattr(
        "src.utils.logger.get_log_file_path",
        lambda log_file_name="tilevision.log": log_path,
    )

    text = format_copy_diagnostics_text(
        {
            "app_version": APP_VERSION,
            "platform": "TestOS-1.0",
            "database": str(plain),
            "catalog_size": 42,
            "device": "cpu",
        }
    )
    assert "App version:" in text
    assert APP_VERSION in text
    assert "OS / platform:" in text
    assert "TestOS-1.0" in text
    assert "Log file:" in text
    assert str(log_path) in text
    assert "Catalogue database encrypted at rest" in text
    assert "Previous quit sealed the database" in text
    assert text.split("Previous quit sealed the database")[1].splitlines()[0].endswith(
        ": yes"
    )
    assert "tiles.db present now (expected while the app is open): yes" in text
    assert "Loose tiles.db beside tiles.db.enc" not in text


def test_copy_diagnostics_unknown_seal_when_no_log_line(tmp_path, monkeypatch):
    log_path = tmp_path / "tilevision.log"
    log_path.write_text("INFO no seal line yet\n", encoding="utf-8")
    monkeypatch.setattr(
        "src.utils.logger.get_log_file_path",
        lambda log_file_name="tilevision.log": log_path,
    )
    text = format_copy_diagnostics_text(
        {"app_version": "1.0.0", "platform": "Linux", "database": str(tmp_path / "tiles.db")}
    )
    assert _UNKNOWN_SEAL_MESSAGE in text
    assert "quit the app once, relaunch, copy again" in text


def test_copy_diagnostics_never_contains_licence_or_password(tmp_path, monkeypatch):
    log_path = tmp_path / "tilevision.log"
    log_path.write_text("ok\n", encoding="utf-8")
    monkeypatch.setattr(
        "src.utils.logger.get_log_file_path",
        lambda log_file_name="tilevision.log": log_path,
    )
    secret_key = "TV-LICENCE-SUPER-SECRET-KEY-999"
    secret_pw = "hunter2-db-password"
    text = format_copy_diagnostics_text(
        {
            "app_version": "9.9.9",
            "platform": "Linux",
            "license_key": secret_key,
            "licence_key": secret_key,
            "password": secret_pw,
            "TILEVISION_DB_PASSWORD": secret_pw,
            "database": str(tmp_path / "tiles.db"),
        }
    )
    assert secret_key not in text
    assert secret_pw not in text
    assert "hunter2" not in text
    assert "LICENCE-SUPER" not in text
    # Required structural fields still present
    assert "App version:" in text
    assert "Log file:" in text
    assert "tiles.db present now (expected while the app is open):" in text
    assert "Previous quit sealed the database" in text


def test_smoke_test_doc_exists():
    doc = Path(__file__).resolve().parents[1] / "docs" / "SMOKE_TEST.md"
    assert doc.is_file()
    body = doc.read_text(encoding="utf-8")
    assert "Copy Diagnostics" in body
    assert "Catalogue database encrypted at rest" in body
    assert "Duplicates" in body
    assert "Relaunch" in body
    # Copy Diagnostics must come after quit + seal checks + relaunch.
    quit_i = body.find("**Quit** the app")
    copy_i = body.find("Settings → Copy Diagnostics")
    relaunch_i = body.find("**Relaunch** the app")
    assert quit_i != -1 and copy_i != -1 and relaunch_i != -1
    assert quit_i < relaunch_i < copy_i
    assert "previous" in body.casefold() or "PREVIOUS" in body
