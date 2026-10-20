"""Copy Diagnostics clipboard text: required fields, no secrets."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.utils.diagnostics import format_copy_diagnostics_text
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
    # Loose plain beside enc (what smoke test looks for after a bad quit)
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
    assert "Last shutdown sealed" in text
    assert ": yes" in text.split("Last shutdown sealed")[1].splitlines()[0]
    assert "Loose tiles.db beside tiles.db.enc: yes" in text


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
    assert "Loose tiles.db beside tiles.db.enc:" in text


def test_smoke_test_doc_exists():
    doc = Path(__file__).resolve().parents[1] / "docs" / "SMOKE_TEST.md"
    assert doc.is_file()
    body = doc.read_text(encoding="utf-8")
    assert "Copy Diagnostics" in body
    assert "Catalogue database encrypted at rest" in body
    assert "Duplicates" in body
