"""Tests for encrypted SQLite at-rest protection."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data.db_protection import (
    decrypt_file,
    derive_db_key,
    encrypt_file,
    encrypted_db_path,
    prepare_working_database,
    seal_database,
)


def test_encrypt_decrypt_round_trip(tmp_path):
    source = tmp_path / "tiles.db"
    source.write_bytes(b"SQLite format 3\x00test payload")
    enc = encrypted_db_path(source)
    key = derive_db_key("test-password")

    encrypt_file(source, enc, key)
    assert enc.exists()
    source.unlink()

    dest = tmp_path / "restored.db"
    decrypt_file(enc, dest, key)
    assert dest.read_bytes() == b"SQLite format 3\x00test payload"


def test_prepare_and_seal_database(tmp_path, monkeypatch):
    db_path = tmp_path / "tiles.db"
    db_path.write_bytes(b"catalogue-data")
    monkeypatch.setenv("TILEVISION_DB_PASSWORD", "site-password")

    seal_database(db_path)
    assert not db_path.exists()
    assert encrypted_db_path(db_path).exists()

    prepare_working_database(db_path)
    assert db_path.exists()
    assert db_path.read_bytes() == b"catalogue-data"
