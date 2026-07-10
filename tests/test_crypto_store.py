"""Tests for EncryptedLicenseStore (AES-256-GCM encrypted license/trial state)."""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.licensing.crypto_store import EncryptedLicenseStore, _derive_key


def test_write_then_read_round_trips(tmp_path):
    store = EncryptedLicenseStore(storage_path=tmp_path / "state.enc")
    store.write({"foo": "bar", "n": 42})

    result = store.read()
    assert result == {"foo": "bar", "n": 42}


def test_missing_file_returns_none(tmp_path):
    store = EncryptedLicenseStore(storage_path=tmp_path / "nope.enc")
    assert store.read() is None


def test_stored_file_is_not_plaintext(tmp_path):
    store = EncryptedLicenseStore(storage_path=tmp_path / "state.enc")
    store.write({"customer_name": "Acme Tiles", "secret_field": "super-secret-value"})

    raw_bytes = (tmp_path / "state.enc").read_bytes()
    assert b"Acme Tiles" not in raw_bytes
    assert b"super-secret-value" not in raw_bytes


def test_corrupted_file_raises_value_error(tmp_path):
    path = tmp_path / "state.enc"
    store = EncryptedLicenseStore(storage_path=path)
    store.write({"a": 1})

    # Corrupt a byte in the ciphertext
    raw = bytearray(path.read_bytes())
    raw[-1] ^= 0xFF
    path.write_bytes(bytes(raw))

    with pytest.raises(ValueError):
        store.read()


def test_key_derivation_is_deterministic_for_same_fingerprint():
    key1 = _derive_key("abc123")
    key2 = _derive_key("abc123")
    assert key1 == key2
    assert len(key1) == 32  # 256 bits


def test_key_derivation_differs_for_different_fingerprints():
    key1 = _derive_key("machine-a-fingerprint")
    key2 = _derive_key("machine-b-fingerprint")
    assert key1 != key2


def test_data_encrypted_with_different_key_cannot_be_read(tmp_path, monkeypatch):
    """Simulates copying the state file to a different machine."""
    path = tmp_path / "state.enc"

    import src.licensing.crypto_store as crypto_store_module

    monkeypatch.setattr(crypto_store_module, "get_machine_fingerprint", lambda: "machine-A")
    store = EncryptedLicenseStore(storage_path=path)
    store.write({"trial": "data"})

    # Now simulate running on a different machine (different fingerprint)
    monkeypatch.setattr(crypto_store_module, "get_machine_fingerprint", lambda: "machine-B")
    with pytest.raises(ValueError):
        store.read()


def test_exists_and_delete(tmp_path):
    path = tmp_path / "state.enc"
    store = EncryptedLicenseStore(storage_path=path)
    assert store.exists() is False

    store.write({"a": 1})
    assert store.exists() is True

    store.delete()
    assert store.exists() is False
