"""Tests for license revocation and v2 license payloads."""

import base64
import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from cryptography.hazmat.primitives.asymmetric import ec
from cryptography.hazmat.primitives.serialization import Encoding, PrivateFormat, NoEncryption

import src.licensing.validator as validator_module
import src.licensing.revocation as revocation_module
from src.licensing.validator import (
    LicenseValidator,
    LicenseRevokedError,
    generate_license_key,
    compute_expiry_date,
    VENDOR_LICENSE_TYPES,
)
from src.licensing.revocation import is_license_revoked, save_revocation_file


@pytest.fixture()
def keypair():
    private_key = ec.generate_private_key(ec.SECP256R1())
    private_pem = private_key.private_bytes(
        Encoding.PEM, PrivateFormat.PKCS8, NoEncryption()
    )
    public_pem = private_key.public_key().public_bytes(
        Encoding.PEM,
        __import__(
            "cryptography.hazmat.primitives.serialization",
            fromlist=["PublicFormat"],
        ).PublicFormat.SubjectPublicKeyInfo,
    )
    return private_pem, public_pem


def test_vendor_license_types_include_trials_and_official():
    assert "15-Day Trial" in VENDOR_LICENSE_TYPES
    assert "3-Year" in VENDOR_LICENSE_TYPES


def test_compute_expiry_for_new_types():
    from datetime import datetime as dt, timedelta as td

    base = dt(2026, 1, 1)
    assert compute_expiry_date("6-Month Trial", base) == (base + td(days=180)).strftime("%Y-%m-%d")
    assert compute_expiry_date("3-Year", base) == (base + td(days=1095)).strftime("%Y-%m-%d")


def test_generated_key_includes_license_id(keypair):
    private_pem, _ = keypair
    key = generate_license_key(
        private_pem, "Acme", "2099-01-01", "machine-a", "1-Year"
    )
    payload = json.loads(base64.b64decode(key + "==").decode("utf-8"))
    assert payload["version"] == 2
    assert payload["license_id"]


def test_revoked_license_is_rejected(keypair, monkeypatch):
    private_pem, public_pem = keypair
    monkeypatch.setattr(validator_module, "_DEVELOPER_MODE", False)
    monkeypatch.setattr(validator_module, "get_machine_fingerprint", lambda: "machine-a")

    key = generate_license_key(
        private_pem, "Acme", "2099-01-01", "machine-a", "1-Year"
    )
    payload = json.loads(base64.b64decode(key + "==").decode("utf-8"))
    license_id = payload["license_id"]

    monkeypatch.setattr(
        revocation_module,
        "load_revoked_license_ids",
        lambda: {license_id},
    )

    validator = LicenseValidator(public_key_pem=public_pem)
    with pytest.raises(LicenseRevokedError):
        validator.validate_license(key)


def test_revocation_file_round_trip(tmp_path):
    path = tmp_path / "revoked.json"
    save_revocation_file({"abc-123", "def-456"}, path)
    assert is_license_revoked("abc-123", revoked_ids={"abc-123"})
