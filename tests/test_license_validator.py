"""
Tests for LicenseValidator, covering the two security fixes in this patch:

1. _DEVELOPER_MODE now defaults OFF (env-var gated) instead of hardcoded True.
2. The wildcard hardware_hash='*' bypass is only honored in dev mode — it's
   no longer a universal master key usable in production.

Also covers license type support and expiry computation.
"""

import sys
from datetime import date, timedelta
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from cryptography.hazmat.primitives.asymmetric import ec
from cryptography.hazmat.primitives.serialization import Encoding, PrivateFormat, NoEncryption

import src.licensing.validator as validator_module
from src.licensing.validator import (
    LicenseValidator,
    LicenseValidationError,
    LicenseHardwareMismatchError,
    LicenseExpiredError,
    generate_license_key,
    compute_expiry_date,
    LICENSE_TYPE_DURATIONS_DAYS,
    LIFETIME_EXPIRY_SENTINEL,
)


@pytest.fixture()
def keypair():
    """A real (test-only) ECDSA P-256 keypair, generated fresh per test."""
    private_key = ec.generate_private_key(ec.SECP256R1())
    private_pem = private_key.private_bytes(
        Encoding.PEM, PrivateFormat.PKCS8, NoEncryption()
    )
    public_pem = private_key.public_key().public_bytes(
        Encoding.PEM, __import__("cryptography.hazmat.primitives.serialization", fromlist=["PublicFormat"]).PublicFormat.SubjectPublicKeyInfo
    )
    return private_pem, public_pem


# ── _DEVELOPER_MODE default / env-var gating ────────────────────────────


def test_developer_mode_defaults_off_when_env_var_unset(monkeypatch):
    monkeypatch.delenv("TILEVISION_DEV_MODE", raising=False)
    assert validator_module._resolve_developer_mode() is False


def test_developer_mode_enabled_only_with_explicit_env_var(monkeypatch):
    monkeypatch.setenv("TILEVISION_DEV_MODE", "1")
    assert validator_module._resolve_developer_mode() is True


def test_developer_mode_not_enabled_by_other_truthy_values(monkeypatch):
    for value in ("true", "yes", "TRUE", "0", ""):
        monkeypatch.setenv("TILEVISION_DEV_MODE", value)
        assert validator_module._resolve_developer_mode() is False, (
            f"Value {value!r} should NOT enable developer mode — only the exact string '1' should."
        )


# ── Production-mode signature verification (real keypair) ──────────────


def test_valid_signed_license_is_accepted_in_production_mode(keypair, monkeypatch):
    private_pem, public_pem = keypair
    monkeypatch.setattr(validator_module, "_DEVELOPER_MODE", False)
    monkeypatch.setattr(
        validator_module, "get_machine_fingerprint", lambda: "my-machine-fingerprint"
    )

    key = generate_license_key(
        private_pem, "Acme Tiles", "2099-01-01", "my-machine-fingerprint", "Lifetime"
    )
    validator = LicenseValidator(public_key_pem=public_pem)
    result = validator.validate_license(key)

    assert result["customer_name"] == "Acme Tiles"
    assert result["license_type"] == "Lifetime"


def test_tampered_signature_is_rejected_in_production_mode(keypair, monkeypatch):
    private_pem, public_pem = keypair
    monkeypatch.setattr(validator_module, "_DEVELOPER_MODE", False)
    monkeypatch.setattr(
        validator_module, "get_machine_fingerprint", lambda: "my-machine-fingerprint"
    )

    # Sign a key for machine X, then splice in a different hardware_hash
    # without re-signing — this must be rejected.
    key = generate_license_key(
        private_pem, "Acme Tiles", "2099-01-01", "attacker-machine", "Lifetime"
    )
    import base64, json

    payload = json.loads(base64.b64decode(key + "=="))
    payload["hardware_hash"] = "my-machine-fingerprint"  # tamper: retarget without re-signing
    tampered_key = base64.b64encode(json.dumps(payload).encode()).decode()

    validator = LicenseValidator(public_key_pem=public_pem)
    with pytest.raises(LicenseValidationError):
        validator.validate_license(tampered_key)


def test_signature_from_wrong_private_key_is_rejected(keypair, monkeypatch):
    _, public_pem = keypair
    other_private_key = ec.generate_private_key(ec.SECP256R1())
    other_private_pem = other_private_key.private_bytes(
        Encoding.PEM, PrivateFormat.PKCS8, NoEncryption()
    )

    monkeypatch.setattr(validator_module, "_DEVELOPER_MODE", False)
    monkeypatch.setattr(validator_module, "get_machine_fingerprint", lambda: "my-machine")

    # Signed with a DIFFERENT private key than the one whose public key the
    # validator trusts.
    key = generate_license_key(other_private_pem, "Acme Tiles", "2099-01-01", "my-machine")
    validator = LicenseValidator(public_key_pem=public_pem)

    with pytest.raises(LicenseValidationError):
        validator.validate_license(key)


# ── Wildcard hardware_hash bypass restricted to dev mode ────────────────


def test_wildcard_hardware_hash_rejected_in_production_mode(keypair, monkeypatch):
    private_pem, public_pem = keypair
    monkeypatch.setattr(validator_module, "_DEVELOPER_MODE", False)

    key = generate_license_key(private_pem, "Acme Tiles", "2099-01-01", "*")
    validator = LicenseValidator(public_key_pem=public_pem)

    with pytest.raises(LicenseValidationError):
        validator.validate_license(key)


def test_wildcard_hardware_hash_accepted_in_dev_mode(keypair, monkeypatch):
    private_pem, public_pem = keypair
    monkeypatch.setattr(validator_module, "_DEVELOPER_MODE", True)
    monkeypatch.setattr(validator_module, "get_machine_fingerprint", lambda: "any-machine-at-all")

    key = generate_license_key(private_pem, "Dev Tester", "2099-01-01", "*")
    validator = LicenseValidator(public_key_pem=public_pem)
    result = validator.validate_license(key)

    assert result["customer_name"] == "Dev Tester"


# ── Hardware mismatch / expiry ──────────────────────────────────────────


def test_hardware_mismatch_is_rejected(keypair, monkeypatch):
    private_pem, public_pem = keypair
    monkeypatch.setattr(validator_module, "_DEVELOPER_MODE", False)
    monkeypatch.setattr(validator_module, "get_machine_fingerprint", lambda: "machine-B")

    key = generate_license_key(private_pem, "Acme Tiles", "2099-01-01", "machine-A")
    validator = LicenseValidator(public_key_pem=public_pem)

    with pytest.raises(LicenseHardwareMismatchError):
        validator.validate_license(key)


def test_expired_license_is_rejected(keypair, monkeypatch):
    private_pem, public_pem = keypair
    monkeypatch.setattr(validator_module, "_DEVELOPER_MODE", False)
    monkeypatch.setattr(validator_module, "get_machine_fingerprint", lambda: "my-machine")

    yesterday = (date.today() - timedelta(days=1)).strftime("%Y-%m-%d")
    key = generate_license_key(private_pem, "Acme Tiles", yesterday, "my-machine")
    validator = LicenseValidator(public_key_pem=public_pem)

    with pytest.raises(LicenseExpiredError):
        validator.validate_license(key)


# ── License type / expiry computation ────────────────────────────────────


def test_compute_expiry_date_for_each_type():
    from datetime import datetime as dt, timedelta as td

    base = dt(2026, 1, 1)
    assert compute_expiry_date("30-Day", base) == (base + td(days=30)).strftime("%Y-%m-%d")
    assert compute_expiry_date("90-Day", base) == (base + td(days=90)).strftime("%Y-%m-%d")
    assert compute_expiry_date("1-Year", base) == (base + td(days=365)).strftime("%Y-%m-%d")
    assert compute_expiry_date("Lifetime") == LIFETIME_EXPIRY_SENTINEL


def test_compute_expiry_date_rejects_unknown_type():
    with pytest.raises(ValueError):
        compute_expiry_date("Not-A-Real-Type")


def test_lifetime_license_never_expires(keypair, monkeypatch):
    private_pem, public_pem = keypair
    monkeypatch.setattr(validator_module, "_DEVELOPER_MODE", False)
    monkeypatch.setattr(validator_module, "get_machine_fingerprint", lambda: "my-machine")

    expiry = compute_expiry_date("Lifetime")
    key = generate_license_key(private_pem, "Acme Tiles", expiry, "my-machine", "Lifetime")
    validator = LicenseValidator(public_key_pem=public_pem)

    result = validator.validate_license(key)
    assert result["license_type"] == "Lifetime"
