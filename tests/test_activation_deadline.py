"""
Task 2 — activation_deadline_at (first-activation window).

Covers the security-sensitive contract:
  (a) Old-format keys without the field still activate.
  (b) Keys with a future deadline activate before it.
  (c) Keys with a past deadline are rejected at first activation.
  (d) An already-installed key whose original deadline has passed still
      passes verify_existing_license (must never lock out paying customers).
"""

from __future__ import annotations

import base64
import json
import sys
from datetime import datetime, timedelta
from pathlib import Path
from unittest.mock import MagicMock

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from cryptography.hazmat.primitives.asymmetric import ec
from cryptography.hazmat.primitives.serialization import (
    Encoding,
    NoEncryption,
    PrivateFormat,
    PublicFormat,
)

import src.licensing.validator as validator_module
from src.core.use_cases.validate_license import ValidateLicenseUseCase
from src.licensing.validator import (
    LicenseValidationError,
    LicenseValidator,
    _signing_message,
    generate_license_key,
)


class _FrozenDateTime:
    """Stand-in for ``validate_license.datetime`` with a fixed ``now()``."""

    def __init__(self, frozen: datetime) -> None:
        self._frozen = frozen

    def now(self, tz=None):
        if tz is not None:
            if self._frozen.tzinfo is None:
                return self._frozen.replace(tzinfo=tz)
            return self._frozen.astimezone(tz)
        if self._frozen.tzinfo is not None:
            return self._frozen.replace(tzinfo=None)
        return self._frozen

    def fromisoformat(self, date_string: str):
        return datetime.fromisoformat(date_string)

    def strptime(self, *args, **kwargs):
        return datetime.strptime(*args, **kwargs)


@pytest.fixture()
def keypair():
    private_key = ec.generate_private_key(ec.SECP256R1())
    private_pem = private_key.private_bytes(
        Encoding.PEM, PrivateFormat.PKCS8, NoEncryption()
    )
    public_pem = private_key.public_key().public_bytes(
        Encoding.PEM, PublicFormat.SubjectPublicKeyInfo
    )
    return private_pem, public_pem


@pytest.fixture()
def production_mode(monkeypatch):
    monkeypatch.setattr(validator_module, "_DEVELOPER_MODE", False)
    monkeypatch.setattr(
        validator_module, "get_machine_fingerprint", lambda: "test-machine-fp"
    )


def _use_case_with_real_validator(public_pem, stored_key=None):
    repo = MagicMock()
    if stored_key is None:
        repo.get_license.return_value = None
    else:
        entity = MagicMock()
        entity.license_key = stored_key
        repo.get_license.return_value = entity
    repo.save_license.return_value = True
    validator = LicenseValidator(public_key_pem=public_pem)
    return ValidateLicenseUseCase(repo, validator), repo


# ── (a) Backward compatibility ──────────────────────────────────────────


def test_old_format_key_without_deadline_still_activates(
    keypair, production_mode
):
    private_pem, public_pem = keypair
    key = generate_license_key(
        private_pem,
        "Acme Tiles",
        "2099-01-01",
        "test-machine-fp",
        "Lifetime",
    )
    payload = json.loads(base64.b64decode(key + "==").decode("utf-8"))
    assert "activation_deadline_at" not in payload

    use_case, repo = _use_case_with_real_validator(public_pem)
    result = use_case.activate_new_license(key)

    assert result["customer_name"] == "Acme Tiles"
    repo.save_license.assert_called_once()


def test_signing_message_omits_deadline_when_absent_matches_legacy():
    """Keys signed before this field existed must produce the same message."""
    payload = {
        "version": 2,
        "license_id": "abc-123",
        "customer_name": "Acme",
        "expires_at": "2099-01-01",
        "hardware_hash": "hw",
        "license_type": "Lifetime",
    }
    assert _signing_message(payload) == b"abc-123|Acme|2099-01-01|hw|Lifetime"


def test_signing_message_appends_deadline_when_present():
    payload = {
        "version": 2,
        "license_id": "abc-123",
        "customer_name": "Acme",
        "expires_at": "2099-01-01",
        "hardware_hash": "hw",
        "license_type": "Lifetime",
        "activation_deadline_at": "2026-10-12T12:00:00",
    }
    assert (
        _signing_message(payload)
        == b"abc-123|Acme|2099-01-01|hw|Lifetime|2026-10-12T12:00:00"
    )


# ── (b) Activate before deadline ────────────────────────────────────────


def test_key_with_future_deadline_activates(keypair, production_mode, monkeypatch):
    private_pem, public_pem = keypair
    now = datetime(2026, 10, 12, 10, 0, 0)
    deadline = now + timedelta(minutes=30)
    monkeypatch.setattr(
        "src.core.use_cases.validate_license.datetime",
        _FrozenDateTime(now),
    )

    key = generate_license_key(
        private_pem,
        "Acme Tiles",
        "2099-01-01",
        "test-machine-fp",
        "1-Year",
        activation_deadline_at=deadline.isoformat(timespec="seconds"),
    )
    use_case, repo = _use_case_with_real_validator(public_pem)
    result = use_case.activate_new_license(key)

    assert result["customer_name"] == "Acme Tiles"
    assert result["activation_deadline_at"] == deadline.isoformat(timespec="seconds")
    repo.save_license.assert_called_once()


# ── (c) Reject after deadline ───────────────────────────────────────────


def test_key_with_past_deadline_rejected_on_activation(
    keypair, production_mode, monkeypatch
):
    private_pem, public_pem = keypair
    now = datetime(2026, 10, 12, 12, 0, 0)
    deadline = now - timedelta(minutes=5)
    monkeypatch.setattr(
        "src.core.use_cases.validate_license.datetime",
        _FrozenDateTime(now),
    )

    key = generate_license_key(
        private_pem,
        "Acme Tiles",
        "2099-01-01",
        "test-machine-fp",
        "1-Year",
        activation_deadline_at=deadline.isoformat(timespec="seconds"),
    )
    use_case, repo = _use_case_with_real_validator(public_pem)

    with pytest.raises(LicenseValidationError, match="activation window has expired"):
        use_case.activate_new_license(key)

    repo.save_license.assert_not_called()


# ── (d) Already-activated key never re-checked ──────────────────────────


def test_verify_existing_ignores_expired_activation_deadline(
    keypair, production_mode, monkeypatch
):
    """Critical regression: installed key must still work after window passes."""
    private_pem, public_pem = keypair
    now = datetime(2026, 10, 12, 18, 0, 0)
    deadline = now - timedelta(hours=4)
    monkeypatch.setattr(
        "src.core.use_cases.validate_license.datetime",
        _FrozenDateTime(now),
    )

    key = generate_license_key(
        private_pem,
        "Paying Customer",
        "2099-01-01",
        "test-machine-fp",
        "Lifetime",
        activation_deadline_at=deadline.isoformat(timespec="seconds"),
    )
    use_case, _repo = _use_case_with_real_validator(public_pem, stored_key=key)

    result = use_case.verify_existing_license()

    assert result is not None
    assert result["customer_name"] == "Paying Customer"
    assert result.get("activation_deadline_at") == deadline.isoformat(timespec="seconds")


def test_validate_license_itself_does_not_enforce_deadline(
    keypair, production_mode
):
    """Deadline gating must live only in activate_new_license."""
    private_pem, public_pem = keypair
    past = (datetime.now() - timedelta(hours=1)).isoformat(timespec="seconds")
    key = generate_license_key(
        private_pem,
        "Acme",
        "2099-01-01",
        "test-machine-fp",
        "Lifetime",
        activation_deadline_at=past,
    )
    validator = LicenseValidator(public_key_pem=public_pem)
    details = validator.validate_license(key)
    assert details["activation_deadline_at"] == past


def test_stripping_deadline_from_signed_key_fails_verification(
    keypair, production_mode
):
    """Removing a signed activation_deadline_at must invalidate the signature."""
    private_pem, public_pem = keypair
    deadline = (datetime.now() + timedelta(hours=1)).isoformat(timespec="seconds")
    key = generate_license_key(
        private_pem,
        "Acme",
        "2099-01-01",
        "test-machine-fp",
        "Lifetime",
        activation_deadline_at=deadline,
    )
    payload = json.loads(base64.b64decode(key + "==").decode("utf-8"))
    del payload["activation_deadline_at"]
    tampered = base64.b64encode(json.dumps(payload).encode("utf-8")).decode("utf-8")

    validator = LicenseValidator(public_key_pem=public_pem)
    with pytest.raises(LicenseValidationError):
        validator.validate_license(tampered)
