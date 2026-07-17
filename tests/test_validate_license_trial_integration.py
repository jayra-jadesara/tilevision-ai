"""Tests for ValidateLicenseUseCase license and trial-key integration."""

import sys
from pathlib import Path
from unittest.mock import MagicMock

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.core.use_cases.validate_license import ValidateLicenseUseCase
from src.licensing.validator import LicenseValidationError


def _make_use_case(license_entity=None, validator_result=None, validator_error=None):
    repo = MagicMock()
    repo.get_license.return_value = license_entity

    validator = MagicMock()
    if validator_error:
        validator.validate_license.side_effect = validator_error
    else:
        validator.validate_license.return_value = validator_result or {
            "customer_name": "Acme",
            "expires_at": "2099-01-01",
            "hardware_hash": "hw",
            "license_type": "Lifetime",
        }

    return ValidateLicenseUseCase(repo, validator), repo, validator


def test_valid_paid_license_returns_enriched_details():
    use_case, repo, validator = _make_use_case(
        license_entity=MagicMock(license_key="somekey")
    )

    result = use_case.verify_existing_license()

    assert result["is_trial"] is False
    assert result["customer_name"] == "Acme"
    assert result["days_remaining"] is not None


def test_trial_license_key_marked_as_trial():
    use_case, repo, validator = _make_use_case(
        license_entity=MagicMock(license_key="trial-key"),
        validator_result={
            "customer_name": "Trial Co",
            "expires_at": "2026-08-01",
            "hardware_hash": "hw",
            "license_type": "15-Day Trial",
        },
    )

    result = use_case.verify_existing_license()

    assert result is not None
    assert result["is_trial"] is True
    assert result["license_type"] == "15-Day Trial"
    assert result["days_remaining"] >= 0


def test_no_license_installed_returns_none():
    use_case, repo, validator = _make_use_case(license_entity=None)

    result = use_case.verify_existing_license()

    assert result is None
    validator.validate_license.assert_not_called()


def test_expired_license_returns_none():
    use_case, repo, validator = _make_use_case(
        license_entity=MagicMock(license_key="expired-key"),
        validator_error=LicenseValidationError("expired"),
    )

    result = use_case.verify_existing_license()
    assert result is None


def test_has_stored_license():
    use_case, repo, validator = _make_use_case(license_entity=MagicMock(license_key="k"))
    assert use_case.has_stored_license() is True

    use_case, repo, validator = _make_use_case(license_entity=None)
    assert use_case.has_stored_license() is False


def test_activate_new_license_enriches_trial_details():
    use_case, repo, validator = _make_use_case(
        validator_result={
            "customer_name": "Trial Co",
            "expires_at": "2026-08-01",
            "hardware_hash": "hw",
            "license_type": "1-Month Trial",
        },
    )
    repo.save_license.return_value = True

    result = use_case.activate_new_license("new-key")

    assert result["is_trial"] is True
    assert result["license_type"] == "1-Month Trial"
    repo.save_license.assert_called_once()
