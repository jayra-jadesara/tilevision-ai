"""Tests for ValidateLicenseUseCase's offline trial fallback integration."""

import sys
from pathlib import Path
from unittest.mock import MagicMock

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.core.use_cases.validate_license import ValidateLicenseUseCase
from src.licensing.trial_manager import TrialStatus
from src.licensing.validator import LicenseValidationError


def _make_use_case(license_entity=None, validator_result=None, validator_error=None, trial_status=None):
    repo = MagicMock()
    repo.get_license.return_value = license_entity

    validator = MagicMock()
    if validator_error:
        validator.validate_license.side_effect = validator_error
    else:
        validator.validate_license.return_value = validator_result or {
            "customer_name": "Acme", "expires_at": "2099-01-01",
            "hardware_hash": "hw", "license_type": "Lifetime",
        }

    trial_manager = MagicMock()
    trial_manager.get_status.return_value = trial_status or TrialStatus(
        is_active=False, is_expired=True, is_tampered=False, days_remaining=0, start_date=None
    )

    return ValidateLicenseUseCase(repo, validator, trial_manager), repo, validator, trial_manager


def test_valid_paid_license_short_circuits_trial_check():
    use_case, repo, validator, trial_manager = _make_use_case(
        license_entity=MagicMock(license_key="somekey")
    )

    result = use_case.verify_existing_license()

    assert result["is_trial"] is False
    assert result["customer_name"] == "Acme"
    trial_manager.get_status.assert_not_called()


def test_no_license_falls_back_to_active_trial():
    active_trial = TrialStatus(
        is_active=True, is_expired=False, is_tampered=False, days_remaining=7, start_date="2026-01-01"
    )
    use_case, repo, validator, trial_manager = _make_use_case(
        license_entity=None, trial_status=active_trial
    )

    result = use_case.verify_existing_license()

    assert result is not None
    assert result["is_trial"] is True
    assert result["days_remaining"] == 7
    assert result["license_type"] == "15-Day Trial"


def test_expired_trial_and_no_license_returns_none():
    expired_trial = TrialStatus(
        is_active=False, is_expired=True, is_tampered=False, days_remaining=0, start_date="2026-01-01"
    )
    use_case, repo, validator, trial_manager = _make_use_case(
        license_entity=None, trial_status=expired_trial
    )

    result = use_case.verify_existing_license()
    assert result is None


def test_tampered_trial_returns_none_even_if_not_marked_expired():
    tampered_trial = TrialStatus(
        is_active=True, is_expired=False, is_tampered=True, days_remaining=5, start_date="2026-01-01"
    )
    use_case, repo, validator, trial_manager = _make_use_case(
        license_entity=None, trial_status=tampered_trial
    )

    result = use_case.verify_existing_license()
    assert result is None


def test_invalid_installed_license_falls_back_to_trial_rather_than_hard_locking():
    """An expired/corrupted paid license shouldn't be worse than having none."""
    active_trial = TrialStatus(
        is_active=True, is_expired=False, is_tampered=False, days_remaining=3, start_date="2026-01-01"
    )
    use_case, repo, validator, trial_manager = _make_use_case(
        license_entity=MagicMock(license_key="expired-key"),
        validator_error=LicenseValidationError("expired"),
        trial_status=active_trial,
    )

    result = use_case.verify_existing_license()
    assert result is not None
    assert result["is_trial"] is True
    assert result["days_remaining"] == 3


def test_no_trial_started_yet_returns_none_without_starting():
    never_started = TrialStatus(
        is_active=False, is_expired=False, is_tampered=False,
        days_remaining=0, start_date=None,
    )
    use_case, repo, validator, trial_manager = _make_use_case(
        license_entity=None, trial_status=never_started
    )

    result = use_case.verify_existing_license()

    assert result is None
    trial_manager.get_or_start_trial.assert_not_called()


def test_start_trial_access_starts_and_returns_details():
    active_trial = TrialStatus(
        is_active=True, is_expired=False, is_tampered=False,
        days_remaining=15, start_date="2026-07-17",
    )
    use_case, repo, validator, trial_manager = _make_use_case()
    trial_manager.get_or_start_trial.return_value = active_trial

    result = use_case.start_trial_access()

    assert result is not None
    assert result["is_trial"] is True
    assert result["days_remaining"] == 15
    trial_manager.get_or_start_trial.assert_called_once()


def test_get_trial_status_delegates_to_trial_manager():
    use_case, repo, validator, trial_manager = _make_use_case()
    trial_manager.get_status.return_value = TrialStatus(
        is_active=True, is_expired=False, is_tampered=False, days_remaining=10, start_date="x"
    )

    status = use_case.get_trial_status()
    assert status.days_remaining == 10
    trial_manager.get_status.assert_called_once()
