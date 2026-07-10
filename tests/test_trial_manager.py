"""Tests for TrialManager (15-day offline trial with tamper detection)."""

import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.licensing.crypto_store import EncryptedLicenseStore
from src.licensing.trial_manager import TrialManager, TRIAL_DURATION_DAYS


@pytest.fixture()
def manager(tmp_path, monkeypatch):
    import src.licensing.trial_manager as trial_manager_module
    import src.licensing.crypto_store as crypto_store_module

    monkeypatch.setattr(crypto_store_module, "get_machine_fingerprint", lambda: "test-machine-fp")
    monkeypatch.setattr(trial_manager_module, "get_machine_fingerprint", lambda: "test-machine-fp")

    store = EncryptedLicenseStore(storage_path=tmp_path / "trial.enc")
    return TrialManager(store=store)


def test_first_run_starts_a_fresh_trial(manager):
    status = manager.get_or_start_trial()
    assert status.is_active is True
    assert status.is_expired is False
    assert status.is_tampered is False
    assert status.days_remaining == TRIAL_DURATION_DAYS


def test_second_check_same_day_keeps_full_days_remaining(manager):
    manager.get_or_start_trial()
    status = manager.get_or_start_trial()
    assert status.days_remaining == TRIAL_DURATION_DAYS
    assert status.is_active is True


def test_trial_expires_after_duration(manager, tmp_path, monkeypatch):
    import src.licensing.trial_manager as tm_module

    # Start the trial "now"
    manager.get_or_start_trial()

    # Fast-forward: pretend it's TRIAL_DURATION_DAYS + 1 days later
    future = datetime.now(timezone.utc) + timedelta(days=TRIAL_DURATION_DAYS + 1)

    class FakeDatetime(datetime):
        @classmethod
        def now(cls, tz=None):
            return future

    monkeypatch.setattr(tm_module, "datetime", FakeDatetime)

    status = manager.get_or_start_trial()
    assert status.is_expired is True
    assert status.is_active is False
    assert status.days_remaining == 0


def test_partial_days_remaining_after_some_elapsed(manager, monkeypatch):
    import src.licensing.trial_manager as tm_module

    manager.get_or_start_trial()

    future = datetime.now(timezone.utc) + timedelta(days=5)

    class FakeDatetime(datetime):
        @classmethod
        def now(cls, tz=None):
            return future

    monkeypatch.setattr(tm_module, "datetime", FakeDatetime)

    status = manager.get_or_start_trial()
    assert status.days_remaining == TRIAL_DURATION_DAYS - 5
    assert status.is_active is True


def test_clock_rollback_is_detected(manager, monkeypatch):
    import src.licensing.trial_manager as tm_module

    # Start trial, then advance forward a few days to establish a
    # "last_seen" high-water mark.
    manager.get_or_start_trial()

    forward = datetime.now(timezone.utc) + timedelta(days=3)

    class ForwardDatetime(datetime):
        @classmethod
        def now(cls, tz=None):
            return forward

    monkeypatch.setattr(tm_module, "datetime", ForwardDatetime)
    manager.get_or_start_trial()  # records last_seen = forward

    # Now simulate the user turning the clock BACK to before start.
    rolled_back = datetime.now(timezone.utc) - timedelta(days=1)

    class RolledBackDatetime(datetime):
        @classmethod
        def now(cls, tz=None):
            return rolled_back

    monkeypatch.setattr(tm_module, "datetime", RolledBackDatetime)

    status = manager.get_or_start_trial()
    assert status.is_tampered is True
    assert status.is_expired is True
    assert status.is_active is False


def test_small_clock_drift_within_grace_period_is_not_flagged(manager, monkeypatch):
    """A few minutes of backward drift (NTP sync, etc.) shouldn't trigger tamper detection."""
    import src.licensing.trial_manager as tm_module

    manager.get_or_start_trial()

    slightly_back = datetime.now(timezone.utc) - timedelta(minutes=2)

    class SlightlyBackDatetime(datetime):
        @classmethod
        def now(cls, tz=None):
            return slightly_back

    monkeypatch.setattr(tm_module, "datetime", SlightlyBackDatetime)

    status = manager.get_or_start_trial()
    assert status.is_tampered is False


def test_copying_trial_file_to_different_machine_is_detected(tmp_path, monkeypatch):
    """Simulates copying the encrypted trial file to a different PC."""
    import src.licensing.trial_manager as tm_module
    import src.licensing.crypto_store as cs_module

    store_path = tmp_path / "trial.enc"

    monkeypatch.setattr(cs_module, "get_machine_fingerprint", lambda: "machine-A")
    monkeypatch.setattr(tm_module, "get_machine_fingerprint", lambda: "machine-A")
    manager_a = TrialManager(store=EncryptedLicenseStore(storage_path=store_path))
    manager_a.get_or_start_trial()

    # Same file, different machine fingerprint this time (simulating copy).
    monkeypatch.setattr(cs_module, "get_machine_fingerprint", lambda: "machine-B")
    monkeypatch.setattr(tm_module, "get_machine_fingerprint", lambda: "machine-B")
    manager_b = TrialManager(store=EncryptedLicenseStore(storage_path=store_path))
    status = manager_b.get_or_start_trial()

    assert status.is_tampered is True
    assert status.is_active is False


def test_get_status_does_not_start_a_trial_as_a_side_effect(manager):
    status = manager.get_status()
    assert status.is_active is False
    assert status.is_expired is False
    assert status.is_tampered is False
    assert status.days_remaining == 0

    # Confirm no trial was actually started
    status2 = manager.get_status()
    assert status2.start_date is None
