"""
Offline trial management for TileVision AI.

Every new installation gets a 15-day free trial with no license key
required. This module owns that lifecycle: starting the trial on first
run, computing days remaining on each subsequent run, and detecting two
common ways client-side trials get abused:

  1. System clock rollback — setting the OS clock backwards to make an
     expired trial look "not yet expired" again.
  2. Hardware copy — copying the encrypted trial state file to a different
     machine to reset/extend it (the file is unreadable there anyway per
     EncryptedLicenseStore's hardware-bound key derivation, which itself
     serves as a tamper signal).

Neither of these is disclosed to the caller beyond a boolean
`is_tampered` flag deliberately — the goal is a UI message like "trial
data appears invalid" without exact detection mechanics, avoiding
scenarios where the response reveals exactly how to route around it.
"""

from dataclasses import dataclass
from datetime import datetime, timezone
import logging
from typing import Optional

from src.licensing.crypto_store import EncryptedLicenseStore
from src.licensing.hardware import get_machine_fingerprint

logger = logging.getLogger("tilevision.licensing.trial_manager")

TRIAL_DURATION_DAYS = 15

# Small grace window for legitimate clock drift (NTP sync, timezone/DST
# changes crossing midnight, etc.) before treating a backwards clock jump
# as deliberate rollback tampering.
_CLOCK_ROLLBACK_GRACE_SECONDS = 6 * 60 * 60  # 6 hours


@dataclass
class TrialStatus:
    """Snapshot of the current trial state for a UI to render."""

    is_active: bool
    is_expired: bool
    is_tampered: bool
    days_remaining: int
    start_date: Optional[str]  # ISO date string


class TrialManager:
    """
    Manages the offline 15-day trial lifecycle using encrypted, hardware-
    bound local storage (see EncryptedLicenseStore).
    """

    def __init__(self, store: Optional[EncryptedLicenseStore] = None) -> None:
        """
        Args:
            store: Optional EncryptedLicenseStore override (for testing).
                Defaults to the standard hidden ProgramData-based store.
        """
        self._store = store or EncryptedLicenseStore()

    def get_or_start_trial(self) -> TrialStatus:
        """
        Load the existing trial record, or create a new one on first run.

        Returns:
            The current TrialStatus. If tampering is detected (clock
            rollback or a hardware-mismatched/corrupted state file), the
            trial is reported as tampered and inactive rather than reset —
            deleting the file and starting a fresh trial is NOT done
            automatically, since that would itself be an easy bypass
            (delete the file, get 15 more days).
        """
        now = datetime.now(timezone.utc)

        try:
            state = self._store.read()
        except ValueError:
            # File exists but couldn't be decrypted: either corrupted, or
            # copied from a different machine (different derived key).
            # Treat as tampered rather than silently starting a new trial.
            logger.warning("Trial state file could not be decrypted — treating as tampered.")
            return TrialStatus(
                is_active=False, is_expired=True, is_tampered=True,
                days_remaining=0, start_date=None,
            )

        if state is None:
            return self._start_new_trial(now)

        return self._evaluate_existing_trial(state, now)

    def get_status(self) -> TrialStatus:
        """
        Read-only check of trial status without starting a new trial if
        none exists yet (used for display purposes where starting a trial
        as a side effect of merely checking would be surprising).

        Returns:
            The current TrialStatus, or an inactive/non-expired status with
            days_remaining=0 if no trial has ever been started.
        """
        try:
            state = self._store.read()
        except ValueError:
            return TrialStatus(
                is_active=False, is_expired=True, is_tampered=True,
                days_remaining=0, start_date=None,
            )

        if state is None:
            return TrialStatus(
                is_active=False, is_expired=False, is_tampered=False,
                days_remaining=0, start_date=None,
            )

        return self._evaluate_existing_trial(state, datetime.now(timezone.utc), persist=False)

    # ── Internal ─────────────────────────────────────────────────────────

    def _start_new_trial(self, now: datetime) -> TrialStatus:
        record = {
            "start_date": now.isoformat(),
            "last_seen": now.isoformat(),
            "hardware_fingerprint": get_machine_fingerprint(),
        }
        self._store.write(record)
        logger.info(f"New {TRIAL_DURATION_DAYS}-day trial started at {now.isoformat()}.")
        return TrialStatus(
            is_active=True, is_expired=False, is_tampered=False,
            days_remaining=TRIAL_DURATION_DAYS, start_date=now.date().isoformat(),
        )

    def _evaluate_existing_trial(
        self, state: dict, now: datetime, persist: bool = True
    ) -> TrialStatus:
        try:
            start_date = datetime.fromisoformat(state["start_date"])
            last_seen = datetime.fromisoformat(state["last_seen"])
            recorded_hw = state.get("hardware_fingerprint", "")
        except (KeyError, ValueError) as e:
            logger.warning(f"Trial state file is malformed — treating as tampered: {e}")
            return TrialStatus(
                is_active=False, is_expired=True, is_tampered=True,
                days_remaining=0, start_date=None,
            )

        current_hw = get_machine_fingerprint()
        if recorded_hw and recorded_hw != current_hw:
            # Decrypted successfully (same machine key) but the recorded
            # fingerprint disagrees — extremely unlikely in practice, but
            # treat defensively as tampered.
            logger.warning("Trial hardware fingerprint mismatch — treating as tampered.")
            return TrialStatus(
                is_active=False, is_expired=True, is_tampered=True,
                days_remaining=0, start_date=start_date.date().isoformat(),
            )

        # Clock rollback detection: if "now" is meaningfully earlier than
        # the last time we legitimately observed the clock, someone (or
        # something) turned the clock back.
        if now < last_seen and (last_seen - now).total_seconds() > _CLOCK_ROLLBACK_GRACE_SECONDS:
            logger.warning("System clock rollback detected — treating trial as tampered.")
            return TrialStatus(
                is_active=False, is_expired=True, is_tampered=True,
                days_remaining=0, start_date=start_date.date().isoformat(),
            )

        # Advance last_seen (monotonic high-water mark) and persist, so a
        # later rollback attempt is measured against the latest genuine
        # time we've observed, not the original start date.
        effective_now = max(now, last_seen)
        elapsed_days = (effective_now - start_date).days
        days_remaining = max(0, TRIAL_DURATION_DAYS - elapsed_days)
        is_expired = days_remaining <= 0

        if persist:
            state["last_seen"] = effective_now.isoformat()
            self._store.write(state)

        return TrialStatus(
            is_active=not is_expired,
            is_expired=is_expired,
            is_tampered=False,
            days_remaining=days_remaining,
            start_date=start_date.date().isoformat(),
        )
