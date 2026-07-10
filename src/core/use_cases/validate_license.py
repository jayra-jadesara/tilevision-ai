"""
License validation use case module for TileVision AI.

Orchestrates checking local licenses on startup and validating/installing new activation
keys provided by the user. Falls back to the offline 15-day trial (see
TrialManager) when no paid license is installed.
"""

from datetime import datetime
import logging
from typing import Dict, Any, Optional

from src.core.models import LicenseInfo
from src.data.repository_interface import ILicenseRepository
from src.licensing.validator import LicenseValidator, LicenseError
from src.licensing.hardware import get_machine_fingerprint
from src.licensing.trial_manager import TrialManager, TrialStatus

logger = logging.getLogger("tilevision.core.use_cases.validate_license")


class ValidateLicenseUseCase:
    """
    Use case to check and register software licenses offline, with an
    automatic 15-day trial fallback when no paid license is present.
    """

    def __init__(
        self,
        license_repository: ILicenseRepository,
        validator: LicenseValidator,
        trial_manager: Optional[TrialManager] = None,
    ) -> None:
        """
        Initialize the use case.

        Args:
            license_repository: Repository interface for database license access.
            validator: Cryptographic LicenseValidator service.
            trial_manager: Optional TrialManager for the offline trial
                fallback. Defaults to a standard TrialManager instance.
        """
        self._repo = license_repository
        self._validator = validator
        self._trial_manager = trial_manager or TrialManager()

    def verify_existing_license(self) -> Optional[Dict[str, Any]]:
        """
        Check if a valid, unexpired, hardware-locked license is currently
        installed. If not, falls back to the offline trial: if a trial is
        active, this still returns access details (with is_trial=True) so
        the app can proceed without forcing the activation dialog.

        Returns:
            A dict with license/trial details (customer_name, expires_at,
            is_trial, days_remaining if trial) if access is currently
            granted, None if the user must activate a license (no paid
            license AND no active/valid trial).
        """
        logger.info("Verifying installed offline license key...")
        license_entity = self._repo.get_license()

        if license_entity:
            try:
                license_details = self._validator.validate_license(license_entity.license_key)
                license_details["is_trial"] = False
                return license_details
            except LicenseError as e:
                logger.error(f"Installed license verification failed: {e}")
                # Fall through to trial check rather than immediately
                # locking the user out — an expired/invalid paid license
                # shouldn't be worse than having no license at all.

        logger.info("No valid paid license installed — checking trial status.")
        trial_status = self._trial_manager.get_or_start_trial()

        if trial_status.is_tampered:
            logger.warning("Trial data appears invalid (tampered or copied from another machine).")
            return None

        if not trial_status.is_active:
            logger.info("Trial has expired.")
            return None

        return {
            "customer_name": "Trial User",
            "expires_at": None,
            "hardware_hash": get_machine_fingerprint(),
            "license_type": "15-Day Trial",
            "is_trial": True,
            "days_remaining": trial_status.days_remaining,
        }

    def get_trial_status(self) -> TrialStatus:
        """
        Read-only trial status check (does not start a trial as a side
        effect), for display purposes (e.g. a "X days left" banner).

        Returns:
            The current TrialStatus.
        """
        return self._trial_manager.get_status()

    def activate_new_license(self, license_string: str) -> Dict[str, Any]:
        """
        Verify and install a new license key in the local repository.

        Args:
            license_string: Base64-encoded license key.

        Returns:
            The verified license metadata dictionary.

        Raises:
            LicenseError: If verification of the license string fails.
        """
        logger.info("Attempting to activate new license key...")
        
        # 1. Verify the license key cryptographically first
        license_details = self._validator.validate_license(license_string)
        
        # 2. Convert to LicenseInfo domain entity
        license_entity = LicenseInfo(
            license_key=license_string,
            hardware_hash=license_details["hardware_hash"],
            customer_name=license_details["customer_name"],
            expires_at=license_details["expires_at"],
            activated_date=datetime.now(),
        )
        
        # 3. Persist in database
        success = self._repo.save_license(license_entity)
        if not success:
            logger.error("Failed to save validated license key to database.")
            raise LicenseError("Database write error during activation.")
            
        logger.info("New license key successfully installed and activated.")
        return license_details

    def get_hardware_fingerprint(self) -> str:
        """
        Return the hardware fingerprint for this machine.

        This value must be shared with the license vendor to generate
        a hardware-locked offline license key.

        Returns:
            A 64-character hexadecimal SHA-256 hardware fingerprint string.
        """
        return get_machine_fingerprint()

    def validate_and_save(self, license_key: str) -> bool:
        """
        Convenience wrapper: validate a license key and save it if valid.

        Args:
            license_key: Raw license key string entered by the user.

        Returns:
            True if the key is valid and was saved, False otherwise.
        """
        try:
            self.activate_new_license(license_key)
            return True
        except Exception as e:
            logger.warning(f"validate_and_save failed: {e}")
            return False
