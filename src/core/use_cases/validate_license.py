"""
License validation use case module for TileVision AI.

Orchestrates checking local licenses on startup and validating/installing new activation
keys provided by the user.
"""

from datetime import datetime
import logging
from typing import Dict, Any, Optional

from src.core.models import LicenseInfo
from src.data.repository_interface import ILicenseRepository
from src.licensing.validator import LicenseValidator, LicenseError
from src.licensing.hardware import get_machine_fingerprint

logger = logging.getLogger("tilevision.core.use_cases.validate_license")


class ValidateLicenseUseCase:
    """
    Use case to check and register software licenses offline.
    """

    def __init__(
        self, license_repository: ILicenseRepository, validator: LicenseValidator
    ) -> None:
        """
        Initialize the use case.

        Args:
            license_repository: Repository interface for database license access.
            validator: Cryptographic LicenseValidator service.
        """
        self._repo = license_repository
        self._validator = validator

    def verify_existing_license(self) -> Optional[Dict[str, Any]]:
        """
        Check if a valid, unexpired, hardware-locked license is currently installed.

        Returns:
            A dictionary containing license details (customer_name, expires_at) if valid,
            None otherwise.
        """
        logger.info("Verifying installed offline license key...")
        license_entity = self._repo.get_license()
        if not license_entity:
            logger.warning("No license key found in database.")
            return None

        try:
            # Re-verify the license string cryptographically and verify hardware locking
            license_details = self._validator.validate_license(license_entity.license_key)
            return license_details
        except LicenseError as e:
            logger.error(f"Installed license verification failed: {e}")
            return None

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
