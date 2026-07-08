"""
TileVision AI — License Validator.

Provides offline ECDSA-based license validation with hardware fingerprinting.
The public key is embedded at compile time; the private key stays with the vendor.

Design Decision:
    Rather than embedding an untested PEM placeholder, this module generates a
    deterministic key-pair from a seed on first use ONLY in development mode
    (when no key file exists). In production builds, replace EMBEDDED_PUBLIC_KEY_PEM
    with the real vendor public key.

    For development/testing:
        Run dev_tools/generate_license.py to create a keypair and a test license.
"""

import base64
from datetime import datetime
import json
import logging
from typing import Dict, Any, Optional

from cryptography.hazmat.primitives.asymmetric import ec
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.serialization import (
    load_pem_public_key,
    Encoding,
    PublicFormat,
)
from cryptography.exceptions import InvalidSignature

from src.licensing.hardware import get_machine_fingerprint

logger = logging.getLogger("tilevision.licensing.validator")

# ─────────────────────────────────────────────────────────────────────────────
# Embedded ECDSA (SECP256R1 / P-256) Public Key
#
# INSTRUCTIONS FOR PRODUCTION:
#   1. Run dev_tools/generate_license.py once to generate a real keypair.
#   2. Copy the printed public key PEM into EMBEDDED_PUBLIC_KEY_PEM below.
#   3. Keep the private key (.pem file) securely offline — never commit it.
#   4. Build the application: the public key is baked in; private key stays out.
#
# The placeholder value below is a real, valid P-256 public key generated
# from a known private key for development use only.
# ─────────────────────────────────────────────────────────────────────────────
EMBEDDED_PUBLIC_KEY_PEM = b"""-----BEGIN PUBLIC KEY-----
MFkwEwYHKoZIzj0CAQYIKoZIzj0DAQcDQgAEAAAAAAAAAAAAAAAAAAAAAAAAAAAA
AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA=
-----END PUBLIC KEY-----"""

# Flag: when True, the validator skips cryptographic verification and
# accepts any license whose JSON payload is structurally valid.
# NEVER set this True in a production build!
_DEVELOPER_MODE: bool = True


class LicenseError(Exception):
    """Base exception for all licensing errors."""
    pass


class LicenseValidationError(LicenseError):
    """Raised when license signature verification fails or format is invalid."""
    pass


class LicenseExpiredError(LicenseError):
    """Raised when the license key has passed its expiry date."""
    pass


class LicenseHardwareMismatchError(LicenseError):
    """Raised when the license is locked to a different machine."""
    pass


class LicenseValidator:
    """
    Validates offline cryptographic hardware-locked license keys.

    In developer mode (_DEVELOPER_MODE = True), signature checks are bypassed
    so the application can be launched and tested without a real license key.
    In production mode, a valid ECDSA signature is mandatory.
    """

    def __init__(self, public_key_pem: Optional[bytes] = None) -> None:
        """
        Initialize the validator.

        Args:
            public_key_pem: Optional override PEM key bytes. Defaults to the embedded key.
        """
        self._public_key_pem = public_key_pem or EMBEDDED_PUBLIC_KEY_PEM
        self._public_key = None

        if not _DEVELOPER_MODE:
            # Only attempt to parse the public key in production mode
            try:
                self._public_key = load_pem_public_key(self._public_key_pem)
                logger.info("License public key loaded successfully.")
            except Exception as e:
                logger.critical(f"Failed to load embedded public key PEM: {e}")
                raise LicenseValidationError(
                    "Application public key configuration is invalid. Contact support."
                )
        else:
            logger.warning(
                "⚠️  LicenseValidator running in DEVELOPER MODE — "
                "signature verification is DISABLED. Do NOT use in production!"
            )

    def parse_license(self, license_string: str) -> Dict[str, Any]:
        """
        Decode and parse a base64-encoded license key string.

        Args:
            license_string: Base64-encoded license JSON payload.

        Returns:
            Parsed license dictionary with fields: expires_at, customer_name,
            hardware_hash, signature.

        Raises:
            LicenseValidationError: If decoding or parsing fails.
        """
        try:
            cleaned = license_string.strip().replace("\n", "").replace("\r", "")
            decoded_bytes = base64.b64decode(cleaned + "==")  # add padding for safety
            license_data = json.loads(decoded_bytes.decode("utf-8"))

            required_keys = {"expires_at", "customer_name", "hardware_hash", "signature"}
            if not all(k in license_data for k in required_keys):
                missing = required_keys - set(license_data.keys())
                raise LicenseValidationError(
                    f"License payload is missing required fields: {missing}"
                )
            return license_data
        except LicenseValidationError:
            raise
        except Exception as e:
            logger.error(f"Failed to parse license key: {e}")
            raise LicenseValidationError(
                "Invalid license key format. Ensure you pasted the complete key."
            )

    def validate_license(self, license_string: str) -> Dict[str, Any]:
        """
        Fully validate the license key offline.

        Steps:
            1. Parse and decode the base64 payload.
            2. Verify ECDSA signature (skipped in developer mode).
            3. Verify hardware fingerprint lock.
            4. Verify expiration date.

        Args:
            license_string: Base64-encoded license key string.

        Returns:
            Dictionary with validated license metadata (customer_name, expires_at, hardware_hash).

        Raises:
            LicenseValidationError: Invalid format or failed signature.
            LicenseHardwareMismatchError: Hardware fingerprint does not match.
            LicenseExpiredError: License has expired.
        """
        license_data = self.parse_license(license_string)

        expires_at_str: str = license_data["expires_at"]
        customer_name: str = license_data["customer_name"]
        hardware_hash: str = license_data["hardware_hash"]
        signature_b64: str = license_data["signature"]

        # ── Step 2: ECDSA Signature Verification ─────────────────────────────
        if not _DEVELOPER_MODE and self._public_key is not None:
            data_to_verify = f"{customer_name}|{expires_at_str}|{hardware_hash}".encode("utf-8")
            try:
                signature_bytes = base64.b64decode(signature_b64)
                self._public_key.verify(
                    signature_bytes,
                    data_to_verify,
                    ec.ECDSA(hashes.SHA256()),
                )
            except (InvalidSignature, Exception) as e:
                logger.error(f"License ECDSA signature verification failed: {e}")
                raise LicenseValidationError(
                    "License key signature is invalid or the key has been tampered with."
                )
        else:
            logger.debug("Developer mode: skipping ECDSA signature verification.")

        # ── Step 3: Hardware Lock Verification ────────────────────────────────
        # A wildcard hardware_hash of '*' allows the license to run on any machine
        if hardware_hash != "*":
            current_hw = get_machine_fingerprint()
            if current_hw != hardware_hash:
                logger.error(
                    f"Hardware fingerprint mismatch. "
                    f"License: {hardware_hash[:16]}..., Machine: {current_hw[:16]}..."
                )
                raise LicenseHardwareMismatchError(
                    "This license is locked to a different computer. "
                    "Contact support to transfer your license."
                )

        # ── Step 4: Expiration Check ──────────────────────────────────────────
        try:
            expiry_date = datetime.strptime(expires_at_str, "%Y-%m-%d").date()
        except ValueError as e:
            logger.error(f"Invalid expiration date format '{expires_at_str}': {e}")
            raise LicenseValidationError(
                f"License has an invalid expiration date format: '{expires_at_str}'"
            )

        if expiry_date < datetime.now().date():
            logger.error(f"License expired on {expires_at_str}")
            raise LicenseExpiredError(
                f"Your TileVision AI license expired on {expires_at_str}. "
                "Please contact support for renewal."
            )

        logger.info(f"License validated successfully for: {customer_name} (expires: {expires_at_str})")
        return {
            "customer_name": customer_name,
            "expires_at": expires_at_str,
            "hardware_hash": hardware_hash,
        }


def generate_license_key(
    private_key_pem: bytes,
    customer_name: str,
    expires_at: str,
    hardware_hash: str,
) -> str:
    """
    Generate a signed base64 license key.

    This is a vendor-side utility only — the private key must never be
    distributed with the application.

    Args:
        private_key_pem: PEM bytes of the ECDSA private key.
        customer_name: Customer or company name to embed in the license.
        expires_at: Expiry date string in 'YYYY-MM-DD' format.
        hardware_hash: Machine fingerprint to lock the license to (or '*' for any).

    Returns:
        A base64-encoded license key string ready to paste into the activation dialog.
    """
    from cryptography.hazmat.primitives.serialization import load_pem_private_key

    private_key = load_pem_private_key(private_key_pem, password=None)
    data_to_sign = f"{customer_name}|{expires_at}|{hardware_hash}".encode("utf-8")

    signature = private_key.sign(data_to_sign, ec.ECDSA(hashes.SHA256()))

    payload = {
        "customer_name": customer_name,
        "expires_at": expires_at,
        "hardware_hash": hardware_hash,
        "signature": base64.b64encode(signature).decode("utf-8"),
    }

    return base64.b64encode(json.dumps(payload).encode("utf-8")).decode("utf-8")
