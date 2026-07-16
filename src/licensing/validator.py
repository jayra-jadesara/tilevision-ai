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
from datetime import datetime, timedelta
import json
import logging
import os
import uuid
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
from src.licensing.revocation import is_license_revoked, load_revoked_license_ids

logger = logging.getLogger("tilevision.licensing.validator")

# ─────────────────────────────────────────────────────────────────────────────
# Embedded ECDSA (SECP256R1 / P-256) Public Key
#
# INSTRUCTIONS FOR PRODUCTION:
#   1. Run admin_tool/main.py -> "Generate New Keypair" to create a real
#      keypair for YOUR business (never reuse this dev one for real sales).
#   2. Copy the printed public key PEM into EMBEDDED_PUBLIC_KEY_PEM below.
#   3. Keep the private key (.pem file) securely offline — never commit it.
#   4. Build the application: the public key is baked in; private key stays out.
#
# The key below IS a real, valid P-256 public key (unlike an earlier
# placeholder that used dummy all-zero bytes and crashed on load) — its
# matching private key is dev_tools/dev_private_key.pem, committed
# intentionally so the app runs and generates test/trial licenses
# out of the box for development. It is NOT secure for production: anyone
# with this repo can mint licenses that validate against it. Replace both
# halves before shipping to real customers.
# ─────────────────────────────────────────────────────────────────────────────
EMBEDDED_PUBLIC_KEY_PEM = b"""-----BEGIN PUBLIC KEY-----
MFkwEwYHKoZIzj0CAQYIKoZIzj0DAQcDQgAE8oV3FXs5Mzc25mxzta6K9Snbtxaa
7iXvMu4Srxuht3u0B7qFavlLuoYhrmqD8zV9sBY5QyJj5Yir8iZhTBGwrA==
-----END PUBLIC KEY-----"""

def _resolve_developer_mode() -> bool:
    """
    Resolve whether developer mode is enabled from the environment.

    Exposed as a standalone function (rather than only inlining the check at
    module scope) so tests can exercise the env-var parsing logic directly
    without reloading this module — reloading would create new class
    objects for LicenseValidationError etc., breaking isinstance checks
    against references imported before the reload.
    """
    return os.environ.get("TILEVISION_DEV_MODE") == "1"


# SECURITY: Developer mode disables cryptographic signature verification so
# the app can be run/tested without a real signed license. This is gated
# behind an explicit environment variable and defaults OFF (secure), unlike
# a hardcoded flag which would silently ship with verification disabled if
# anyone forgot to flip it back before building a release. Production
# builds must never set TILEVISION_DEV_MODE, and CI/packaging should assert
# it is unset before producing a release artifact.
_DEVELOPER_MODE: bool = _resolve_developer_mode()

if _DEVELOPER_MODE:
    logger.warning(
        "⚠️⚠️⚠️  TILEVISION_DEV_MODE=1 is set — license signature verification "
        "and hardware-lock bypass ('*') are ENABLED. This must NEVER be set "
        "in a production build or shipped installer. ⚠️⚠️⚠️"
    )


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


class LicenseRevokedError(LicenseError):
    """Raised when the license ID appears on the vendor revocation list."""
    pass


# License types: vendor-issued trials and official licences.
# Enforcement is always by expires_at (signed), not by the label alone.
LICENSE_TYPE_DURATIONS_DAYS: Dict[str, Optional[int]] = {
    "15-Day Trial": 15,
    "1-Month Trial": 30,
    "2-Month Trial": 60,
    "3-Month Trial": 90,
    "6-Month Trial": 180,
    "1-Year": 365,
    "3-Year": 1095,
    # Legacy aliases kept for older keys/admin records
    "30-Day": 30,
    "90-Day": 90,
    "Lifetime": None,
}

# Types shown first in the vendor admin tool (trials, then official).
VENDOR_LICENSE_TYPES: tuple[str, ...] = (
    "15-Day Trial",
    "1-Month Trial",
    "2-Month Trial",
    "3-Month Trial",
    "6-Month Trial",
    "1-Year",
    "3-Year",
    "Lifetime",
)
LIFETIME_EXPIRY_SENTINEL = "9999-12-31"
LICENSE_PAYLOAD_VERSION = 2


def _signing_message(payload: Dict[str, Any]) -> bytes:
    """Build the signed message for v1 (legacy) or v2 (license_id) payloads."""
    version = int(payload.get("version", 1))
    if version >= 2 and payload.get("license_id"):
        return (
            f"{payload['license_id']}|{payload['customer_name']}|"
            f"{payload['expires_at']}|{payload['hardware_hash']}|"
            f"{payload.get('license_type', 'Custom')}"
        ).encode("utf-8")
    return (
        f"{payload['customer_name']}|{payload['expires_at']}|{payload['hardware_hash']}"
    ).encode("utf-8")


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
            license_data.setdefault("license_type", "Custom")
            license_data.setdefault("version", 1 if "license_id" not in license_data else 2)
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
        license_id: Optional[str] = license_data.get("license_id")

        if is_license_revoked(license_id):
            logger.error("License ID %s is on the revocation list.", license_id)
            raise LicenseRevokedError(
                "This license has been cancelled by the vendor. Contact support."
            )

        # ── Step 2: ECDSA Signature Verification ─────────────────────────────
        if not _DEVELOPER_MODE and self._public_key is not None:
            data_to_verify = _signing_message(license_data)
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
        # A wildcard hardware_hash of '*' is a development/testing convenience
        # only (lets a license run on any machine without regenerating it for
        # every test VM). It must NEVER be honored outside dev mode — an
        # unconditional wildcard bypass would be a universal master key that
        # unlocks the product on any customer's machine.
        if hardware_hash == "*" and not _DEVELOPER_MODE:
            logger.error(
                "License uses wildcard hardware_hash='*' but the app is not "
                "running in developer mode. Rejecting as invalid."
            )
            raise LicenseValidationError(
                "This license key is not valid for production use."
            )

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
            "license_type": license_data.get("license_type", "Custom"),
            "license_id": license_id,
        }


def compute_expiry_date(license_type: str, from_date: Optional[datetime] = None) -> str:
    """
    Compute the expiry date string for a given license type.

    Args:
        license_type: One of LICENSE_TYPE_DURATIONS_DAYS's keys
            ("15-Day Trial", "30-Day", "90-Day", "1-Year", "Lifetime").
        from_date: The activation/generation date to count from. Defaults
            to now.

    Returns:
        A 'YYYY-MM-DD' expiry date string.

    Raises:
        ValueError: If license_type is not recognized.
    """
    if license_type not in LICENSE_TYPE_DURATIONS_DAYS:
        raise ValueError(
            f"Unknown license type '{license_type}'. "
            f"Must be one of: {sorted(LICENSE_TYPE_DURATIONS_DAYS.keys())}"
        )

    duration_days = LICENSE_TYPE_DURATIONS_DAYS[license_type]
    if duration_days is None:  # Lifetime
        return LIFETIME_EXPIRY_SENTINEL

    base = from_date or datetime.now()
    expiry = base + timedelta(days=duration_days)
    return expiry.strftime("%Y-%m-%d")


def generate_license_key(
    private_key_pem: bytes,
    customer_name: str,
    expires_at: str,
    hardware_hash: str,
    license_type: str = "Custom",
    license_id: Optional[str] = None,
) -> str:
    """
    Generate a signed base64 license key.

    This is a vendor-side utility only — the private key must never be
    distributed with the application. Used by the standalone Admin License
    Manager tool (admin_tool/), never by the end-user application itself.

    Args:
        private_key_pem: PEM bytes of the ECDSA private key.
        customer_name: Customer or company name to embed in the license.
        expires_at: Expiry date string in 'YYYY-MM-DD' format. Use
            compute_expiry_date() to derive this from a license_type.
        hardware_hash: Machine fingerprint to lock the license to. Using
            '*' (any machine) is only honored by the validator when
            TILEVISION_DEV_MODE=1 — never use it for a real customer key.
        license_type: One of LICENSE_TYPE_DURATIONS_DAYS's keys, embedded
            for display/audit purposes. NOTE: the actual enforcement is
            driven entirely by expires_at, not this label — mismatching
            the two (e.g. "Lifetime" with a 30-day expires_at) won't
            confer any extra access.

    Returns:
        A base64-encoded license key string ready to paste into the activation dialog.
    """
    from cryptography.hazmat.primitives.serialization import load_pem_private_key

    private_key = load_pem_private_key(private_key_pem, password=None)
    license_id = license_id or str(uuid.uuid4())

    payload = {
        "version": LICENSE_PAYLOAD_VERSION,
        "license_id": license_id,
        "customer_name": customer_name,
        "expires_at": expires_at,
        "hardware_hash": hardware_hash,
        "license_type": license_type,
    }

    data_to_sign = _signing_message(payload)
    signature = private_key.sign(data_to_sign, ec.ECDSA(hashes.SHA256()))
    payload["signature"] = base64.b64encode(signature).decode("utf-8")

    return base64.b64encode(json.dumps(payload).encode("utf-8")).decode("utf-8")
