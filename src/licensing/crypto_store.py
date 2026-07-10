"""
Encrypted local license/trial storage for TileVision AI.

Stores license and trial state as an AES-256-GCM encrypted blob on disk,
rather than plain text or an easily-copied SQLite row. The encryption key
is derived (via PBKDF2-HMAC-SHA256) from this machine's hardware
fingerprint combined with a static application pepper, so:

  - The file is unreadable without running on the same physical machine it
    was created on (copying the file to another PC yields a different
    derived key, so decryption fails and the copy is treated as invalid).
  - The stored fields (trial start date, license type, expiry, etc.) are
    never visible in plain text on disk.

This is defense-in-depth alongside the ECDSA-signed license key mechanism
in validator.py, not a replacement for it — a sufficiently motivated
attacker with full control of their own machine can defeat any purely
client-side protection. The goal is to raise the bar for casual copying
and tampering, per the spec's requirement to avoid storing license data in
plain text.

Storage location:
    Windows: %PROGRAMDATA%\\TileVisionAI\\.lic\\state.enc (hidden folder)
    Fallback (non-Windows dev/test): ~/.tilevision_ai/.lic/state.enc
"""

import base64
import json
import logging
import os
from pathlib import Path
from typing import Any, Dict, Optional

from cryptography.hazmat.primitives.ciphers.aead import AESGCM
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
from cryptography.hazmat.primitives import hashes

from src.licensing.hardware import get_machine_fingerprint

logger = logging.getLogger("tilevision.licensing.crypto_store")

# Static application pepper mixed into key derivation. This does NOT need to
# be kept secret for the scheme's security to hold (the hardware fingerprint
# is the actual secret-ish component); its purpose is just to make the
# derived key application-specific rather than reusable for any hardware
# fingerprint-based key elsewhere.
_APP_PEPPER = b"TileVisionAI-License-Store-v1"

_PBKDF2_ITERATIONS = 200_000
_SALT = b"TileVisionAI-Static-Salt-2026"  # fixed salt: fine since the "password" (HW fingerprint) is already high-entropy
_NONCE_SIZE = 12  # 96-bit nonce, standard for AES-GCM


def _get_storage_dir() -> Path:
    """
    Resolve the hidden storage directory for encrypted license/trial state.

    Returns:
        Path to the storage directory (created if it doesn't exist).
    """
    program_data = os.environ.get("PROGRAMDATA")
    if program_data:
        base = Path(program_data) / "TileVisionAI" / ".lic"
    else:
        # Non-Windows dev/test fallback
        base = Path.home() / ".tilevision_ai" / ".lic"

    base.mkdir(parents=True, exist_ok=True)
    _try_hide_folder(base)
    return base


def _try_hide_folder(path: Path) -> None:
    """Best-effort: set the Windows FILE_ATTRIBUTE_HIDDEN flag on a folder."""
    try:
        import ctypes

        FILE_ATTRIBUTE_HIDDEN = 0x02
        ctypes.windll.kernel32.SetFileAttributesW(str(path), FILE_ATTRIBUTE_HIDDEN)
    except Exception:
        # Not on Windows, or the call failed — not fatal, storage still works,
        # it just won't have the hidden attribute set.
        pass


def _derive_key(hardware_fingerprint: str) -> bytes:
    """Derive a 256-bit AES key from the hardware fingerprint via PBKDF2."""
    kdf = PBKDF2HMAC(
        algorithm=hashes.SHA256(),
        length=32,
        salt=_SALT,
        iterations=_PBKDF2_ITERATIONS,
    )
    return kdf.derive(hardware_fingerprint.encode("utf-8") + _APP_PEPPER)


class EncryptedLicenseStore:
    """
    Encrypted, hardware-bound key-value store for license/trial state.

    A thin JSON-over-AES-256-GCM persistence layer. Callers (TrialManager,
    license repository decorators, etc.) read/write plain Python dicts;
    this class handles deriving the key, encrypting, and writing to disk
    (and the reverse on read).
    """

    def __init__(self, storage_path: Optional[Path] = None) -> None:
        """
        Args:
            storage_path: Optional override path for the encrypted state
                file (primarily for testing). Defaults to the hidden
                ProgramData-based location.
        """
        self._storage_path = storage_path or (_get_storage_dir() / "state.enc")

    def read(self) -> Optional[Dict[str, Any]]:
        """
        Decrypt and return the stored state dict.

        Returns:
            The decrypted dict, or None if no state file exists.

        Raises:
            ValueError: If the file exists but cannot be decrypted (wrong
                machine, corrupted, or tampered with) — this is treated as
                a signal, not silently swallowed, so callers can respond to
                tampering explicitly rather than quietly starting over.
        """
        if not self._storage_path.exists():
            return None

        try:
            raw = self._storage_path.read_bytes()
            nonce, ciphertext = raw[:_NONCE_SIZE], raw[_NONCE_SIZE:]

            key = _derive_key(get_machine_fingerprint())
            aesgcm = AESGCM(key)
            plaintext = aesgcm.decrypt(nonce, ciphertext, associated_data=None)
            return json.loads(plaintext.decode("utf-8"))
        except Exception as e:
            logger.error(
                f"Failed to decrypt license/trial state (file may be corrupted, "
                f"tampered with, or copied from another machine): {e}"
            )
            raise ValueError("License/trial state could not be decrypted.") from e

    def write(self, data: Dict[str, Any]) -> None:
        """
        Encrypt and persist a state dict, overwriting any existing file.

        Args:
            data: JSON-serializable dict to encrypt and store.
        """
        key = _derive_key(get_machine_fingerprint())
        aesgcm = AESGCM(key)
        nonce = os.urandom(_NONCE_SIZE)

        plaintext = json.dumps(data).encode("utf-8")
        ciphertext = aesgcm.encrypt(nonce, plaintext, associated_data=None)

        self._storage_path.parent.mkdir(parents=True, exist_ok=True)
        self._storage_path.write_bytes(nonce + ciphertext)
        logger.debug(f"Encrypted state written to {self._storage_path}")

    def delete(self) -> None:
        """Remove the stored state file, if present."""
        try:
            self._storage_path.unlink(missing_ok=True)
        except Exception as e:
            logger.warning(f"Failed to delete license/trial state file: {e}")

    def exists(self) -> bool:
        return self._storage_path.exists()
