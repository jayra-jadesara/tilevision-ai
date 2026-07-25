"""Password gate for the vendor admin tool."""

from __future__ import annotations

import hashlib
import json
import secrets
from pathlib import Path

DEFAULT_ADMIN_PASSWORD = "raj!RAJ!"
_VENDOR_DIR = Path.home() / ".tilevision_ai_vendor"
_SETTINGS_PATH = _VENDOR_DIR / "admin_settings.json"
_PBKDF2_ITERATIONS = 200_000


def _hash_password(password: str, salt: bytes) -> str:
    digest = hashlib.pbkdf2_hmac(
        "sha256",
        password.encode("utf-8"),
        salt,
        _PBKDF2_ITERATIONS,
    )
    return digest.hex()


def _load_settings() -> dict:
    if not _SETTINGS_PATH.is_file():
        return {}
    try:
        return json.loads(_SETTINGS_PATH.read_text(encoding="utf-8"))
    except Exception:
        return {}


def _save_password_hash(password_hash: str, password_salt: str) -> None:
    _VENDOR_DIR.mkdir(parents=True, exist_ok=True)
    data = _load_settings()
    data["access_password_hash"] = password_hash
    data["access_password_salt"] = password_salt
    _SETTINGS_PATH.write_text(json.dumps(data, indent=2), encoding="utf-8")


def ensure_default_password_recorded() -> None:
    """Store a hash for the default password on first use (no plaintext on disk)."""
    data = _load_settings()
    if data.get("access_password_hash") and data.get("access_password_salt"):
        return
    salt = secrets.token_bytes(16)
    _save_password_hash(_hash_password(DEFAULT_ADMIN_PASSWORD, salt), salt.hex())


def verify_admin_password(password: str) -> bool:
    """Return True when the password matches the configured admin gate."""
    ensure_default_password_recorded()
    data = _load_settings()
    stored_hash = str(data.get("access_password_hash", ""))
    salt_hex = str(data.get("access_password_salt", ""))
    if not stored_hash or not salt_hex:
        return password == DEFAULT_ADMIN_PASSWORD
    try:
        salt = bytes.fromhex(salt_hex)
    except ValueError:
        return password == DEFAULT_ADMIN_PASSWORD
    return secrets.compare_digest(_hash_password(password, salt), stored_hash)
