"""
Password-protected SQLite storage for TileVision AI.

The tiles catalogue database is encrypted at rest (AES-256-GCM) so casual
tools cannot open tables.db in DB Browser. While the app is running a
decrypted working copy exists; on exit it is re-encrypted.

The encryption key is derived from:
  - Optional TILEVISION_DB_PASSWORD environment variable (vendor-set password), or
  - This machine's hardware fingerprint (automatic per-PC protection).
"""

from __future__ import annotations

import logging
import os
from pathlib import Path

from cryptography.hazmat.primitives.ciphers.aead import AESGCM
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
from cryptography.hazmat.primitives import hashes

from src.licensing.hardware import get_machine_fingerprint

logger = logging.getLogger("tilevision.data.db_protection")

_DB_PEPPER = b"TileVisionAI-Database-v1"
_PBKDF2_ITERATIONS = 200_000
_SALT = b"TileVisionAI-DB-Salt-2026"
_NONCE_SIZE = 12
_ENCRYPTED_SUFFIX = ".enc"


def encrypted_db_path(db_path: Path) -> Path:
    return Path(f"{db_path}{_ENCRYPTED_SUFFIX}")


def resolve_db_password() -> str:
    """Return vendor password or machine-bound default."""
    env_password = os.environ.get("TILEVISION_DB_PASSWORD", "").strip()
    if env_password:
        return env_password
    return get_machine_fingerprint()


def derive_db_key(password: str) -> bytes:
    kdf = PBKDF2HMAC(
        algorithm=hashes.SHA256(),
        length=32,
        salt=_SALT,
        iterations=_PBKDF2_ITERATIONS,
    )
    return kdf.derive(password.encode("utf-8") + _DB_PEPPER)


def encrypt_file(source: Path, destination: Path, key: bytes) -> None:
    plaintext = source.read_bytes()
    aesgcm = AESGCM(key)
    nonce = os.urandom(_NONCE_SIZE)
    ciphertext = aesgcm.encrypt(nonce, plaintext, associated_data=None)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_bytes(nonce + ciphertext)


def decrypt_file(source: Path, destination: Path, key: bytes) -> None:
    raw = source.read_bytes()
    nonce, ciphertext = raw[:_NONCE_SIZE], raw[_NONCE_SIZE:]
    aesgcm = AESGCM(key)
    plaintext = aesgcm.decrypt(nonce, ciphertext, associated_data=None)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_bytes(plaintext)


def prepare_working_database(db_path: Path, password: str | None = None) -> None:
    """
    Ensure a decrypted working database exists at db_path.

    If only the encrypted file exists, decrypt it. If neither exists, SQLite
    will create a new database on first connection.
    """
    db_path = Path(db_path)
    enc_path = encrypted_db_path(db_path)
    key = derive_db_key(password or resolve_db_password())

    if db_path.exists():
        return

    if enc_path.exists():
        logger.info("Decrypting protected database: %s", enc_path.name)
        decrypt_file(enc_path, db_path, key)
        return

    logger.debug("No existing database — a new encrypted catalogue DB will be created.")


def seal_database(db_path: Path, password: str | None = None) -> None:
    """Encrypt the working database and remove the plain file."""
    db_path = Path(db_path)
    if not db_path.exists():
        return

    enc_path = encrypted_db_path(db_path)
    key = derive_db_key(password or resolve_db_password())
    logger.info("Encrypting catalogue database at rest: %s", enc_path.name)
    encrypt_file(db_path, enc_path, key)
    db_path.unlink(missing_ok=True)
