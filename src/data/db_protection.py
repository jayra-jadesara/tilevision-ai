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

    A corrupt ``.enc`` (interrupted seal during update hard-exit) must not
    prevent startup — fall back to an existing plaintext DB or start fresh.
    """
    db_path = Path(db_path)
    enc_path = encrypted_db_path(db_path)
    key = derive_db_key(password or resolve_db_password())

    if db_path.exists():
        return

    if enc_path.exists():
        logger.info("Decrypting protected database: %s", enc_path.name)
        try:
            decrypt_file(enc_path, db_path, key)
            return
        except Exception:
            logger.exception(
                "Failed to decrypt %s — moving aside and starting a recoverable DB.",
                enc_path.name,
            )
            try:
                corrupt = enc_path.with_suffix(enc_path.suffix + ".corrupt")
                enc_path.replace(corrupt)
            except OSError:
                logger.exception("Could not quarantine corrupt encrypted database")
            return

    logger.debug("No existing database — a new encrypted catalogue DB will be created.")


def _sidecar_paths(db_path: Path) -> tuple[Path, Path]:
    """SQLite WAL / shared-memory sidecars next to the working DB."""
    return Path(f"{db_path}-wal"), Path(f"{db_path}-shm")


def remove_sqlite_sidecars(db_path: Path) -> None:
    """Delete leftover ``-wal`` / ``-shm`` files after a successful seal."""
    for side in _sidecar_paths(Path(db_path)):
        try:
            side.unlink(missing_ok=True)
        except OSError as exc:
            logger.warning("Could not remove SQLite sidecar %s: %s", side.name, exc)


def _unlink_with_retries(
    path: Path,
    *,
    attempts: int = 5,
    delay_s: float = 0.2,
) -> None:
    """
    Remove ``path``, retrying briefly on Windows sharing violations.

    Raises the last OSError if every attempt fails. Does not delete on
    persistent lock — callers must leave the plain DB in place.
    """
    import time

    last_exc: OSError | None = None
    for attempt in range(1, max(1, attempts) + 1):
        try:
            path.unlink(missing_ok=True)
            return
        except PermissionError as exc:
            last_exc = exc
            if attempt >= attempts:
                break
            logger.debug(
                "Plain DB still locked (attempt %s/%s): %s",
                attempt,
                attempts,
                exc,
            )
            time.sleep(delay_s)
        except OSError as exc:
            # Other OS errors (e.g. missing mid-retry) — surface immediately.
            raise
    assert last_exc is not None
    raise last_exc


def seal_database(
    db_path: Path,
    password: str | None = None,
    *,
    unlink_attempts: int = 5,
    unlink_delay_s: float = 0.2,
) -> None:
    """
    Encrypt the working database and remove the plain file (+ WAL/SHM).

    Callers must close every SQLite connection (and checkpoint WAL) first.
    On Windows, pooled connections left open cause WinError 32 and the
    encrypt-at-rest promise silently fails — see shutdown path in app.py.

    If the plain file cannot be removed after retries, the encrypted copy
    is discarded and the plain DB is kept (never delete a locked DB).
    """
    db_path = Path(db_path)
    if not db_path.exists():
        remove_sqlite_sidecars(db_path)
        return

    enc_path = encrypted_db_path(db_path)
    key = derive_db_key(password or resolve_db_password())
    logger.info("Encrypting catalogue database at rest: %s", enc_path.name)
    encrypt_file(db_path, enc_path, key)
    try:
        _unlink_with_retries(
            db_path, attempts=unlink_attempts, delay_s=unlink_delay_s
        )
    except PermissionError as exc:
        # Keep the working DB; drop a half-written .enc so next launch
        # does not prefer a stale encrypted snapshot over live data.
        try:
            enc_path.unlink(missing_ok=True)
        except OSError:
            pass
        logger.error(
            "Catalogue DB left unencrypted (plain %s kept): still locked "
            "after close — holder prevented delete (%s). Close all app "
            "copies and quit cleanly so encryption can finish.",
            db_path.name,
            exc,
        )
        raise
    remove_sqlite_sidecars(db_path)
