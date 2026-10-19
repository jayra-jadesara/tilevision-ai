"""
A1: Catalogue DB must encrypt on exit even with pooled WAL connections.

Customer log (Windows): 34× WinError 32 on seal, zero successful
"Catalogue database encrypted at rest" lines — pooled connections held
tiles.db open so unlink failed. Root cause: check_same_thread=True made
close_all() on the main thread raise ProgrammingError for worker conns
(swallowed) so handles stayed open.
"""

from __future__ import annotations

import sys
import threading
from pathlib import Path

import pytest
import sqlite3

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data.db_context import DatabaseContext
from src.data.db_protection import (
    encrypted_db_path,
    prepare_working_database,
    seal_database,
)


def _sidecar(db_path: Path) -> tuple[Path, Path]:
    return Path(f"{db_path}-wal"), Path(f"{db_path}-shm")


def _assert_conn_closed(conn: sqlite3.Connection) -> None:
    with pytest.raises(sqlite3.ProgrammingError, match="closed database"):
        conn.execute("SELECT 1")


def test_prepare_for_seal_then_encrypt_with_two_thread_pool(tmp_path, monkeypatch):
    """
    Open pooled connections on 2 threads (WAL mode), then the real shutdown
    sequence: prepare_for_seal() → seal_database(). Assert every pooled
    connection is closed, .enc exists, and plain / -wal / -shm are gone.
    """
    monkeypatch.setenv("TILEVISION_DB_PASSWORD", "seal-shutdown-test")
    db_path = tmp_path / "tiles.db"
    db = DatabaseContext(str(db_path))

    barrier = threading.Barrier(3)
    errors: list[BaseException] = []
    worker_conns: list[sqlite3.Connection] = []

    def _worker(name: str) -> None:
        try:
            with db.session() as conn:
                conn.execute(
                    "INSERT INTO tiles(file_path,file_name,file_size,dimensions) "
                    "VALUES (?,?,?,?)",
                    (f"/{name}.jpg", f"{name}.jpg", 1, "1x1"),
                )
            # Hold the thread-local connection until the main thread joins
            # the barrier — same shape as live Search + Index workers.
            with db.session() as conn:
                worker_conns.append(conn)
                conn.execute("SELECT COUNT(*) FROM tiles")
                barrier.wait(timeout=5.0)
        except BaseException as exc:  # noqa: BLE001 — surface in main thread
            errors.append(exc)
            try:
                barrier.abort()
            except Exception:
                pass

    t1 = threading.Thread(target=_worker, args=("a",), daemon=True)
    t2 = threading.Thread(target=_worker, args=("b",), daemon=True)
    t1.start()
    t2.start()
    barrier.wait(timeout=5.0)
    t1.join(timeout=5.0)
    t2.join(timeout=5.0)
    assert not errors, errors
    assert len(worker_conns) == 2

    # Main thread also has a pooled connection (UI path).
    with db.session() as conn:
        n = conn.execute("SELECT COUNT(*) FROM tiles").fetchone()[0]
    assert int(n) == 2

    wal, shm = _sidecar(db_path)
    assert db_path.exists()

    pooled_before = db.pooled_connections()
    assert len(pooled_before) >= 3  # main + 2 workers

    db.prepare_for_seal()

    # Meaningful assertion: every pooled connection is actually closed.
    # (Previously close_all swallowed ProgrammingError and left them open.)
    for conn in pooled_before:
        _assert_conn_closed(conn)
    assert db.pooled_connections() == []

    # On Windows, a locked tiles.db cannot be renamed (WinError 32). After a
    # real close_all this probe must succeed — the failing assertion CI saw.
    if sys.platform == "win32":
        probe = db_path.with_suffix(".db.probe")
        db_path.replace(probe)
        probe.replace(db_path)

    seal_database(db_path)

    enc = encrypted_db_path(db_path)
    assert enc.exists(), "tiles.db.enc must exist after seal"
    assert not db_path.exists(), "plain tiles.db must be removed"
    assert not wal.exists(), f"WAL leftover: {wal}"
    assert not shm.exists(), f"SHM leftover: {shm}"

    # Next launch decrypts the working copy.
    prepare_working_database(db_path)
    assert db_path.exists()
    db2 = DatabaseContext(str(db_path))
    with db2.session() as conn:
        rows = conn.execute("SELECT file_name FROM tiles ORDER BY file_name").fetchall()
    assert [r["file_name"] for r in rows] == ["a.jpg", "b.jpg"]
    db2.close_all()


def test_close_all_closes_worker_thread_connections(tmp_path):
    """Regression: worker connections must close from the main thread."""
    db = DatabaseContext(str(tmp_path / "tiles.db"))
    held: list[sqlite3.Connection] = []
    ready = threading.Event()

    def _worker() -> None:
        with db.session() as conn:
            held.append(conn)
            conn.execute("SELECT 1")
            ready.set()
            # Keep the thread alive briefly so pool still owns the conn.
            threading.Event().wait(0.2)

    t = threading.Thread(target=_worker, daemon=True)
    t.start()
    assert ready.wait(timeout=5.0)
    assert held
    # Before the fix, this execute still worked after close_all().
    db.close_all()
    _assert_conn_closed(held[0])
    t.join(timeout=5.0)


def test_seal_keeps_plain_db_when_unlink_blocked(tmp_path, monkeypatch):
    """Persistent PermissionError must not delete the working catalogue."""
    monkeypatch.setenv("TILEVISION_DB_PASSWORD", "seal-keep-plain")
    db_path = tmp_path / "tiles.db"
    db_path.write_bytes(b"catalogue-live-data")

    import src.data.db_protection as prot

    calls = {"n": 0}

    def _always_locked(path, *, attempts=5, delay_s=0.2):  # noqa: ARG001
        calls["n"] += 1
        raise PermissionError(32, "file in use")

    monkeypatch.setattr(prot, "_unlink_with_retries", _always_locked)

    with pytest.raises(PermissionError):
        seal_database(db_path, unlink_attempts=3, unlink_delay_s=0.01)

    assert calls["n"] == 1
    assert db_path.exists()
    assert db_path.read_bytes() == b"catalogue-live-data"
    assert not encrypted_db_path(db_path).exists()


def test_seal_removes_wal_shm_sidecars(tmp_path, monkeypatch):
    monkeypatch.setenv("TILEVISION_DB_PASSWORD", "seal-sidecars")
    db_path = tmp_path / "tiles.db"
    db_path.write_bytes(b"payload")
    wal, shm = _sidecar(db_path)
    wal.write_bytes(b"wal")
    shm.write_bytes(b"shm")

    seal_database(db_path)
    assert encrypted_db_path(db_path).exists()
    assert not db_path.exists()
    assert not wal.exists()
    assert not shm.exists()
