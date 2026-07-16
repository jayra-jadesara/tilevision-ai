"""
Vendor-side license registry for TileVision AI Admin License Manager.

Tracks every issued key, customer, trial/official type, and cancellation
status. Stored locally on the vendor machine — not shipped to customers.
"""

from __future__ import annotations

import hashlib
import json
import sqlite3
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Generator, List, Optional


@dataclass(slots=True)
class LicenseRecord:
    license_id: str
    customer_name: str
    machine_id: str
    license_type: str
    issued_at: str
    expires_at: str
    status: str
    notes: str = ""
    contact: str = ""
    license_key: str = ""


@dataclass(slots=True)
class LicenseStats:
    total: int
    active: int
    cancelled: int
    trials_active: int
    official_active: int
    expiring_soon: int


def is_trial_license_type(license_type: str) -> bool:
    return "Trial" in license_type


class LicenseLedger:
    """SQLite-backed registry of issued licenses."""

    def __init__(self, db_path: Optional[Path] = None) -> None:
        default = Path.home() / ".tilevision_ai_vendor" / "license_ledger.db"
        self._db_path = db_path or default
        self._db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_schema()

    @property
    def db_path(self) -> Path:
        return self._db_path

    @contextmanager
    def _conn(self) -> Generator[sqlite3.Connection, None, None]:
        conn = sqlite3.connect(str(self._db_path))
        conn.row_factory = sqlite3.Row
        try:
            yield conn
            conn.commit()
        finally:
            conn.close()

    def _init_schema(self) -> None:
        with self._conn() as conn:
            conn.executescript(
                """
                CREATE TABLE IF NOT EXISTS licenses (
                    license_id TEXT PRIMARY KEY,
                    customer_name TEXT NOT NULL,
                    machine_id TEXT NOT NULL,
                    license_type TEXT NOT NULL,
                    issued_at TEXT NOT NULL,
                    expires_at TEXT NOT NULL,
                    status TEXT NOT NULL DEFAULT 'active',
                    notes TEXT DEFAULT '',
                    contact TEXT DEFAULT '',
                    key_fingerprint TEXT NOT NULL,
                    license_key TEXT DEFAULT ''
                );
                CREATE INDEX IF NOT EXISTS idx_licenses_status ON licenses(status);
                CREATE INDEX IF NOT EXISTS idx_licenses_customer ON licenses(customer_name);
                CREATE INDEX IF NOT EXISTS idx_licenses_machine ON licenses(machine_id);
                """
            )
            self._migrate_schema(conn)

    def _migrate_schema(self, conn: sqlite3.Connection) -> None:
        columns = {row[1] for row in conn.execute("PRAGMA table_info(licenses)").fetchall()}
        if "license_key" not in columns:
            conn.execute("ALTER TABLE licenses ADD COLUMN license_key TEXT DEFAULT ''")

    @staticmethod
    def fingerprint_key(license_key: str) -> str:
        return hashlib.sha256(license_key.encode("utf-8")).hexdigest()[:16]

    @staticmethod
    def _row_to_record(row: sqlite3.Row) -> LicenseRecord:
        return LicenseRecord(
            license_id=row["license_id"],
            customer_name=row["customer_name"],
            machine_id=row["machine_id"],
            license_type=row["license_type"],
            issued_at=row["issued_at"],
            expires_at=row["expires_at"],
            status=row["status"],
            notes=row["notes"] or "",
            contact=row["contact"] or "",
            license_key=row["license_key"] or "",
        )

    def get_license(self, license_id: str) -> Optional[LicenseRecord]:
        with self._conn() as conn:
            row = conn.execute(
                "SELECT * FROM licenses WHERE license_id = ?", (license_id,)
            ).fetchone()
        return self._row_to_record(row) if row else None

    def record_issue(
        self,
        *,
        license_id: str,
        customer_name: str,
        machine_id: str,
        license_type: str,
        expires_at: str,
        license_key: str,
        notes: str = "",
        contact: str = "",
        supersede_license_id: Optional[str] = None,
    ) -> None:
        issued_at = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        with self._conn() as conn:
            if supersede_license_id:
                conn.execute(
                    """
                    UPDATE licenses
                    SET status = 'superseded',
                        notes = COALESCE(notes, '') || ' Superseded by renewal.'
                    WHERE license_id = ?
                    """,
                    (supersede_license_id,),
                )
            conn.execute(
                """
                INSERT INTO licenses (
                    license_id, customer_name, machine_id, license_type,
                    issued_at, expires_at, status, notes, contact,
                    key_fingerprint, license_key
                ) VALUES (?, ?, ?, ?, ?, ?, 'active', ?, ?, ?, ?)
                """,
                (
                    license_id,
                    customer_name,
                    machine_id,
                    license_type,
                    issued_at,
                    expires_at,
                    notes,
                    contact,
                    self.fingerprint_key(license_key),
                    license_key,
                ),
            )

    def cancel_license(self, license_id: str, reason: str = "") -> bool:
        with self._conn() as conn:
            row = conn.execute(
                "SELECT license_id FROM licenses WHERE license_id = ?", (license_id,)
            ).fetchone()
            if row is None:
                return False
            note_suffix = f" Cancelled: {reason}" if reason else " Cancelled."
            conn.execute(
                """
                UPDATE licenses
                SET status = 'cancelled',
                    notes = COALESCE(notes, '') || ?
                WHERE license_id = ?
                """,
                (note_suffix, license_id),
            )
            return True

    def is_machine_blocked(self, machine_id: str) -> bool:
        with self._conn() as conn:
            row = conn.execute(
                """
                SELECT 1 FROM licenses
                WHERE machine_id = ? AND status = 'cancelled'
                LIMIT 1
                """,
                (machine_id,),
            ).fetchone()
            return row is not None

    def list_licenses(
        self,
        status_filter: Optional[str] = None,
        category_filter: Optional[str] = None,
        search_text: str = "",
    ) -> List[LicenseRecord]:
        clauses: list[str] = []
        params: list = []

        if status_filter and status_filter != "all":
            clauses.append("status = ?")
            params.append(status_filter)

        if category_filter == "trials":
            clauses.append("license_type LIKE '%Trial%'")
        elif category_filter == "official":
            clauses.append("license_type NOT LIKE '%Trial%'")

        if search_text.strip():
            clauses.append(
                "(customer_name LIKE ? OR machine_id LIKE ? OR contact LIKE ? OR license_id LIKE ?)"
            )
            needle = f"%{search_text.strip()}%"
            params.extend([needle, needle, needle, needle])

        query = "SELECT * FROM licenses"
        if clauses:
            query += " WHERE " + " AND ".join(clauses)
        query += " ORDER BY issued_at DESC"

        with self._conn() as conn:
            rows = conn.execute(query, tuple(params)).fetchall()

        return [self._row_to_record(row) for row in rows]

    def get_stats(self) -> LicenseStats:
        records = self.list_licenses()
        today = datetime.now().date()
        soon = today + timedelta(days=30)
        active = [r for r in records if r.status == "active"]
        cancelled = [r for r in records if r.status == "cancelled"]
        trials_active = [r for r in active if is_trial_license_type(r.license_type)]
        official_active = [r for r in active if not is_trial_license_type(r.license_type)]
        expiring_soon = 0
        for rec in active:
            try:
                expiry = datetime.strptime(rec.expires_at, "%Y-%m-%d").date()
                if today <= expiry <= soon:
                    expiring_soon += 1
            except ValueError:
                continue
        return LicenseStats(
            total=len(records),
            active=len(active),
            cancelled=len(cancelled),
            trials_active=len(trials_active),
            official_active=len(official_active),
            expiring_soon=expiring_soon,
        )

    def active_revoked_ids(self) -> List[str]:
        with self._conn() as conn:
            rows = conn.execute(
                "SELECT license_id FROM licenses WHERE status = 'cancelled'"
            ).fetchall()
        return [row["license_id"] for row in rows]

    def export_csv(self, destination: Path) -> None:
        records = self.list_licenses()
        lines = [
            "license_id,customer_name,machine_id,license_type,issued_at,expires_at,status,contact,notes"
        ]
        for rec in records:
            lines.append(
                ",".join(
                    [
                        rec.license_id,
                        _csv(rec.customer_name),
                        rec.machine_id,
                        _csv(rec.license_type),
                        rec.issued_at,
                        rec.expires_at,
                        rec.status,
                        _csv(rec.contact),
                        _csv(rec.notes),
                    ]
                )
            )
        destination.write_text("\n".join(lines), encoding="utf-8")

    def export_revocation_manifest(self, destination: Path) -> None:
        payload = {"revoked_license_ids": self.active_revoked_ids()}
        destination.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    def export_revocation_python_snippet(self) -> str:
        ids = self.active_revoked_ids()
        if not ids:
            return "EMBEDDED_REVOKED_LICENSE_IDS: FrozenSet[str] = frozenset()"
        quoted = ",\n    ".join(f'"{item}"' for item in ids)
        return f"EMBEDDED_REVOKED_LICENSE_IDS: FrozenSet[str] = frozenset({{\n    {quoted},\n}})"


def _csv(value: str) -> str:
    if "," in value or '"' in value:
        return '"' + value.replace('"', '""') + '"'
    return value
