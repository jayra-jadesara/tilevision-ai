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
from datetime import datetime
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
                    key_fingerprint TEXT NOT NULL
                );
                CREATE INDEX IF NOT EXISTS idx_licenses_status ON licenses(status);
                CREATE INDEX IF NOT EXISTS idx_licenses_customer ON licenses(customer_name);
                CREATE INDEX IF NOT EXISTS idx_licenses_machine ON licenses(machine_id);
                """
            )

    @staticmethod
    def fingerprint_key(license_key: str) -> str:
        return hashlib.sha256(license_key.encode("utf-8")).hexdigest()[:16]

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
    ) -> None:
        issued_at = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        with self._conn() as conn:
            conn.execute(
                """
                INSERT INTO licenses (
                    license_id, customer_name, machine_id, license_type,
                    issued_at, expires_at, status, notes, contact, key_fingerprint
                ) VALUES (?, ?, ?, ?, ?, ?, 'active', ?, ?, ?)
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

    def list_licenses(self, status_filter: Optional[str] = None) -> List[LicenseRecord]:
        query = "SELECT * FROM licenses"
        params: tuple = ()
        if status_filter and status_filter != "all":
            query += " WHERE status = ?"
            params = (status_filter,)
        query += " ORDER BY issued_at DESC"

        with self._conn() as conn:
            rows = conn.execute(query, params).fetchall()

        return [
            LicenseRecord(
                license_id=row["license_id"],
                customer_name=row["customer_name"],
                machine_id=row["machine_id"],
                license_type=row["license_type"],
                issued_at=row["issued_at"],
                expires_at=row["expires_at"],
                status=row["status"],
                notes=row["notes"] or "",
                contact=row["contact"] or "",
            )
            for row in rows
        ]

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


def _csv(value: str) -> str:
    if "," in value or '"' in value:
        return '"' + value.replace('"', '""') + '"'
    return value
