"""
Compatibility / upgrade checker for TileVision AI Enterprise (v1.2+).

Validates database reachability, embedding identity, FAISS metadata, and
configured index backend. Surfaces a guided rebuild recommendation when
anything is incompatible — without changing production search behavior.
"""

from __future__ import annotations

import logging
import sqlite3
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Callable, Optional

from src.ai.feature_versions import (
    CURRENT_EMBEDDING_DIMENSION,
    CURRENT_EMBEDDING_MODEL,
    CURRENT_FEATURE_VERSION,
    FeatureVersionStatus,
)
from src.ai.index_backends import IndexBackend
from src.ai.index_metadata import read_index_metadata
from src.version import APP_VERSION

logger = logging.getLogger("tilevision.core.compatibility")


@dataclass(slots=True)
class CompatibilityIssue:
    code: str
    severity: str  # info | warning | error
    message: str
    remediation: str


@dataclass(slots=True)
class CompatibilityReport:
    is_compatible: bool
    requires_rebuild: bool
    app_version: str
    issues: list[CompatibilityIssue] = field(default_factory=list)
    details: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "is_compatible": self.is_compatible,
            "requires_rebuild": self.requires_rebuild,
            "app_version": self.app_version,
            "issues": [asdict(i) for i in self.issues],
            "details": self.details,
        }

    def summary_message(self) -> str:
        if self.is_compatible and not self.requires_rebuild:
            return "Catalogue and search index are compatible with this application version."
        if not self.issues:
            return "Compatibility check reported issues."
        return self.issues[0].message


def check_database_schema(db_path: str | Path) -> list[CompatibilityIssue]:
    issues: list[CompatibilityIssue] = []
    path = Path(db_path)
    if not path.exists():
        issues.append(
            CompatibilityIssue(
                code="db_missing",
                severity="error",
                message=f"Database file not found: {path}",
                remediation="Re-run indexing or restore a database backup.",
            )
        )
        return issues
    try:
        conn = sqlite3.connect(f"file:{path}?mode=ro", uri=True)
        try:
            cols = {
                row[1]
                for row in conn.execute("PRAGMA table_info(tiles)").fetchall()
            }
            required = {
                "id",
                "file_path",
                "embedding_blob",
                "feature_version",
                "embedding_model",
                "embedding_dimension",
            }
            missing = sorted(required - cols)
            if missing:
                issues.append(
                    CompatibilityIssue(
                        code="db_schema_missing_columns",
                        severity="error",
                        message=f"Database schema missing columns: {', '.join(missing)}",
                        remediation="Let the app migrate the schema on next start, or restore a backup.",
                    )
                )
            # Smoke-query to catch corruption.
            conn.execute("SELECT COUNT(*) FROM tiles").fetchone()
        finally:
            conn.close()
    except sqlite3.DatabaseError as exc:
        issues.append(
            CompatibilityIssue(
                code="db_corrupt",
                severity="error",
                message=f"Database appears corrupted: {exc}",
                remediation="Restore from Backup Database, then rebuild the FAISS index.",
            )
        )
    except Exception as exc:
        issues.append(
            CompatibilityIssue(
                code="db_unreadable",
                severity="error",
                message=f"Could not open database: {exc}",
                remediation="Check file permissions and disk health.",
            )
        )
    return issues


def check_index_metadata(
    index_path: str | Path,
    *,
    expected_backend: IndexBackend = IndexBackend.FLAT_IP,
) -> list[CompatibilityIssue]:
    issues: list[CompatibilityIssue] = []
    path = Path(index_path)
    if not path.exists() or path.stat().st_size == 0:
        # Empty catalog is fine for a fresh install.
        return issues

    meta = read_index_metadata(path)
    if meta is None:
        issues.append(
            CompatibilityIssue(
                code="index_meta_missing",
                severity="warning",
                message="FAISS index has no metadata sidecar (.meta.json).",
                remediation="Rebuild FAISS Index once to write compatibility metadata.",
            )
        )
        return issues

    if not meta.is_compatible():
        issues.append(
            CompatibilityIssue(
                code="index_meta_incompatible",
                severity="error",
                message=(
                    "FAISS index metadata is incompatible with this app "
                    f"(model={meta.embedding_model}, dim={meta.embedding_dimension}, "
                    f"feature_v={meta.feature_version})."
                ),
                remediation="Use Settings → Rebuild FAISS Index (guided rebuild).",
            )
        )

    if meta.index_backend and meta.index_backend != expected_backend.value:
        issues.append(
            CompatibilityIssue(
                code="index_backend_mismatch",
                severity="warning",
                message=(
                    f"On-disk index backend={meta.index_backend} differs from "
                    f"configured={expected_backend.value}."
                ),
                remediation=(
                    "Rebuild FAISS Index to switch backends, or set Settings → "
                    "Index Backend back to the on-disk type."
                ),
            )
        )
    return issues


def check_feature_status(
    status: FeatureVersionStatus | None,
) -> list[CompatibilityIssue]:
    if status is None:
        return []
    if status.is_compatible or status.stale_count <= 0:
        return []
    return [
        CompatibilityIssue(
            code="features_stale",
            severity="error",
            message=status.message
            or f"{status.stale_count} of {status.indexed_count} tiles have stale features.",
            remediation="Use Settings → Rebuild FAISS Index after re-scanning folders.",
        )
    ]


def run_compatibility_check(
    *,
    database_path: str | Path,
    index_path: str | Path,
    expected_backend: IndexBackend | str = IndexBackend.FLAT_IP,
    feature_status_provider: Optional[Callable[[], FeatureVersionStatus]] = None,
    catalog_size: int | None = None,
) -> CompatibilityReport:
    """Run the full enterprise compatibility suite."""
    backend = IndexBackend.parse(
        expected_backend.value
        if isinstance(expected_backend, IndexBackend)
        else expected_backend
    )
    issues: list[CompatibilityIssue] = []
    issues.extend(check_database_schema(database_path))
    issues.extend(check_index_metadata(index_path, expected_backend=backend))

    feature_status = None
    if feature_status_provider is not None:
        try:
            feature_status = feature_status_provider()
            issues.extend(check_feature_status(feature_status))
        except Exception as exc:
            issues.append(
                CompatibilityIssue(
                    code="feature_status_failed",
                    severity="warning",
                    message=f"Could not read feature version status: {exc}",
                    remediation="Retry after indexing completes.",
                )
            )

    rebuild_codes = {
        "index_meta_incompatible",
        "features_stale",
        "index_backend_mismatch",
        "db_corrupt",
    }
    requires_rebuild = any(i.code in rebuild_codes for i in issues)
    hard_errors = [i for i in issues if i.severity == "error"]
    report = CompatibilityReport(
        is_compatible=len(hard_errors) == 0,
        requires_rebuild=requires_rebuild,
        app_version=APP_VERSION,
        issues=issues,
        details={
            "embedding_model": CURRENT_EMBEDDING_MODEL,
            "embedding_dimension": CURRENT_EMBEDDING_DIMENSION,
            "feature_version": CURRENT_FEATURE_VERSION,
            "expected_backend": backend.value,
            "catalog_size": catalog_size,
            "database_path": str(database_path),
            "index_path": str(index_path),
            "feature_status": (
                {
                    "is_compatible": feature_status.is_compatible,
                    "indexed_count": feature_status.indexed_count,
                    "stale_count": feature_status.stale_count,
                    "message": feature_status.message,
                }
                if feature_status is not None
                else None
            ),
        },
    )
    if requires_rebuild:
        logger.warning("Compatibility check recommends rebuild: %s", report.summary_message())
    else:
        logger.info("Compatibility check OK (%d issue(s))", len(issues))
    return report
