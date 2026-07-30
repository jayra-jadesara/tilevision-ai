"""
FAISS index metadata sidecar for TileVision AI.

Stores embedding model / dimension / app version / backend next to the binary
index so incompatible indexes can be detected before a silent empty search.
"""

from __future__ import annotations

import json
import logging
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from src.ai.feature_versions import (
    CURRENT_EMBEDDING_DIMENSION,
    CURRENT_EMBEDDING_MODEL,
    CURRENT_FEATURE_VERSION,
)
from src.version import APP_VERSION

logger = logging.getLogger("tilevision.ai.index_metadata")


@dataclass(slots=True)
class FaissIndexMetadata:
    embedding_model: str
    embedding_dimension: int
    feature_version: int
    app_version: str
    faiss_type: str
    ntotal: int
    build_date: str
    catalog_version: int
    index_backend: str = "flat_ip"

    def is_compatible(self, *, expected_backend: str | None = None) -> bool:
        ok = (
            self.embedding_model == CURRENT_EMBEDDING_MODEL
            and int(self.embedding_dimension) == CURRENT_EMBEDDING_DIMENSION
            and int(self.feature_version) == CURRENT_FEATURE_VERSION
            and int(self.catalog_version) == CURRENT_FEATURE_VERSION
        )
        if expected_backend is not None and self.index_backend:
            ok = ok and self.index_backend == expected_backend
        return ok


def metadata_path_for(index_path: str | Path) -> Path:
    path = Path(index_path)
    return path.with_suffix(path.suffix + ".meta.json")


def write_index_metadata(
    index_path: str | Path,
    *,
    faiss_type: str,
    ntotal: int,
    index_backend: str = "flat_ip",
) -> Path:
    meta = FaissIndexMetadata(
        embedding_model=CURRENT_EMBEDDING_MODEL,
        embedding_dimension=CURRENT_EMBEDDING_DIMENSION,
        feature_version=CURRENT_FEATURE_VERSION,
        app_version=APP_VERSION,
        faiss_type=faiss_type,
        ntotal=int(ntotal),
        build_date=datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        catalog_version=CURRENT_FEATURE_VERSION,
        index_backend=str(index_backend or "flat_ip"),
    )
    out = metadata_path_for(index_path)
    out.write_text(json.dumps(asdict(meta), indent=2), encoding="utf-8")
    logger.info("Wrote FAISS metadata sidecar: %s", out.name)
    return out


def read_index_metadata(index_path: str | Path) -> Optional[FaissIndexMetadata]:
    path = metadata_path_for(index_path)
    if not path.exists():
        return None
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        return FaissIndexMetadata(
            embedding_model=str(data.get("embedding_model", "")),
            embedding_dimension=int(data.get("embedding_dimension", 0)),
            feature_version=int(data.get("feature_version", 0)),
            app_version=str(data.get("app_version", "")),
            faiss_type=str(data.get("faiss_type", "")),
            ntotal=int(data.get("ntotal", 0)),
            build_date=str(data.get("build_date", "")),
            catalog_version=int(data.get("catalog_version", 0)),
            index_backend=str(data.get("index_backend", "flat_ip") or "flat_ip"),
        )
    except Exception as exc:
        logger.warning("Failed to read FAISS metadata sidecar: %s", exc)
        return None
