"""Skip / fingerprint helpers for incremental indexing."""

from __future__ import annotations

from pathlib import Path
from typing import Optional

from src.core.models import TileImage
from src.utils.image_utils import compute_sha256


def is_unchanged_indexed_file(
    path: Path,
    existing_record: Optional[TileImage],
    *,
    force: bool,
) -> bool:
    """
    Return True when an indexed file can be skipped without re-embedding.

    Uses size+mtime fingerprint first (no disk read). Falls back to SHA256
    for legacy rows that predate file_mtime storage.
    """
    if force or existing_record is None or not existing_record.is_indexed:
        return False

    stat = path.stat()
    if stat.st_size != existing_record.file_size:
        return False

    stored_mtime = float(existing_record.file_mtime or 0.0)
    if stored_mtime > 0 and abs(stat.st_mtime - stored_mtime) < 1e-6:
        return True

    if stored_mtime <= 0:
        current_sha = compute_sha256(path)
        return current_sha == existing_record.sha256_hash

    return False
