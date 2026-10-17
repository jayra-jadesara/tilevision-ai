"""Decide whether a tile needs full re-embed, CLIP-only backfill, or skip."""

from __future__ import annotations

from enum import Enum
from pathlib import Path
from typing import Optional, Set

from src.ai.feature_versions import is_tile_features_compatible
from src.core.models import TileImage
from src.core.use_cases.index_skip import is_unchanged_indexed_file


class IndexWorkKind(str, Enum):
    FULL = "full"
    CLIP_ONLY = "clip_only"
    METADATA_ONLY = "metadata_only"
    SKIP = "skip"


def classify_index_work(
    path: Path,
    existing_record: Optional[TileImage],
    *,
    force: bool,
    clip_ids_present: Optional[Set[int]] = None,
    clip_enabled: bool = False,
) -> IndexWorkKind:
    """
    Per-tile work for Update Search Index / rebuild.

    - ``force`` → always full DINOv2 (+ CLIP) re-embed.
    - File missing / changed / feature version stale → full.
    - DINOv2 current but CLIP id absent (and CLIP is configured) → CLIP-only.
    - Otherwise skip.
    """
    if force:
        return IndexWorkKind.FULL

    if existing_record is None or not existing_record.is_indexed:
        return IndexWorkKind.FULL

    if not is_unchanged_indexed_file(path, existing_record, force=False):
        return IndexWorkKind.FULL

    if not _features_are_current(existing_record):
        return IndexWorkKind.FULL

    if clip_enabled and clip_ids_present is not None:
        db_id = existing_record.id
        if db_id is not None and int(db_id) not in clip_ids_present:
            return IndexWorkKind.CLIP_ONLY
        if db_id is not None and int(db_id) in clip_ids_present:
            if _color_needs_backfill(existing_record):
                return IndexWorkKind.METADATA_ONLY

    return IndexWorkKind.SKIP


def _color_needs_backfill(record: TileImage) -> bool:
    """True when showroom colour was never resolved (Unknown / empty)."""
    label = (getattr(record, "color", None) or "").strip().lower()
    return label in ("", "unknown")


def _features_are_current(record: TileImage) -> bool:
    """True when stored feature metadata matches the current pipeline."""
    # Missing metadata on legacy rows → treat as stale (force full re-embed).
    if getattr(record, "feature_version", None) in (None, 0):
        return False
    pattern_size = None
    color_size = None
    features = record.features
    if features is not None:
        try:
            pattern_size = int(len(features.pattern_features))
        except Exception:
            pattern_size = None
        try:
            color_size = int(len(features.color_histogram))
        except Exception:
            color_size = None
    return is_tile_features_compatible(
        feature_version=getattr(record, "feature_version", None),
        pattern_feature_version=getattr(record, "pattern_feature_version", None),
        embedding_model=getattr(record, "embedding_model", None),
        embedding_dimension=getattr(record, "embedding_dimension", None),
        pattern_feature_size=pattern_size,
        color_histogram_size=color_size,
    )
