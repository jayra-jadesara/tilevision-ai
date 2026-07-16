"""
Feature versioning for TileVision AI.

Tracks embedding pipeline and handcrafted descriptor versions so stale
indexed features are detected instead of silently compared.
"""

from __future__ import annotations

from dataclasses import dataclass

# Bump when the DINOv2 embedding pipeline changes (model, views, fusion).
CURRENT_FEATURE_VERSION = 2

# Bump when pattern descriptor layout or algorithm changes.
CURRENT_PATTERN_FEATURE_VERSION = 2

CURRENT_EMBEDDING_MODEL = "facebook/dinov2-large"
CURRENT_EMBEDDING_DIMENSION = 1024
CURRENT_PATTERN_FEATURE_SIZE = 8


@dataclass(frozen=True, slots=True)
class FeatureVersionStatus:
    is_compatible: bool
    indexed_count: int
    stale_count: int
    message: str


def is_tile_features_compatible(
    *,
    feature_version: int | None,
    pattern_feature_version: int | None,
    embedding_model: str | None,
    embedding_dimension: int | None,
    pattern_feature_size: int | None = None,
) -> bool:
    """Return True when stored feature metadata matches the current pipeline."""
    if feature_version != CURRENT_FEATURE_VERSION:
        return False
    if pattern_feature_version != CURRENT_PATTERN_FEATURE_VERSION:
        return False
    if embedding_model != CURRENT_EMBEDDING_MODEL:
        return False
    if embedding_dimension != CURRENT_EMBEDDING_DIMENSION:
        return False
    if (
        pattern_feature_size is not None
        and pattern_feature_size != CURRENT_PATTERN_FEATURE_SIZE
    ):
        return False
    return True
