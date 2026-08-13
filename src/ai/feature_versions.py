"""
Feature versioning for TileVision AI.

Tracks embedding pipeline and handcrafted descriptor versions so stale
indexed features are detected instead of silently compared.
"""

from __future__ import annotations

from dataclasses import dataclass

# Bump when the DINOv2 embedding pipeline changes (model, views, fusion).
# v6: index-time secondary texture-panel vector for wide catalog sheets.
# v7: lower panel contrast gate so high-key white marble/onyx sheets
#     (e.g. PGYS2319) actually receive an aux FAISS vector; rebuild required.
# v8: multi-scale center-50% aux for large tiles + skip scene auto-crop on
#     catalog marketing sheets (index/query preprocess alignment); rebuild.
# v9: Strategy E heuristic multi-view indexer (image analysis gates panel /
#     center aux). Golden bakeoff winner vs primary-only; rebuild required.
# v10: Strategy E + force adaptive content crop (320-tile optimization study:
#      +1.57pp R@5 / +1.19pp R@1 vs E at +0.13 vectors/tile); rebuild required.
# v11: marketing-sheet text detection + aspect gate (PGYS2319 @ 1.063 with
#      preview grid now gets left-panel aux index view); rebuild required.
# v12: panel aux crop shaves top/left caption band (PGYS2319 top-left bleed);
#      rebuild required.
# v13: widen panel top caption band 10% → 13% (residual clipped line on real
#      PGYS2319 at 2x zoom); rebuild required.
# v14: catalog-sheet primary TileFeatures (embedding + color/texture/edge/
#      pattern/dominant) come from isolated panel, not full marketing sheet;
#      full-sheet kept as FAISS aux for sheet self-hit; rebuild required.
# v15: normalize_lighting skips high-key low-chroma materials (cream marble)
#      so panel primary is not posterized; rebuild required for catalog sheets.
# v16: EdgeDescriptor adaptive Canny (+ empty-hist similarity); fixed 80/180
#      returned all-zero hists on subtle marble → cosine 0.0; rebuild required.
CURRENT_FEATURE_VERSION = 16

# Bump when pattern descriptor layout or algorithm changes.
CURRENT_PATTERN_FEATURE_VERSION = 3

CURRENT_EMBEDDING_MODEL = "facebook/dinov2-large"
CURRENT_EMBEDDING_DIMENSION = 1024
CURRENT_PATTERN_FEATURE_SIZE = 12
CURRENT_COLOR_HISTOGRAM_SIZE = 2884


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
    color_histogram_size: int | None = None,
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
    if (
        color_histogram_size is not None
        and color_histogram_size != CURRENT_COLOR_HISTOGRAM_SIZE
    ):
        return False
    return True
