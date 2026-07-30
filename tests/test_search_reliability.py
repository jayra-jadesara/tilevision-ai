"""Reliability tests for search silent-failure fixes."""

from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.ai.inference_guard import InferenceBusyError
from src.ai.models import TileFeatures
from src.core.models import TileImage
from src.core.use_cases.search_tiles import SearchTilesUseCase
from src.utils.image_formats import query_image_extensions
from src.utils.search_stages import STAGE_FAISS_SEARCH, log_search_stage


def _features(dim: int = 8) -> TileFeatures:
    emb = np.ones(dim, dtype=np.float32)
    emb /= float(np.linalg.norm(emb))
    return TileFeatures(
        embedding=emb,
        color_histogram=np.zeros(8, dtype=np.float32),
        texture_histogram=np.zeros(8, dtype=np.float32),
        edge_histogram=np.zeros(8, dtype=np.float32),
        pattern_features=np.zeros(12, dtype=np.float32),
        dominant_color=(128, 128, 128),
        width=64,
        height=64,
    )


def test_query_extensions_include_required_formats():
    exts = query_image_extensions()
    for required in (".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tiff", ".tif"):
        assert required in exts
    assert ".jfif" in exts


def test_get_searchable_count_propagates_busy():
    index = MagicMock()
    index.get_total_count.side_effect = InferenceBusyError("busy")
    uc = SearchTilesUseCase(
        image_repository=MagicMock(),
        feature_extractor=MagicMock(),
        vector_index=index,
        thumbnail_dir="/tmp",
    )
    with pytest.raises(InferenceBusyError):
        uc.get_searchable_count()


def test_faiss_ids_without_sqlite_raises(tmp_path):
    """Orphan FAISS IDs must not become a silent empty result list."""
    query = tmp_path / "q.jpg"
    from PIL import Image

    Image.new("RGB", (64, 64), color=(200, 100, 50)).save(query)

    repo = MagicMock()
    repo.get_feature_version_status.return_value = SimpleNamespace(
        is_compatible=True, stale_count=0, indexed_count=10
    )
    repo.get_by_path.return_value = None
    repo.get_by_ids.return_value = []  # FAISS IDs missing from SQLite

    extractor = MagicMock()
    extractor.load_model = MagicMock()
    feats = _features(8)
    extractor.extract_for_search.return_value = (feats, [feats.embedding])
    extractor.last_timings = SimpleNamespace(preprocessing=0.0, dinov2=0.0, descriptors=0.0)

    index = MagicMock()
    index._index = object()
    index.embedding_dimension.return_value = 8
    index.get_total_count.return_value = 10
    index.index_type_name.return_value = "IndexIDMap(IndexFlatIP)"
    index.search_vectors.return_value = ([101, 102], [0.9, 0.8])

    # Patch CURRENT dim check by matching 8 — monkeypatch feature_versions in use case path.
    from src.ai import feature_versions

    original = feature_versions.CURRENT_EMBEDDING_DIMENSION
    feature_versions.CURRENT_EMBEDDING_DIMENSION = 8
    try:
        uc = SearchTilesUseCase(repo, extractor, index, str(tmp_path))
        # Bypass multi-crop helper by stubbing it
        uc._search_faiss_multi_crop = MagicMock(return_value=[101, 102])
        with pytest.raises(RuntimeError, match="out of sync|Rebuild FAISS"):
            uc.execute(str(query), top_k=5)
    finally:
        feature_versions.CURRENT_EMBEDDING_DIMENSION = original


def test_filter_mismatch_raises_not_silent_empty(tmp_path):
    query = tmp_path / "q.jpg"
    from PIL import Image

    Image.new("RGB", (32, 32), color=(10, 20, 30)).save(query)

    repo = MagicMock()
    repo.get_feature_version_status.return_value = SimpleNamespace(
        is_compatible=True, stale_count=0, indexed_count=5
    )
    repo.get_ids_matching_filters.return_value = []
    repo.get_by_path.return_value = None

    extractor = MagicMock()
    extractor.load_model = MagicMock()
    feats = _features(8)
    extractor.extract_for_search.return_value = (feats, [feats.embedding])
    extractor.last_timings = SimpleNamespace(preprocessing=0.0, dinov2=0.0, descriptors=0.0)

    index = MagicMock()
    index._index = object()
    index.embedding_dimension.return_value = 8
    index.get_total_count.return_value = 5
    index.index_type_name.return_value = "IndexFlatIP"

    from src.ai import feature_versions

    original = feature_versions.CURRENT_EMBEDDING_DIMENSION
    feature_versions.CURRENT_EMBEDDING_DIMENSION = 8
    try:
        uc = SearchTilesUseCase(repo, extractor, index, str(tmp_path))
        with pytest.raises(RuntimeError, match="active search filters"):
            uc.execute(str(query), top_k=5, filters={"brand": "NoSuchBrand"})
    finally:
        feature_versions.CURRENT_EMBEDDING_DIMENSION = original


def test_log_search_stage_invokes_callback(caplog):
    import logging

    called = []
    with caplog.at_level(logging.INFO):
        log_search_stage(
            logging.getLogger("tilevision.test"),
            STAGE_FAISS_SEARCH,
            detail="20 IDs",
            on_stage=called.append,
        )
    assert called
    assert "20 IDs" in called[0]
