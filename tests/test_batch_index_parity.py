"""
Batch folder indexing must write the same TileFeatures as index_single_file.

Regression for PGYS2319: scan_and_index_directory() used extract_batch()
(full-sheet ImagePreprocessor.preprocess) while index_single_file used
extract_index_vectors() (prepare_index_primary panel isolation).
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

faiss = pytest.importorskip("faiss")

from src.ai.feature_extractor import FeatureExtractor
from src.ai.feature_versions import CURRENT_EMBEDDING_DIMENSION
from src.ai.vector_index import FaissIndexManager
from src.core.use_cases.index_images import IndexImagesUseCase
from src.data.db_context import DatabaseContext
from src.data.sqlite_repository import SQLiteImageRepository
from tests.fake_ai import FakeEmbedder
from tests.test_crop_search_consistency import _make_catalog_sheet


def _histograms_equal(stored, fresh) -> bool:
    """Match SQLite float16 histogram storage (see sqlite_repository)."""
    def _roundtrip(hist: np.ndarray) -> np.ndarray:
        return hist.astype(np.float16).astype(np.float32)

    return (
        np.allclose(_roundtrip(stored.color_histogram), _roundtrip(fresh.color_histogram))
        and np.allclose(
            _roundtrip(stored.texture_histogram), _roundtrip(fresh.texture_histogram)
        )
        and np.allclose(_roundtrip(stored.edge_histogram), _roundtrip(fresh.edge_histogram))
        and np.allclose(
            _roundtrip(stored.pattern_features), _roundtrip(fresh.pattern_features)
        )
    )


def test_extract_batch_diverges_from_index_vectors_on_catalog_sheet(tmp_path):
    """Documents the bug class: extract_batch must not be used for folder scans."""
    sheet_path, _ = _make_catalog_sheet(tmp_path)
    feature_extractor = FeatureExtractor(embedder=FakeEmbedder())
    batch_features = feature_extractor.extract_batch([str(sheet_path)])[0]
    index_features, _aux = feature_extractor.extract_index_vectors(str(sheet_path))
    assert not np.allclose(
        batch_features.edge_histogram,
        index_features.edge_histogram,
        rtol=0,
        atol=1e-6,
    )


def _make_use_case(tmp_path):
    db_context = DatabaseContext(str(tmp_path / "db" / "tiles.db"))
    repo = SQLiteImageRepository(db_context)
    embedder = FakeEmbedder()
    feature_extractor = FeatureExtractor(embedder=embedder)
    vector_index = FaissIndexManager(
        str(tmp_path / "index" / "tiles.index"),
        dimension=CURRENT_EMBEDDING_DIMENSION,
    )
    use_case = IndexImagesUseCase(
        image_repository=repo,
        feature_extractor=feature_extractor,
        vector_index=vector_index,
        thumbnail_dir=str(tmp_path / "thumbs"),
    )
    return use_case, repo, feature_extractor


def test_batch_scan_matches_standalone_extract_index_vectors(tmp_path):
    use_case, repo, feature_extractor = _make_use_case(tmp_path)

    images_dir = tmp_path / "images"
    images_dir.mkdir()
    sheet_path, _ = _make_catalog_sheet(tmp_path)
    catalog_copy = images_dir / "PGYS2319.jpg"
    catalog_copy.write_bytes(sheet_path.read_bytes())
    filler = images_dir / "filler.jpg"
    filler.write_bytes(sheet_path.read_bytes())

    result = use_case.scan_and_index_directory(images_dir)
    assert result.indexed_count == 2
    assert result.is_completed is True

    stored = repo.get_by_path(str(catalog_copy.resolve()))
    assert stored is not None and stored.features is not None

    fresh, _aux = feature_extractor.extract_index_vectors(str(catalog_copy))
    assert _histograms_equal(stored.features, fresh), (
        "batch scan stored different handcrafted descriptors than "
        "extract_index_vectors standalone"
    )


def test_batch_scan_matches_index_single_file(tmp_path):
    use_case, repo, _feature_extractor = _make_use_case(tmp_path)

    sheet_path, _ = _make_catalog_sheet(tmp_path)
    batch_dir = tmp_path / "batch"
    batch_dir.mkdir()
    target = batch_dir / "PGYS2319.jpg"
    target.write_bytes(sheet_path.read_bytes())

    use_case.scan_and_index_directory(batch_dir)
    batch_tile = repo.get_by_path(str(target.resolve()))
    assert batch_tile is not None and batch_tile.features is not None

    single_path = tmp_path / "single" / "PGYS2319.jpg"
    single_path.parent.mkdir()
    single_path.write_bytes(sheet_path.read_bytes())
    use_case.index_single_file(single_path, persist=False)
    single_tile = repo.get_by_path(str(single_path.resolve()))
    assert single_tile is not None and single_tile.features is not None

    assert _histograms_equal(batch_tile.features, single_tile.features)
