"""Tests for query embedding LRU cache and search pipeline optimizations."""

from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import MagicMock

import numpy as np
import pytest
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.ai.models import TileFeatures
from src.ai.preprocess.image_preprocessor import ImagePreprocessor
from src.ai.query_cache import QUERY_EMBEDDING_CACHE, QueryEmbeddingCache
from src.core.models import TileImage
from src.core.use_cases.search_tiles import SearchTilesUseCase
from src.utils.pipeline_timing import PipelineTimer
from src.utils.image_utils import compute_dhash, compute_dhash_from_image


def _features() -> TileFeatures:
    emb = np.ones(8, dtype=np.float32)
    emb /= np.linalg.norm(emb)
    return TileFeatures(
        embedding=emb,
        color_histogram=np.zeros(32, dtype=np.float32),
        texture_histogram=np.zeros(16, dtype=np.float32),
        edge_histogram=np.zeros(16, dtype=np.float32),
        pattern_features=np.zeros(8, dtype=np.float32),
        dominant_color=(128, 128, 128),
        width=100,
        height=100,
    )


def test_query_embedding_cache_hit_miss(tmp_path):
    cache = QueryEmbeddingCache(capacity=4)
    path = tmp_path / "tile.jpg"
    Image.new("RGB", (64, 64), color=(10, 20, 30)).save(path)

    assert cache.get(path) is None
    feats = _features()
    cache.put(path, feats, [feats.embedding])
    hit = cache.get(path)
    assert hit is not None
    assert np.allclose(hit.features.embedding, feats.embedding)

    # Touch file → cache miss
    path.write_bytes(path.read_bytes() + b"\x00")
    assert cache.get(path) is None


def test_prepare_query_views_max_views_one_skips_second_load(tmp_path, monkeypatch):
    path = tmp_path / "room.jpg"
    # Wide image looks like a scene photo.
    Image.new("RGB", (800, 400), color=(180, 180, 180)).save(path)

    loads = {"count": 0}
    original_load = ImagePreprocessor.load

    def counting_load(cls, image_path, **kwargs):
        loads["count"] += 1
        return original_load(image_path, **kwargs)

    monkeypatch.setattr(
        ImagePreprocessor,
        "load",
        classmethod(counting_load),
    )

    views = ImagePreprocessor.prepare_query_views(path, max_views=1)
    assert len(views) == 1
    # One load inside preprocess_for_query — no second decode for extras.
    assert loads["count"] == 1


def test_prepare_query_views_multi_may_reload(tmp_path, monkeypatch):
    path = tmp_path / "room.jpg"
    Image.new("RGB", (800, 400), color=(180, 180, 180)).save(path)

    loads = {"count": 0}
    original_load = ImagePreprocessor.load

    def counting_load(cls, image_path, **kwargs):
        loads["count"] += 1
        return original_load(image_path, **kwargs)

    monkeypatch.setattr(
        ImagePreprocessor,
        "load",
        classmethod(counting_load),
    )
    monkeypatch.setattr(
        ImagePreprocessor,
        "_capped_query_max_views",
        classmethod(lambda cls, requested: max(1, int(requested))),
    )
    monkeypatch.setattr(
        ImagePreprocessor,
        "_looks_like_scene_photo",
        classmethod(lambda cls, image: True),
    )
    monkeypatch.setattr(
        ImagePreprocessor,
        "_isolate_query_tile",
        classmethod(lambda cls, image: image),
    )

    from src.ai.preprocess import fast_tile_crop

    class FakeCrop:
        def __init__(self, image):
            self.image = image
            self.method = "test"
            self.confidence = 1.0

    monkeypatch.setattr(
        fast_tile_crop,
        "list_tile_region_candidates",
        lambda image, limit=3: [FakeCrop(image), FakeCrop(image.crop((10, 10, 100, 100)))],
    )

    views = ImagePreprocessor.prepare_query_views(path, max_views=2)
    assert len(views) >= 1
    assert loads["count"] >= 2


def test_find_catalog_tile_by_stem_uses_sql_helper():
    repo = MagicMock()
    tile = TileImage(
        file_path="/catalog/foo.jpg",
        file_name="foo.jpg",
        file_size=1,
        dimensions="1x1",
        id=7,
        is_indexed=True,
    )
    repo.get_indexed_by_file_stem.return_value = tile
    use_case = SearchTilesUseCase(
        image_repository=repo,
        feature_extractor=MagicMock(),
        vector_index=MagicMock(),
        thumbnail_dir="/tmp",
    )
    found = use_case._find_catalog_tile_by_stem("foo")
    assert found is tile
    repo.get_indexed_by_file_stem.assert_called_once_with("foo")
    repo.get_all.assert_not_called()


def test_pipeline_timer_prints_required_labels(monkeypatch, capsys):
    monkeypatch.setenv("TILEVISION_PROFILE", "1")
    from importlib import reload
    import src.utils.pipeline_timing as pt

    reload(pt)
    timer = pt.PipelineTimer("SEARCH TIMING")
    timer.set_meta(
        cache="miss",
        catalog_size=100000,
        faiss_index="IndexIDMap(IndexFlatIP)",
        embedding_dim=1024,
    )
    timer.timings.record("image_load", 0.045)
    timer.timings.record("crop", 0.080)
    timer.timings.record("embedding", 0.310)
    timer.timings.record("faiss", 0.018)
    timer.timings.record("metadata", 0.007)
    timer.timings.record("thumbnail", 0.055)
    timer.log_summary()
    out = capsys.readouterr().out
    assert "Image Load" in out
    assert "Crop" in out
    assert "Embedding" in out
    assert "FAISS" in out
    assert "SQLite" in out
    assert "Thumbnail" in out
    assert "Cache" in out
    assert "Catalog Size" in out
    assert "FAISS Index" in out
    assert "Embedding Dim" in out
    assert "Thread ID" in out
    assert "TOTAL" in out


def test_pipeline_timer_respects_profile_env(monkeypatch, capsys):
    monkeypatch.setenv("TILEVISION_PROFILE", "0")
    from importlib import reload
    import src.utils.pipeline_timing as pt

    reload(pt)
    assert pt.profiling_enabled() is False
    timer = pt.PipelineTimer("SEARCH TIMING")
    with timer.measure("faiss"):
        pass
    timer.log_summary()
    assert capsys.readouterr().out == ""
    assert timer.timings.stages == {}


def test_faiss_index_type_is_flat_ip(tmp_path):
    faiss = pytest.importorskip("faiss")
    from src.ai.vector_index import FaissIndexManager

    manager = FaissIndexManager(str(tmp_path / "t.index"), dimension=8)
    manager.load_index()
    assert manager.index_type_name() == "IndexIDMap(IndexFlatIP)"
    assert manager.embedding_dimension() == 8


def test_global_query_cache_roundtrip(tmp_path):
    QUERY_EMBEDDING_CACHE.clear()
    path = tmp_path / "q.jpg"
    Image.new("RGB", (32, 32), color=(1, 2, 3)).save(path)
    feats = _features()
    QUERY_EMBEDDING_CACHE.put(path, feats, [feats.embedding])
    hit = QUERY_EMBEDDING_CACHE.get(path)
    assert hit is not None
    QUERY_EMBEDDING_CACHE.clear()
    assert QUERY_EMBEDDING_CACHE.get(path) is None


def test_sqlite_stem_lookup_avoids_full_scan(tmp_path):
    from src.data.db_context import DatabaseContext
    from src.data.sqlite_repository import SQLiteImageRepository

    db = DatabaseContext(str(tmp_path / "tiles.db"))
    repo = SQLiteImageRepository(db)
    tile = TileImage(
        file_path=str(tmp_path / "marble-white-60x60.jpg"),
        file_name="marble-white-60x60.jpg",
        file_size=10,
        dimensions="32x32",
        is_indexed=True,
    )
    tile_id = repo.add(tile)
    repo.mark_as_indexed(tile_id, True)

    found = repo.get_indexed_by_file_stem("marble-white-60x60")
    assert found is not None
    assert found.id == tile_id
    assert found.file_name == "marble-white-60x60.jpg"
    assert repo.get_indexed_by_file_stem("missing-product") is None


def test_dhash_from_image_matches_path_hash(tmp_path):
    path = tmp_path / "tile.jpg"
    Image.new("RGB", (64, 64), color=(40, 80, 120)).save(path)
    with Image.open(path) as img:
        from_mem = compute_dhash_from_image(img)
    from_path = compute_dhash(path)
    assert from_mem == from_path
    assert len(from_mem) == 16


def test_query_cache_key_uses_abs_path_size_mtime(tmp_path):
    path = tmp_path / "tile.jpg"
    Image.new("RGB", (16, 16), color=(1, 2, 3)).save(path)
    key = QueryEmbeddingCache.key_for_path(path)
    assert key is not None
    assert Path(key.path).is_absolute()
    assert key.size == path.stat().st_size
    assert key.mtime_ns == path.stat().st_mtime_ns
