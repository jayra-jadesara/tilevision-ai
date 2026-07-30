"""v1.1 performance optimization unit tests."""

from __future__ import annotations

import json
import sys
import tempfile
import threading
import time
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.ai.index_metadata import (
    FaissIndexMetadata,
    read_index_metadata,
    write_index_metadata,
)
from src.ai.vector_index import FaissIndexManager
from src.data.db_context import DatabaseContext
from src.data.sqlite_repository import SQLiteImageRepository
from src.core.models import TileImage
from tests.fake_ai import make_tile_features


def test_faiss_accepts_ndarray_query_without_tolist(tmp_path):
    faiss = pytest.importorskip("faiss")
    manager = FaissIndexManager(str(tmp_path / "t.index"), dimension=8)
    manager.load_index()
    vec = np.ones(8, dtype=np.float32)
    vec /= np.linalg.norm(vec)
    manager.add_vectors([1], [vec], persist=False)
    ids, scores = manager.search_vectors(vec, top_k=1)
    assert ids == [1]
    assert scores[0] > 0.99


def test_index_metadata_roundtrip_and_compat(tmp_path):
    index_path = tmp_path / "tiles.index"
    index_path.write_bytes(b"x")
    write_index_metadata(index_path, faiss_type="IndexIDMap(IndexFlatIP)", ntotal=3)
    meta = read_index_metadata(index_path)
    assert meta is not None
    assert meta.is_compatible()
    assert meta.ntotal == 3


def test_index_metadata_incompatible_flag():
    meta = FaissIndexMetadata(
        embedding_model="other",
        embedding_dimension=512,
        feature_version=1,
        app_version="0.0.1",
        faiss_type="IndexFlatIP",
        ntotal=1,
        build_date="x",
        catalog_version=1,
    )
    assert meta.is_compatible() is False


def test_sqlite_thread_local_pool_reuses_connection(tmp_path):
    db = DatabaseContext(str(tmp_path / "t.db"))
    ids = []
    with db.session() as conn:
        ids.append(id(conn))
        conn.execute(
            "INSERT INTO tiles(file_path,file_name,file_size,dimensions) VALUES (?,?,?,?)",
            ("/a.jpg", "a.jpg", 1, "1x1"),
        )
    with db.session() as conn:
        ids.append(id(conn))
        row = conn.execute("SELECT file_name FROM tiles WHERE file_path=?", ("/a.jpg",)).fetchone()
        assert row["file_name"] == "a.jpg"
    assert ids[0] == ids[1]

    other = []

    def worker():
        with db.session() as conn:
            other.append(id(conn))

    t = threading.Thread(target=worker)
    t.start()
    t.join()
    assert other and other[0] != ids[0]
    db.close_all()


def test_feature_version_status_is_cached(tmp_path):
    db = DatabaseContext(str(tmp_path / "t.db"))
    repo = SQLiteImageRepository(db)
    path = tmp_path / "a.jpg"
    Image.new("RGB", (16, 16)).save(path)
    emb = np.ones(1024, dtype=np.float32)
    emb /= np.linalg.norm(emb)
    tile = TileImage(
        file_path=str(path),
        file_name="a.jpg",
        file_size=1,
        dimensions="16x16",
        features=make_tile_features(emb),
        is_indexed=True,
    )
    tid = repo.add(tile)
    repo.mark_as_indexed(tid, True)

    t0 = time.perf_counter()
    a = repo.get_feature_version_status()
    first = time.perf_counter() - t0
    t1 = time.perf_counter()
    b = repo.get_feature_version_status()
    second = time.perf_counter() - t1
    assert a.is_compatible and b.is_compatible
    assert second < first * 0.5 or second < 0.001
    repo.mark_as_indexed(tid, False)
    c = repo.get_feature_version_status()
    # Cache invalidated — may still be compatible with 0 indexed.
    assert c.indexed_count == 0


def test_wal_mode_enabled(tmp_path):
    db = DatabaseContext(str(tmp_path / "t.db"))
    with db.session() as conn:
        mode = conn.execute("PRAGMA journal_mode;").fetchone()[0]
    assert str(mode).lower() == "wal"
    db.close_all()


def test_thumbnail_cache_lru(qapp=None):
    pytest.importorskip("PySide6")
    from PySide6.QtWidgets import QApplication
    from src.presentation.thumbnail_cache import ThumbnailPixmapCache

    app = QApplication.instance() or QApplication(sys.argv)
    cache = ThumbnailPixmapCache(capacity=2, max_edge=64)
    with tempfile.TemporaryDirectory() as td:
        paths = []
        for i in range(3):
            p = Path(td) / f"{i}.png"
            Image.new("RGB", (80, 80), color=(i * 40, 10, 10)).save(p)
            paths.append(p)
        cache.get(paths[0])
        cache.get(paths[1])
        assert len(cache) == 2
        cache.get(paths[2])
        assert len(cache) == 2
        assert app is not None
