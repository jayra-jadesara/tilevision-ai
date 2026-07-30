"""
Enterprise reliability / scale stress harness for TileVision AI v1.2.

Runs without DINOv2 (synthetic vectors) so CI stays offline and fast enough
for the default suite. Heavy loops are gated behind markers.

Usage:
  pytest tests/test_v12_stress.py -q
  pytest tests/test_v12_stress.py -q -m slow   # 10k searches
"""

from __future__ import annotations

import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

faiss = pytest.importorskip("faiss")

from src.ai.index_backends import BackendParams, IndexBackend
from src.ai.inference_guard import begin_search_priority, end_search_priority
from src.ai.vector_index import FaissIndexManager
from src.core.compatibility import check_database_schema, check_index_metadata
from src.data.db_context import DatabaseContext


def _fill(manager: FaissIndexManager, n: int, dim: int = 32) -> np.ndarray:
    rng = np.random.default_rng(42)
    vectors = rng.standard_normal((n, dim), dtype=np.float32)
    vectors /= np.linalg.norm(vectors, axis=1, keepdims=True).clip(min=1e-12)
    ids = list(range(1, n + 1))
    # Add in chunks so IVF can train on first batch.
    chunk = max(64, min(512, n))
    for start in range(0, n, chunk):
        end = min(n, start + chunk)
        manager.add_vectors(ids[start:end], vectors[start:end], persist=False)
    return vectors


@pytest.fixture()
def small_flat(tmp_path):
    mgr = FaissIndexManager(str(tmp_path / "s.index"), dimension=32)
    mgr.load_index()
    vectors = _fill(mgr, 500)
    return mgr, vectors


def test_cancel_restart_lock_cycles(small_flat):
    """1000 begin/end search-priority cycles must not deadlock."""
    manager, vectors = small_flat
    errors: list[str] = []

    def worker(i: int) -> None:
        try:
            begin_search_priority()
            manager.search_vectors(vectors[i % len(vectors)], top_k=10)
        except Exception as exc:  # pragma: no cover
            errors.append(str(exc))
        finally:
            end_search_priority()

    for i in range(1000):
        worker(i)
    assert not errors


def test_indexing_while_searching(tmp_path):
    mgr = FaissIndexManager(str(tmp_path / "c.index"), dimension=32)
    mgr.load_index()
    base = _fill(mgr, 200)
    errors: list[str] = []
    stop = threading.Event()

    def search_loop() -> None:
        i = 0
        while not stop.is_set():
            try:
                begin_search_priority()
                mgr.search_vectors(base[i % len(base)], top_k=5)
            except Exception as exc:
                errors.append(f"search:{exc}")
            finally:
                end_search_priority()
            i += 1

    t = threading.Thread(target=search_loop, daemon=True)
    t.start()
    extra = np.random.default_rng(7).standard_normal((100, 32), dtype=np.float32)
    extra /= np.linalg.norm(extra, axis=1, keepdims=True).clip(min=1e-12)
    for i in range(100):
        mgr.update_vectors([10_000 + i], [extra[i]], persist=False)
        time.sleep(0.001)
    stop.set()
    t.join(timeout=5)
    assert not errors
    assert mgr.get_total_count() >= 300


def test_corrupt_index_recovers_to_empty(tmp_path):
    path = tmp_path / "corrupt.index"
    path.write_bytes(b"\xff\x00not-faiss")
    mgr = FaissIndexManager(str(path), dimension=8)
    mgr.load_index()  # should recreate empty rather than crash
    assert mgr.get_total_count() == 0


def test_corrupt_metadata_is_non_fatal(tmp_path):
    path = tmp_path / "t.index"
    path.write_bytes(b"x" * 32)
    meta = path.with_suffix(path.suffix + ".meta.json")
    meta.write_text("{not-json", encoding="utf-8")
    issues = check_index_metadata(path)
    # Missing/unreadable sidecar → warning, not crash
    assert isinstance(issues, list)


def test_missing_database_reported(tmp_path):
    issues = check_database_schema(tmp_path / "nope.db")
    assert any(i.code == "db_missing" for i in issues)


def test_sqlite_survives_pool_stress(tmp_path):
    db = DatabaseContext(str(tmp_path / "pool.db"))

    def poke(_: int) -> None:
        with db.session() as conn:
            conn.execute(
                "INSERT INTO tiles(file_path,file_name,file_size,dimensions) VALUES (?,?,?,?)",
                (f"/t{_}.jpg", f"t{_}.jpg", 1, "1x1"),
            )

    with ThreadPoolExecutor(max_workers=8) as pool:
        list(pool.map(poke, range(200)))
    with db.session() as conn:
        n = conn.execute("SELECT COUNT(*) AS c FROM tiles").fetchone()["c"]
    assert n == 200


@pytest.mark.slow
def test_ten_thousand_searches_flat(tmp_path):
    mgr = FaissIndexManager(str(tmp_path / "big.index"), dimension=32)
    mgr.load_index()
    vectors = _fill(mgr, 2000)
    t0 = time.perf_counter()
    for i in range(10_000):
        mgr.search_vectors(vectors[i % len(vectors)], top_k=10)
    elapsed = time.perf_counter() - t0
    # Soft ceiling — environment dependent; primarily a leak/deadlock check.
    assert elapsed < 180.0
    assert mgr.get_total_count() == 2000


@pytest.mark.slow
@pytest.mark.parametrize("backend", [IndexBackend.HNSW, IndexBackend.IVF])
def test_backend_scale_latency_sample(tmp_path, backend):
    dim = 64
    n = 20_000
    mgr = FaissIndexManager(
        str(tmp_path / f"{backend.value}.index"),
        dimension=dim,
        backend=backend,
        backend_params=BackendParams(ivf_nlist=64, ivf_nprobe=8, hnsw_m=16, hnsw_ef_search=32),
    )
    mgr.load_index()
    vectors = _fill(mgr, n, dim=dim)
    # Warm
    for i in range(20):
        mgr.search_vectors(vectors[i], top_k=10)
    times = []
    for i in range(100):
        t0 = time.perf_counter()
        mgr.search_vectors(vectors[i], top_k=10)
        times.append((time.perf_counter() - t0) * 1000.0)
    median = float(np.median(times))
    assert median < 50.0  # synthetic; ensures approximate path is fast
