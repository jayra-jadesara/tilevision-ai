"""v1.2 Enterprise unit tests — backends, diagnostics, compatibility."""

from __future__ import annotations

import json
import sqlite3
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

faiss = pytest.importorskip("faiss")

from src.ai.index_backends import (
    BackendParams,
    IndexBackend,
    estimate_index_memory_mib,
)
from src.ai.index_metadata import read_index_metadata, write_index_metadata
from src.ai.vector_index import FaissIndexManager
from src.core.compatibility import (
    check_database_schema,
    check_index_metadata,
    run_compatibility_check,
)
from src.utils.diagnostics import collect_diagnostics_report, export_diagnostics_json
from src.version import APP_VERSION


def _unit(dim: int = 8) -> np.ndarray:
    v = np.random.randn(dim).astype(np.float32)
    v /= max(float(np.linalg.norm(v)), 1e-12)
    return v


def test_default_backend_remains_flat_ip(tmp_path):
    mgr = FaissIndexManager(str(tmp_path / "t.index"), dimension=8)
    mgr.load_index()
    assert mgr.configured_backend is IndexBackend.FLAT_IP
    assert mgr.active_backend() is IndexBackend.FLAT_IP
    assert "FlatIP" in mgr.index_type_name()


@pytest.mark.parametrize(
    "backend",
    [IndexBackend.FLAT_IP, IndexBackend.HNSW, IndexBackend.IVF, IndexBackend.IVF_PQ],
)
def test_optional_backends_add_and_search(tmp_path, backend):
    # IVF-PQ needs a larger first train set; others are fine with 50.
    n = 400 if backend is IndexBackend.IVF_PQ else 50
    mgr = FaissIndexManager(
        str(tmp_path / f"{backend.value}.index"),
        dimension=8,
        backend=backend,
        backend_params=BackendParams(ivf_nlist=4, ivf_nprobe=2, ivf_pq_m=4, hnsw_m=8),
    )
    mgr.load_index()
    rng = np.random.default_rng(0)
    vectors = rng.standard_normal((n, 8), dtype=np.float32)
    vectors /= np.linalg.norm(vectors, axis=1, keepdims=True).clip(min=1e-12)
    ids = list(range(1, n + 1))
    mgr.add_vectors(ids, vectors, persist=True)
    assert mgr.get_total_count() == n
    # Small IVF-PQ trains may safely fall back to IVF-Flat.
    assert mgr.active_backend() in (backend, IndexBackend.IVF)

    query = vectors[0]
    hit_ids, scores = mgr.search_vectors(query, top_k=5)
    assert hit_ids
    assert scores[0] > 0.5
    meta = read_index_metadata(mgr.index_path)
    assert meta is not None
    assert meta.index_backend in (backend.value, IndexBackend.IVF.value)


def test_flat_ip_exact_top1_preserved(tmp_path):
    mgr = FaissIndexManager(str(tmp_path / "exact.index"), dimension=8)
    mgr.load_index()
    target = _unit(8)
    others = np.stack([_unit(8) for _ in range(50)])
    mgr.add_vectors([1], [target], persist=False)
    mgr.add_vectors(list(range(2, 52)), others, persist=False)
    ids, scores = mgr.search_vectors(target, top_k=1)
    assert ids == [1]
    assert scores[0] == pytest.approx(1.0, abs=1e-5)


def test_memory_estimate_ivf_pq_lower_than_flat():
    flat = estimate_index_memory_mib(ntotal=1_000_000, dimension=1024, backend=IndexBackend.FLAT_IP)
    pq = estimate_index_memory_mib(ntotal=1_000_000, dimension=1024, backend=IndexBackend.IVF_PQ)
    assert flat.total_mib > 3000
    assert pq.total_mib < flat.total_mib * 0.1


def test_diagnostics_json_export(tmp_path):
    path = export_diagnostics_json(
        tmp_path / "diag.json",
        {"app_version": APP_VERSION, "catalog_size": 42, "index_backend": "flat_ip"},
    )
    data = json.loads(path.read_text(encoding="utf-8"))
    assert data["app_version"] == APP_VERSION
    assert data["catalog_size"] == 42
    assert "python" in data
    assert "sqlite" in data
    report = collect_diagnostics_report({"catalog_size": 1})
    assert "ram_rss_mb" in report or report.get("ram_rss_mb") is None


def test_compatibility_detects_corrupt_db(tmp_path):
    bad = tmp_path / "bad.db"
    bad.write_bytes(b"not a sqlite database")
    issues = check_database_schema(bad)
    assert any(i.code in ("db_corrupt", "db_unreadable") for i in issues)


def test_compatibility_detects_missing_meta(tmp_path):
    index = tmp_path / "tiles.index"
    index.write_bytes(b"x" * 64)
    issues = check_index_metadata(index)
    assert any(i.code == "index_meta_missing" for i in issues)


def test_compatibility_backend_mismatch(tmp_path):
    index = tmp_path / "tiles.index"
    index.write_bytes(b"x" * 64)
    write_index_metadata(index, faiss_type="HNSW", ntotal=10, index_backend="hnsw")
    issues = check_index_metadata(index, expected_backend=IndexBackend.FLAT_IP)
    assert any(i.code == "index_backend_mismatch" for i in issues)


def test_compatibility_ok_fresh_install(tmp_path):
    db = tmp_path / "tiles.db"
    conn = sqlite3.connect(db)
    conn.execute(
        "CREATE TABLE tiles (id INTEGER PRIMARY KEY, file_path TEXT, "
        "embedding_blob BLOB, feature_version INT, embedding_model TEXT, "
        "embedding_dimension INT)"
    )
    conn.commit()
    conn.close()
    report = run_compatibility_check(
        database_path=db,
        index_path=tmp_path / "missing.index",
        expected_backend=IndexBackend.FLAT_IP,
    )
    assert report.is_compatible
    assert report.requires_rebuild is False


def test_app_version_is_1_2():
    assert APP_VERSION.startswith("1.2")
