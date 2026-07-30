"""
Benchmark optional FAISS backends vs IndexFlatIP (exact).

Writes JSON under /tmp/cursor/artifacts when run as a script.
Also importable from tests.

  python3 dev_tools/benchmark_index_backends.py
"""

from __future__ import annotations

import json
import sys
import tempfile
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.ai.index_backends import (  # noqa: E402
    BackendParams,
    IndexBackend,
    estimate_index_memory_mib,
)
from src.ai.vector_index import FaissIndexManager  # noqa: E402


def _build(backend: IndexBackend, n: int, dim: int) -> tuple[FaissIndexManager, np.ndarray, float]:
    tmp = Path(tempfile.mkdtemp()) / f"{backend.value}.index"
    params = BackendParams(
        hnsw_m=32,
        hnsw_ef_search=64,
        ivf_nlist=max(16, int(4 * (n**0.5)) // 4 * 4),
        ivf_nprobe=16,
        ivf_pq_m=16 if dim % 16 == 0 else 8,
    )
    mgr = FaissIndexManager(str(tmp), dimension=dim, backend=backend, backend_params=params)
    mgr.load_index()
    rng = np.random.default_rng(0)
    vectors = rng.standard_normal((n, dim), dtype=np.float32)
    vectors /= np.linalg.norm(vectors, axis=1, keepdims=True).clip(min=1e-12)
    t0 = time.perf_counter()
    chunk = 1000
    ids = list(range(1, n + 1))
    for start in range(0, n, chunk):
        end = min(n, start + chunk)
        mgr.add_vectors(ids[start:end], vectors[start:end], persist=False)
    build_ms = (time.perf_counter() - t0) * 1000.0
    return mgr, vectors, build_ms


def _recall_at_k(exact_ids: list[int], approx_ids: list[int], k: int = 10) -> float:
    if not exact_ids:
        return 1.0
    truth = set(exact_ids[:k])
    hit = len(truth.intersection(approx_ids[:k]))
    return hit / float(k)


def run_benchmark(n: int = 50_000, dim: int = 128, queries: int = 50) -> dict:
    flat_mgr, vectors, flat_build = _build(IndexBackend.FLAT_IP, n, dim)
    # Warm + baseline exact results
    exact_results = []
    flat_times = []
    for i in range(queries):
        t0 = time.perf_counter()
        ids, _ = flat_mgr.search_vectors(vectors[i], top_k=10)
        flat_times.append((time.perf_counter() - t0) * 1000.0)
        exact_results.append(ids)

    rows = [
        {
            "backend": "flat_ip",
            "ntotal": n,
            "dimension": dim,
            "build_ms": round(flat_build, 1),
            "search_median_ms": round(float(np.median(flat_times)), 3),
            "search_p95_ms": round(float(np.percentile(flat_times, 95)), 3),
            "recall@10": 1.0,
            "memory_estimate_mib": estimate_index_memory_mib(
                ntotal=n, dimension=dim, backend=IndexBackend.FLAT_IP
            ).total_mib,
            "exact": True,
        }
    ]

    for backend in (IndexBackend.HNSW, IndexBackend.IVF, IndexBackend.IVF_PQ):
        mgr, _, build_ms = _build(backend, n, dim)
        times = []
        recalls = []
        for i in range(queries):
            t0 = time.perf_counter()
            ids, _ = mgr.search_vectors(vectors[i], top_k=10)
            times.append((time.perf_counter() - t0) * 1000.0)
            recalls.append(_recall_at_k(exact_results[i], ids, 10))
        rows.append(
            {
                "backend": backend.value,
                "ntotal": n,
                "dimension": dim,
                "build_ms": round(build_ms, 1),
                "search_median_ms": round(float(np.median(times)), 3),
                "search_p95_ms": round(float(np.percentile(times, 95)), 3),
                "recall@10": round(float(np.mean(recalls)), 4),
                "memory_estimate_mib": estimate_index_memory_mib(
                    ntotal=n, dimension=dim, backend=backend
                ).total_mib,
                "exact": False,
            }
        )

    return {
        "title": "TileVision FAISS backend comparison",
        "note": "Production default remains flat_ip (exact). Approx backends are optional.",
        "rows": rows,
    }


def main() -> None:
    report = run_benchmark()
    out_dir = Path("/tmp/cursor/artifacts")
    out_dir.mkdir(parents=True, exist_ok=True)
    out = out_dir / "v12_backend_benchmark.json"
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
