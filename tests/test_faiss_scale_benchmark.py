"""
FAISS FlatIP search latency benchmarks at catalog scale.

Builds synthetic L2-normalized 1024-D vectors and measures pure FAISS
search latency (no DINOv2). Reports median of several queries for:

  1k, 10k, 50k, 100k, 250k

Run:
  python3 -m pytest tests/test_faiss_scale_benchmark.py -q -s
  # or:
  python3 tests/test_faiss_scale_benchmark.py
"""

from __future__ import annotations

import statistics
import sys
import time
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

faiss = pytest.importorskip("faiss")

from src.ai.vector_index import FaissIndexManager

CATALOG_SIZES = (1_000, 10_000, 50_000, 100_000, 250_000)
DIMENSION = 1024
TOP_K = 20
QUERY_REPEATS = 7
WARMUP = 2


def _build_index(tmp_path: Path, n_vectors: int) -> FaissIndexManager:
    manager = FaissIndexManager(
        str(tmp_path / f"bench_{n_vectors}.index"),
        dimension=DIMENSION,
    )
    manager.load_index()
    rng = np.random.default_rng(42)
    batch = 5_000
    for start in range(0, n_vectors, batch):
        ids = list(range(start, min(start + batch, n_vectors)))
        vectors = rng.normal(size=(len(ids), DIMENSION)).astype(np.float32)
        vectors /= np.linalg.norm(vectors, axis=1, keepdims=True) + 1e-8
        manager.add_vectors(ids, vectors.tolist(), persist=False)
    assert manager.get_total_count() == n_vectors
    assert "FlatIP" in manager.index_type_name()
    return manager


def _measure_search_ms(manager: FaissIndexManager, rng: np.random.Generator) -> list[float]:
    samples: list[float] = []
    for i in range(WARMUP + QUERY_REPEATS):
        query = rng.normal(size=(DIMENSION,)).astype(np.float32)
        query /= np.linalg.norm(query) + 1e-8
        start = time.perf_counter()
        ids, scores = manager.search_vectors(query.tolist(), top_k=TOP_K)
        elapsed_ms = (time.perf_counter() - start) * 1000.0
        assert len(ids) == TOP_K
        assert len(scores) == TOP_K
        if i >= WARMUP:
            samples.append(elapsed_ms)
    return samples


@pytest.mark.slow
@pytest.mark.parametrize("n_vectors", CATALOG_SIZES)
def test_faiss_flatip_latency_at_scale(tmp_path, n_vectors):
    manager = _build_index(tmp_path, n_vectors)
    rng = np.random.default_rng(7)
    samples = _measure_search_ms(manager, rng)
    median = statistics.median(samples)
    p95 = sorted(samples)[max(0, int(0.95 * (len(samples) - 1)))]
    print(
        f"\nFAISS IndexIDMap(IndexFlatIP) n={n_vectors:>7,}  "
        f"dim={DIMENSION} top_k={TOP_K}  "
        f"median={median:7.2f} ms  p95={p95:7.2f} ms  "
        f"type={manager.index_type_name()}",
        flush=True,
    )
    # Exact FlatIP stays interactive even at 250k on CPU.
    assert median < 2_000.0, f"median {median:.1f} ms too slow for n={n_vectors}"


def main() -> None:
    import tempfile

    print("=== TileVision FAISS FlatIP scale benchmark ===")
    print(f"dimension={DIMENSION} top_k={TOP_K} repeats={QUERY_REPEATS}")
    rows = []
    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        for n in CATALOG_SIZES:
            manager = _build_index(tmp_path, n)
            samples = _measure_search_ms(manager, np.random.default_rng(7))
            median = statistics.median(samples)
            p95 = sorted(samples)[max(0, int(0.95 * (len(samples) - 1)))]
            rows.append((n, median, p95, manager.index_type_name()))
            print(
                f"n={n:>7,}  median={median:7.2f} ms  p95={p95:7.2f} ms  "
                f"type={manager.index_type_name()}",
                flush=True,
            )
    print("\nSummary")
    print(f"{'Catalog':>10}  {'Median ms':>10}  {'p95 ms':>10}  Index")
    for n, median, p95, name in rows:
        print(f"{n:>10,}  {median:>10.2f}  {p95:>10.2f}  {name}")


if __name__ == "__main__":
    main()
