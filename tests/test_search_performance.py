"""
Performance test for Feature 2's <2s @ 50,000 images target.

Builds a real FAISS index (IndexFlatIP via FaissIndexManager) with 50,000
synthetic 512-dim vectors and measures pure vector-search latency. Model
inference time (CLIP forward pass) isn't reproducible without torch/GPU in
this environment, so this isolates and verifies the piece that scales with
catalog size: the FAISS query itself must not become the bottleneck.
"""

import sys
import time
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

faiss = pytest.importorskip("faiss")

from src.ai.vector_index import FaissIndexManager


@pytest.mark.slow
def test_faiss_search_under_2s_at_50k_scale(tmp_path):
    dimension = 512
    n_vectors = 50_000

    manager = FaissIndexManager(str(tmp_path / "perf.index"), dimension=dimension)
    manager.load_index()

    rng = np.random.default_rng(42)
    batch = 5_000
    for start in range(0, n_vectors, batch):
        ids = list(range(start, min(start + batch, n_vectors)))
        vectors = rng.normal(size=(len(ids), dimension)).astype(np.float32)
        # L2-normalize, matching how real CLIP embeddings are stored.
        vectors /= np.linalg.norm(vectors, axis=1, keepdims=True)
        manager.add_vectors(ids, vectors.tolist(), persist=False)

    assert manager._index.ntotal == n_vectors

    query = rng.normal(size=(dimension,)).astype(np.float32)
    query /= np.linalg.norm(query)

    start_time = time.monotonic()
    ids, scores = manager.search_vectors(query.tolist(), top_k=20)
    elapsed = time.monotonic() - start_time

    assert len(ids) == 20
    # The FAISS query portion alone should be a tiny fraction of the 2s
    # budget, leaving ample headroom for CLIP inference + DB hydration.
    assert elapsed < 0.5, f"FAISS search took {elapsed:.3f}s for {n_vectors} vectors — too slow"
