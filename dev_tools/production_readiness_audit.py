#!/usr/bin/env python3
"""
Production-readiness audit harness (read-mostly).

Measures FAISS/RSS memory at catalog scales, runs 1,000 consecutive
searches with a fake embedder, and reports latency/RSS/thread trends.
Does not require the real DINOv2 weights.
"""

from __future__ import annotations

import gc
import os
import statistics
import sys
import tempfile
import threading
import time
import tracemalloc
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import faiss  # noqa: E402

from src.ai.feature_versions import CURRENT_EMBEDDING_DIMENSION  # noqa: E402
from src.ai.query_cache import QUERY_EMBEDDING_CACHE  # noqa: E402
from src.ai.vector_index import FaissIndexManager  # noqa: E402
from src.core.models import TileImage  # noqa: E402
from src.core.use_cases.search_tiles import SearchTilesUseCase  # noqa: E402
from src.data.db_context import DatabaseContext  # noqa: E402
from src.data.sqlite_repository import SQLiteImageRepository  # noqa: E402
from tests.fake_ai import FakeFeatureExtractor, make_tile_features  # noqa: E402


def rss_mb() -> float:
    try:
        with open("/proc/self/status", encoding="utf-8") as handle:
            for line in handle:
                if line.startswith("VmRSS:"):
                    return int(line.split()[1]) / 1024.0
    except Exception:
        pass
    import resource

    usage = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return usage / 1024.0 if usage < 10_000_000 else usage / (1024.0 * 1024.0)


def thread_count() -> int:
    return threading.active_count()


def faiss_bytes(n: int, dim: int = CURRENT_EMBEDDING_DIMENSION) -> int:
    # FlatIP float32 vectors + IDMap int64 ids (approx).
    return n * dim * 4 + n * 8


def build_faiss(tmp: Path, n: int, dim: int = CURRENT_EMBEDDING_DIMENSION) -> FaissIndexManager:
    manager = FaissIndexManager(str(tmp / f"scale_{n}.index"), dimension=dim)
    manager.load_index()
    rng = np.random.default_rng(42)
    batch = 5_000
    for start in range(0, n, batch):
        ids = list(range(start, min(start + batch, n)))
        vecs = rng.normal(size=(len(ids), dim)).astype(np.float32)
        vecs /= np.linalg.norm(vecs, axis=1, keepdims=True) + 1e-8
        manager.add_vectors(ids, vecs.tolist(), persist=False)
    return manager


def measure_catalog_memory(sizes: list[int]) -> list[dict]:
    rows = []
    baseline = rss_mb()
    print(f"Baseline RSS before FAISS builds: {baseline:.1f} MiB")
    with tempfile.TemporaryDirectory() as td:
        tmp = Path(td)
        for n in sizes:
            gc.collect()
            before = rss_mb()
            manager = build_faiss(tmp, n)
            after = rss_mb()
            # Probe search latency
            q = np.random.default_rng(1).normal(size=(CURRENT_EMBEDDING_DIMENSION,)).astype(np.float32)
            q /= np.linalg.norm(q) + 1e-8
            samples = []
            for _ in range(5):
                t0 = time.perf_counter()
                manager.search_vectors(q.tolist(), top_k=20)
                samples.append((time.perf_counter() - t0) * 1000.0)
            row = {
                "n": n,
                "rss_mb": after,
                "delta_rss_mb": after - before,
                "faiss_est_mb": faiss_bytes(n) / (1024 * 1024),
                "median_search_ms": statistics.median(samples),
                "index_type": manager.index_type_name(),
            }
            rows.append(row)
            print(
                f"n={n:>7,}  RSS={after:8.1f} MiB  Δ={row['delta_rss_mb']:7.1f} MiB  "
                f"FAISS≈{row['faiss_est_mb']:7.1f} MiB  search_med={row['median_search_ms']:6.2f} ms  "
                f"{row['index_type']}",
                flush=True,
            )
            del manager
            gc.collect()
    return rows


def setup_search_env(tmp: Path, catalog: int = 2_000):
    db = DatabaseContext(str(tmp / "tiles.db"))
    repo = SQLiteImageRepository(db)
    embedder_dim = CURRENT_EMBEDDING_DIMENSION
    fe = FakeFeatureExtractor()
    index = FaissIndexManager(str(tmp / "search.index"), dimension=embedder_dim)
    index.load_index()
    thumbs = tmp / "thumbs"
    thumbs.mkdir()
    rng = np.random.default_rng(0)
    images = tmp / "images"
    images.mkdir()
    for i in range(catalog):
        path = images / f"tile_{i:05d}.jpg"
        Image.new("RGB", (32, 32), color=(i % 255, 40, 80)).save(path)
        emb = rng.normal(size=(embedder_dim,)).astype(np.float32)
        emb /= np.linalg.norm(emb) + 1e-8
        tile = TileImage(
            file_path=str(path),
            file_name=path.name,
            file_size=path.stat().st_size,
            dimensions="32x32",
            is_indexed=False,
        )
        # FakeFeatureExtractor.extract will produce features; for FAISS we add vectors directly
        # and mark indexed via a minimal feature blob path using repo after fake extract.
        from src.utils.image_utils import compute_sha256

        features = make_tile_features(emb)
        tile.features = features
        tile.sha256_hash = compute_sha256(path)
        tile_id = repo.add(tile)
        index.add_vectors([tile_id], [emb.tolist()], persist=False)
        repo.mark_as_indexed(tile_id, True)
    index.save_index()
    use_case = SearchTilesUseCase(repo, fe, index, str(thumbs))
    query_paths = sorted(images.glob("*.jpg"))
    return use_case, query_paths


def run_1000_searches(n: int = 1000) -> dict:
    QUERY_EMBEDDING_CACHE.clear()
    with tempfile.TemporaryDirectory() as td:
        tmp = Path(td)
        print("Building 2k fake catalog for stability run…", flush=True)
        use_case, paths = setup_search_env(tmp, catalog=2_000)
        gc.collect()
        rss0 = rss_mb()
        threads0 = thread_count()
        latencies = []
        tracemalloc.start()
        snap0 = tracemalloc.take_snapshot()
        t_start = time.perf_counter()
        for i in range(n):
            path = paths[i % len(paths)]
            # Alternate cache hit (same path twice) and miss patterns.
            if i % 3 == 0:
                QUERY_EMBEDDING_CACHE.clear()
            t0 = time.perf_counter()
            results = use_case.execute(str(path), top_k=20)
            latencies.append(time.perf_counter() - t0)
            assert isinstance(results, list)
            if (i + 1) % 200 == 0:
                print(
                    f"  completed {i+1}/{n}  last={latencies[-1]*1000:.1f} ms  "
                    f"RSS={rss_mb():.1f} MiB  threads={thread_count()}",
                    flush=True,
                )
        wall = time.perf_counter() - t_start
        snap1 = tracemalloc.take_snapshot()
        top = snap1.compare_to(snap0, "lineno")[:5]
        rss1 = rss_mb()
        threads1 = thread_count()
        # Latency trend: first 100 vs last 100
        first = statistics.median(latencies[:100]) * 1000
        last = statistics.median(latencies[-100:]) * 1000
        report = {
            "n": n,
            "wall_s": wall,
            "median_ms": statistics.median(latencies) * 1000,
            "p95_ms": sorted(latencies)[int(0.95 * (len(latencies) - 1))] * 1000,
            "first100_median_ms": first,
            "last100_median_ms": last,
            "rss_start_mb": rss0,
            "rss_end_mb": rss1,
            "rss_delta_mb": rss1 - rss0,
            "threads_start": threads0,
            "threads_end": threads1,
            "top_alloc_growth": [
                (str(stat.traceback), stat.size_diff / 1024)
                for stat in top
            ],
        }
        print("\n=== 1000-search stability ===")
        for k, v in report.items():
            if k == "top_alloc_growth":
                print("top_alloc_growth:")
                for item in v:
                    print(f"  {item[1]:8.1f} KiB  {item[0]}")
            else:
                print(f"{k}: {v}")
        tracemalloc.stop()
        return report


def stress_cancel_pattern() -> None:
    """Simulate rapid cancel/restart at use-case level + concurrent FAISS."""
    from src.ai.inference_guard import (
        begin_search_priority,
        end_search_priority,
        synchronized_inference,
    )

    errors = []
    def indexer():
        try:
            for _ in range(50):
                with synchronized_inference(timeout=5.0, purpose="index-stress"):
                    time.sleep(0.01)
                time.sleep(0.005)
        except Exception as exc:
            errors.append(exc)

    def searcher():
        try:
            for _ in range(50):
                begin_search_priority()
                try:
                    with synchronized_inference(timeout=5.0, purpose="search-stress"):
                        time.sleep(0.01)
                finally:
                    end_search_priority()
                time.sleep(0.002)
        except Exception as exc:
            errors.append(exc)

    threads = [threading.Thread(target=indexer), threading.Thread(target=searcher)]
    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout=30)
    print(f"\n=== Concurrent search/index stress === errors={errors}")


def main() -> None:
    print("=== MEMORY / FAISS SCALE ===")
    sizes = [10_000, 50_000, 100_000, 250_000, 500_000]
    mem_rows = measure_catalog_memory(sizes)
    print("\n=== STABILITY ===")
    stab = run_1000_searches(1000)
    stress_cancel_pattern()
    print("\n=== DONE ===")
    return mem_rows, stab


if __name__ == "__main__":
    main()
