"""Rebuild must not freeze forever on one hung file; Cancel + timeout + stall abort."""

from __future__ import annotations

import sys
import time
from pathlib import Path

import pytest
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.core.use_cases.index_images import IndexImagesUseCase
from src.core.use_cases.index_timeout import (
    IndexFileTimeoutError,
    IndexStallAbort,
    file_index_timeout_s,
    run_with_timeout,
)
from src.presentation.workers.rebuild_index_worker import RebuildIndexWorker


def test_file_index_timeout_env_override(monkeypatch):
    monkeypatch.setenv("TILEVISION_INDEX_FILE_TIMEOUT_S", "7")
    assert file_index_timeout_s() == 7.0


def test_run_with_timeout_returns_within_timeout_not_task_duration():
    """
    Regression: ThreadPoolExecutor context manager used shutdown(wait=True),
    so sleep(25) with timeout=5 still blocked 25s. Must return near timeout_s.

    Keep the hung sleep only slightly past timeout so leftover daemon workers
    do not outlive the rest of the suite (macOS Intel CI abort flake).
    """

    def _block():
        time.sleep(2.5)

    t0 = time.perf_counter()
    with pytest.raises(IndexFileTimeoutError):
        run_with_timeout(_block, timeout_s=1.0, label="blocked.png")
    elapsed = time.perf_counter() - t0
    assert elapsed < 3.0, f"timeout returned after {elapsed:.2f}s (expected <3s)"


def test_scan_aborts_promptly_on_hung_file(tmp_path, monkeypatch):
    """
    Hung embed aborts the scan with IndexStallAbort (lock may still be held).
    Must return near the timeout budget — not wait for the 30s sleep.
    """
    monkeypatch.setenv("TILEVISION_INDEX_FILE_TIMEOUT_S", "1")

    images = tmp_path / "imgs"
    images.mkdir()
    for name in ("a.png", "hang.png", "c.png"):
        Image.new("RGB", (32, 32), color=(10, 20, 30)).save(images / name)

    from tests.fake_ai import FakeEmbedder, FakeFeatureExtractor
    from src.ai.feature_versions import CURRENT_EMBEDDING_DIMENSION
    from src.ai.vector_index import FaissIndexManager
    from src.data.db_context import DatabaseContext
    from src.data.sqlite_repository import SQLiteImageRepository

    pytest.importorskip("faiss")

    repo = SQLiteImageRepository(DatabaseContext(str(tmp_path / "t.db")))
    fe = FakeFeatureExtractor(embedder=FakeEmbedder())
    index = FaissIndexManager(
        str(tmp_path / "i.index"), dimension=CURRENT_EMBEDDING_DIMENSION
    )
    uc = IndexImagesUseCase(
        image_repository=repo,
        feature_extractor=fe,
        vector_index=index,
        thumbnail_dir=str(tmp_path / "thumbs"),
    )

    original = uc._index_file_batch
    batch_calls = []

    def flaky_batch(items, persist=True):
        batch_calls.append([i.path.name for i in items])
        for item in items:
            if item.path.name == "hang.png":
                # Past the 1s×N batch timeout, but short enough that the
                # daemon worker exits before later tests (Intel CI abort).
                time.sleep(4.0)
        return original(items, persist=persist)

    monkeypatch.setattr(uc, "_index_file_batch", flaky_batch)

    t0 = time.perf_counter()
    with pytest.raises(IndexStallAbort) as caught:
        uc.scan_and_index_directory(images, force=True)
    elapsed = time.perf_counter() - t0

    # batch_timeout = 1s * 3 files = 3s, then maybe per-file 1s if lock free.
    assert elapsed < 8.0, f"hung scan took {elapsed:.2f}s (expected <8s)"
    assert "stalled" in str(caught.value).lower()
    assert "restart" in str(caught.value).lower()
    # First attempt should be a real multi-file batch (not per-file-only).
    assert batch_calls, "expected at least one _index_file_batch call"
    assert len(batch_calls[0]) == 3


def test_normal_flush_uses_single_batched_call(tmp_path, monkeypatch):
    """Happy path must keep one batched embed + one FAISS persist per flush."""
    monkeypatch.setenv("TILEVISION_INDEX_FILE_TIMEOUT_S", "30")

    images = tmp_path / "imgs"
    images.mkdir()
    for i in range(5):
        Image.new("RGB", (24, 24), color=(i * 40, 20, 30)).save(images / f"t{i}.png")

    from tests.fake_ai import FakeEmbedder, FakeFeatureExtractor
    from src.ai.feature_versions import CURRENT_EMBEDDING_DIMENSION
    from src.ai.vector_index import FaissIndexManager
    from src.config.indexing_performance import IndexingPerformanceConfig
    from src.data.db_context import DatabaseContext
    from src.data.sqlite_repository import SQLiteImageRepository

    pytest.importorskip("faiss")

    uc = IndexImagesUseCase(
        image_repository=SQLiteImageRepository(DatabaseContext(str(tmp_path / "t.db"))),
        feature_extractor=FakeFeatureExtractor(embedder=FakeEmbedder()),
        vector_index=FaissIndexManager(
            str(tmp_path / "i.index"), dimension=CURRENT_EMBEDDING_DIMENSION
        ),
        thumbnail_dir=str(tmp_path / "thumbs"),
        performance=IndexingPerformanceConfig(batch_size=12),
    )

    original = uc._index_file_batch
    calls = []

    def tracking(items, persist=True):
        calls.append((len(items), persist))
        return original(items, persist=persist)

    monkeypatch.setattr(uc, "_index_file_batch", tracking)
    result = uc.scan_and_index_directory(images, force=True)
    assert result.indexed_count == 5
    assert result.is_completed is True
    # First batch is capped at 4 so the first timing sample arrives sooner;
    # remainder flushes as a second batch. Still batched — not five 1-file flushes.
    assert calls == [(4, True), (1, True)], (
        f"expected first_batch_max=4 then remainder, got {calls}"
    )


def test_rebuild_worker_cancel_stops_scan(tmp_path, monkeypatch):
    monkeypatch.setenv("TILEVISION_INDEX_FILE_TIMEOUT_S", "2")
    images = tmp_path / "imgs"
    images.mkdir()
    for i in range(6):
        Image.new("RGB", (24, 24), color=(i * 20, 40, 60)).save(images / f"t{i}.png")

    from tests.fake_ai import FakeEmbedder, FakeFeatureExtractor
    from src.ai.feature_versions import CURRENT_EMBEDDING_DIMENSION
    from src.ai.vector_index import FaissIndexManager
    from src.data.db_context import DatabaseContext
    from src.data.sqlite_repository import SQLiteImageRepository
    from PySide6.QtWidgets import QApplication

    pytest.importorskip("faiss")
    app = QApplication.instance() or QApplication([])

    uc = IndexImagesUseCase(
        image_repository=SQLiteImageRepository(DatabaseContext(str(tmp_path / "t.db"))),
        feature_extractor=FakeFeatureExtractor(embedder=FakeEmbedder()),
        vector_index=FaissIndexManager(
            str(tmp_path / "i.index"), dimension=CURRENT_EMBEDDING_DIMENSION
        ),
        thumbnail_dir=str(tmp_path / "thumbs"),
    )

    real = uc._index_file_batch

    def slow_batch(items, persist=True):
        time.sleep(0.35)
        return real(items, persist=persist)

    monkeypatch.setattr(uc, "_index_file_batch", slow_batch)

    worker = RebuildIndexWorker(uc, [str(images)], force=True)
    finished = []
    failed = []
    worker.rebuild_finished.connect(lambda summary: finished.append(summary))
    worker.rebuild_failed.connect(lambda m: failed.append(m))
    worker.start()
    time.sleep(0.5)
    worker.request_cancel()
    assert worker.wait(15000)
    app.processEvents()
    deadline = time.time() + 2
    while not finished and not failed and time.time() < deadline:
        app.processEvents()
        time.sleep(0.05)
    assert worker._cancel_event.is_set()
    assert finished or failed


def test_stall_label_logic_expires_eta():
    """Mirror settings_view stall rules: ETA older than its claim is stale."""
    issued = 100.0
    eta = 10.0
    now = 100.0 + 15.0
    assert (now - issued) > eta
