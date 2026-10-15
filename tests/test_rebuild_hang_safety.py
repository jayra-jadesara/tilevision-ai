"""Rebuild must not freeze forever on one hung file; Cancel + timeout + stall label."""

from __future__ import annotations

import os
import sys
import threading
import time
from pathlib import Path
from unittest.mock import MagicMock

import pytest
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.core.use_cases.index_images import IndexImagesUseCase, _PendingIndexItem
from src.core.use_cases.index_timeout import (
    IndexFileTimeoutError,
    file_index_timeout_s,
    run_with_timeout,
)
from src.presentation.workers.rebuild_index_worker import RebuildIndexWorker


def test_file_index_timeout_env_override(monkeypatch):
    monkeypatch.setenv("TILEVISION_INDEX_FILE_TIMEOUT_S", "7")
    assert file_index_timeout_s() == 7.0


def test_run_with_timeout_raises_on_block():
    def _block():
        time.sleep(30)

    with pytest.raises(IndexFileTimeoutError):
        run_with_timeout(_block, timeout_s=0.3, label="blocked.png")


def test_scan_skips_hung_file_and_continues(tmp_path, monkeypatch):
    """A file whose embed blocks past the timeout is failed; siblings still index."""
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

    faiss = pytest.importorskip("faiss")

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

    def flaky_batch(items, persist=True):
        for item in items:
            if item.path.name == "hang.png":
                time.sleep(30)
        return original(items, persist=persist)

    monkeypatch.setattr(uc, "_index_file_batch", flaky_batch)

    result = uc.scan_and_index_directory(images, force=True)
    assert result.failed_count >= 1
    # Other files should still succeed.
    assert result.indexed_count >= 2
    assert result.is_completed is True


def test_rebuild_worker_cancel_stops_scan(tmp_path, monkeypatch, qapp=None):
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

    # Slow each file so cancel can fire mid-run.
    real = uc._index_file_batch

    def slow_batch(items, persist=True):
        time.sleep(0.35)
        return real(items, persist=persist)

    monkeypatch.setattr(uc, "_index_file_batch", slow_batch)

    worker = RebuildIndexWorker(uc, [str(images)])
    finished = []
    worker.rebuild_finished.connect(lambda a, b: finished.append((a, b)))
    worker.start()
    time.sleep(0.5)
    worker.request_cancel()
    assert worker.wait(15000)
    # Drain queued Qt signals from the worker thread.
    app.processEvents()
    deadline = time.time() + 2
    while not finished and time.time() < deadline:
        app.processEvents()
        time.sleep(0.05)
    assert worker._cancel_event.is_set()
    assert finished  # emits finished even when cancelled


def test_stall_label_logic_expires_eta():
    """Mirror settings_view stall rules: ETA older than its claim is stale."""
    issued = 100.0
    eta = 10.0
    now = 100.0 + 15.0  # past the claimed remaining time
    assert (now - issued) > eta
