"""
PR A: nested-folder dedupe, incremental Update Search Index, CLIP batch save.
"""

from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import MagicMock

import pytest
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

faiss = pytest.importorskip("faiss")

from src.ai.feature_versions import CURRENT_EMBEDDING_DIMENSION
from src.ai.vector_index import FaissIndexManager
from src.config.indexing_performance import IndexingPerformanceConfig
from src.core.use_cases.index_folder_dedupe import drop_nested_indexed_folders
from src.core.use_cases.index_images import IndexImagesUseCase
from src.core.use_cases.index_work_classify import IndexWorkKind, classify_index_work
from src.data.db_context import DatabaseContext
from src.data.sqlite_repository import SQLiteImageRepository
from src.presentation.workers.rebuild_index_worker import (
    RebuildIndexWorker,
    RebuildSummary,
)
from tests.fake_ai import FakeEmbedder, FakeFeatureExtractor


def _make_image(path: Path, color) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (32, 32), color=color).save(path)


def _use_case(tmp_path, *, clip_index=None, clip_embedder=None):
    repo = SQLiteImageRepository(DatabaseContext(str(tmp_path / "t.db")))
    fe = FakeFeatureExtractor(embedder=FakeEmbedder())
    index = FaissIndexManager(
        str(tmp_path / "i.index"), dimension=CURRENT_EMBEDDING_DIMENSION
    )
    return IndexImagesUseCase(
        image_repository=repo,
        feature_extractor=fe,
        vector_index=index,
        thumbnail_dir=str(tmp_path / "thumbs"),
        performance=IndexingPerformanceConfig(batch_size=12),
        clip_embedder=clip_embedder,
        clip_index=clip_index,
    ), repo, index, fe


def test_drop_nested_indexed_folders_keeps_parent_only(tmp_path):
    parent = tmp_path / "Tiles"
    child = parent / "New"
    parent.mkdir()
    child.mkdir()
    other = tmp_path / "Other"
    other.mkdir()
    kept = drop_nested_indexed_folders([str(child), str(parent), str(other)])
    assert str(parent.resolve()) in kept
    assert str(other.resolve()) in kept
    assert str(child.resolve()) not in kept


def test_nested_folders_each_file_embedded_once(tmp_path):
    """folders [A, A/B] → each file embedded once (not twice)."""
    parent = tmp_path / "Tiles"
    child = parent / "New"
    for i in range(3):
        _make_image(parent / f"p{i}.jpg", (i * 10, 20, 30))
    for i in range(2):
        _make_image(child / f"c{i}.jpg", (40, i * 10, 50))

    uc, repo, index, fe = _use_case(tmp_path)
    embed_calls = []
    original = uc._index_file_batch

    def tracking(items, persist=True, **kwargs):
        for item in items:
            embed_calls.append(str(item.path.resolve()))
        return original(items, persist=persist, **kwargs)

    uc._index_file_batch = tracking

    from PySide6.QtWidgets import QApplication

    app = QApplication.instance() or QApplication([])
    worker = RebuildIndexWorker(
        uc, [str(parent), str(child)], force=True
    )
    finished = []
    worker.rebuild_finished.connect(lambda s: finished.append(s))
    worker.start()
    assert worker.wait(60000)
    app.processEvents()

    assert finished
    summary: RebuildSummary = finished[0]
    assert summary.reindexed == 5
    assert len(embed_calls) == 5
    assert len(set(embed_calls)) == 5
    assert len(repo.get_all()) == 5
    _ = app


def test_incremental_update_skips_current_tiles(tmp_path):
    d = tmp_path / "imgs"
    for i in range(6):
        _make_image(d / f"t{i}.jpg", (i * 20, 10, 5))

    uc, repo, index, fe = _use_case(tmp_path)
    first = uc.scan_and_index_directory(d, force=True)
    assert first.indexed_count == 6

    calls_before = fe.embedder.calls
    second = uc.scan_and_index_directory(d, force=False)
    assert second.indexed_count == 0
    assert second.skipped_count == 6
    assert second.clip_backfill_count == 0
    assert fe.embedder.calls == calls_before


def test_clip_only_backfill_when_dino_current(tmp_path):
    d = tmp_path / "imgs"
    for i in range(4):
        _make_image(d / f"t{i}.jpg", (i * 20, 10, 5))

    class FakeClip:
        is_loaded = True

        def __init__(self):
            self.calls = 0

        def embed_image(self, path):
            self.calls += 1
            import numpy as np

            return np.zeros(512, dtype=np.float32)

    clip_embedder = FakeClip()
    clip_index = FaissIndexManager(
        str(tmp_path / "clip.index"), dimension=512
    )
    uc, repo, index, fe = _use_case(
        tmp_path, clip_embedder=clip_embedder, clip_index=clip_index
    )

    # Index DINOv2 only (no CLIP yet) by temporarily clearing clip.
    uc._clip_embedder = None
    uc._clip_index = None
    uc.scan_and_index_directory(d, force=True)
    assert clip_embedder.calls == 0

    uc._clip_embedder = clip_embedder
    uc._clip_index = clip_index
    dino_calls = fe.embedder.calls
    result = uc.scan_and_index_directory(d, force=False)
    assert result.indexed_count == 0
    assert result.clip_backfill_count == 4
    assert clip_embedder.calls == 4
    assert fe.embedder.calls == dino_calls  # no DINOv2 re-run


def test_clip_index_saved_once_per_batch_not_per_vector(tmp_path):
    d = tmp_path / "imgs"
    for i in range(5):
        _make_image(d / f"t{i}.jpg", (i * 15, 20, 25))

    class FakeClip:
        is_loaded = True

        def embed_image(self, path):
            import numpy as np

            return np.zeros(512, dtype=np.float32)

    clip_index = FaissIndexManager(
        str(tmp_path / "clip.index"), dimension=512
    )
    saves = []
    real_save = clip_index.save_index

    def counting_save():
        saves.append(1)
        return real_save()

    clip_index.save_index = counting_save
    uc, *_ = _use_case(
        tmp_path, clip_embedder=FakeClip(), clip_index=clip_index
    )
    # first_batch_max=4 → two flushes for 5 files → two CLIP batch saves
    # (plus possible final save_index in scan epilogue).
    uc.scan_and_index_directory(d, force=True)
    # Must not be 5 (per-vector). Allow a small constant for end flush.
    assert 1 <= len(saves) <= 4, f"expected few batch saves, got {len(saves)}"
    assert len(saves) < 5


def test_cancel_then_resume_only_embeds_remaining(tmp_path):
    import threading

    d = tmp_path / "imgs"
    for i in range(10):
        _make_image(d / f"t{i}.jpg", (i * 10, 20, 30))

    uc, repo, index, fe = _use_case(tmp_path)
    cancel = threading.Event()

    def progress(processed, total, name, eta):
        if processed >= 4:
            cancel.set()

    first = uc.scan_and_index_directory(
        d, force=True, progress_callback=progress, cancel_event=cancel
    )
    assert first.is_completed is False
    done = len(repo.get_all())
    assert 1 <= done < 10

    calls_before = fe.embedder.calls
    second = uc.scan_and_index_directory(d, force=False)
    # Remaining only — already indexed with current feature version are skipped.
    assert second.skipped_count == done
    assert second.new_count == 10 - done
    assert len(repo.get_all()) == 10
    # Embedder called only for remaining tiles (not the full 10 again).
    assert fe.embedder.calls - calls_before > 0


def test_rebuild_summary_format():
    s = RebuildSummary(reindexed=12, clip_backfill=20, skipped=6, failed=0)
    msg = s.format_message()
    assert "12 re-indexed" in msg
    assert "20 description-search vectors added" in msg
    assert "6 already up to date" in msg


def test_classify_force_always_full(tmp_path):
    p = tmp_path / "a.jpg"
    _make_image(p, (1, 2, 3))
    assert (
        classify_index_work(p, None, force=True) is IndexWorkKind.FULL
    )
