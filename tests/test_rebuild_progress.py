"""
Rebuild progress must report COMPLETED files, not QUEUED ones.

Real Windows log (38 tiles, batch 12): UI jumped to 11/38 in ~1s with
"~10s remaining" then sat idle ~3 min while batch 1 embedded. Root cause
was progress_callback on queue into pending_batch. Also, closing the
QProgressDialog on success emitted canceled → spurious cancel log.
"""

from __future__ import annotations

import sys
import time
import types
from pathlib import Path
from unittest.mock import MagicMock

import pytest
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

faiss = pytest.importorskip("faiss")

from src.ai.feature_versions import CURRENT_EMBEDDING_DIMENSION
from src.ai.vector_index import FaissIndexManager
from src.config.indexing_performance import IndexingPerformanceConfig
from src.core.use_cases.index_images import IndexImagesUseCase
from src.data.db_context import DatabaseContext
from src.data.sqlite_repository import SQLiteImageRepository
from tests.fake_ai import FakeEmbedder, FakeFeatureExtractor


def _make_image(path: Path, color) -> None:
    Image.new("RGB", (32, 32), color=color).save(path)


@pytest.fixture()
def env(tmp_path):
    db_context = DatabaseContext(str(tmp_path / "db" / "tiles.db"))
    repo = SQLiteImageRepository(db_context)
    embedder = FakeEmbedder()
    feature_extractor = FakeFeatureExtractor(embedder=embedder)
    vector_index = FaissIndexManager(
        str(tmp_path / "index" / "tiles.index"),
        dimension=CURRENT_EMBEDDING_DIMENSION,
    )
    use_case = IndexImagesUseCase(
        image_repository=repo,
        feature_extractor=feature_extractor,
        vector_index=vector_index,
        thumbnail_dir=str(tmp_path / "thumbs"),
        performance=IndexingPerformanceConfig(batch_size=12),
    )
    images_dir = tmp_path / "images"
    images_dir.mkdir()
    return {
        "use_case": use_case,
        "repo": repo,
        "embedder": embedder,
        "vector_index": vector_index,
        "images_dir": images_dir,
        "feature_extractor": feature_extractor,
    }


def test_progress_reports_completed_not_queued_batch12_of_38(env):
    """
    With batch_size=12 and 38 files, callbacks must never report processed
    ahead of files actually finished. First flush is capped at 4 so the
    first completion sample arrives sooner; until then processed stays 0
    (aside from the initial scanning callback).
    """
    d = env["images_dir"]
    n = 38
    for i in range(n):
        _make_image(d / f"tile_{i:03d}.jpg", (i % 200, (i * 3) % 200, (i * 7) % 200))

    calls = []
    processed_while_first_batch_runs = []

    original_batch = env["use_case"]._index_file_batch
    flush_n = {"n": 0}

    def slowish_batch(items, persist=True, **kwargs):
        flush_n["n"] += 1
        if flush_n["n"] == 1:
            # Snapshot every progress report observed so far; first flush
            # has not finished yet so completed must still be 0.
            processed_while_first_batch_runs.extend(c[0] for c in calls)
            time.sleep(0.05)
        return original_batch(items, persist=persist, **kwargs)

    env["use_case"]._index_file_batch = slowish_batch

    def progress_cb(processed, total, filename, eta):
        calls.append((processed, total, filename, eta, time.perf_counter()))

    result = env["use_case"].scan_and_index_directory(
        d, progress_callback=progress_cb, force=True
    )
    assert result.is_completed is True
    assert result.new_count == n
    assert result.failed_count == 0

    # Never report more completed than files that have finished.
    # Monotonic non-decreasing completed count (allow final Done = total).
    prev = 0
    for processed, total, name, eta, _t in calls:
        assert total == n
        assert 0 <= processed <= total
        assert processed >= prev or name == "Done"
        prev = processed

    # Before first batch finishes, processed must still be 0 (scanning /
    # "Embedding first batch…"). Old bug: queued 11 within ~1s.
    assert processed_while_first_batch_runs, "first batch never ran"
    assert max(processed_while_first_batch_runs) == 0, (
        f"progress jumped to {max(processed_while_first_batch_runs)} "
        f"before first batch completed; calls={calls[:8]}"
    )

    # First non-scan callback should be the honest first-batch message.
    non_scan = [c for c in calls if c[2] != "Scanning catalogue…"]
    assert non_scan, "expected progress during embed"
    first_name = non_scan[0][2]
    assert "Embedding first batch" in first_name
    assert non_scan[0][0] == 0
    assert non_scan[0][3] < 0  # no fabricated numeric ETA

    # After first flush completes, processed should jump by first_batch_max (4),
    # not by a full queued batch of 12.
    after_first = [c for c in calls if c[0] > 0 and c[2] != "Done"]
    assert after_first, f"expected completion callbacks, got {calls}"
    assert after_first[0][0] == 4
    # No callback may report a processed count that was never a completed
    # batch boundary (4, then +12, +12, +10) — never mid-queue values like 11.
    allowed_mid = {0, 4, 16, 28, 38}
    mid_values = {c[0] for c in calls}
    assert mid_values <= allowed_mid | {n}, (
        f"unexpected processed values (queued mid-batch?): {sorted(mid_values)}"
    )


def test_first_callbacks_have_no_numeric_eta_until_batch_sample(env):
    """ETA must be -1 until a post-warmup batch timing sample exists."""
    d = env["images_dir"]
    for i in range(10):
        _make_image(d / f"t{i}.jpg", (i * 20, 40, 80))

    calls = []

    def progress_cb(processed, total, filename, eta):
        calls.append((processed, total, filename, eta))

    env["use_case"].scan_and_index_directory(d, progress_callback=progress_cb)
    assert calls
    # Before any completion, and right after warm-up batch 1, ETA is -1.
    before_second = [c for c in calls if c[0] < 8]
    assert before_second
    assert all(c[3] < 0 for c in before_second if c[2] != "Done"), (
        f"fabricated ETA before post-warmup sample: {before_second}"
    )
    # After second batch (processed >= 8 for 10 files: 4 then 6), ETA >= 0.
    after_second = [c for c in calls if c[0] >= 8 and c[2] != "Done"]
    assert after_second, f"expected post-warmup callbacks, got {calls}"
    assert any(c[3] >= 0 for c in after_second), (
        f"expected numeric ETA after second batch, got {after_second}"
    )


def test_slow_embedder_honest_message_then_sane_eta(env):
    """Faked slow batches: honest first-batch copy, then ETA after batch 2."""
    d = env["images_dir"]
    for i in range(8):
        _make_image(d / f"slow_{i}.jpg", (10, i * 20, 50))

    calls = []
    original = env["use_case"]._index_file_batch

    def slow_batch(items, persist=True, **kwargs):
        time.sleep(0.15 * max(1, len(items)))
        return original(items, persist=persist, **kwargs)

    env["use_case"]._index_file_batch = slow_batch

    def progress_cb(processed, total, filename, eta):
        calls.append((processed, total, filename, eta))

    result = env["use_case"].scan_and_index_directory(
        d, progress_callback=progress_cb
    )
    assert result.new_count == 8

    first_embed = next(c for c in calls if "Embedding first batch" in c[2])
    assert first_embed[0] == 0
    assert first_embed[3] < 0

    # After warm-up batch only: still estimating (eta < 0).
    after_first = [c for c in calls if c[0] == 4]
    assert after_first
    assert after_first[0][3] < 0 or "estimating" in after_first[0][2].lower()

    # After second batch: numeric ETA (8 files = 4 + 4).
    after_second = [c for c in calls if c[0] >= 8 and c[2] != "Done"]
    assert after_second
    assert after_second[0][3] >= 0
    if after_second[0][3] > 0:
        assert after_second[0][3] < 5.0, f"ETA inflated?: {after_second[0]}"


def test_closing_rebuild_dialog_after_finish_does_not_request_cancel():
    """Completion close must not log/behave as user cancel."""
    pytest.importorskip("PySide6")
    from PySide6.QtWidgets import QApplication, QProgressDialog

    from src.presentation.views.settings_view import SettingsView

    app = QApplication.instance() or QApplication(sys.argv)

    # Minimal stand-in with the real cancel/close helpers bound on.
    holder = types.SimpleNamespace()
    holder._rebuild_stall_timer = MagicMock()
    holder._rebuild_button = MagicMock()
    holder._rebuild_progress_dialog = QProgressDialog("Working", "Cancel", 0, 10)
    holder._rebuild_worker = MagicMock()
    holder._rebuild_closing_cleanly = False
    cancel_calls: list[int] = []

    def request_cancel():
        cancel_calls.append(1)

    holder._rebuild_worker.request_cancel = request_cancel
    holder._on_rebuild_cancel_clicked = types.MethodType(
        SettingsView._on_rebuild_cancel_clicked, holder
    )
    holder._stop_rebuild_ui = types.MethodType(SettingsView._stop_rebuild_ui, holder)

    holder._rebuild_progress_dialog.canceled.connect(holder._on_rebuild_cancel_clicked)

    # Successful finish closes the dialog (which would emit canceled).
    holder._stop_rebuild_ui()

    assert holder._rebuild_closing_cleanly is True
    assert cancel_calls == [], (
        "closing dialog after finish must not call request_cancel "
        f"(got {len(cancel_calls)} call(s))"
    )
    # Guard still holds if canceled somehow fires afterward.
    holder._on_rebuild_cancel_clicked()
    assert cancel_calls == []
    _ = app
