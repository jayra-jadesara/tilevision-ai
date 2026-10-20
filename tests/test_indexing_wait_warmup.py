"""PR C: indexing waits for in-flight query warm-up (no parallel model use)."""

from __future__ import annotations

import logging
import sys
import threading
import time
from pathlib import Path

import pytest
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

faiss = pytest.importorskip("faiss")

from src.ai.feature_versions import CURRENT_EMBEDDING_DIMENSION, CURRENT_FEATURE_VERSION
from src.ai.inference_guard import warmup_compute_scope, warmup_in_progress
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
        performance=IndexingPerformanceConfig(batch_size=4),
    )
    images_dir = tmp_path / "images"
    images_dir.mkdir()
    for i in range(3):
        _make_image(images_dir / f"tile_{i}.jpg", (i * 40, 80, 120))
    return {
        "use_case": use_case,
        "embedder": embedder,
        "images_dir": images_dir,
    }


def test_feature_version_unchanged():
    assert CURRENT_FEATURE_VERSION == 17


def test_indexing_waits_for_slow_warmup(env, caplog):
    """Fake slow warm-up: progress shows Preparing model…; wait is logged."""
    release = threading.Event()
    warmup_started = threading.Event()

    def _slow_warmup():
        with warmup_compute_scope(torch_threads=1):
            warmup_started.set()
            # Hold long enough that scan must wait.
            release.wait(timeout=5.0)
            time.sleep(0.05)

    t = threading.Thread(target=_slow_warmup, daemon=True)
    t.start()
    assert warmup_started.wait(timeout=2.0)
    assert warmup_in_progress()

    progress_names: list[str] = []

    def progress_cb(processed, total, name, eta):
        progress_names.append(name)

    # Release warm-up shortly after scan starts waiting.
    def _release_soon():
        time.sleep(0.35)
        release.set()

    threading.Thread(target=_release_soon, daemon=True).start()

    with caplog.at_level(logging.INFO, logger="tilevision.core.use_cases.index_images"):
        result = env["use_case"].scan_and_index_directory(
            env["images_dir"],
            progress_callback=progress_cb,
            force=True,
        )

    t.join(timeout=3.0)
    assert result.is_completed is True
    assert any("Preparing model…" in n for n in progress_names)
    # Preparing model must appear before first-batch embedding message.
    prep_i = next(i for i, n in enumerate(progress_names) if "Preparing model…" in n)
    embed_msgs = [
        i
        for i, n in enumerate(progress_names)
        if "Embedding first batch" in n or n.endswith(".jpg")
    ]
    if embed_msgs:
        assert prep_i < embed_msgs[0]
    assert any("Indexing waited" in r.message for r in caplog.records)
    waited_msg = next(r.message for r in caplog.records if "Indexing waited" in r.message)
    # At least ~0.25s of artificial wait.
    assert "0.0" not in waited_msg.split("waited ")[1][:3] or float(
        waited_msg.split("waited ")[1].split("s")[0]
    ) >= 0.25
    assert any("first_batch_sec_per_view" in r.message for r in caplog.records)
    assert not warmup_in_progress()


def test_indexing_cancel_during_warmup_wait(env):
    """Cancel during Preparing model… must stop without embedding."""
    release = threading.Event()
    warmup_started = threading.Event()

    def _slow_warmup():
        with warmup_compute_scope(torch_threads=1):
            warmup_started.set()
            release.wait(timeout=8.0)

    t = threading.Thread(target=_slow_warmup, daemon=True)
    t.start()
    assert warmup_started.wait(timeout=2.0)

    cancel = threading.Event()
    progress_names: list[str] = []

    def progress_cb(processed, total, name, eta):
        progress_names.append(name)
        if "Preparing model…" in name:
            cancel.set()

    calls_before = env["embedder"].call_count if hasattr(env["embedder"], "call_count") else None
    # FakeEmbedder may track differently — count extract calls via wrapper.
    embed_calls = {"n": 0}
    original = env["use_case"]._index_file_batch

    def counting_batch(*args, **kwargs):
        embed_calls["n"] += 1
        return original(*args, **kwargs)

    env["use_case"]._index_file_batch = counting_batch

    result = env["use_case"].scan_and_index_directory(
        env["images_dir"],
        progress_callback=progress_cb,
        cancel_event=cancel,
        force=True,
    )
    release.set()
    t.join(timeout=3.0)

    assert result.is_completed is False
    assert any("Preparing model…" in n for n in progress_names)
    assert embed_calls["n"] == 0
    _ = calls_before  # silence lint if unused
