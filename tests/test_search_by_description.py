"""
Tests for description / text search (CLIP index path).

Uses fakes for CLIP + FAISS so CI does not need model weights.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.core.models import SearchResult, TileImage
from src.core.use_cases.search_by_description import SearchByDescriptionUseCase


class FakeClip:
    def __init__(self, vec=None):
        self._vec = np.asarray(
            vec if vec is not None else np.ones(512, dtype=np.float32),
            dtype=np.float32,
        )
        self._vec = self._vec / np.linalg.norm(self._vec)
        self.ensure_calls = 0
        self.embed_calls = []
        self._loaded = False
        self.progress_messages = []

    @property
    def is_loaded(self) -> bool:
        return self._loaded

    def ensure_loaded(self, on_progress=None) -> None:
        self.ensure_calls += 1
        if on_progress is not None:
            on_progress("Loading description-search model…")
            self.progress_messages.append("Loading description-search model…")
        self._loaded = True

    def embed_text(self, text: str) -> np.ndarray:
        self.embed_calls.append(text)
        return self._vec.copy()


class FakeClipIndex:
    def __init__(self, ids=None, scores=None, total=None):
        self._ids = list(ids or [])
        self._scores = list(scores or [])
        self._total = total if total is not None else len(self._ids)
        self._index = object()  # non-None so load is skipped
        self.load_calls = 0
        self.search_calls = []

    def load_index(self) -> None:
        self.load_calls += 1

    def get_total_count(self) -> int:
        return int(self._total)

    def search_vectors(self, query_vec, k):
        self.search_calls.append((np.asarray(query_vec).shape, k))
        return list(self._ids[:k]), list(self._scores[:k])


class FakeRepo:
    def __init__(self, tiles):
        self._by_id = {t.id: t for t in tiles}

    def get_by_ids(self, ids):
        return [self._by_id[i] for i in ids if i in self._by_id]


def _tile(tid: int, brand="Acme", category="Floor", path=None) -> TileImage:
    p = path or f"/tmp/tile_{tid}.jpg"
    return TileImage(
        id=tid,
        file_path=p,
        file_name=Path(p).name,
        file_size=1,
        dimensions="1x1",
        brand=brand,
        category=category,
        product_code=f"P{tid}",
    )


def test_description_search_returns_ranked_results(tmp_path):
    tiles = [_tile(1, path=str(tmp_path / "a.jpg")), _tile(2, path=str(tmp_path / "b.jpg"))]
    for t in tiles:
        Path(t.file_path).write_bytes(b"x")
    uc = SearchByDescriptionUseCase(
        image_repository=FakeRepo(tiles),
        clip_embedder=FakeClip(),
        clip_index=FakeClipIndex(ids=[2, 1], scores=[0.91, 0.55], total=2),
        thumbnail_dir=str(tmp_path / "thumbs"),
    )
    results = uc.execute("white marble look", top_k=10)
    assert len(results) == 2
    assert results[0].tile.id == 2
    assert results[0].similarity_score > results[1].similarity_score
    assert isinstance(results[0], SearchResult)


def test_description_search_applies_filters(tmp_path):
    tiles = [
        _tile(1, brand="Acme", path=str(tmp_path / "a.jpg")),
        _tile(2, brand="Other", path=str(tmp_path / "b.jpg")),
    ]
    for t in tiles:
        Path(t.file_path).write_bytes(b"x")
    uc = SearchByDescriptionUseCase(
        image_repository=FakeRepo(tiles),
        clip_embedder=FakeClip(),
        clip_index=FakeClipIndex(ids=[1, 2], scores=[0.9, 0.8], total=2),
        thumbnail_dir=str(tmp_path / "thumbs"),
    )
    results = uc.execute("grey stone", top_k=10, filters={"brand": "Acme"})
    assert len(results) == 1
    assert results[0].tile.brand == "Acme"


def test_description_search_empty_text_raises():
    uc = SearchByDescriptionUseCase(
        image_repository=FakeRepo([]),
        clip_embedder=FakeClip(),
        clip_index=FakeClipIndex(total=0),
        thumbnail_dir="/tmp",
    )
    with pytest.raises(ValueError, match="description"):
        uc.execute("   ")


def test_description_search_empty_index_raises():
    uc = SearchByDescriptionUseCase(
        image_repository=FakeRepo([]),
        clip_embedder=FakeClip(),
        clip_index=FakeClipIndex(total=0),
        thumbnail_dir="/tmp",
    )
    with pytest.raises(RuntimeError, match="empty"):
        uc.execute("blue ceramic")


def test_clip_index_skipped_when_weights_missing(monkeypatch, tmp_path):
    """Indexing must not force a Hub download when CLIP is not local."""
    from src.core.use_cases.index_images import IndexImagesUseCase

    class StubEmbedder:
        is_loaded = False

        def embed_image(self, path):
            raise AssertionError("embed_image must not run without local CLIP")

    class StubClipIndex:
        def load_index(self):
            raise AssertionError("load_index must not run")

        def update_vectors(self, *a, **k):
            raise AssertionError("update_vectors must not run")

    monkeypatch.setattr(
        "src.ai.clip_model_paths.clip_model_available_locally",
        lambda: False,
    )
    # Minimal IndexImagesUseCase wiring — only exercise _index_clip_vector.
    uc = IndexImagesUseCase.__new__(IndexImagesUseCase)
    uc._clip_embedder = StubEmbedder()
    uc._clip_index = StubClipIndex()
    uc._index_clip_vector(1, tmp_path / "x.jpg", persist=False)


@pytest.fixture
def qapp():
    pytest.importorskip("PySide6")
    from PySide6.QtWidgets import QApplication

    app = QApplication.instance() or QApplication(sys.argv)
    yield app


def test_viewmodel_description_search(qapp):
    from PySide6.QtCore import QCoreApplication
    from src.presentation.viewmodels.search_viewmodel import SearchViewModel, SearchState

    class FakeImageUC:
        def get_filter_options(self):
            return {}

        def get_index_health(self):
            return SimpleNamespace(is_compatible=True, stale_count=0, indexed_count=1)

        def get_searchable_count(self):
            return 1

    class FakeDescUC:
        def __init__(self):
            self.calls = []

        def execute(self, query_text, top_k=20, filters=None, on_stage=None):
            self.calls.append(query_text)
            tile = _tile(9, path="/tmp/t.jpg")
            return [
                SearchResult(
                    tile=tile, similarity_score=88.0, thumbnail_path="/tmp/t.jpg"
                )
            ]

    desc = FakeDescUC()
    vm = SearchViewModel(use_case=FakeImageUC(), description_use_case=desc)
    states = []
    results_holder = []
    vm.state_changed.connect(states.append)
    vm.results_ready.connect(results_holder.append)

    vm.search_by_description("matte white subway")
    deadline = time.monotonic() + 5.0
    while time.monotonic() < deadline:
        QCoreApplication.processEvents()
        if results_holder:
            break
        time.sleep(0.01)

    assert results_holder
    assert results_holder[0][0].tile.id == 9
    assert SearchState.RESULTS in states or vm.state == SearchState.RESULTS
    assert desc.calls == ["matte white subway"]

    worker = getattr(vm, "_worker", None)
    if worker is not None and worker.isRunning():
        deadline = time.monotonic() + 3.0
        while time.monotonic() < deadline and worker.isRunning():
            QCoreApplication.processEvents()
            time.sleep(0.01)
    for _ in range(10):
        QCoreApplication.processEvents()
        time.sleep(0.01)


def test_clip_model_paths_constants():
    from src.ai.clip_model_paths import (
        CLIP_EMBEDDING_DIM,
        DEFAULT_CLIP_MODEL_ID,
        bundled_clip_model_dir,
    )

    assert CLIP_EMBEDDING_DIM == 512
    assert DEFAULT_CLIP_MODEL_ID == "openai/clip-vit-base-patch32"
    assert "clip-vit-base-patch32" in str(bundled_clip_model_dir())


def test_use_case_forwards_clip_load_progress(tmp_path, monkeypatch):
    """ensure_loaded(on_progress=…) must surface progress to on_stage."""
    tiles = [_tile(1, path=str(tmp_path / "a.jpg"))]
    Path(tiles[0].file_path).write_bytes(b"x")
    clip = FakeClip()

    def _slow_ensure(on_progress=None):
        clip.ensure_calls += 1
        if on_progress is not None:
            on_progress("Downloading description-search model… 12 / ~581 MB")
        clip._loaded = True

    clip.ensure_loaded = _slow_ensure  # type: ignore[method-assign]
    monkeypatch.setattr(
        "src.ai.clip_model_paths.clip_model_available_locally",
        lambda: False,
    )
    stages = []
    uc = SearchByDescriptionUseCase(
        image_repository=FakeRepo(tiles),
        clip_embedder=clip,
        clip_index=FakeClipIndex(ids=[1], scores=[0.9], total=1),
        thumbnail_dir=str(tmp_path / "thumbs"),
    )
    uc.execute("gray tile", top_k=5, on_stage=stages.append)
    assert any("Downloading description-search model" in s for s in stages)
    assert any("12 / ~581 MB" in s for s in stages)


def test_description_search_survives_slow_clip_load_with_heartbeats(qapp, monkeypatch):
    """
    Simulated first-use CLIP load longer than the stall window must NOT be
    killed as long as heartbeats/progress keep flowing.
    """
    from PySide6.QtCore import QCoreApplication
    from src.presentation.viewmodels import search_viewmodel as svm_mod
    from src.presentation.viewmodels.search_viewmodel import SearchViewModel, SearchState

    # Stall window much shorter than the simulated download — would have
    # false-fired before heartbeats were wired during ensure_loaded.
    monkeypatch.setattr(svm_mod, "_SEARCH_STALL_MS", 250)
    monkeypatch.setattr(
        "src.ai.clip_model_paths.clip_model_available_locally",
        lambda: False,
    )

    class FakeImageUC:
        def get_filter_options(self):
            return {}

        def get_index_health(self):
            return SimpleNamespace(is_compatible=True, stale_count=0, indexed_count=1)

        def get_searchable_count(self):
            return 1

    class SlowFirstUseDescUC:
        def __init__(self):
            self._clip = SimpleNamespace(is_loaded=False)

        def execute(self, query_text, top_k=20, filters=None, on_stage=None):
            # Simulate ~1s blocking download with progress every 80ms —
            # well past the 250ms stall window if heartbeats were absent.
            deadline = time.monotonic() + 1.0
            n = 0
            while time.monotonic() < deadline:
                n += 1
                if on_stage is not None:
                    on_stage(
                        f"Downloading description-search model… "
                        f"{n * 40} / ~581 MB"
                    )
                time.sleep(0.08)
            self._clip.is_loaded = True
            tile = _tile(3, path="/tmp/slow.jpg")
            return [
                SearchResult(
                    tile=tile, similarity_score=77.0, thumbnail_path="/tmp/slow.jpg"
                )
            ]

    desc = SlowFirstUseDescUC()
    vm = SearchViewModel(use_case=FakeImageUC(), description_use_case=desc)
    errors = []
    results_holder = []
    statuses = []
    vm.search_error.connect(errors.append)
    vm.results_ready.connect(results_holder.append)
    vm.status_message.connect(statuses.append)

    vm.search_by_description("gray tile")
    assert any("Downloading description-search model" in s for s in statuses)

    deadline = time.monotonic() + 8.0
    while time.monotonic() < deadline:
        QCoreApplication.processEvents()
        if results_holder or vm.state == SearchState.ERROR:
            break
        time.sleep(0.01)

    assert results_holder, f"expected results; state={vm.state} errors={errors}"
    assert vm.state == SearchState.RESULTS
    assert not any("stopped responding" in (e or "").lower() for e in errors)
    assert not any("description-search model is taking longer" in (e or "") for e in errors)

    worker = getattr(vm, "_worker", None)
    if worker is not None and worker.isRunning():
        deadline = time.monotonic() + 3.0
        while time.monotonic() < deadline and worker.isRunning():
            QCoreApplication.processEvents()
            time.sleep(0.01)
    for _ in range(15):
        QCoreApplication.processEvents()
        time.sleep(0.01)


def test_description_stall_message_differs_from_image_stall(qapp):
    from src.presentation.viewmodels.search_viewmodel import SearchViewModel, SearchState

    class FakeImageUC:
        def get_filter_options(self):
            return {}

        def get_index_health(self):
            return SimpleNamespace(is_compatible=True, stale_count=0, indexed_count=1)

        def get_searchable_count(self):
            return 1

    vm = SearchViewModel(use_case=FakeImageUC(), description_use_case=object())

    image_errors = []
    vm.search_error.connect(image_errors.append)
    vm._active_search_kind = "image"
    vm._set_state(SearchState.SEARCHING)
    vm._on_search_stall()
    assert image_errors
    assert "Auto Crop" in image_errors[0]
    assert "Pause Indexing" in image_errors[0]
    assert "description-search model" not in image_errors[0].lower()

    desc_errors = []
    vm.search_error.connect(desc_errors.append)
    vm._active_search_kind = "description"
    vm._set_state(SearchState.SEARCHING)
    vm._on_search_stall()
    assert desc_errors
    assert "description-search model" in desc_errors[0].lower()
    assert "internet connection" in desc_errors[0].lower()
    assert "Auto Crop" not in desc_errors[0]
    assert "Pause Indexing" not in desc_errors[0]


def test_download_progress_ticker_emits_while_blocked(monkeypatch, tmp_path):
    """Unit-level: ticker fires on_progress during a blocking download body."""
    import threading

    import huggingface_hub

    from src.ai.clip_embedder import ClipEmbedder

    out = tmp_path / "clip-vit-base-patch32"
    out.mkdir()
    monkeypatch.setattr(
        "src.ai.clip_model_paths.bundled_clip_model_dir",
        lambda: out,
    )

    messages: list[str] = []
    started = threading.Event()
    release = threading.Event()

    def _fake_snapshot_download(**kwargs):
        started.set()
        # Block long enough for the 2s ticker to fire at least once.
        assert release.wait(timeout=6.0)
        (out / "config.json").write_text("{}", encoding="utf-8")
        (out / "pytorch_model.bin").write_bytes(b"x" * 1024)

    monkeypatch.setattr(huggingface_hub, "snapshot_download", _fake_snapshot_download)

    t = threading.Thread(
        target=lambda: ClipEmbedder._download_to_bundled(
            "openai/clip-vit-base-patch32", on_progress=messages.append
        ),
        daemon=True,
    )
    t.start()
    assert started.wait(timeout=3.0)
    # Wait for at least one ticker pulse (~2s) beyond the initial message.
    deadline = time.monotonic() + 5.0
    while time.monotonic() < deadline and len(messages) < 2:
        time.sleep(0.05)
    release.set()
    t.join(timeout=5.0)
    assert messages, "expected progress messages during blocked download"
    assert any("Downloading description-search model" in m for m in messages)
    assert len(messages) >= 2, f"ticker should pulse during block; got {messages!r}"
