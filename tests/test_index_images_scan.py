"""
Integration tests for IndexImagesUseCase.scan_and_index_directory(), using a
fake feature extractor (no torch needed) but the real SQLite repository and
real FAISS index.
"""

import sys
from pathlib import Path

import pytest
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

faiss = pytest.importorskip("faiss")

from src.ai.feature_versions import CURRENT_EMBEDDING_DIMENSION
from src.ai.vector_index import FaissIndexManager
from src.core.use_cases.index_images import IndexImagesUseCase
from src.data.db_context import DatabaseContext
from src.data.sqlite_repository import SQLiteImageRepository
from tests.fake_ai import FakeEmbedder, FakeFeatureExtractor


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
    )

    images_dir = tmp_path / "images"
    images_dir.mkdir()

    return {
        "use_case": use_case,
        "repo": repo,
        "embedder": embedder,
        "vector_index": vector_index,
        "images_dir": images_dir,
    }


def _make_image(path: Path, color) -> None:
    Image.new("RGB", (32, 32), color=color).save(path)


def test_full_scan_indexes_all_supported_files(env):
    d = env["images_dir"]
    _make_image(d / "a.jpg", (255, 0, 0))
    _make_image(d / "b.png", (0, 255, 0))
    _make_image(d / "c.webp", (0, 0, 255))
    (d / "readme.txt").write_text("not an image")

    result = env["use_case"].scan_and_index_directory(d)

    assert result.is_completed is True
    assert result.indexed_count == 3
    assert result.new_count == 3
    assert result.modified_count == 0
    assert result.skipped_count == 0
    assert env["vector_index"]._index.ntotal == 3
    assert len(env["repo"].get_all()) == 3


def test_second_scan_of_unchanged_folder_skips_everything(env):
    d = env["images_dir"]
    _make_image(d / "a.jpg", (255, 0, 0))
    _make_image(d / "b.jpg", (0, 255, 0))

    env["use_case"].scan_and_index_directory(d)
    calls_after_first = env["embedder"].calls

    result = env["use_case"].scan_and_index_directory(d)

    assert result.indexed_count == 0
    assert result.skipped_count == 2
    assert result.has_any_changes is False
    assert env["embedder"].calls == calls_after_first


@pytest.mark.faiss_search
def test_changed_file_reindexes_without_duplicating_vector(env):
    d = env["images_dir"]
    target = d / "a.jpg"
    _make_image(target, (10, 10, 10))

    env["use_case"].scan_and_index_directory(d)
    assert env["vector_index"]._index.ntotal == 1

    _make_image(target, (250, 5, 5))
    result = env["use_case"].scan_and_index_directory(d)

    assert result.indexed_count == 1
    assert result.new_count == 0
    assert result.modified_count == 1
    assert result.skipped_count == 0
    assert env["vector_index"]._index.ntotal == 1

    tile = env["repo"].get_all()[0]
    ids, scores = env["vector_index"].search_vectors(
        env["embedder"].get_embedding(str(target)), top_k=5
    )
    assert ids.count(tile.id) == 1


def test_checkpoint_saves_after_each_batch_for_crash_safe_resume(env, monkeypatch):
    """Every flush persists FAISS so a killed rebuild can resume via skips."""
    d = env["images_dir"]
    for i in range(60):
        _make_image(d / f"tile_{i}.jpg", (i % 255, (i * 3) % 255, (i * 7) % 255))

    save_calls = {"count": 0}
    original_save = env["vector_index"].save_index

    def counting_save():
        save_calls["count"] += 1
        return original_save()

    monkeypatch.setattr(env["vector_index"], "save_index", counting_save)

    env["use_case"].scan_and_index_directory(d)

    # batch_size=12 → ~5 flushes (+ final save). Must not be a single end-only save.
    assert save_calls["count"] >= 5


def test_index_single_file_persist_false_does_not_write_disk(env):
    d = env["images_dir"]
    target = d / "a.jpg"
    _make_image(target, (100, 100, 100))

    env["use_case"].index_single_file(target, persist=False)

    assert not env["vector_index"]._index_path.exists()
    assert env["vector_index"]._index.ntotal == 1


def test_deleted_file_is_removed_from_faiss_and_sqlite(env):
    d = env["images_dir"]
    a_path = d / "a.jpg"
    b_path = d / "b.jpg"
    _make_image(a_path, (255, 0, 0))
    _make_image(b_path, (0, 255, 0))

    first = env["use_case"].scan_and_index_directory(d)
    assert first.new_count == 2
    assert env["vector_index"]._index.ntotal == 2

    a_path.unlink()
    second = env["use_case"].scan_and_index_directory(d)

    assert second.deleted_count == 1
    assert second.new_count == 0
    assert second.modified_count == 0
    assert env["vector_index"]._index.ntotal == 1
    remaining_paths = {t.file_path for t in env["repo"].get_all()}
    assert str(a_path.resolve()) not in remaining_paths
    assert str(b_path.resolve()) in remaining_paths


def test_deletion_not_detected_on_cancelled_scan(env):
    d = env["images_dir"]
    for i in range(5):
        _make_image(d / f"tile_{i}.jpg", (i, i, i))
    env["use_case"].scan_and_index_directory(d)

    import threading

    cancel_event = threading.Event()
    call_count = {"n": 0}

    def progress_cb(processed, total, filename, eta):
        call_count["n"] += 1
        if call_count["n"] == 2:
            cancel_event.set()

    result = env["use_case"].scan_and_index_directory(
        d, progress_callback=progress_cb, cancel_event=cancel_event
    )

    assert result.is_completed is False
    assert result.deleted_count == 0


def test_everything_already_indexed_has_no_changes(env):
    d = env["images_dir"]
    _make_image(d / "a.jpg", (1, 2, 3))
    _make_image(d / "b.jpg", (4, 5, 6))

    env["use_case"].scan_and_index_directory(d)
    result = env["use_case"].scan_and_index_directory(d)

    assert result.has_any_changes is False
    assert result.new_count == 0
    assert result.modified_count == 0
    assert result.deleted_count == 0
    assert result.skipped_count == 2


def test_time_saved_is_positive_when_files_are_skipped(env):
    d = env["images_dir"]
    for i in range(4):
        _make_image(d / f"tile_{i}.jpg", (i * 10, i * 20, i * 30))

    env["use_case"].scan_and_index_directory(d)
    result = env["use_case"].scan_and_index_directory(d)

    assert result.skipped_count == 4
    assert result.time_saved_seconds > 0


def test_mixed_new_modified_and_unchanged_in_one_scan(env):
    d = env["images_dir"]
    unchanged_path = d / "unchanged.jpg"
    modified_path = d / "modified.jpg"
    _make_image(unchanged_path, (1, 1, 1))
    _make_image(modified_path, (2, 2, 2))

    env["use_case"].scan_and_index_directory(d)

    _make_image(modified_path, (250, 10, 10))
    _make_image(d / "new_file.jpg", (3, 3, 3))

    result = env["use_case"].scan_and_index_directory(d)

    assert result.new_count == 1
    assert result.modified_count == 1
    assert result.skipped_count == 1


def test_force_rebuild_reembeds_unchanged_files(env):
    d = env["images_dir"]
    _make_image(d / "a.jpg", (1, 2, 3))
    _make_image(d / "b.jpg", (4, 5, 6))

    env["use_case"].scan_and_index_directory(d)
    calls_after_first = env["embedder"].calls
    assert calls_after_first == 2

    result = env["use_case"].scan_and_index_directory(d, force=True)

    assert result.skipped_count == 0
    assert result.modified_count == 2
    assert env["embedder"].calls == calls_after_first + 2


def test_folder_is_recorded_after_successful_scan(tmp_path):
    from src.data.sqlite_repository import SQLiteIndexedFolderRepository

    db_context = DatabaseContext(str(tmp_path / "db" / "tiles.db"))
    repo = SQLiteImageRepository(db_context)
    folder_repo = SQLiteIndexedFolderRepository(db_context)
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
        folder_repository=folder_repo,
    )

    images_dir = tmp_path / "images"
    images_dir.mkdir()
    _make_image(images_dir / "a.jpg", (1, 2, 3))
    _make_image(images_dir / "b.jpg", (4, 5, 6))

    assert use_case.get_last_indexed_folder_status() is None

    use_case.scan_and_index_directory(images_dir)

    status = use_case.get_last_indexed_folder_status()
    assert status is not None
    assert status.folder_path == str(images_dir.resolve())
    assert status.indexed_image_count == 2
    assert status.last_indexed_at is not None


def test_folder_not_recorded_when_scan_is_cancelled(tmp_path):
    from src.data.sqlite_repository import SQLiteIndexedFolderRepository
    import threading

    db_context = DatabaseContext(str(tmp_path / "db" / "tiles.db"))
    repo = SQLiteImageRepository(db_context)
    folder_repo = SQLiteIndexedFolderRepository(db_context)
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
        folder_repository=folder_repo,
    )

    images_dir = tmp_path / "images"
    images_dir.mkdir()
    for i in range(5):
        _make_image(images_dir / f"tile_{i}.jpg", (i, i, i))

    cancel_event = threading.Event()

    def progress_cb(processed, total, filename, eta):
        if processed == 1:
            cancel_event.set()

    use_case.scan_and_index_directory(
        images_dir, progress_callback=progress_cb, cancel_event=cancel_event
    )

    assert use_case.get_last_indexed_folder_status() is None


def _build_nested_image_tree(images_dir: Path) -> dict[str, Path]:
    """
    Nested catalogue layout used by the recursive-scan regression tests:

        images_dir/
          top_level.jpg
          Marble/
            marble_a.jpg
            Polished/
              marble_polished_b.jpg
          Granite/
            granite_c.jpg
    """
    marble = images_dir / "Marble"
    polished = marble / "Polished"
    granite = images_dir / "Granite"
    polished.mkdir(parents=True)
    granite.mkdir(parents=True)

    paths = {
        "top": images_dir / "top_level.jpg",
        "marble_a": marble / "marble_a.jpg",
        "marble_b": polished / "marble_polished_b.jpg",
        "granite_c": granite / "granite_c.jpg",
    }
    _make_image(paths["top"], (255, 0, 0))
    _make_image(paths["marble_a"], (0, 255, 0))
    _make_image(paths["marble_b"], (0, 0, 255))
    _make_image(paths["granite_c"], (255, 255, 0))
    return paths


def test_scan_finds_images_in_nested_subfolders(env):
    """Picking one parent folder must index images at every nested depth."""
    d = env["images_dir"]
    nested = _build_nested_image_tree(d)

    result = env["use_case"].scan_and_index_directory(d)

    assert result.new_count == 4
    assert result.total_files_scanned == 4
    assert result.indexed_count == 4
    assert env["vector_index"]._index.ntotal == 4

    for path in nested.values():
        tile = env["repo"].get_by_path(str(path.resolve()))
        assert tile is not None, f"expected indexed tile for {path}"
        assert tile.is_indexed is True


def test_second_scan_of_nested_folder_skips_unchanged_and_detects_deletion(env):
    """Incremental scan must handle nested adds/deletes, not only top-level."""
    d = env["images_dir"]
    nested = _build_nested_image_tree(d)

    baseline = env["use_case"].scan_and_index_directory(d)
    assert baseline.new_count == 4
    assert env["vector_index"]._index.ntotal == 4

    deleted_path = nested["marble_b"]
    deleted_resolved = str(deleted_path.resolve())
    deleted_path.unlink()

    new_nested = d / "Granite" / "granite_d.jpg"
    _make_image(new_nested, (128, 64, 32))

    result = env["use_case"].scan_and_index_directory(d)

    assert result.deleted_count == 1
    assert result.new_count == 1
    assert result.skipped_count == 3
    assert result.modified_count == 0
    assert env["vector_index"]._index.ntotal == 4

    remaining_paths = {t.file_path for t in env["repo"].get_all()}
    assert deleted_resolved not in remaining_paths
    assert env["repo"].get_by_path(deleted_resolved) is None
    assert env["repo"].get_by_path(str(new_nested.resolve())) is not None
    assert env["repo"].get_by_path(str(nested["top"].resolve())) is not None
    assert env["repo"].get_by_path(str(nested["marble_a"].resolve())) is not None
    assert env["repo"].get_by_path(str(nested["granite_c"].resolve())) is not None


def test_count_indexed_tiles_under_folder_includes_nested_files(tmp_path):
    """
    Index-page restore uses get_last_indexed_folder_status(), which live-counts
    via _count_indexed_tiles_under(). Nested files must be included.
    """
    from src.data.sqlite_repository import SQLiteIndexedFolderRepository

    db_context = DatabaseContext(str(tmp_path / "db" / "tiles.db"))
    repo = SQLiteImageRepository(db_context)
    folder_repo = SQLiteIndexedFolderRepository(db_context)
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
        folder_repository=folder_repo,
    )

    images_dir = tmp_path / "images"
    images_dir.mkdir()
    _build_nested_image_tree(images_dir)

    result = use_case.scan_and_index_directory(images_dir)
    assert result.new_count == 4

    status = use_case.get_last_indexed_folder_status()
    assert status is not None
    assert status.folder_path == str(images_dir.resolve())
    assert status.indexed_image_count == 4
    assert use_case._count_indexed_tiles_under(str(images_dir.resolve())) == 4


def test_mid_rebuild_cancel_then_resume_skips_completed_files(env):
    """
    Task 3: kill/cancel mid-rebuild must not restart from zero — already
    flushed tiles are skipped on the next scan (FAISS persisted per batch).
    """
    import threading

    d = env["images_dir"]
    for i in range(10):
        _make_image(d / f"tile_{i}.jpg", (i * 20, i * 10, i * 5))

    cancel_event = threading.Event()
    calls = {"n": 0}

    def progress_cb(processed, total, filename, eta):
        calls["n"] += 1
        # Cancel once a few files have been offered so at least one batch
        # of work has a chance to flush (batch_size defaults to 12, so we
        # cancel after several files and rely on the cancel-path flush).
        if processed >= 4:
            cancel_event.set()

    first = env["use_case"].scan_and_index_directory(
        d, progress_callback=progress_cb, cancel_event=cancel_event
    )
    assert first.is_completed is False
    indexed_after_cancel = len(env["repo"].get_all())
    assert indexed_after_cancel >= 1
    assert env["vector_index"]._index.ntotal == indexed_after_cancel

    second = env["use_case"].scan_and_index_directory(d)
    assert second.is_completed is True
    assert second.skipped_count == indexed_after_cancel
    assert second.new_count == 10 - indexed_after_cancel
    assert len(env["repo"].get_all()) == 10
    assert env["vector_index"]._index.ntotal == 10


def test_progress_eta_uses_embed_work_not_skip_dilution(env):
    """ETA should stay meaningful once real embeds have been timed."""
    d = env["images_dir"]
    for i in range(6):
        _make_image(d / f"tile_{i}.jpg", (i * 30, i * 15, i * 8))

    env["use_case"].scan_and_index_directory(d)

    # Second scan: all skips. ETA may be 0/-- early; when work_samples exist
    # from a prior flush in the same process they still inform remaining work
    # rate. Force a rebuild so ETA is driven by real embed samples.
    etas = []

    def progress_cb(processed, total, filename, eta):
        etas.append(eta)

    result = env["use_case"].scan_and_index_directory(
        d, progress_callback=progress_cb, force=True
    )
    assert result.modified_count == 6
    # After the first batch flush, later progress callbacks should report a
    # positive remaining ETA (until the final file).
    positive = [e for e in etas if e > 0]
    assert positive, f"expected positive ETA samples during rebuild, got {etas}"
