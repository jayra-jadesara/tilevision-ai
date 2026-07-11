"""
Integration tests for IndexImagesUseCase.scan_and_index_directory(), using a
fake embedder (no torch/open_clip needed — see conftest.py) but the real
SQLite repository and real FAISS index, to verify the folder-scan pipeline
end-to-end after the bug fixes in this patch:

- Re-indexing a changed file must not leave a duplicate vector in FAISS.
- The FAISS index must not be written to disk on every single file.
- Unchanged files must be skipped (incremental indexing).
"""

import sys
from pathlib import Path

import pytest
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

faiss = pytest.importorskip("faiss")

from src.ai.vector_index import FaissIndexManager
from src.core.use_cases.index_images import IndexImagesUseCase
from src.data.db_context import DatabaseContext
from src.data.sqlite_repository import SQLiteImageRepository


class FakeEmbedder:
    """Deterministic fake embedder: encodes the image's average RGB as a
    4-dim vector so different content produces different (but stable)
    embeddings, without needing torch/open_clip installed."""

    def __init__(self):
        self.calls = 0

    def load_model(self):
        pass

    def get_embedding(self, image_path: str):
        self.calls += 1
        with Image.open(image_path) as img:
            img = img.convert("RGB").resize((8, 8))
            r, g, b = 0.0, 0.0, 0.0
            pixels = list(img.getdata())
            for pr, pg, pb in pixels:
                r += pr
                g += pg
                b += pb
            n = len(pixels) * 255.0
            return [r / n, g / n, b / n, 1.0]


@pytest.fixture()
def env(tmp_path):
    db_context = DatabaseContext(str(tmp_path / "db" / "tiles.db"))
    repo = SQLiteImageRepository(db_context)
    embedder = FakeEmbedder()
    vector_index = FaissIndexManager(str(tmp_path / "index" / "tiles.index"), dimension=4)

    use_case = IndexImagesUseCase(
        image_repository=repo,
        embedder=embedder,
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
    assert env["embedder"].calls == calls_after_first  # no re-embedding


def test_changed_file_reindexes_without_duplicating_vector(env):
    d = env["images_dir"]
    target = d / "a.jpg"
    _make_image(target, (10, 10, 10))

    env["use_case"].scan_and_index_directory(d)
    assert env["vector_index"]._index.ntotal == 1

    # Overwrite with very different content -> hash changes -> must re-embed,
    # and must NOT leave a second stale vector behind for the same tile id.
    _make_image(target, (250, 5, 5))
    result = env["use_case"].scan_and_index_directory(d)

    assert result.indexed_count == 1
    assert result.new_count == 0
    assert result.modified_count == 1
    assert result.skipped_count == 0
    assert env["vector_index"]._index.ntotal == 1  # still exactly one vector

    tile = env["repo"].get_all()[0]
    ids, scores = env["vector_index"].search_vectors(
        env["embedder"].get_embedding(str(target)), top_k=5
    )
    assert ids.count(tile.id) == 1  # the tile appears exactly once in results


def test_checkpoint_saves_periodically_not_per_file(env, monkeypatch):
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

    # 60 files with a checkpoint every 25 => saves at 25, 50, and one final
    # save at the end (60) — a handful of saves, not 60.
    assert 2 <= save_calls["count"] <= 4


def test_index_single_file_persist_false_does_not_write_disk(env):
    d = env["images_dir"]
    target = d / "a.jpg"
    _make_image(target, (100, 100, 100))

    env["use_case"].index_single_file(target, persist=False)

    assert not env["vector_index"]._index_path.exists()
    assert env["vector_index"]._index.ntotal == 1


# ── Task 2: Smart Re-index (new/modified/deleted/skipped breakdown) ────


def test_deleted_file_is_removed_from_faiss_and_sqlite(env):
    d = env["images_dir"]
    a_path = d / "a.jpg"
    b_path = d / "b.jpg"
    _make_image(a_path, (255, 0, 0))
    _make_image(b_path, (0, 255, 0))

    first = env["use_case"].scan_and_index_directory(d)
    assert first.new_count == 2
    assert env["vector_index"]._index.ntotal == 2

    # Delete one file from disk, then re-scan.
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
    """A cancelled scan sees only a partial file listing — treating every
    unseen file as 'deleted' in that case would be wrong."""
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

    result = env["use_case"].scan_and_index_directory(d, progress_callback=progress_cb, cancel_event=cancel_event)

    assert result.is_completed is False
    assert result.deleted_count == 0  # must not treat unscanned files as deleted


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

    env["use_case"].scan_and_index_directory(d)  # indexes all 4
    result = env["use_case"].scan_and_index_directory(d)  # skips all 4

    assert result.skipped_count == 4
    assert result.time_saved_seconds > 0


def test_mixed_new_modified_and_unchanged_in_one_scan(env):
    d = env["images_dir"]
    unchanged_path = d / "unchanged.jpg"
    modified_path = d / "modified.jpg"
    _make_image(unchanged_path, (1, 1, 1))
    _make_image(modified_path, (2, 2, 2))

    env["use_case"].scan_and_index_directory(d)

    # Modify one existing file and add a brand new one.
    _make_image(modified_path, (250, 10, 10))
    _make_image(d / "new_file.jpg", (3, 3, 3))

    result = env["use_case"].scan_and_index_directory(d)

    assert result.new_count == 1
    assert result.modified_count == 1
    assert result.skipped_count == 1  # unchanged.jpg


# ── Task 1: Persistent Indexed Folder ────────────────────────────────────


def test_folder_is_recorded_after_successful_scan(tmp_path):
    from src.data.sqlite_repository import SQLiteIndexedFolderRepository

    db_context = DatabaseContext(str(tmp_path / "db" / "tiles.db"))
    repo = SQLiteImageRepository(db_context)
    folder_repo = SQLiteIndexedFolderRepository(db_context)
    embedder = FakeEmbedder()
    vector_index = FaissIndexManager(str(tmp_path / "index" / "tiles.index"), dimension=4)

    use_case = IndexImagesUseCase(
        image_repository=repo, embedder=embedder, vector_index=vector_index,
        thumbnail_dir=str(tmp_path / "thumbs"), folder_repository=folder_repo,
    )

    images_dir = tmp_path / "images"
    images_dir.mkdir()
    _make_image(images_dir / "a.jpg", (1, 2, 3))
    _make_image(images_dir / "b.jpg", (4, 5, 6))

    assert use_case.get_last_indexed_folder_status() is None  # nothing yet

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
    vector_index = FaissIndexManager(str(tmp_path / "index" / "tiles.index"), dimension=4)

    use_case = IndexImagesUseCase(
        image_repository=repo, embedder=embedder, vector_index=vector_index,
        thumbnail_dir=str(tmp_path / "thumbs"), folder_repository=folder_repo,
    )

    images_dir = tmp_path / "images"
    images_dir.mkdir()
    for i in range(5):
        _make_image(images_dir / f"tile_{i}.jpg", (i, i, i))

    cancel_event = threading.Event()

    def progress_cb(processed, total, filename, eta):
        if processed == 1:
            cancel_event.set()

    use_case.scan_and_index_directory(images_dir, progress_callback=progress_cb, cancel_event=cancel_event)

    assert use_case.get_last_indexed_folder_status() is None
