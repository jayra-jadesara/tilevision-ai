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

    indexed, skipped, completed = env["use_case"].scan_and_index_directory(d)

    assert completed is True
    assert indexed == 3
    assert skipped == 0
    assert env["vector_index"]._index.ntotal == 3
    assert len(env["repo"].get_all()) == 3


def test_second_scan_of_unchanged_folder_skips_everything(env):
    d = env["images_dir"]
    _make_image(d / "a.jpg", (255, 0, 0))
    _make_image(d / "b.jpg", (0, 255, 0))

    env["use_case"].scan_and_index_directory(d)
    calls_after_first = env["embedder"].calls

    indexed, skipped, completed = env["use_case"].scan_and_index_directory(d)

    assert indexed == 0
    assert skipped == 2
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
    indexed, skipped, completed = env["use_case"].scan_and_index_directory(d)

    assert indexed == 1
    assert skipped == 0
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
