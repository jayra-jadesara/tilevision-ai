"""Reuse DINOv2/CLIP/descriptors for byte-identical files across batches."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

faiss = pytest.importorskip("faiss")

from src.ai.feature_versions import CURRENT_EMBEDDING_DIMENSION
from src.ai.vector_index import FaissIndexManager
from src.config.indexing_performance import IndexingPerformanceConfig
from src.core.use_cases.index_images import ContentEmbedCache, IndexImagesUseCase
from src.data.db_context import DatabaseContext
from src.data.sqlite_repository import SQLiteImageRepository
from src.utils.image_utils import compute_sha256
from tests.fake_ai import FakeEmbedder, FakeFeatureExtractor


class FakeClip:
    def __init__(self):
        self.calls = 0
        self._loaded = True
        self._vec = np.ones(512, dtype=np.float32)
        self._vec = self._vec / np.linalg.norm(self._vec)

    @property
    def is_loaded(self) -> bool:
        return self._loaded

    def ensure_loaded(self, on_progress=None) -> None:
        self._loaded = True

    def embed_image(self, path) -> np.ndarray:
        self.calls += 1
        return self._vec.copy()


def _make_use_case(tmp_path, *, batch_size: int = 12, with_clip: bool = False):
    db_context = DatabaseContext(str(tmp_path / "db" / "tiles.db"))
    repo = SQLiteImageRepository(db_context)
    embedder = FakeEmbedder()
    feature_extractor = FakeFeatureExtractor(embedder=embedder)
    vector_index = FaissIndexManager(
        str(tmp_path / "index" / "tiles.index"),
        dimension=CURRENT_EMBEDDING_DIMENSION,
    )
    clip = FakeClip() if with_clip else None
    clip_index = None
    if with_clip:
        clip_index = FaissIndexManager(
            str(tmp_path / "index" / "tiles_clip.index"),
            dimension=512,
        )
    use_case = IndexImagesUseCase(
        image_repository=repo,
        feature_extractor=feature_extractor,
        vector_index=vector_index,
        thumbnail_dir=str(tmp_path / "thumbs"),
        performance=IndexingPerformanceConfig(batch_size=batch_size),
        clip_embedder=clip,
        clip_index=clip_index,
    )
    return use_case, repo, embedder, vector_index, clip, clip_index


def _write_jpeg(path: Path, color: tuple[int, int, int]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (48, 48), color=color).save(path, format="JPEG")


def test_identical_copies_share_embeddings_one_forward_each(tmp_path):
    use_case, repo, embedder, vector_index, _clip, _ci = _make_use_case(tmp_path)

    images_dir = tmp_path / "catalog"
    images_dir.mkdir()
    shared_bytes = None
    for name in ("copy_a.jpg", "copy_b.jpg", "copy_c.jpg"):
        path = images_dir / name
        _write_jpeg(path, (40, 120, 200))
        if shared_bytes is None:
            shared_bytes = path.read_bytes()
        else:
            path.write_bytes(shared_bytes)

    _write_jpeg(images_dir / "unique.jpg", (10, 10, 10))

    result = use_case.scan_and_index_directory(images_dir, force=True)
    assert result.is_completed
    assert result.indexed_count == 4

    # One FakeEmbedder forward per unique sha256 (3 identical + 1 different).
    assert embedder.calls == 2

    tiles = [repo.get_by_path(str(p.resolve())) for p in sorted(images_dir.glob("*.jpg"))]
    assert len(tiles) == 4
    assert all(t is not None and t.features is not None for t in tiles)

    by_name = {t.file_name: t for t in tiles}
    assert by_name["copy_a.jpg"].file_path != by_name["copy_b.jpg"].file_path
    assert (
        compute_sha256(Path(by_name["copy_a.jpg"].file_path))
        == compute_sha256(Path(by_name["copy_b.jpg"].file_path))
    )

    shared_emb = by_name["copy_a.jpg"].features.embedding
    for name in ("copy_b.jpg", "copy_c.jpg"):
        assert (by_name[name].features.embedding == shared_emb).all()
    assert not (by_name["unique.jpg"].features.embedding == shared_emb).all()

    vector_index.load_index()
    assert len(vector_index.known_ids()) == 4


def test_nested_folder_copies_reuse_across_batches(tmp_path):
    """
    Repro of customer layout: 13 unique images in 3 nested folders.

    Directory-sorted rglob + batch_size 12 / first_batch_max 4 puts copies of
    the same sha256 in different flush batches — in-batch-only dedupe embeds
    every path. Scan-wide cache must yield 13 forwards for 39 files (13×3).
    (Customer log had 38 ≈ same shape; one copy missing does not change the bug.)
    """
    use_case, repo, embedder, vector_index, clip, clip_index = _make_use_case(
        tmp_path, batch_size=12, with_clip=True
    )

    root = tmp_path / "Tiles"
    folders = [root, root / "New", root / "New" / "New2"]
    for folder in folders:
        folder.mkdir(parents=True, exist_ok=True)

    unique = 13
    file_count = unique * 3
    palette = [
        (10 + i * 15, 20 + i * 10, 30 + i * 5) for i in range(unique)
    ]
    for i, color in enumerate(palette):
        primary = folders[0] / f"tile_{i:02d}.jpg"
        _write_jpeg(primary, color)
        blob = primary.read_bytes()
        for folder in folders[1:]:
            (folder / f"tile_{i:02d}.jpg").write_bytes(blob)

    all_files = list(root.rglob("*.jpg"))
    assert len(all_files) == file_count

    result = use_case.scan_and_index_directory(root, force=True)
    assert result.is_completed
    assert result.indexed_count == file_count
    assert embedder.calls == unique, (
        f"expected {unique} forwards for nested copies, got {embedder.calls}"
    )
    assert clip.calls == unique, (
        f"expected {unique} CLIP embeds, got {clip.calls}"
    )

    vector_index.load_index()
    assert len(vector_index.known_ids()) == file_count
    clip_index.load_index()
    assert len(clip_index.known_ids()) == file_count

    # Per-path metadata stays distinct; shared content shares embedding bytes.
    t0 = repo.get_by_path(str((folders[0] / "tile_00.jpg").resolve()))
    t1 = repo.get_by_path(str((folders[1] / "tile_00.jpg").resolve()))
    t2 = repo.get_by_path(str((folders[2] / "tile_00.jpg").resolve()))
    assert t0 and t1 and t2
    assert t0.file_path != t1.file_path != t2.file_path
    assert t0.sha256_hash == t1.sha256_hash == t2.sha256_hash
    assert (t0.features.embedding == t1.features.embedding).all()
    assert (t0.features.embedding == t2.features.embedding).all()


def test_changing_one_identical_copy_reembeds_only_that_path(tmp_path):
    use_case, repo, embedder, _vector_index, _c, _ci = _make_use_case(tmp_path)

    images_dir = tmp_path / "catalog"
    images_dir.mkdir()
    path_a = images_dir / "a.jpg"
    path_b = images_dir / "b.jpg"
    _write_jpeg(path_a, (90, 90, 90))
    path_b.write_bytes(path_a.read_bytes())

    use_case.scan_and_index_directory(images_dir, force=True)
    assert embedder.calls == 1

    _write_jpeg(path_b, (5, 5, 5))
    second = use_case.scan_and_index_directory(images_dir, force=False)
    assert second.is_completed
    assert second.modified_count == 1
    assert second.skipped_count == 1
    # Only the changed file should trigger a new forward.
    assert embedder.calls == 2

    tile_a = repo.get_by_path(str(path_a.resolve()))
    tile_b = repo.get_by_path(str(path_b.resolve()))
    assert tile_a is not None and tile_b is not None
    assert tile_a.sha256_hash != tile_b.sha256_hash


def test_failed_representative_does_not_poison_cache(tmp_path, monkeypatch):
    """A failed/timed-out first occurrence must not block later copies."""
    use_case, repo, embedder, _vi, _c, _ci = _make_use_case(tmp_path, batch_size=4)

    root = tmp_path / "Tiles"
    nested = root / "New"
    nested.mkdir(parents=True)
    primary = root / "same.jpg"
    copy = nested / "same.jpg"
    _write_jpeg(primary, (50, 60, 70))
    copy.write_bytes(primary.read_bytes())

    real_extract = use_case._feature_extractor.extract_index_vectors
    calls = {"n": 0}

    def flaky(path):
        calls["n"] += 1
        # Fail the first attempt (representative in first batch); succeed after.
        if calls["n"] == 1:
            raise RuntimeError("simulated embed failure")
        return real_extract(path)

    monkeypatch.setattr(
        use_case._feature_extractor, "extract_index_vectors", flaky
    )

    # First scan: primary may fail (counted as failed); copy in a later batch
    # must still be attempted and succeed via a fresh embed (no poisoned cache).
    result = use_case.scan_and_index_directory(root, force=True)
    copy_tile = repo.get_by_path(str(copy.resolve()))
    assert copy_tile is not None and copy_tile.features is not None
    # At least one successful forward happened for the copy (call #2+).
    assert calls["n"] >= 2


def test_content_embed_cache_lru_cap():
    cache = ContentEmbedCache(max_entries=2)
    feat = type("F", (), {"embedding": np.zeros(4, dtype=np.float32)})()
    cache.put_features("aaa", feat, [])
    cache.put_features("bbb", feat, [])
    cache.put_features("ccc", feat, [])
    assert len(cache) == 2
    assert cache.get("aaa") is None  # evicted
    assert cache.get("bbb") is not None
    assert cache.get("ccc") is not None
