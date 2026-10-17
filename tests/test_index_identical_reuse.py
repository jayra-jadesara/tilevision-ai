"""Reuse DINOv2/CLIP/descriptors for byte-identical files in one index batch."""

from __future__ import annotations

import sys
from pathlib import Path

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
from src.utils.image_utils import compute_sha256
from tests.fake_ai import FakeEmbedder, FakeFeatureExtractor


def _make_use_case(tmp_path, *, batch_size: int = 12):
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
        performance=IndexingPerformanceConfig(batch_size=batch_size),
    )
    return use_case, repo, embedder, vector_index


def _write_jpeg(path: Path, color: tuple[int, int, int]) -> None:
    Image.new("RGB", (48, 48), color=color).save(path, format="JPEG")


def test_identical_copies_share_embeddings_one_forward_each(tmp_path):
    use_case, repo, embedder, vector_index = _make_use_case(tmp_path)

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


def test_changing_one_identical_copy_reembeds_only_that_path(tmp_path):
    use_case, repo, embedder, _vector_index = _make_use_case(tmp_path)

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
