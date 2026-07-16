"""Tests for large-catalog indexing performance helpers."""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.config.indexing_performance import IndexingPerformanceConfig
from src.core.models import TileImage
from src.core.use_cases.index_images import _is_unchanged_indexed_file


def test_adaptive_batch_size_shrinks_for_huge_files():
    perf = IndexingPerformanceConfig(
        batch_size=12,
        large_file_mb=10,
        huge_file_mb=50,
    )
    assert perf.adaptive_batch_size(5 * 1024 * 1024) == 12
    assert perf.adaptive_batch_size(15 * 1024 * 1024) == 4
    assert perf.adaptive_batch_size(80 * 1024 * 1024) == 2


def test_adaptive_preprocess_workers_shrinks_for_huge_files():
    perf = IndexingPerformanceConfig(preprocess_workers=4)
    assert perf.adaptive_preprocess_workers(5 * 1024 * 1024) == 4
    assert perf.adaptive_preprocess_workers(15 * 1024 * 1024) == 2
    assert perf.adaptive_preprocess_workers(80 * 1024 * 1024) == 1


def test_fast_skip_uses_mtime_without_sha256(tmp_path, monkeypatch):
    image_path = tmp_path / "tile.jpg"
    image_path.write_bytes(b"tile-bytes-v1")

    record = TileImage(
        file_path=str(image_path),
        file_name=image_path.name,
        file_size=image_path.stat().st_size,
        dimensions="1x1",
        file_mtime=image_path.stat().st_mtime,
        sha256_hash="stale-hash",
        is_indexed=True,
    )

    def _fail_sha256(_path):
        raise AssertionError("SHA256 should not run when mtime+size match")

    monkeypatch.setattr(
        "src.core.use_cases.index_images.compute_sha256",
        _fail_sha256,
    )

    assert _is_unchanged_indexed_file(image_path, record, force=False)


def test_size_change_forces_reindex(tmp_path):
    image_path = tmp_path / "tile.jpg"
    image_path.write_bytes(b"tile-bytes-v1")

    record = TileImage(
        file_path=str(image_path),
        file_name=image_path.name,
        file_size=999,
        dimensions="1x1",
        file_mtime=image_path.stat().st_mtime,
        sha256_hash="anything",
        is_indexed=True,
    )

    assert not _is_unchanged_indexed_file(image_path, record, force=False)


def test_legacy_record_without_mtime_falls_back_to_sha256(tmp_path):
    image_path = tmp_path / "tile.jpg"
    image_path.write_bytes(b"tile-bytes-v1")

    from src.utils.image_utils import compute_sha256

    record = TileImage(
        file_path=str(image_path),
        file_name=image_path.name,
        file_size=image_path.stat().st_size,
        dimensions="1x1",
        file_mtime=0.0,
        sha256_hash=compute_sha256(image_path),
        is_indexed=True,
    )

    assert _is_unchanged_indexed_file(image_path, record, force=False)
