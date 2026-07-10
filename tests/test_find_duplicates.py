"""Tests for FindDuplicatesUseCase (Feature 5: Duplicate Detection)."""

import sys
from pathlib import Path
from unittest.mock import MagicMock

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.core.models import TileImage
from src.core.use_cases.find_duplicates import FindDuplicatesUseCase


def _tile(id_, path, sha256="", phash="") -> TileImage:
    return TileImage(
        id=id_, file_path=path, file_name=Path(path).name, file_size=1, dimensions="1x1",
        sha256_hash=sha256, perceptual_hash=phash,
    )


def _use_case(tiles):
    repo = MagicMock()
    repo.get_all.return_value = tiles
    return FindDuplicatesUseCase(repo)


# ── Exact duplicates ─────────────────────────────────────────────────────


def test_exact_duplicates_grouped_by_identical_hash():
    tiles = [
        _tile(1, "/a.jpg", sha256="HASH_A"),
        _tile(2, "/b.jpg", sha256="HASH_A"),
        _tile(3, "/c.jpg", sha256="HASH_B"),
    ]
    groups = _use_case(tiles).find_exact_duplicates()

    assert len(groups) == 1
    assert {t.id for t in groups[0]} == {1, 2}


def test_no_exact_duplicates_returns_empty_list():
    tiles = [_tile(1, "/a.jpg", sha256="A"), _tile(2, "/b.jpg", sha256="B")]
    groups = _use_case(tiles).find_exact_duplicates()
    assert groups == []


def test_tiles_with_empty_hash_are_ignored():
    tiles = [_tile(1, "/a.jpg", sha256=""), _tile(2, "/b.jpg", sha256="")]
    groups = _use_case(tiles).find_exact_duplicates()
    assert groups == []


def test_three_way_exact_duplicate_group():
    tiles = [
        _tile(1, "/a.jpg", sha256="X"),
        _tile(2, "/b.jpg", sha256="X"),
        _tile(3, "/c.jpg", sha256="X"),
    ]
    groups = _use_case(tiles).find_exact_duplicates()
    assert len(groups) == 1
    assert {t.id for t in groups[0]} == {1, 2, 3}


# ── Near duplicates ──────────────────────────────────────────────────────


def test_near_duplicates_within_threshold_are_grouped():
    # Hashes differing by 2 bits (well within default threshold of 8)
    tiles = [
        _tile(1, "/a.jpg", phash="0000000000000000"),
        _tile(2, "/b.jpg", phash="0000000000000003"),  # differs by 2 bits
        _tile(3, "/c.jpg", phash="ffffffffffffffff"),  # completely different
    ]
    groups = _use_case(tiles).find_near_duplicates(threshold=8)

    assert len(groups) == 1
    assert {t.id for t in groups[0]} == {1, 2}


def test_transitive_near_duplicate_chain_merges_into_one_group():
    # A~B (distance small), B~C (distance small), but A~C might exceed
    # threshold directly — union-find should still merge all three.
    tiles = [
        _tile(1, "/a.jpg", phash="0000000000000000"),  # A
        _tile(2, "/b.jpg", phash="0000000000000007"),  # B: 3 bits from A
        _tile(3, "/c.jpg", phash="00000000000000ff"),  # C: 5 bits from B, 8 bits from A
    ]
    groups = _use_case(tiles).find_near_duplicates(threshold=5)

    assert len(groups) == 1
    assert {t.id for t in groups[0]} == {1, 2, 3}


def test_tiles_beyond_threshold_are_not_grouped():
    tiles = [
        _tile(1, "/a.jpg", phash="0000000000000000"),
        _tile(2, "/b.jpg", phash="ffffffffffffffff"),  # maximally different (64 bits)
    ]
    groups = _use_case(tiles).find_near_duplicates(threshold=8)
    assert groups == []


def test_tiles_without_perceptual_hash_are_excluded():
    tiles = [
        _tile(1, "/a.jpg", phash="0000000000000000"),
        _tile(2, "/b.jpg", phash=""),  # no hash — should be skipped entirely
    ]
    groups = _use_case(tiles).find_near_duplicates()
    assert groups == []


def test_exact_duplicates_also_appear_as_near_duplicates_at_distance_zero():
    tiles = [
        _tile(1, "/a.jpg", phash="abc0000000000000"),
        _tile(2, "/b.jpg", phash="abc0000000000000"),
    ]
    groups = _use_case(tiles).find_near_duplicates(threshold=0)
    assert len(groups) == 1
