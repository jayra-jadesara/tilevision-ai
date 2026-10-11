"""Tests for bulk per-folder indexed tile counts (longest-prefix assignment)."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.core.models import TileImage
from src.data.db_context import DatabaseContext
from src.data.sqlite_repository import SQLiteImageRepository


def _repo(tmp_path):
    ctx = DatabaseContext(str(tmp_path / "tiles.db"))
    return SQLiteImageRepository(ctx)


def _add_indexed(repo: SQLiteImageRepository, path: str) -> None:
    repo.add(
        TileImage(
            file_path=path,
            file_name=Path(path).name,
            file_size=100,
            dimensions="100x100",
            is_indexed=True,
        )
    )


def test_count_indexed_tiles_by_folders_longest_prefix(tmp_path):
    repo = _repo(tmp_path)
    _add_indexed(repo, "/tiles/A/one.jpg")
    _add_indexed(repo, "/tiles/A/nested/two.jpg")
    _add_indexed(repo, "/tiles/B/three.jpg")
    _add_indexed(repo, "/tiles/other.jpg")

    counts = repo.count_indexed_tiles_by_folders(
        ["/tiles", "/tiles/A", "/tiles/B"]
    )
    assert counts["/tiles/A"] == 2
    assert counts["/tiles/B"] == 1
    # Parent keeps only tiles not claimed by a longer prefix.
    assert counts["/tiles"] == 1


def test_count_matches_per_folder_live_scan_semantics(tmp_path):
    """Bulk counts match IndexImagesUseCase._count_indexed_tiles_under style."""
    repo = _repo(tmp_path)
    folders = ["/showroom/porcelain", "/showroom/porcelain/matte", "/showroom"]
    _add_indexed(repo, "/showroom/porcelain/matte/a.jpg")
    _add_indexed(repo, "/showroom/porcelain/b.jpg")
    _add_indexed(repo, "/showroom/c.jpg")
    # Not indexed — ignored.
    repo.add(
        TileImage(
            file_path="/showroom/porcelain/skip.jpg",
            file_name="skip.jpg",
            file_size=100,
            dimensions="100x100",
            is_indexed=False,
        )
    )

    bulk = repo.count_indexed_tiles_by_folders(folders)

    def live_count(folder: str) -> int:
        return sum(
            1
            for tile in repo.get_all()
            if tile.file_path.startswith(folder) and tile.is_indexed
        )

    # Longest-prefix bulk ≠ naive per-folder startswith sums (nested would
    # double-count). Compare each folder's bulk value to a live scan that
    # excludes tiles belonging to a longer registered prefix.
    ordered = sorted(folders, key=len, reverse=True)
    expected = {f: 0 for f in folders}
    for tile in repo.get_all():
        if not tile.is_indexed:
            continue
        for folder in ordered:
            if tile.file_path.startswith(folder):
                expected[folder] += 1
                break

    assert bulk == expected
    # Sanity: naive live_count on nested folder still matches bulk for that folder.
    assert bulk["/showroom/porcelain/matte"] == live_count("/showroom/porcelain/matte")
    assert bulk["/showroom/porcelain/matte"] == 1
    assert bulk["/showroom/porcelain"] == 1
    assert bulk["/showroom"] == 1


def test_activity_and_search_history_offset(tmp_path):
    from src.data.sqlite_repository import (
        SQLiteActivityLogRepository,
        SQLiteSearchHistoryRepository,
    )

    ctx = DatabaseContext(str(tmp_path / "hist.db"))
    search = SQLiteSearchHistoryRepository(ctx)
    activity = SQLiteActivityLogRepository(ctx)
    for i in range(5):
        search.record_search(f"/q/{i}.jpg", result_count=i)
        activity.record_activity("search", f"msg-{i}")

    page = search.get_recent_searches(limit=2, offset=2)
    assert [e.result_count for e in page] == [2, 1]

    act = activity.get_recent_activity(limit=2, offset=2)
    assert [e.message for e in act] == ["msg-2", "msg-1"]
