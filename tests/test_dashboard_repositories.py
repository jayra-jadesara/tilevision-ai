"""Tests for SQLiteSearchHistoryRepository and SQLiteActivityLogRepository."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data.db_context import DatabaseContext
from src.data.sqlite_repository import SQLiteSearchHistoryRepository, SQLiteActivityLogRepository


def _search_repo(tmp_path):
    ctx = DatabaseContext(str(tmp_path / "test.db"))
    return SQLiteSearchHistoryRepository(ctx)


def _activity_repo(tmp_path):
    ctx = DatabaseContext(str(tmp_path / "test.db"))
    return SQLiteActivityLogRepository(ctx)


# ── Search History ────────────────────────────────────────────────────────


def test_no_searches_returns_empty_and_none(tmp_path):
    repo = _search_repo(tmp_path)
    assert repo.get_recent_searches() == []
    assert repo.get_last_search() is None


def test_record_and_retrieve_search(tmp_path):
    repo = _search_repo(tmp_path)
    repo.record_search("/tmp/query.jpg", result_count=12, elapsed_seconds=0.42)

    results = repo.get_recent_searches()
    assert len(results) == 1
    assert results[0].query_image_path == "/tmp/query.jpg"
    assert results[0].result_count == 12
    assert results[0].elapsed_seconds == 0.42
    assert results[0].searched_at is not None


def test_recent_searches_ordered_newest_first(tmp_path):
    repo = _search_repo(tmp_path)
    repo.record_search("/tmp/a.jpg", result_count=1)
    repo.record_search("/tmp/b.jpg", result_count=2)
    repo.record_search("/tmp/c.jpg", result_count=3)

    results = repo.get_recent_searches()
    assert [r.query_image_path for r in results] == ["/tmp/c.jpg", "/tmp/b.jpg", "/tmp/a.jpg"]


def test_get_last_search_returns_most_recent(tmp_path):
    repo = _search_repo(tmp_path)
    repo.record_search("/tmp/a.jpg", result_count=1)
    repo.record_search("/tmp/b.jpg", result_count=2)

    last = repo.get_last_search()
    assert last.query_image_path == "/tmp/b.jpg"


def test_recent_searches_respects_limit(tmp_path):
    repo = _search_repo(tmp_path)
    for i in range(5):
        repo.record_search(f"/tmp/{i}.jpg", result_count=i)

    results = repo.get_recent_searches(limit=2)
    assert len(results) == 2


# ── Activity Log ─────────────────────────────────────────────────────────


def test_no_activity_returns_empty(tmp_path):
    repo = _activity_repo(tmp_path)
    assert repo.get_recent_activity() == []


def test_record_and_retrieve_activity(tmp_path):
    repo = _activity_repo(tmp_path)
    repo.record_activity("index", "Indexed 'E:\\Tiles' — 42 new")

    results = repo.get_recent_activity()
    assert len(results) == 1
    assert results[0].activity_type == "index"
    assert "42 new" in results[0].message
    assert results[0].created_at is not None


def test_activity_ordered_newest_first(tmp_path):
    repo = _activity_repo(tmp_path)
    repo.record_activity("index", "first")
    repo.record_activity("search", "second")
    repo.record_activity("duplicate_scan", "third")

    results = repo.get_recent_activity()
    assert [r.message for r in results] == ["third", "second", "first"]
