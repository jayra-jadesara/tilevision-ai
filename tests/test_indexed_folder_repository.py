"""Tests for SQLiteIndexedFolderRepository (Task 1: Persistent Indexed Folder)."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data.db_context import DatabaseContext
from src.data.sqlite_repository import SQLiteIndexedFolderRepository


def _repo(tmp_path):
    ctx = DatabaseContext(str(tmp_path / "test.db"))
    return SQLiteIndexedFolderRepository(ctx)


def test_no_folders_indexed_returns_none(tmp_path):
    repo = _repo(tmp_path)
    assert repo.get_last_indexed_folder() is None
    assert repo.get_folder_state("/tiles/A") is None


def test_record_and_retrieve_folder(tmp_path):
    repo = _repo(tmp_path)
    repo.record_folder_indexed("/tiles/A")

    state = repo.get_folder_state("/tiles/A")
    assert state is not None
    assert state.folder_path == "/tiles/A"
    assert state.last_indexed_at is not None


def test_most_recently_indexed_folder_is_returned(tmp_path):
    """
    Regression test: SQLite's CURRENT_TIMESTAMP only has second-resolution,
    so recording two folders in quick succession (well within the same
    second, as happens in real usage and in fast test runs) used to tie
    and sort unpredictably. record_folder_indexed() now uses
    millisecond-resolution timestamps to avoid this.
    """
    repo = _repo(tmp_path)
    repo.record_folder_indexed("/tiles/A")
    repo.record_folder_indexed("/tiles/B")

    last = repo.get_last_indexed_folder()
    assert last.folder_path == "/tiles/B"


def test_reindexing_a_folder_makes_it_most_recent_again(tmp_path):
    repo = _repo(tmp_path)
    repo.record_folder_indexed("/tiles/A")
    repo.record_folder_indexed("/tiles/B")
    repo.record_folder_indexed("/tiles/A")  # re-index A

    last = repo.get_last_indexed_folder()
    assert last.folder_path == "/tiles/A"


def test_recording_same_folder_twice_does_not_duplicate_rows(tmp_path):
    repo = _repo(tmp_path)
    repo.record_folder_indexed("/tiles/A")
    repo.record_folder_indexed("/tiles/A")
    repo.record_folder_indexed("/tiles/A")

    state = repo.get_folder_state("/tiles/A")
    assert state.id is not None
    # Only one row should exist for this path — verified indirectly via
    # get_last_indexed_folder() still resolving to exactly this one folder.
    last = repo.get_last_indexed_folder()
    assert last.folder_path == "/tiles/A"
    assert last.id == state.id
