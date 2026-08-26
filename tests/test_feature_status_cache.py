"""Feature-version health must not look like an empty catalog on DB errors."""

from __future__ import annotations

import sqlite3
import sys
from pathlib import Path
from unittest.mock import MagicMock

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.ai.feature_versions import FeatureVersionStatus
from src.data.sqlite_repository import SQLiteImageRepository


def test_feature_status_db_error_uses_last_good_not_zero():
    repo = SQLiteImageRepository.__new__(SQLiteImageRepository)
    repo._db = MagicMock()
    repo._feature_status_cache = None
    repo._last_good_feature_status_cache = FeatureVersionStatus(
        is_compatible=True,
        indexed_count=847,
        stale_count=0,
        message="ok",
    )

    cm = MagicMock()
    cm.__enter__.side_effect = sqlite3.Error("database is locked")
    cm.__exit__.return_value = False
    repo._db.session.return_value = cm

    status = repo.get_feature_version_status()
    assert status.indexed_count == 847
    assert status.is_compatible is True


def test_feature_status_db_error_without_cache_raises():
    repo = SQLiteImageRepository.__new__(SQLiteImageRepository)
    repo._db = MagicMock()
    repo._feature_status_cache = None
    repo._last_good_feature_status_cache = None

    cm = MagicMock()
    cm.__enter__.side_effect = sqlite3.Error("database is locked")
    cm.__exit__.return_value = False
    repo._db.session.return_value = cm

    with pytest.raises(RuntimeError, match="Could not verify feature versions"):
        repo.get_feature_version_status()
