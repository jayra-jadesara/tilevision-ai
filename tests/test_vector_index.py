"""
Regression tests for FaissIndexManager, covering the two bugs fixed in this
patch:

1. add_vectors() no longer forces a disk write on every call (persist=False
   lets callers batch writes during a folder scan).
2. update_vectors() removes any pre-existing vector for an id before
   re-adding, so re-indexing a changed file never leaves a duplicate/stale
   vector under the same id (which previously caused the same tile to
   appear twice in search results).
"""

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

faiss = pytest.importorskip("faiss")

from src.ai.vector_index import FaissIndexManager


@pytest.fixture()
def manager(tmp_path):
    index_path = tmp_path / "index" / "tiles.index"
    mgr = FaissIndexManager(index_path=str(index_path), dimension=4)
    mgr.load_index()
    return mgr


def test_add_vectors_with_persist_false_does_not_write_to_disk(manager, tmp_path):
    index_file = manager._index_path
    assert not index_file.exists()

    manager.add_vectors([1], [[1.0, 0.0, 0.0, 0.0]], persist=False)

    assert not index_file.exists()  # no disk write happened
    assert manager._index.ntotal == 1  # but the vector IS in memory


def test_add_vectors_with_persist_true_writes_to_disk(manager):
    manager.add_vectors([1], [[1.0, 0.0, 0.0, 0.0]], persist=True)
    assert manager._index_path.exists()


def test_update_vectors_replaces_rather_than_duplicates(manager):
    manager.update_vectors([1], [[1.0, 0.0, 0.0, 0.0]], persist=False)
    assert manager._index.ntotal == 1

    # Re-index the same tile id with new (changed-file) content.
    manager.update_vectors([1], [[0.0, 1.0, 0.0, 0.0]], persist=False)

    # Must still be exactly one vector for this id, not two.
    assert manager._index.ntotal == 1

    ids, scores = manager.search_vectors([0.0, 1.0, 0.0, 0.0], top_k=5)
    assert ids.count(1) == 1  # tile 1 appears exactly once in results, not twice


def test_update_vectors_safe_for_brand_new_id(manager):
    # Should behave like a normal add when the id has never been seen.
    manager.update_vectors([99], [[0.5, 0.5, 0.5, 0.5]], persist=False)
    assert manager._index.ntotal == 1


def test_search_after_update_returns_fresh_embedding(manager):
    manager.update_vectors([1], [[1.0, 0.0, 0.0, 0.0]], persist=False)
    manager.update_vectors([1], [[0.0, 0.0, 1.0, 0.0]], persist=False)

    ids, scores = manager.search_vectors([0.0, 0.0, 1.0, 0.0], top_k=1)
    assert ids == [1]
    assert scores[0] == pytest.approx(1.0, abs=1e-5)
