"""Tests for multi-crop FAISS candidate merge."""

from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.core.use_cases.search_tiles import SearchTilesUseCase


class _FakeIndex:
    def __init__(self, responses):
        self._responses = list(responses)
        self.calls = 0

    def search_vectors(self, _vector, _top_k):
        ids, scores = self._responses[self.calls]
        self.calls += 1
        return ids, scores


def test_search_faiss_multi_crop_keeps_best_score_per_id():
    index = _FakeIndex(
        [
            ([10, 20, 30], [0.50, 0.80, 0.40]),
            ([20, 40], [0.90, 0.70]),
        ]
    )
    use_case = SearchTilesUseCase.__new__(SearchTilesUseCase)
    use_case._index = index

    embeddings = [[0.1] * 4, [0.2] * 4]
    ordered, scores, view_map = use_case._search_faiss_multi_crop(embeddings, search_k=10)

    assert ordered[0] == 20  # best score 0.90 from second crop
    assert set(ordered) == {10, 20, 30, 40}
    assert scores[20] == pytest.approx(0.90)
    assert scores[10] == pytest.approx(0.50)
    assert view_map[20] == 1
    assert view_map[10] == 0
    assert index.calls == 2


def test_search_faiss_multi_crop_single_embedding():
    index = _FakeIndex([([7, 8], [0.9, 0.5])])
    use_case = SearchTilesUseCase.__new__(SearchTilesUseCase)
    use_case._index = index
    ordered, scores, _view_map = use_case._search_faiss_multi_crop([[0.3] * 3], search_k=5)
    assert ordered == [7, 8]
    assert scores[7] == pytest.approx(0.9)


def test_search_faiss_multi_crop_rrf_orders_by_rank_keeps_max_cosine(monkeypatch):
    """RRF can reorder vs MAX while faiss_scores stay MAX cosine."""
    monkeypatch.delenv("TILEVISION_MULTI_CROP_FUSION", raising=False)
    monkeypatch.delenv("TILEVISION_MULTI_CROP_RRF_K", raising=False)
    # Tile 1: rank-1 then rank-5. Tile 2: rank-2 both views → RRF prefers 2.
    responses = [
        ([1, 2, 3, 4, 5], [0.95, 0.90, 0.80, 0.70, 0.60]),
        ([2, 3, 4, 5, 1], [0.91, 0.85, 0.75, 0.65, 0.40]),
    ]
    embeddings = [[0.1] * 2, [0.2] * 2]

    use_case = SearchTilesUseCase.__new__(SearchTilesUseCase)
    use_case._index = _FakeIndex(responses)
    use_case._multi_crop_fusion_configured = "max"
    use_case._multi_crop_rrf_k_configured = 60
    ordered_max, _, _ = use_case._search_faiss_multi_crop(embeddings, search_k=10)
    assert ordered_max[0] == 1  # MAX cosine leader

    use_case._index = _FakeIndex(responses)
    use_case._multi_crop_fusion_configured = "rrf"
    ordered, scores, view_map = use_case._search_faiss_multi_crop(
        embeddings, search_k=10
    )
    assert ordered[0] == 2  # stronger dual ranks under RRF
    assert scores[1] == pytest.approx(0.95)  # MAX cosine preserved
    assert scores[2] == pytest.approx(0.91)
    assert view_map[1] == 0
    assert view_map[2] == 1
