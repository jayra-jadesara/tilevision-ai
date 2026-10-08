"""Multi-view FAISS merge: MAX (default) vs RRF (opt-in)."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.ai.search_quality.fusion import (
    FusionMethod,
    ScoredHit,
    fuse_hits,
    hits_from_faiss_lists,
    merge_multi_view_faiss,
    resolve_multi_crop_fusion,
)


def test_resolve_multi_crop_fusion_defaults_to_max(monkeypatch):
    monkeypatch.delenv("TILEVISION_MULTI_CROP_FUSION", raising=False)
    monkeypatch.delenv("TILEVISION_MULTI_CROP_RRF_K", raising=False)
    method, k = resolve_multi_crop_fusion(None)
    assert method is FusionMethod.MAX
    assert k == 60


def test_resolve_multi_crop_fusion_env_overrides_config(monkeypatch):
    monkeypatch.setenv("TILEVISION_MULTI_CROP_FUSION", "rrf")
    monkeypatch.setenv("TILEVISION_MULTI_CROP_RRF_K", "30")
    method, k = resolve_multi_crop_fusion("max")
    assert method is FusionMethod.RRF
    assert k == 30


def test_max_first_then_rrf_is_order_identical_to_max():
    """Documents the old bakeoff bug: RRF after MAX collapse ≠ multi-view RRF."""
    hits = [
        ScoredHit(tile_id=i, score=1.0 - 0.01 * i, rank_in_list=i)
        for i in range(1, 8)
    ]
    max_order = [t for t, _ in fuse_hits(hits, FusionMethod.MAX)]
    rrf_order = [t for t, _ in fuse_hits(hits, FusionMethod.RRF, rrf_k=60)]
    assert max_order == rrf_order


def test_true_multi_view_rrf_can_outrank_max_leader():
    """
    Tile A: best score (rank 1 view0) but weak view1.
    Tile B: strong ranks in both views → RRF prefers B; MAX prefers A.
    """
    view0_ids = [1, 2, 3]
    view0_scores = [0.95, 0.90, 0.80]
    view1_ids = [2, 3, 1]
    view1_scores = [0.92, 0.88, 0.50]
    merged_max = merge_multi_view_faiss(
        [(view0_ids, view0_scores), (view1_ids, view1_scores)],
        FusionMethod.MAX,
    )
    merged_rrf = merge_multi_view_faiss(
        [(view0_ids, view0_scores), (view1_ids, view1_scores)],
        FusionMethod.RRF,
        rrf_k=60,
    )
    assert merged_max.matching_ids[0] == 1
    assert merged_rrf.matching_ids[0] == 2
    # Cosine dict stays MAX for aux-boost compatibility.
    assert merged_rrf.best_cosine[1] == 0.95
    assert merged_rrf.best_cosine[2] == 0.92


def test_hits_from_faiss_lists_collapses_dup_tile_within_view():
    ids = [7, 7, 8]
    scores = [0.5, 0.9, 0.8]
    hits, best, views = hits_from_faiss_lists([(ids, scores)])
    assert best[7] == 0.9
    assert [h.tile_id for h in hits] == [7, 8]
    assert hits[0].rank_in_list == 1
    assert hits[1].rank_in_list == 2
    assert views[7] == 0


def test_rrf_k_changes_mass_not_single_list_order():
    hits = [
        ScoredHit(1, 0.9, rank_in_list=1),
        ScoredHit(2, 0.8, rank_in_list=2),
    ]
    a = fuse_hits(hits, FusionMethod.RRF, rrf_k=30)
    b = fuse_hits(hits, FusionMethod.RRF, rrf_k=90)
    assert [t for t, _ in a] == [t for t, _ in b] == [1, 2]
    assert a[0][1] > b[0][1]
