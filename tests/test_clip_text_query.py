"""Prompt ensemble + relevance cutoff for description search."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.core.use_cases.clip_text_query import (
    CLIP_ABS_FLOOR,
    CLIP_MIN_MARGIN,
    CLIP_SPREAD_FRACTION,
    apply_relevance_cutoff,
    average_l2_normalized,
    description_prompt_variants,
    distribution_aware_margin,
    score_distribution_stats,
)
from src.core.use_cases.search_by_description import SearchByDescriptionUseCase
from tests.test_search_by_description import FakeClip, FakeClipIndex, FakeRepo, _tile


def test_prompt_variants_include_raw_and_templates():
    variants = description_prompt_variants("gray")
    assert "gray" in variants
    assert any("photo of a gray tile" in v for v in variants)
    assert len(variants) >= 3


def test_average_l2_normalized():
    a = np.array([1.0, 0.0, 0.0], dtype=np.float32)
    b = np.array([0.0, 1.0, 0.0], dtype=np.float32)
    out = average_l2_normalized([a, b])
    assert abs(float(np.linalg.norm(out)) - 1.0) < 1e-5
    assert out[0] > 0 and out[1] > 0


def test_distribution_aware_margin_scales_with_spread():
    # Spread large enough that 0.35*(top−median) beats the 0.02 floor.
    m = distribution_aware_margin(0.30, 0.20)
    assert abs(m - CLIP_SPREAD_FRACTION * 0.10) < 1e-9
    # Narrower gap floors at CLIP_MIN_MARGIN (0.35*0.05 = 0.0175 < 0.02).
    assert distribution_aware_margin(0.29, 0.24) == CLIP_MIN_MARGIN
    # Flat catalogue: top≈median → floor at CLIP_MIN_MARGIN.
    assert distribution_aware_margin(0.25, 0.25) == CLIP_MIN_MARGIN
    assert distribution_aware_margin(0.25, 0.249) == CLIP_MIN_MARGIN


def _narrow_band_scores(
    *,
    n: int = 38,
    lo: float = 0.18,
    hi: float = 0.30,
    peak: float = 0.295,
    peak_count: int = 4,
) -> list[tuple[int, float]]:
    """Synthetic scores shaped like real CLIP catalogue spreads (~0.18–0.30)."""
    assert peak_count < n
    body = np.linspace(lo, hi - 0.01, n - peak_count)
    peaks = [peak - i * 0.002 for i in range(peak_count)]
    scores = list(peaks) + [float(x) for x in body]
    return [(i + 1, scores[i]) for i in range(n)]


def test_relevance_cutoff_trims_narrow_clip_band():
    """Fixed 0.10 margin would keep ~all; adaptive margin keeps the top cluster."""
    scored = _narrow_band_scores()
    scores = [sc for _, sc in scored]
    stats = score_distribution_stats(scores)
    assert stats["spread"] <= 0.13
    assert stats["margin"] < 0.10  # tighter than the old fixed margin

    kept, fallback, before = apply_relevance_cutoff(scored, top_k=25)
    assert before == 38
    assert fallback is False
    # Must shrink well below catalogue / old top_k dump.
    assert 3 <= len(kept) <= 12
    # Peak ids (1..4) survive; deep tail does not.
    ids = {t for t, _ in kept}
    assert 1 in ids and 2 in ids
    assert (38 not in ids) or kept[-1][1] >= CLIP_ABS_FLOOR


def test_relevance_cutoff_drops_below_absolute_floor():
    scored = [(1, 0.55), (2, 0.52), (3, 0.50), (4, 0.18)] + [
        (i, 0.10) for i in range(5, 30)
    ]
    kept, fallback, before = apply_relevance_cutoff(scored, top_k=25)
    assert before == len(scored)
    assert fallback is False
    ids = [t for t, _ in kept]
    assert 1 in ids and 2 in ids and 3 in ids
    assert 4 not in ids  # below absolute floor
    assert len(kept) < 25


def test_relevance_cutoff_keeps_min_three_with_fallback():
    scored = [(1, 0.11), (2, 0.10), (3, 0.09), (4, 0.08)]
    kept, fallback, _ = apply_relevance_cutoff(scored, top_k=20)
    assert fallback is True
    assert len(kept) == 3


def test_score_distribution_stats_top10_and_counts():
    scored = _narrow_band_scores()
    scores = [sc for _, sc in scored]
    stats = score_distribution_stats(scores)
    assert len(stats["top10"]) == 10
    assert stats["max"] >= stats["median"] >= stats["min"]
    assert abs(stats["spread"] - (stats["max"] - stats["min"])) < 1e-6
    assert stats["above_floor"] == sum(1 for s in scores if s >= CLIP_ABS_FLOOR)
    assert stats["within_margin"] >= 1
    assert stats["margin"] >= CLIP_MIN_MARGIN


def test_description_search_batches_templates_and_cuts(tmp_path, caplog):
    tiles = [_tile(i, path=str(tmp_path / f"{i}.jpg")) for i in range(1, 39)]
    for t in tiles:
        Path(t.file_path).write_bytes(b"x")

    scored = _narrow_band_scores()
    # Search returns oversampled ids; FakeClipIndex zips ids/scores.
    ids = [tid for tid, _ in scored]
    scores = [sc for _, sc in scored]

    clip = FakeClip()
    uc = SearchByDescriptionUseCase(
        image_repository=FakeRepo(tiles),
        clip_embedder=clip,
        clip_index=FakeClipIndex(ids=ids, scores=scores, total=38),
        thumbnail_dir=str(tmp_path / "thumbs"),
    )
    import logging

    with caplog.at_level(logging.INFO, logger="tilevision.core.use_cases.search_by_description"):
        results = uc.execute("gray", top_k=25)

    # One batched embed_texts call, not four sequential-only embed_text.
    assert getattr(clip, "embed_batch_calls", 0) == 1
    assert len(clip.embed_calls) >= 3
    assert "gray" in clip.embed_calls
    assert 1 <= len(results) <= 12
    assert uc.last_result_meta.get("catalogue") == 38
    assert "gray" in str(uc.last_result_meta.get("status", ""))
    # Score log must include raw numbers for tuning.
    joined = " ".join(r.message for r in caplog.records)
    assert "description_search_scores" in joined
    assert "top10=" in joined
    assert "spread=" in joined
    assert "margin=" in joined
