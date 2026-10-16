"""Prompt ensemble + relevance cutoff for description search."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.core.use_cases.clip_text_query import (
    apply_relevance_cutoff,
    average_l2_normalized,
    description_prompt_variants,
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


def test_relevance_cutoff_drops_weak_tail():
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


def test_description_search_ensembles_and_cuts(tmp_path):
    tiles = [_tile(i, path=str(tmp_path / f"{i}.jpg")) for i in range(1, 39)]
    for t in tiles:
        Path(t.file_path).write_bytes(b"x")
    # 25 high-ish scores like the user's log (unfiltered top_k fill).
    ids = list(range(1, 26))
    scores = [0.55 - i * 0.01 for i in range(25)]
    # Make the tail clearly weak so cutoff shrinks the list.
    scores = [0.55 if i < 5 else 0.15 for i in range(25)]

    clip = FakeClip()
    uc = SearchByDescriptionUseCase(
        image_repository=FakeRepo(tiles),
        clip_embedder=clip,
        clip_index=FakeClipIndex(ids=ids, scores=scores, total=38),
        thumbnail_dir=str(tmp_path / "thumbs"),
    )
    results = uc.execute("gray", top_k=25)
    # Ensemble called embed_text multiple times.
    assert len(clip.embed_calls) >= 3
    assert "gray" in clip.embed_calls
    # Cutoff should not return all 25 weak-filled hits.
    assert 1 <= len(results) <= 10
    assert uc.last_result_meta.get("catalogue") == 38
    assert "gray" in str(uc.last_result_meta.get("status", ""))
