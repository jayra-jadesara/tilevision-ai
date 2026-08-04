"""Tests for FAISS candidate pool sizing (Phase 7 + multi-vector over-fetch)."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.core.use_cases.search_tiles import SearchTilesUseCase


def test_search_k_respects_50_floor_with_2x_overfetch():
    # unique_target = max(5*5, 50) = 50 → search_k = 100
    assert SearchTilesUseCase._compute_faiss_search_k(top_k=5, total_vectors=1000) == 100


def test_search_k_caps_at_200_unique_then_2x():
    # unique_target = min(50*5, 200) = 200 → search_k = 400
    assert SearchTilesUseCase._compute_faiss_search_k(top_k=50, total_vectors=10000) == 400


def test_search_k_scales_with_top_k_then_2x():
    # unique_target = 20*5 = 100 → search_k = 200
    assert SearchTilesUseCase._compute_faiss_search_k(top_k=20, total_vectors=1000) == 200


def test_search_k_never_exceeds_total_vectors():
    assert SearchTilesUseCase._compute_faiss_search_k(top_k=20, total_vectors=30) == 30
