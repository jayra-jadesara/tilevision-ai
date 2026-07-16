"""Tests for FAISS candidate pool sizing (Phase 7)."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.core.use_cases.search_tiles import SearchTilesUseCase


def test_search_k_respects_50_floor():
    assert SearchTilesUseCase._compute_faiss_search_k(top_k=5, total_vectors=1000) == 50


def test_search_k_caps_at_200():
    assert SearchTilesUseCase._compute_faiss_search_k(top_k=50, total_vectors=10000) == 200


def test_search_k_scales_with_top_k():
    assert SearchTilesUseCase._compute_faiss_search_k(top_k=20, total_vectors=1000) == 100


def test_search_k_never_exceeds_total_vectors():
    assert SearchTilesUseCase._compute_faiss_search_k(top_k=20, total_vectors=30) == 30
