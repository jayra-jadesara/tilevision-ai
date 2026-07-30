"""Unit tests for Index Advisor (read-only recommendations)."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.ai.index_advisor import (
    advise_index_backend,
    comparison_table,
    estimate_search_latency_ms,
    expected_recall,
)
from src.ai.index_backends import IndexBackend


def test_under_100k_recommends_flat_ip():
    advice = advise_index_backend(
        catalog_size=50_000,
        current_backend=IndexBackend.FLAT_IP,
        available_ram_mib=16_000,
        cpu_cores=8,
    )
    assert advice.recommended_backend is IndexBackend.FLAT_IP
    assert advice.rebuild_required is False
    assert advice.expected_recall == 1.0
    assert advice.optional_backend is None


def test_100k_to_500k_flat_with_optional_hnsw():
    advice = advise_index_backend(
        catalog_size=250_000,
        current_backend=IndexBackend.FLAT_IP,
        available_ram_mib=32_000,
        cpu_cores=8,
    )
    assert advice.recommended_backend is IndexBackend.FLAT_IP
    assert advice.optional_backend is IndexBackend.HNSW
    assert "optional" in advice.optional_note.lower() or "HNSW" in advice.optional_note


def test_500k_to_1m_recommends_hnsw():
    advice = advise_index_backend(
        catalog_size=750_000,
        current_backend=IndexBackend.FLAT_IP,
        available_ram_mib=32_000,
        cpu_cores=8,
    )
    assert advice.recommended_backend is IndexBackend.HNSW
    assert advice.rebuild_required is True
    assert advice.expected_recall < 1.0
    assert "approximate" in advice.approximate_warning.lower()


def test_over_1m_hnsw_when_ram_sufficient():
    advice = advise_index_backend(
        catalog_size=1_500_000,
        current_backend=IndexBackend.FLAT_IP,
        available_ram_mib=64_000,
        cpu_cores=16,
        embedding_dimension=1024,
    )
    assert advice.recommended_backend is IndexBackend.HNSW
    assert advice.recommended_backend is not IndexBackend.IVF_PQ


def test_over_1m_ivf_pq_when_ram_insufficient():
    advice = advise_index_backend(
        catalog_size=1_500_000,
        current_backend=IndexBackend.FLAT_IP,
        available_ram_mib=4096,  # tight — FlatIP/HNSW won't fit budget
        cpu_cores=4,
        embedding_dimension=1024,
    )
    assert advice.recommended_backend is IndexBackend.IVF_PQ
    assert advice.rebuild_required is True
    assert "RAM" in advice.reason or "memory" in advice.reason.lower()


def test_advisor_never_mutates_implicitly():
    """Advise is pure — calling it twice with same inputs is stable."""
    a = advise_index_backend(catalog_size=10, current_backend="flat_ip")
    b = advise_index_backend(catalog_size=10, current_backend="flat_ip")
    assert a.to_dict()["recommended_backend"] == b.to_dict()["recommended_backend"]
    assert a.rebuild_required is False


def test_comparison_table_has_four_backends():
    rows = comparison_table(catalog_size=100_000, dimension=1024)
    assert len(rows) == 4
    assert rows[0].exact is True
    assert rows[1].exact is False
    assert "FlatIP" in rows[0].backend or "flat" in rows[0].backend.lower()


def test_latency_scales_with_catalog():
    small = estimate_search_latency_ms(
        IndexBackend.FLAT_IP, catalog_size=10_000, dimension=1024, cpu_cores=4
    )
    large = estimate_search_latency_ms(
        IndexBackend.FLAT_IP, catalog_size=1_000_000, dimension=1024, cpu_cores=4
    )
    assert large > small
    assert expected_recall(IndexBackend.FLAT_IP) == 1.0
    assert expected_recall(IndexBackend.IVF_PQ) < expected_recall(IndexBackend.HNSW)
