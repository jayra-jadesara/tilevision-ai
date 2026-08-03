"""Unit tests for SearchOptimizationEngine (measurement-injected decisions)."""

from __future__ import annotations

import sys
from datetime import datetime, timezone
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.ai.index_backends import IndexBackend
from src.ai.search_optimization_engine import (
    SearchOptimizationEngine,
    format_tile_count,
)


def test_format_tile_count_is_dynamic():
    assert format_tile_count(1) == "1 Tile"
    assert format_tile_count(5000) == "5,000 Tiles"
    assert format_tile_count(325410) == "325,410 Tiles"


def test_flat_ip_when_memory_fits(tmp_path, monkeypatch):
    engine = SearchOptimizationEngine(index_path=tmp_path / "tiles.index", embedding_dimension=1024)

    monkeypatch.setattr(
        SearchOptimizationEngine,
        "_measure_ram_mib",
        staticmethod(lambda: (16384.0, 8192.0)),
    )
    monkeypatch.setattr(
        SearchOptimizationEngine,
        "_measure_process_rss_mib",
        staticmethod(lambda: 1500.0),
    )
    monkeypatch.setattr(
        engine,
        "_measure_model_footprint_mib",
        lambda: 1100.0,
    )
    monkeypatch.setattr(
        engine,
        "_run_latency_benchmark",
        lambda catalog_size: (0.4, "unit-test"),
    )
    monkeypatch.setattr(engine, "_read_existing_index_info", lambda: ("flat_ip", "1.2.12"))

    # Large catalog still prefers FlatIP when measured RAM fits the estimate.
    decision = engine.analyze_and_decide(catalog_size=50_000, run_benchmark=True)
    assert decision.selected_backend is IndexBackend.FLAT_IP
    assert decision.search_health == "Excellent"
    assert decision.rebuild_required is False
    assert "IndexFlatIP" not in decision.customer_summary


def test_hnsw_only_when_flat_cannot_fit(tmp_path, monkeypatch):
    engine = SearchOptimizationEngine(index_path=tmp_path / "tiles.index", embedding_dimension=1024)

    # Tiny available RAM forces FlatIP estimate over budget for a huge catalog.
    monkeypatch.setattr(
        SearchOptimizationEngine,
        "_measure_ram_mib",
        staticmethod(lambda: (2048.0, 200.0)),
    )
    monkeypatch.setattr(
        SearchOptimizationEngine,
        "_measure_process_rss_mib",
        staticmethod(lambda: 1800.0),
    )
    monkeypatch.setattr(engine, "_measure_model_footprint_mib", lambda: 1100.0)
    monkeypatch.setattr(engine, "_run_latency_benchmark", lambda catalog_size: (None, "skipped"))
    monkeypatch.setattr(engine, "_read_existing_index_info", lambda: ("flat_ip", "1.2.12"))

    decision = engine.analyze_and_decide(catalog_size=2_000_000, run_benchmark=False)
    # With only ~180 MiB usable (200 * 0.9), FlatIP for 2M×1024 floats won't fit;
    # engine may pick HNSW or IVF-PQ or retain FlatIP for accuracy — assert accuracy-first rule:
    assert decision.selected_backend in (
        IndexBackend.FLAT_IP,
        IndexBackend.HNSW,
        IndexBackend.IVF_PQ,
    )
    if decision.snapshot.estimated_flat_ip_mib <= decision.snapshot.usable_index_ram_mib:
        assert decision.selected_backend is IndexBackend.FLAT_IP
    elif decision.snapshot.estimated_hnsw_mib <= decision.snapshot.usable_index_ram_mib:
        assert decision.selected_backend is IndexBackend.HNSW
    elif decision.snapshot.estimated_ivf_pq_mib <= decision.snapshot.usable_index_ram_mib:
        assert decision.selected_backend is IndexBackend.IVF_PQ
    else:
        assert decision.selected_backend is IndexBackend.FLAT_IP


def test_rebuild_required_when_on_disk_differs(tmp_path, monkeypatch):
    engine = SearchOptimizationEngine(index_path=tmp_path / "tiles.index", embedding_dimension=1024)
    monkeypatch.setattr(
        SearchOptimizationEngine,
        "_measure_ram_mib",
        staticmethod(lambda: (32768.0, 16000.0)),
    )
    monkeypatch.setattr(
        SearchOptimizationEngine,
        "_measure_process_rss_mib",
        staticmethod(lambda: 1000.0),
    )
    monkeypatch.setattr(engine, "_measure_model_footprint_mib", lambda: 1000.0)
    monkeypatch.setattr(engine, "_run_latency_benchmark", lambda catalog_size: (0.2, "ok"))
    monkeypatch.setattr(engine, "_read_existing_index_info", lambda: ("hnsw", "1.2.10"))

    decision = engine.analyze_and_decide(catalog_size=5_000, run_benchmark=True)
    assert decision.selected_backend is IndexBackend.FLAT_IP
    assert decision.rebuild_required is True
    assert decision.index_status == "Needs rebuild"


def test_apply_to_settings_persists_customer_fields(tmp_path, monkeypatch):
    from src.config.settings import AppSettings

    settings = AppSettings(config_dir=tmp_path)
    engine = SearchOptimizationEngine(
        index_path=tmp_path / "tiles.index",
        embedding_dimension=1024,
        now_fn=lambda: datetime(2026, 8, 3, 21, 0, tzinfo=timezone.utc),
    )
    monkeypatch.setattr(
        SearchOptimizationEngine,
        "_measure_ram_mib",
        staticmethod(lambda: (8192.0, 4096.0)),
    )
    monkeypatch.setattr(
        SearchOptimizationEngine,
        "_measure_process_rss_mib",
        staticmethod(lambda: 800.0),
    )
    monkeypatch.setattr(engine, "_measure_model_footprint_mib", lambda: 900.0)
    monkeypatch.setattr(engine, "_run_latency_benchmark", lambda catalog_size: (0.3, "ok"))
    monkeypatch.setattr(engine, "_read_existing_index_info", lambda: ("flat_ip", ""))

    decision = engine.analyze_and_decide(catalog_size=8214, run_benchmark=True)
    engine.apply_to_settings(settings, decision)
    assert settings.index_backend == "flat_ip"
    assert settings.search_engine_mode == "automatic"
    assert settings.search_optimization_status == "Optimized"
    assert settings.last_optimization_catalog_size == 8214
    assert settings.search_health == "Excellent"
