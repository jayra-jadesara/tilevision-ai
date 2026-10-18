"""PR B: warm-up yields to indexing; aux drop logging; prefilter opt-in."""

from __future__ import annotations

import logging
import sys
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.ai.aux_crop_prefilter import (
    aux_prefilter_enabled,
    decide_aux_prefilter,
)
from src.ai.feature_extractor import FeatureExtractor
from src.ai.inference_guard import (
    begin_indexing_busy,
    end_indexing_busy,
    indexing_busy_active,
)
from src.ai.query_warmup import run_query_path_warmup


def test_indexing_busy_blocks_query_warmup(caplog):
    begin_indexing_busy()
    try:
        assert indexing_busy_active()

        class Boom:
            def warmup_query_inference(self, *, shapes=(1,)):
                raise AssertionError("warmup must not run while indexing")

        with caplog.at_level(logging.INFO, logger="tilevision.ai.query_warmup"):
            result = run_query_path_warmup(Boom())
        assert result == {}
        assert any("indexing/rebuild already running" in r.message for r in caplog.records)
    finally:
        end_indexing_busy()
    assert not indexing_busy_active()


def test_aux_drop_logs_cos(caplog):
    class StubEmbedder:
        pass

    fe = FeatureExtractor(embedder=StubEmbedder())
    primary = np.ones(8, dtype=np.float32)
    primary /= np.linalg.norm(primary)
    # Near-identical candidate → dropped at 0.985
    candidate = primary.copy()
    with caplog.at_level(logging.INFO, logger="tilevision.ai.feature_extractor"):
        kept = fe._maybe_append_aux(
            [], primary, candidate, label="center", image_name="x.jpg"
        )
    assert kept is False
    assert any("dropped" in r.message and "cos_vs_primary" in r.message for r in caplog.records)


def test_aux_keep_logs_vector(caplog):
    class StubEmbedder:
        pass

    fe = FeatureExtractor(embedder=StubEmbedder())
    primary = np.zeros(8, dtype=np.float32)
    primary[0] = 1.0
    candidate = np.zeros(8, dtype=np.float32)
    candidate[1] = 1.0  # orthogonal
    aux: list = []
    with caplog.at_level(logging.INFO, logger="tilevision.ai.feature_extractor"):
        kept = fe._maybe_append_aux(
            aux, primary, candidate, label="adaptive", image_name="y.jpg"
        )
    assert kept is True
    assert len(aux) == 1
    assert any("Index aux adaptive vector" in r.message for r in caplog.records)


def test_prefilter_default_off(monkeypatch):
    monkeypatch.delenv("TILEVISION_AUX_PREFILTER", raising=False)
    assert aux_prefilter_enabled() is False


def test_prefilter_skips_near_identical_large_crop(monkeypatch):
    monkeypatch.setenv("TILEVISION_AUX_PREFILTER", "1")
    assert aux_prefilter_enabled() is True
    primary = Image.new("RGB", (200, 200), color=(120, 80, 40))
    # Same texture, large area
    crop = primary.crop((10, 10, 190, 190))
    decision = decide_aux_prefilter(
        primary,
        crop,
        crop_box=(10, 10, 190, 190),
        primary_size=(200, 200),
    )
    assert decision.skip is True
    assert decision.reason == "thumb_near_dup"


def test_prefilter_keeps_different_crop(monkeypatch):
    monkeypatch.setenv("TILEVISION_AUX_PREFILTER", "1")
    primary = Image.new("RGB", (200, 200), color=(10, 10, 10))
    # Paint a bright corner so the crop looks different
    crop = Image.new("RGB", (60, 60), color=(240, 240, 10))
    decision = decide_aux_prefilter(
        primary,
        crop,
        crop_box=(0, 0, 60, 60),
        primary_size=(200, 200),
    )
    assert decision.skip is False


def test_bench_script_imports():
    """Harness must load without model weights (help / argparse only)."""
    import importlib.util

    path = Path(__file__).resolve().parents[1] / "dev_tools" / "bench_indexing_speed.py"
    spec = importlib.util.spec_from_file_location("bench_indexing_speed", path)
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    assert callable(mod.main)
