"""
Regression gate for production search accuracy.

Fast tests cover analysis / fusion / view builders.
Slow tests (DINOv2) verify the winning strategy does not regress
customer-critical variants relative to primary-only on a tiny catalog.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.ai.search_quality.fusion import FusionMethod, ScoredHit, fuse_hits
from src.ai.search_quality.views import IndexStrategy, build_index_views


def test_all_strategies_include_primary():
    img = Image.fromarray(
        np.random.default_rng(0).integers(100, 200, (640, 640, 3), dtype=np.uint8)
    )
    for strategy in IndexStrategy:
        views = build_index_views(img, strategy)
        assert views, strategy
        assert views[0].view_type.value == "primary"


def test_fusion_methods_are_deterministic():
    hits = [
        ScoredHit(3, 0.8, 1.0, 1),
        ScoredHit(3, 0.7, 0.9, 2),
        ScoredHit(5, 0.85, 1.0, 3),
    ]
    for method in FusionMethod:
        a = fuse_hits(hits, method)
        b = fuse_hits(hits, method)
        assert a == b
        assert {tid for tid, _ in a} == {3, 5}


@pytest.mark.slow
def test_winning_strategy_beats_or_ties_primary_on_sheet_crop(tmp_path):
    """
    Tiny live DINOv2 gate: sheet texture crop must retrieve parent under
    heuristic / production multi-view, and must not under primary-only.
    """
    weights = Path("model_weights/dinov2-large/config.json")
    if not weights.is_file():
        pytest.skip("DINOv2 weights unavailable")

    pytest.importorskip("torch")
    from src.ai.embedder import DINOv2Embedder
    from src.ai.feature_extractor import FeatureExtractor
    from src.ai.preprocess.image_preprocessor import ImagePreprocessor
    from src.ai.vector_index import FaissIndexManager
    from src.ai.search_quality.views import IndexViewType
    from tests.test_crop_search_consistency import _make_catalog_sheet

    sheet_path, crop_path = _make_catalog_sheet(tmp_path)
    emb = DINOv2Embedder()
    emb.load_model()
    fx = FeatureExtractor(embedder=emb)

    def cos(a, b):
        a = np.asarray(a, np.float32).ravel()
        b = np.asarray(b, np.float32).ravel()
        return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-8))

    raw = ImagePreprocessor.to_rgb(ImagePreprocessor.load(sheet_path))
    primary = fx.extract(str(sheet_path), for_query=False).embedding
    crop_q, _ = fx.extract_for_search(str(crop_path))

    # Primary-only: crop should be weakly aligned
    assert cos(primary, crop_q.embedding) < 0.75

    views = build_index_views(raw, IndexStrategy.E_HEURISTIC_MULTIVIEW)
    assert any(v.view_type in {IndexViewType.PANEL, IndexViewType.PANEL_CENTER} for v in views)

    # Production extract_index_vectors must return aux for this sheet
    _feat, aux = fx.extract_index_vectors(str(sheet_path))
    assert aux, "sheet must produce aux under production indexer"
    assert max(cos(a, crop_q.embedding) for a in aux) > 0.85

    mgr = FaissIndexManager(str(tmp_path / "g.index"), dimension=len(primary))
    mgr.load_index()
    mgr.update_vectors([1] * (1 + len(aux)), [primary, *aux], persist=False)
    # distractor
    d = Image.fromarray(np.full((600, 600, 3), 40, dtype=np.uint8))
    dp = tmp_path / "d.jpg"
    d.save(dp)
    de = fx.extract(str(dp), for_query=False).embedding
    mgr.update_vectors([9], [de], persist=False)

    ids, scores = mgr.search_vectors(crop_q.embedding, top_k=5)
    best = {}
    for i, s in zip(ids, scores):
        if i not in best or s > best[i]:
            best[i] = s
    top = sorted(best, key=best.get, reverse=True)
    assert top[0] == 1
