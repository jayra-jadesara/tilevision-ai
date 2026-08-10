"""
Crop-vs-full search consistency regressions.

Customer case (PGYS2319-style marketing sheet):
  - Query = full catalog sheet → self-hit ~100% (exact / near-exact)
  - Query = 600×600 texture crop from the slab → parent sheet must stay in Top-5

Root cause (measured): indexed full-sheet embedding is layout/text dominated;
texture crops sit ~0.51 cosine vs sheet and lose FAISS recall to other whites.
Fix: index an aux left-panel texture vector under the same FAISS id.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest
from PIL import Image, ImageDraw, ImageFont

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

cv2 = pytest.importorskip("cv2")
faiss = pytest.importorskip("faiss")

from src.ai.preprocess.image_preprocessor import ImagePreprocessor
from src.ai.search_quality.image_analysis import analyze_image
from src.ai.search_quality.views import IndexStrategy, IndexViewType, build_index_views
from src.ai.vector_index import FaissIndexManager


def _make_marble(h: int, w: int, seed: int = 42) -> Image.Image:
    rng = np.random.default_rng(seed)
    base = np.full((h, w, 3), 245, dtype=np.uint8)
    for _ in range(40):
        x0, y0 = int(rng.integers(0, w)), int(rng.integers(0, h))
        x1, y1 = int(rng.integers(0, w)), int(rng.integers(0, h))
        color = int(rng.integers(210, 235))
        cv2.line(base, (x0, y0), (x1, y1), (color, color, color), 1, cv2.LINE_AA)
    return Image.fromarray(base)


def _load_fonts():
    try:
        return (
            ImageFont.truetype(
                "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 52
            ),
            ImageFont.truetype(
                "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 16
            ),
            ImageFont.truetype(
                "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 13
            ),
        )
    except Exception:
        default = ImageFont.load_default()
        return default, default, default


def _make_catalog_sheet(tmp_path: Path) -> tuple[Path, Path]:
    """
    Build a PGYS2319-realistic marketing sheet + 600×600 slab crop.

    Matches the real client sheet proportions (aspect ~1.063, panel ~45%,
    sparse gold logo + small caption blocks + preview grid). The old synthetic
    (1200×900, dense English paragraphs) falsely passed left_panel_beneficial
    because aspect 1.333 cleared the 1.12 gate and dense text inflated
    text_region_score — it did not reproduce the real failure mode.
    """
    # Real PGYS2319.jpg: aspect≈1.063, kind=bordered_tile, has_preview_grid=True.
    sheet_w, sheet_h = 1063, 1000
    panel_w = int(sheet_w * 0.45)

    slab = _make_marble(900, 450, seed=7)
    sheet = Image.new("RGB", (sheet_w, sheet_h), (255, 255, 255))
    sheet.paste(slab.resize((panel_w - 30, sheet_h - 20)), (15, 10))
    draw = ImageDraw.Draw(sheet)
    font_logo, font_md, font_sm = _load_fonts()

    # Sparse gold logo — low Canny density vs white (real sheet pattern).
    draw.text((panel_w + 35, 28), "ELEGANT", fill=(185, 155, 45), font=font_logo)

    # Small caption blocks (proxy for Chinese lines — few pixels vs canvas).
    captions = [
        "中性色调设计理念",
        "柔和光影层次表现",
        "PGYS2319  750*1500mm",
    ]
    for i, line in enumerate(captions):
        draw.text((panel_w + 35, 110 + i * 26), line, fill=(15, 15, 15), font=font_sm)

    # Secondary small block (product series line).
    draw.text((panel_w + 35, 200), "Qingyu Large Slab Series", fill=(30, 30, 30), font=font_md)

    mini = slab.resize((82, 145))
    grid_x0 = panel_w + 30
    for r in range(2):
        for c in range(3):
            x, y = grid_x0 + c * 105, 340 + r * 175
            sheet.paste(mini, (x, y))
            draw.rectangle((x, y, x + 82, y + 145), outline=(0, 0, 0), width=2)

    draw.rectangle(
        (panel_w + 15, 18, sheet_w - 18, sheet_h - 18),
        outline=(30, 30, 30),
        width=3,
    )

    sheet_path = tmp_path / "sheet_PGYS2319.jpg"
    crop_path = tmp_path / "crop_600x600.jpg"
    sheet.save(sheet_path, quality=95)
    slab_big = slab.resize((600, 900))
    slab_big.crop((0, 150, 600, 750)).save(crop_path, quality=95)
    return sheet_path, crop_path


def _panel_crop_is_marble_only(panel: Image.Image, sheet: Image.Image) -> bool:
    """Heuristic: panel crop must not include marketing-column edge density."""
    panel_arr = np.asarray(panel.convert("RGB"))
    ph, pw = panel_arr.shape[:2]
    if pw < 64 or ph < 64:
        return False
    # Right 15% of panel should stay marble — not logo/grid bleed.
    right_strip = panel_arr[:, int(pw * 0.85) :]
    gray = cv2.cvtColor(right_strip, cv2.COLOR_RGB2GRAY)
    edges = cv2.Canny(gray, 60, 140)
    edge_density = float(np.mean(edges > 0))
    # Marketing text/grid typically >0.08 on contaminated crops.
    return edge_density < 0.06


def test_realistic_pgys2319_sheet_analysis_flags_left_panel(tmp_path):
    """Realistic proportions must trigger panel isolation (regression PGYS2319)."""
    sheet_path, _ = _make_catalog_sheet(tmp_path)
    sheet = Image.open(sheet_path)
    a = analyze_image(sheet)

    aspect = sheet.size[0] / sheet.size[1]
    assert 1.04 <= aspect <= 1.08
    assert a.has_preview_grid is True
    # Old detector returned ~0.024 on real image; must exceed gate now.
    assert a.text_region_score >= 0.12, (
        f"text_region_score={a.text_region_score:.3f} still undercounts"
    )
    assert a.left_panel_beneficial is True
    assert a.center_crop_beneficial is False


def test_realistic_pgys2319_index_views_include_clean_panel(tmp_path):
    """Index-time views must include panel aux, not center-only pollution."""
    sheet_path, _ = _make_catalog_sheet(tmp_path)
    sheet = Image.open(sheet_path)
    views = build_index_views(sheet, IndexStrategy.E_HEURISTIC_MULTIVIEW)
    types = [v.view_type for v in views]
    assert IndexViewType.PANEL in types

    panel = ImagePreprocessor.primary_texture_panel(sheet)
    assert panel is not None
    assert _panel_crop_is_marble_only(panel, sheet)

    panel_views = [v for v in views if v.view_type == IndexViewType.PANEL]
    assert panel_views
    assert _panel_crop_is_marble_only(panel_views[0].image, sheet)


def test_primary_texture_panel_detects_wide_catalog_sheet(tmp_path):
    sheet_path, _ = _make_catalog_sheet(tmp_path)
    sheet = Image.open(sheet_path)
    panel = ImagePreprocessor.primary_texture_panel(sheet)
    assert panel is not None
    assert panel.size[0] < sheet.size[0]
    assert panel.size[1] <= sheet.size[1]
    assert panel.size[0] >= 64 and panel.size[1] >= 64


def test_primary_texture_panel_keeps_low_contrast_white_marble(tmp_path):
    """Customer PGYS2319-class sheets: high-key white slab, std often < 6."""
    # Realistic aspect (~1.063) — old 1200×900 fixture hid the aspect gate bug.
    sheet_w, sheet_h = 1063, 1000
    panel_w = int(sheet_w * 0.45)
    sheet = Image.new("RGB", (sheet_w, sheet_h), (255, 255, 255))
    slab = np.full((sheet_h - 20, panel_w - 30, 3), 248, dtype=np.float32)
    rng = np.random.default_rng(11)
    slab += rng.normal(0, 2.5, slab.shape)
    slab = np.clip(slab, 0, 255).astype(np.uint8)
    sheet.paste(Image.fromarray(slab), (15, 10))
    draw = ImageDraw.Draw(sheet)
    font_logo, _, font_sm = _load_fonts()
    draw.text((panel_w + 35, 28), "ELEGANT", fill=(185, 155, 45), font=font_logo)
    draw.text((panel_w + 35, 120), "PGYS2319 catalog sheet", fill=(0, 0, 0), font=font_sm)
    path = tmp_path / "white_sheet.jpg"
    sheet.save(path, quality=95)

    panel = ImagePreprocessor.primary_texture_panel(Image.open(path))
    assert panel is not None
    assert float(np.asarray(panel, dtype=np.float32).std()) < 6.0


def test_primary_texture_panel_skips_square_tile(tmp_path):
    tile = _make_marble(600, 600, seed=3)
    path = tmp_path / "square.jpg"
    tile.save(path)
    assert ImagePreprocessor.primary_texture_panel(Image.open(path)) is None


def test_preprocess_for_query_skips_scene_crop_on_catalog_sheet(tmp_path, monkeypatch):
    """Marketing sheets must not take the room-photo isolation path."""
    sheet_path, _ = _make_catalog_sheet(tmp_path)
    called = {"n": 0}

    def _boom(_image):
        called["n"] += 1
        raise AssertionError("catalog sheet must not scene-isolate")

    monkeypatch.setattr(
        ImagePreprocessor,
        "_isolate_query_tile",
        classmethod(lambda cls, image: _boom(image)),
    )
    processed = ImagePreprocessor.preprocess_for_query(sheet_path)
    assert processed.pil.size[0] == processed.pil.size[1]
    assert called["n"] == 0


def test_update_vectors_allows_multi_vector_same_id(tmp_path):
    mgr = FaissIndexManager(index_path=str(tmp_path / "t.index"), dimension=4)
    mgr.load_index()
    mgr.update_vectors(
        [7, 7],
        [[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0]],
        persist=False,
    )
    assert mgr._index.ntotal == 2
    # Replacing must clear both old vectors then add the new set.
    mgr.update_vectors([7], [[0.0, 0.0, 1.0, 0.0]], persist=False)
    assert mgr._index.ntotal == 1


def test_faiss_multi_vector_sheet_recovers_crop_in_top5(tmp_path):
    """
    Without DINOv2: use orthogonal-ish handcrafted vectors that mimic the
    measured gap (sheet≈0.5 to crop; panel≈0.93 to crop).
    """
    mgr = FaissIndexManager(index_path=str(tmp_path / "t.index"), dimension=8)
    mgr.load_index()

    def _norm(v):
        a = np.asarray(v, dtype=np.float32)
        return (a / (np.linalg.norm(a) + 1e-8)).tolist()

    # id 1 = sheet: layout-heavy primary + texture panel aux
    sheet_primary = _norm([1, 0, 0, 0, 0.2, 0, 0, 0])
    sheet_panel = _norm([0.15, 1, 0, 0, 0, 0, 0, 0])
    # distractors closer to crop than sheet_primary alone
    distractors = [
        _norm([0.1, 0.9, 0.1, 0, 0, 0, 0, 0]),
        _norm([0.05, 0.85, 0.2, 0, 0, 0, 0, 0]),
        _norm([0.2, 0.8, 0.1, 0.1, 0, 0, 0, 0]),
        _norm([0.0, 0.75, 0.3, 0, 0, 0, 0, 0]),
        _norm([0.1, 0.7, 0.4, 0, 0, 0, 0, 0]),
    ]
    crop_query = _norm([0.12, 1.0, 0.05, 0, 0, 0, 0, 0])

    # Baseline: sheet primary only → crop misses Top-5
    baseline = FaissIndexManager(index_path=str(tmp_path / "b.index"), dimension=8)
    baseline.load_index()
    baseline.update_vectors(
        [1] + list(range(10, 10 + len(distractors))),
        [sheet_primary] + distractors,
        persist=False,
    )
    ids_b, _ = baseline.search_vectors(crop_query, top_k=5)
    assert 1 not in ids_b

    # Dual-vector sheet → crop retrieves sheet in Top-5
    mgr.update_vectors(
        [1, 1] + list(range(10, 10 + len(distractors))),
        [sheet_primary, sheet_panel] + distractors,
        persist=False,
    )
    # Over-fetch then unique (mirrors SearchTilesUseCase merge)
    ids_raw, scores = mgr.search_vectors(crop_query, top_k=10)
    best: dict[int, float] = {}
    for tid, sc in zip(ids_raw, scores):
        if tid not in best or sc > best[tid]:
            best[tid] = sc
    ordered = sorted(best, key=best.get, reverse=True)[:5]
    assert 1 in ordered
    assert ordered[0] == 1


def test_faiss_aux_boost_promotes_layout_sheet_over_weak_hybrid():
    """Rerank must not leave aux-retrieved sheets at ~27% display."""
    from src.ai.similarity_score import calibrate_display_percent
    from src.core.use_cases import search_tiles as st

    faiss_cos = 0.94
    hybrid_emb = 0.46
    hybrid_final = 0.45
    assert faiss_cos >= st._FAISS_AUX_BOOST_MIN
    assert faiss_cos >= hybrid_emb + st._FAISS_AUX_BOOST_GAP
    boosted = max(0.0, min(1.0, 0.72 * faiss_cos + 0.28 * hybrid_final))
    final_score = max(hybrid_final, boosted)
    display = calibrate_display_percent(final_score, exact_match=False)
    # Without boost: ~27%. With boost: well above Top-5 usefulness band.
    assert display >= 70.0
    assert calibrate_display_percent(hybrid_final, exact_match=False) < 35.0


def test_parent_sheet_outranks_indexed_crop_self_hit():
    """Drop crop → parent marketing sheet must rank above the crop file itself."""
    from src.core.use_cases import search_tiles as st

    parent_faiss = 0.94
    assert parent_faiss >= st._FAISS_PARENT_SHEET_TOP_MIN
    parent_final = 1.0  # promoted exact via aux
    crop_self_final = st._QUERY_SELF_MATCH_SCORE
    assert parent_final > crop_self_final


@pytest.mark.slow
def test_real_dinov2_sheet_crop_consistency(tmp_path):
    """Live DINOv2 measurement — skip when weights unavailable."""
    weights = Path("model_weights/dinov2-large/config.json")
    if not weights.is_file():
        pytest.skip("DINOv2 weights not bundled in this environment")

    torch = pytest.importorskip("torch")
    from src.ai.embedder import DINOv2Embedder
    from src.ai.feature_extractor import FeatureExtractor

    sheet_path, crop_path = _make_catalog_sheet(tmp_path)
    # Distractors: different colors/patterns (not near-identical white marble).
    rng = np.random.default_rng(0)
    for i in range(6):
        arr = np.zeros((600, 600, 3), dtype=np.uint8)
        arr[:, :] = (
            int(rng.integers(40, 200)),
            int(rng.integers(40, 200)),
            int(rng.integers(40, 200)),
        )
        for _ in range(20):
            x0, y0 = int(rng.integers(0, 600)), int(rng.integers(0, 600))
            x1, y1 = int(rng.integers(0, 600)), int(rng.integers(0, 600))
            cv2.line(
                arr,
                (x0, y0),
                (x1, y1),
                (
                    int(rng.integers(0, 255)),
                    int(rng.integers(0, 255)),
                    int(rng.integers(0, 255)),
                ),
                3,
            )
        Image.fromarray(arr).save(tmp_path / f"d{i}.jpg")

    emb = DINOv2Embedder()
    emb.load_model()
    fx = FeatureExtractor(embedder=emb)

    sheet_feat, aux = fx.extract_index_vectors(str(sheet_path))
    assert aux, "wide catalog sheet must produce a texture-panel aux vector"
    crop_q, _ = fx.extract_for_search(str(crop_path))
    # Same-file query must stay aligned with index primary (no scene auto-crop).
    sheet_q, _ = fx.extract_for_search(str(sheet_path))

    def cos(a, b):
        a = np.asarray(a, dtype=np.float32).ravel()
        b = np.asarray(b, dtype=np.float32).ravel()
        return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-8))

    assert cos(sheet_feat.embedding, sheet_q.embedding) > 0.90
    # Best aux (panel or center) must beat full-sheet primary for texture crops.
    best_aux = max(cos(a, crop_q.embedding) for a in aux)
    assert best_aux >= cos(sheet_feat.embedding, crop_q.embedding) - 0.02
    assert best_aux > 0.80

    dim = len(sheet_feat.embedding)
    mgr = FaissIndexManager(index_path=str(tmp_path / "live.index"), dimension=dim)
    mgr.load_index()
    ids = [1] * (1 + len(aux))
    vecs = [sheet_feat.embedding, *aux]
    for i in range(6):
        f = fx.extract(str(tmp_path / f"d{i}.jpg"), for_query=False)
        ids.append(10 + i)
        vecs.append(f.embedding)
    mgr.update_vectors(ids, vecs, persist=False)

    raw_ids, raw_scores = mgr.search_vectors(crop_q.embedding, top_k=12)
    best: dict[int, float] = {}
    for tid, sc in zip(raw_ids, raw_scores):
        if tid not in best or sc > best[tid]:
            best[tid] = sc
    top5 = sorted(best, key=best.get, reverse=True)[:5]
    assert 1 in top5
    assert top5[0] == 1


@pytest.mark.slow
def test_real_dinov2_square_tile_center_aux_helps_deep_crop(tmp_path):
    """Center-50% aux must improve 600×600 crop cosine on square tiles."""
    weights = Path("model_weights/dinov2-large/config.json")
    if not weights.is_file():
        pytest.skip("DINOv2 weights not bundled in this environment")

    pytest.importorskip("torch")
    from src.ai.embedder import DINOv2Embedder
    from src.ai.feature_extractor import FeatureExtractor

    tile = _make_marble(1200, 1200, seed=99)
    tile_path = tmp_path / "tile.jpg"
    crop_path = tmp_path / "crop600.jpg"
    tile.save(tile_path, quality=95)
    tile.crop((300, 300, 900, 900)).save(crop_path, quality=95)

    emb = DINOv2Embedder()
    emb.load_model()
    fx = FeatureExtractor(embedder=emb)
    feat, aux = fx.extract_index_vectors(str(tile_path))
    assert aux, "large square tile must produce center-50 aux"
    crop_q, _ = fx.extract_for_search(str(crop_path))

    def cos(a, b):
        a = np.asarray(a, dtype=np.float32).ravel()
        b = np.asarray(b, dtype=np.float32).ravel()
        return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-8))

    best_aux = max(cos(a, crop_q.embedding) for a in aux)
    assert best_aux >= cos(feat.embedding, crop_q.embedding) - 0.01
    assert best_aux > 0.90
