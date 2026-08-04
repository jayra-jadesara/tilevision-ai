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


def _make_catalog_sheet(tmp_path: Path) -> tuple[Path, Path]:
    """Build PGYS2319-like sheet + 600×600 slab crop."""
    slab = _make_marble(900, 450, seed=7)
    sheet = Image.new("RGB", (1200, 900), (255, 255, 255))
    sheet.paste(slab.resize((500, 880)), (20, 10))
    draw = ImageDraw.Draw(sheet)
    try:
        font = ImageFont.truetype(
            "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 64
        )
        font_sm = ImageFont.truetype(
            "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 22
        )
    except Exception:
        font = ImageFont.load_default()
        font_sm = font
    draw.text((560, 40), "ELEGANT", fill=(180, 150, 40), font=font)
    for i, line in enumerate(
        [
            "Design from neutral tones",
            "Soft light and shadow",
            "Delicate jade-like touch",
            "Created for aesthetics",
            "Qingyu Large Slab Series",
        ]
    ):
        draw.text((560, 130 + i * 34), line, fill=(10, 10, 10), font=font_sm)
    mini = slab.resize((90, 160))
    for r in range(2):
        for c in range(3):
            x, y = 560 + c * 110, 360 + r * 180
            sheet.paste(mini, (x, y))
            draw.rectangle((x, y, x + 90, y + 160), outline=(0, 0, 0), width=2)
    draw.text((560, 760), "PGYS2319  750*1500mm", fill=(0, 0, 0), font=font_sm)
    draw.rectangle((540, 20, 1180, 880), outline=(30, 30, 30), width=3)

    sheet_path = tmp_path / "sheet_PGYS2319.jpg"
    crop_path = tmp_path / "crop_600x600.jpg"
    sheet.save(sheet_path, quality=95)
    slab_big = slab.resize((600, 900))
    slab_big.crop((0, 150, 600, 750)).save(crop_path, quality=95)
    return sheet_path, crop_path


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
    h, w = 900, 1200
    sheet = Image.new("RGB", (w, h), (255, 255, 255))
    # Nearly flat white left slab with faint veins (std ~2–4).
    slab = np.full((h - 20, w // 2 - 40, 3), 248, dtype=np.float32)
    rng = np.random.default_rng(11)
    slab += rng.normal(0, 2.5, slab.shape)
    slab = np.clip(slab, 0, 255).astype(np.uint8)
    sheet.paste(Image.fromarray(slab), (20, 10))
    draw = ImageDraw.Draw(sheet)
    draw.text((w // 2 + 40, 40), "ELEGANT", fill=(180, 150, 40))
    draw.text((w // 2 + 40, 120), "PGYS2319 catalog sheet", fill=(0, 0, 0))
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

    def cos(a, b):
        a = np.asarray(a, dtype=np.float32).ravel()
        b = np.asarray(b, dtype=np.float32).ravel()
        return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-8))

    # Aux panel must be at least as aligned to the crop as the full sheet.
    assert cos(aux[0], crop_q.embedding) >= cos(sheet_feat.embedding, crop_q.embedding) - 0.02
    assert cos(aux[0], crop_q.embedding) > 0.80

    dim = len(sheet_feat.embedding)
    mgr = FaissIndexManager(index_path=str(tmp_path / "live.index"), dimension=dim)
    mgr.load_index()
    ids = [1, 1]
    vecs = [sheet_feat.embedding, aux[0]]
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
