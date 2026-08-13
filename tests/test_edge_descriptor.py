"""EdgeDescriptor: adaptive detection + empty-hist similarity."""

from __future__ import annotations

import cv2
import numpy as np
from PIL import Image, ImageDraw

from src.ai.descriptors.edge_descriptor import EdgeDescriptor
from src.ai.preprocess.image_preprocessor import ImagePreprocessor


def _cream_marble_bgr(size: int = 256) -> np.ndarray:
    """High-key low-contrast marble with subtle diagonal veins."""
    img = Image.new("RGB", (size, size), (238, 236, 232))
    pix = img.load()
    for y in range(size):
        for x in range(size):
            n = ((x * 13 + y * 29) % 89) / 89.0
            vein = 1.0 if abs((x * 0.4 + y * 0.5) % 70 - 35) < 2.0 else 0.0
            g = int(230 + 16 * n - 22 * vein)
            pix[x, y] = (g, g - 1, g - 3)
    return cv2.cvtColor(np.asarray(img), cv2.COLOR_RGB2BGR)


def _patterned_granite_bgr(size: int = 256) -> np.ndarray:
    rng = np.random.RandomState(1)
    arr = rng.randint(40, 220, (size, size, 3), dtype=np.uint8)
    for _ in range(80):
        x0, y0 = int(rng.randint(0, size)), int(rng.randint(0, size))
        x1, y1 = int(rng.randint(0, size)), int(rng.randint(0, size))
        color = int(rng.randint(10, 250))
        cv2.line(arr, (x0, y0), (x1, y1), (color, color, color), thickness=2)
    return arr


def _solid_bgr(size: int = 256, value: int = 200) -> np.ndarray:
    return np.full((size, size, 3), value, dtype=np.uint8)


def test_extract_finds_edges_on_low_contrast_marble():
    """Fixed Canny 80/180 found 0 edges here; adaptive must not return zeros."""
    bgr = _cream_marble_bgr()
    hist = EdgeDescriptor.extract(bgr)
    assert hist.shape == (EdgeDescriptor.ORIENTATION_BINS,)
    assert float(np.linalg.norm(hist)) > 0.5
    assert int(np.count_nonzero(hist)) >= 2


def test_similarity_marble_pair_not_exactly_zero():
    a = _cream_marble_bgr()
    b = _cream_marble_bgr()
    # Slight translation of the same material field.
    b = np.roll(b, shift=3, axis=1)
    sim = EdgeDescriptor.similarity(
        EdgeDescriptor.extract(a),
        EdgeDescriptor.extract(b),
    )
    assert sim > 0.5, f"marble-marble edge sim collapsed to {sim:.4f}"


def test_similarity_both_empty_is_one_not_zero():
    z = np.zeros(EdgeDescriptor.ORIENTATION_BINS, dtype=np.float32)
    assert EdgeDescriptor.similarity(z, z) == 1.0


def test_similarity_one_empty_is_zero():
    z = np.zeros(EdgeDescriptor.ORIENTATION_BINS, dtype=np.float32)
    patterned = EdgeDescriptor.extract(_patterned_granite_bgr())
    assert float(np.linalg.norm(patterned)) > 0.5
    assert EdgeDescriptor.similarity(z, patterned) == 0.0
    assert EdgeDescriptor.similarity(patterned, z) == 0.0


def test_granite_vs_solid_stays_low():
    """Edge must still discriminate structured vs unstructured surfaces."""
    gran = EdgeDescriptor.extract(_patterned_granite_bgr())
    solid = EdgeDescriptor.extract(_solid_bgr())
    # Solid may be empty (sim path) or near-empty mag-mask; either way low vs granite.
    sim = EdgeDescriptor.similarity(gran, solid)
    assert sim < 0.35, f"granite-solid edge sim too high: {sim:.4f}"


def test_letterboxed_cream_panels_edge_similarity(tmp_path):
    """Closest synthetic stand-in for the real xx vs PGYS2319 letterbox pair."""
    sheet = Image.new("RGB", (1063, 1000), (248, 248, 250))
    d = ImageDraw.Draw(sheet)
    panel_w = int(1063 * 0.42)
    for y in range(1000):
        for x in range(panel_w):
            n = ((x * 17 + y * 31) % 97) / 97.0
            vein = 1.0 if abs((x * 0.35 + y * 0.55) % 80 - 40) < 2.2 else 0.0
            g = int(236 + 14 * n - 28 * vein)
            sheet.putpixel((x, y), (g, g - 1, g - 2))
    d.rectangle([panel_w, 0, 1062, 999], fill=(252, 252, 252))
    d.rectangle([panel_w + 40, 40, 1023, 140], fill=(20, 20, 20))
    gx0, gy0 = panel_w + 30, 180
    for r in range(5):
        for c in range(4):
            x0 = gx0 + c * 152
            y0 = gy0 + r * 122
            shade = 200 + ((r * 3 + c) % 5) * 8
            d.rectangle([x0, y0, x0 + 140, y0 + 110], fill=(shade, shade - 4, shade - 8))
    d.rectangle([0, 0, panel_w - 1, 110], fill=(245, 245, 248))
    d.text((18, 22), "PGYS2319", fill=(55, 55, 60))
    sheet_path = tmp_path / "PGYS2319.jpg"
    sheet.save(sheet_path)

    from src.ai.debug.index_crop_debug import show_index_crops

    report = show_index_crops(sheet_path, output_dir=tmp_path / "crops")
    primary_path = next(
        p for p in report.saved_paths if p.endswith("_primary_preprocess_letterbox.png")
    )
    panel_bgr = cv2.imread(primary_path)

    query = Image.new("RGB", (640, 480), (238, 236, 232))
    pix = query.load()
    for y in range(480):
        for x in range(640):
            n = ((x * 13 + y * 29) % 89) / 89.0
            vein = 1.0 if abs((x * 0.4 + y * 0.5) % 70 - 35) < 2.0 else 0.0
            g = int(230 + 16 * n - 22 * vein)
            pix[x, y] = (g, g - 1, g - 3)
    q_path = tmp_path / "xx.jpg"
    query.save(q_path)
    xx_bgr = ImagePreprocessor.preprocess_for_query(q_path).bgr

    ex = EdgeDescriptor.extract(xx_bgr)
    ep = EdgeDescriptor.extract(panel_bgr)
    assert float(np.linalg.norm(ep)) > 0.5, "panel edge hist still all-zero"
    assert float(np.linalg.norm(ex)) > 0.5, "query edge hist still all-zero"
    sim = EdgeDescriptor.similarity(ex, ep)
    assert sim > 0.25, f"letterbox marble edge sim={sim:.4f} (was exact 0.0)"
    # Must not be the degenerate exact-zero signature.
    assert sim != 0.0
