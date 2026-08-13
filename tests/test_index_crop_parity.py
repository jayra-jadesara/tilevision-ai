"""Debug tool must share the production index-primary letterbox path."""

from __future__ import annotations

import numpy as np
from PIL import Image, ImageDraw

from src.ai.debug.index_crop_debug import show_index_crops
from src.ai.descriptors.edge_descriptor import EdgeDescriptor
from src.ai.preprocess.image_preprocessor import ImagePreprocessor
from src.ai.preprocess.index_primary import prepare_index_primary
from src.ai.feature_extractor import FeatureExtractor


def _make_sheet(path) -> None:
    w, h = 1063, 1000
    img = Image.new("RGB", (w, h), (248, 248, 250))
    d = ImageDraw.Draw(img)
    panel_w = int(w * 0.42)
    for y in range(h):
        for x in range(panel_w):
            n = ((x * 17 + y * 31) % 97) / 97.0
            vein = 1.0 if abs((x * 0.35 + y * 0.55) % 80 - 40) < 2.2 else 0.0
            g = int(236 + 14 * n - 28 * vein)
            img.putpixel((x, y), (g, g - 1, g - 2))
    d.rectangle([panel_w, 0, w - 1, h - 1], fill=(252, 252, 252))
    d.rectangle([panel_w + 40, 40, w - 40, 140], fill=(20, 20, 20))
    gx0, gy0 = panel_w + 30, 180
    for r in range(5):
        for c in range(4):
            x0 = gx0 + c * 152
            y0 = gy0 + r * 122
            shade = 200 + ((r * 3 + c) % 5) * 8
            d.rectangle(
                [x0, y0, x0 + 140, y0 + 110],
                fill=(shade, shade - 4, shade - 8),
            )
    d.rectangle([0, 0, panel_w - 1, 110], fill=(245, 245, 248))
    d.text((18, 22), "PGYS2319", fill=(55, 55, 60))
    img.save(path)


def _make_query(path) -> None:
    img = Image.new("RGB", (640, 480), (238, 236, 232))
    pix = img.load()
    for y in range(480):
        for x in range(640):
            n = ((x * 13 + y * 29) % 89) / 89.0
            vein = 1.0 if abs((x * 0.4 + y * 0.5) % 70 - 35) < 2.0 else 0.0
            g = int(230 + 16 * n - 22 * vein)
            pix[x, y] = (g, g - 1, g - 3)
    img.save(path)


def test_show_index_crop_primary_matches_prepare_index_primary(tmp_path):
    sheet = tmp_path / "PGYS2319.jpg"
    _make_sheet(sheet)
    prep = prepare_index_primary(sheet)
    assert prep.primary_source == "panel"

    report = show_index_crops(sheet, output_dir=tmp_path / "crops")
    assert report.primary_source == "panel"
    primary_path = next(
        p for p in report.saved_paths if p.endswith("_primary_preprocess_letterbox.png")
    )
    from PIL import Image as PilImage

    saved = np.asarray(PilImage.open(primary_path).convert("RGB"))
    expected = np.asarray(prep.primary.pil.convert("RGB"))
    assert saved.shape == expected.shape
    assert np.array_equal(saved, expected), "debug PNG diverged from production prep"


def test_finalize_index_pil_wrapper_matches_shared_helper(tmp_path):
    sheet = tmp_path / "PGYS2319.jpg"
    _make_sheet(sheet)
    prep = prepare_index_primary(sheet)
    assert prep.panel is not None
    wrapped = FeatureExtractor._finalize_index_pil(
        prep.panel,
        original_size=prep.raw.size,
        match_pad_to_content=True,
    )
    assert np.array_equal(
        np.asarray(wrapped.pil),
        np.asarray(prep.primary.pil),
    )


def test_query_parity_edge_below_index_only_pair(tmp_path):
    """
    Production hybrid uses query preprocess (gray pad) vs index primary
    (content pad). Comparing two content-matched letterboxes overstates edge.
    """
    sheet = tmp_path / "PGYS2319.jpg"
    query = tmp_path / "xx.jpg"
    _make_sheet(sheet)
    _make_query(query)

    report = show_index_crops(
        sheet,
        output_dir=tmp_path / "crops",
        query_path=query,
    )
    assert report.parity is not None

    # Inflated comparison: index primary vs content-padded query (NOT production).
    q_img = Image.open(query).convert("RGB")
    mean = np.asarray(q_img, dtype=np.float32).mean(axis=(0, 1))
    pad = tuple(int(np.clip(c, 0, 255)) for c in mean)
    q_content = ImagePreprocessor.resize_letterbox(
        ImagePreprocessor.normalize_lighting(q_img),
        pad_color=pad,
    )
    prep = prepare_index_primary(sheet)
    import cv2

    q_bgr = cv2.cvtColor(np.asarray(q_content.convert("RGB")), cv2.COLOR_RGB2BGR)
    inflated = EdgeDescriptor.similarity(
        EdgeDescriptor.extract(q_bgr),
        EdgeDescriptor.extract(prep.primary.bgr),
    )
    # Production-like parity from the tool.
    assert report.parity.edge < inflated + 0.05 or report.parity.edge < 0.85
    # Must not be the degenerate exact-zero signature.
    assert report.parity.edge != 0.0
    assert report.parity.color > 0.5


def test_sam2_not_required_for_index_primary(tmp_path):
    """Indexing must not depend on SAM2 precise-crop being enabled."""
    from src.ai.preprocess import sam2_backend

    sheet = tmp_path / "PGYS2319.jpg"
    _make_sheet(sheet)
    sam2_backend.configure_sam2_from_settings(True)
    prep_on = prepare_index_primary(sheet)
    sam2_backend.configure_sam2_from_settings(False)
    prep_off = prepare_index_primary(sheet)
    assert np.array_equal(
        np.asarray(prep_on.primary.pil),
        np.asarray(prep_off.primary.pil),
    )
