"""Debug tool must share the production index-primary letterbox path."""

from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

from src.ai.debug.index_crop_debug import show_index_crops
from src.ai.descriptors.color_descriptor import ColorDescriptor
from src.ai.descriptors.edge_descriptor import EdgeDescriptor
from src.ai.descriptors.pattern_descriptor import PatternDescriptor
from src.ai.descriptors.texture_descriptor import TextureDescriptor
from src.ai.feature_extractor import FeatureExtractor
from src.ai.models import TileFeatures
from src.ai.preprocess.image_preprocessor import ImagePreprocessor
from src.ai.preprocess.index_primary import prepare_index_primary
from src.core.models import TileImage


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


def _features_from_prep(pre) -> TileFeatures:
    return TileFeatures(
        embedding=np.zeros(8, dtype=np.float32),
        color_histogram=ColorDescriptor.extract(pre.bgr),
        texture_histogram=TextureDescriptor.extract(pre.bgr),
        edge_histogram=EdgeDescriptor.extract(pre.bgr),
        pattern_features=PatternDescriptor.extract(pre.bgr),
        dominant_color=ColorDescriptor.dominant_color_rgb(pre.bgr),
        width=pre.width,
        height=pre.height,
    )


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


def test_auto_mode_reports_catalog_and_fresh(tmp_path):
    sheet = tmp_path / "PGYS2319.jpg"
    query = tmp_path / "xx.jpg"
    _make_sheet(sheet)
    _make_query(query)

    report = show_index_crops(
        sheet,
        output_dir=tmp_path / "crops",
        query_path=query,
        query_mode="auto",
    )
    assert report.parity is not None
    assert report.parity.mode.startswith("catalog")
    assert report.parity_alt is not None
    assert report.parity_alt.mode == "fresh"
    assert report.parity.edge != 0.0
    assert report.parity_alt.edge != 0.0


def test_catalog_mode_uses_prepare_index_primary_on_query(tmp_path):
    """UI catalog-cache simulation must not use preprocess_for_query."""
    sheet = tmp_path / "PGYS2319.jpg"
    query = tmp_path / "xx.jpg"
    _make_sheet(sheet)
    _make_query(query)

    report = show_index_crops(
        sheet,
        output_dir=tmp_path / "crops",
        query_path=query,
        query_mode="catalog",
    )
    assert report.parity is not None
    assert report.parity.mode == "catalog_sim"
    assert report.parity_alt is None

    q_prep = prepare_index_primary(query)
    c_prep = prepare_index_primary(sheet)
    expected = EdgeDescriptor.similarity(
        EdgeDescriptor.extract(q_prep.primary.bgr),
        EdgeDescriptor.extract(c_prep.primary.bgr),
    )
    assert abs(report.parity.edge - expected) < 1e-5


def test_catalog_stored_mode_uses_repo_features(tmp_path):
    sheet = tmp_path / "PGYS2319.jpg"
    query = tmp_path / "xx.jpg"
    _make_sheet(sheet)
    _make_query(query)

    q_feats = _features_from_prep(prepare_index_primary(query).primary)
    c_feats = _features_from_prep(prepare_index_primary(sheet).primary)

    # Deliberately mutate stored edge hist so catalog_stored ≠ live recompute.
    mutated = q_feats.edge_histogram.copy()
    mutated[:] = 0.0
    mutated[0] = 1.0
    q_stored = TileFeatures(
        embedding=q_feats.embedding,
        color_histogram=q_feats.color_histogram,
        texture_histogram=q_feats.texture_histogram,
        edge_histogram=mutated,
        pattern_features=q_feats.pattern_features,
        dominant_color=q_feats.dominant_color,
        width=q_feats.width,
        height=q_feats.height,
    )

    def _sha(path: Path) -> str:
        return hashlib.sha256(path.read_bytes()).hexdigest()

    class _FakeRepo:
        def get_by_path(self, file_path: str):
            p = Path(file_path).resolve()
            if p == query.resolve():
                return TileImage(
                    id=1,
                    file_path=str(p),
                    file_name=p.name,
                    file_size=query.stat().st_size,
                    dimensions="640x480",
                    is_indexed=True,
                    sha256_hash=_sha(query),
                    features=q_stored,
                )
            if p == sheet.resolve():
                return TileImage(
                    id=2,
                    file_path=str(p),
                    file_name=p.name,
                    file_size=sheet.stat().st_size,
                    dimensions="1063x1000",
                    is_indexed=True,
                    sha256_hash=_sha(sheet),
                    features=c_feats,
                )
            return None

    report = show_index_crops(
        sheet,
        output_dir=tmp_path / "crops",
        query_path=query,
        query_mode="catalog",
        catalog_repo=_FakeRepo(),
    )
    assert report.parity is not None
    assert report.parity.mode == "catalog_stored"
    expected = EdgeDescriptor.similarity(
        q_stored.edge_histogram, c_feats.edge_histogram
    )
    assert abs(report.parity.edge - expected) < 1e-5
    # Live recompute would differ because we mutated the stored edge hist.
    live = EdgeDescriptor.similarity(
        EdgeDescriptor.extract(prepare_index_primary(query).primary.bgr),
        EdgeDescriptor.extract(prepare_index_primary(sheet).primary.bgr),
    )
    assert abs(report.parity.edge - live) > 0.05


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


def test_fresh_can_diverge_from_catalog_when_isolation_differs(tmp_path):
    """
    Non-square marble triggers _looks_like_scene_photo; preprocess_for_query
    may isolate while index-time preprocess does not — the catalog/fresh gap.
    """
    sheet = tmp_path / "PGYS2319.jpg"
    query = tmp_path / "xx.jpg"
    _make_sheet(sheet)
    _make_query(query)  # 640x480 → looks_like_scene_photo True

    report = show_index_crops(
        sheet,
        output_dir=tmp_path / "crops",
        query_path=query,
        query_mode="both",
    )
    assert report.parity is not None and report.parity_alt is not None
    # Document both numbers; they need not always differ on every fixture,
    # but modes must be labeled distinctly.
    assert report.parity.mode.startswith("catalog")
    assert report.parity_alt.mode == "fresh"
    text = "\n".join(
        [
            report.parity.notes,
            report.parity_alt.notes,
        ]
    )
    assert "catalog" in text.lower() or "stored" in text.lower()
    assert "preprocess_for_query" in text or "ad-hoc" in text.lower()
