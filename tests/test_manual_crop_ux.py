"""Manual Crop UX: Auto-aligned default + tight-selection hint (no silent override)."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
from PIL import Image
from PySide6.QtWidgets import QApplication

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.ai.descriptors.edge_descriptor import EdgeDescriptor
from src.ai.descriptors.pattern_descriptor import PatternDescriptor
from src.ai.feature_extractor import FeatureExtractor
from src.ai.preprocess.fast_tile_crop import resolve_auto_tile_crop, save_auto_tile_crop
from src.ai.preprocess.index_primary import prepare_index_primary
from src.presentation.views.crop_dialog import CropDialog
from src.presentation.views.manual_crop_ux import (
    format_selection_status,
    is_tight_selection,
    selection_keep_ratio,
    suggest_manual_default_box,
)
from tests.fake_ai import FakeEmbedder
from tests.test_crop_search_consistency import _make_catalog_sheet
from tests.test_fast_tile_crop import _make_room_like_photo


def _ensure_qapp() -> QApplication:
    app = QApplication.instance()
    if app is None:
        app = QApplication([])
    return app


def test_suggest_manual_default_matches_auto_on_clean_tile(tmp_path):
    _sheet, crop_path = _make_catalog_sheet(tmp_path)
    src = Image.open(crop_path).convert("RGB")
    suggestion = suggest_manual_default_box(src)
    auto = resolve_auto_tile_crop(src)
    assert suggestion.box == auto.box
    assert suggestion.method in {"already_clean", "already_clean_overcrop_guard"}
    assert suggestion.keep_ratio >= 0.99
    assert not is_tight_selection(src.size, suggestion.box)


def test_suggest_manual_default_uses_auto_isolation_on_room(tmp_path):
    path = tmp_path / "room.jpg"
    _make_room_like_photo(path)
    src = Image.open(path).convert("RGB")
    suggestion = suggest_manual_default_box(src)
    auto = resolve_auto_tile_crop(src)
    assert suggestion.box == auto.box
    assert suggestion.method not in {"already_clean", "already_clean_overcrop_guard"}
    assert suggestion.keep_ratio < 1.0


def test_tight_selection_status_warns_below_half():
    size = (1000, 1000)
    tight = (300, 300, 700, 700)  # 16% of area
    assert selection_keep_ratio(size, tight) == 0.16
    assert is_tight_selection(size, tight)
    label, warn = format_selection_status(size, tight)
    assert warn is True
    assert "small region" in label.lower()

    full = (0, 0, 1000, 1000)
    label2, warn2 = format_selection_status(size, full)
    assert warn2 is False
    assert "100%" in label2


def test_manual_default_dialog_enables_search_without_drag(tmp_path):
    """Opening Manual Crop should pre-fill Auto's box and enable Search."""
    _ensure_qapp()
    _sheet, crop_path = _make_catalog_sheet(tmp_path)
    dialog = CropDialog(str(crop_path), theme="dark")
    try:
        assert dialog._canvas.selection_rect is not None
        assert dialog._use_selection_button.isEnabled()
        assert "Selection:" in dialog._status_label.text()
        # Clean tile → full-frame default → not a tight warning.
        assert dialog._status_label.property("warn") in (False, "false", None)
    finally:
        dialog.close()


def test_manual_accept_writes_last_manual_without_override(tmp_path):
    """Accepting the default must crop exactly that rectangle — no full-frame force."""
    _ensure_qapp()
    _sheet, crop_path = _make_catalog_sheet(tmp_path)
    dialog = CropDialog(str(crop_path), theme="dark")
    try:
        dialog._on_use_selection()
        assert dialog.cropped_image_path is not None
        out = Path(dialog.cropped_image_path)
        assert out.is_file()
        with Image.open(crop_path) as src, Image.open(out) as cropped:
            # Clean tile default is full frame (matches Auto).
            assert cropped.size == src.size
        last = Path(out).parent / "last_manual.jpg"
        assert last.is_file()
    finally:
        dialog.close()


def test_manual_default_edge_pat_track_auto_not_tight_center(tmp_path):
    """
    Real failure mode: a tight center manual crop collapses edge/pattern vs
    the index panel (client Manual Crop: edge 0.276 / pat 0.266). The new
    Auto-aligned default on a clean close-up is full-frame and must track
    Auto Crop descriptors.
    """
    sheet_path, crop_path = _make_catalog_sheet(tmp_path)
    index_pre = prepare_index_primary(sheet_path).primary
    fx = FeatureExtractor(embedder=FakeEmbedder())
    index = fx.extract_from_preprocessed(index_pre, for_query=False)

    src = Image.open(crop_path).convert("RGB")
    w, h = src.size
    # Simulate the old empty-canvas bias: user draws a ~35% center box.
    tight_keep = 0.35
    side = int(round((tight_keep**0.5) * min(w, h)))
    left = (w - side) // 2
    top = (h - side) // 2
    tight_img = src.crop((left, top, left + side, top + side))

    suggestion = suggest_manual_default_box(src)
    default_img = src.crop(suggestion.box)
    _out, auto = save_auto_tile_crop(crop_path)
    assert auto.image.size == src.size

    crops = tmp_path / "tilevision_crops"
    crops.mkdir(exist_ok=True)
    tight_p = crops / "manual_tight.jpg"
    default_p = crops / "manual_default.jpg"
    auto_p = crops / "autocrop.jpg"
    tight_img.save(tight_p, quality=95)
    default_img.save(default_p, quality=95)
    auto.image.save(auto_p, quality=95)

    tight_f, _ = fx.extract_for_search(str(tight_p))
    default_f, _ = fx.extract_for_search(str(default_p))
    auto_f, _ = fx.extract_for_search(str(auto_p))

    def edge(feat):
        return EdgeDescriptor.similarity(feat.edge_histogram, index.edge_histogram)

    def pat(feat):
        return PatternDescriptor.similarity(feat.pattern_features, index.pattern_features)

    # Default tracks Auto (same pixels on clean tile) — the UX fix invariant.
    assert abs(edge(default_f) - edge(auto_f)) < 0.05
    assert abs(pat(default_f) - pat(auto_f)) < 0.05
    # Tight center is a different region (client damage class when under-selected).
    assert tight_img.size != default_img.size


def test_lighting_gradient_manual_default_is_full_frame(tmp_path):
    """Gradient close-up: Auto guard → full frame; Manual default must match."""
    _sheet, crop_path = _make_catalog_sheet(tmp_path)
    base = Image.open(crop_path).convert("RGB")
    arr = np.asarray(base).astype(np.float32)
    height = arr.shape[0]
    for y in range(height):
        arr[y] *= 0.50 + 0.50 * (y / max(1, height - 1))
    gradient = Image.fromarray(arr.astype(np.uint8))
    suggestion = suggest_manual_default_box(gradient)
    assert suggestion.method == "already_clean_overcrop_guard"
    assert suggestion.box == (0, 0, gradient.size[0], gradient.size[1])
