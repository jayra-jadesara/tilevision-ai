"""Crop-tool search routing: skip over-crop on clean tiles, multi-view without straighten."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.ai.descriptors.edge_descriptor import EdgeDescriptor
from src.ai.descriptors.texture_descriptor import TextureDescriptor
from src.ai.feature_extractor import FeatureExtractor
from src.ai.preprocess.fast_tile_crop import isolate_tile_region, save_auto_tile_crop
from src.ai.preprocess.index_primary import prepare_index_primary
from src.ai.search_quality.query_analyzer import analyze_query
from src.ai.search_quality.query_origin import QueryOrigin, resolve_query_origin
from src.core.use_cases.search_tiles import SearchTilesUseCase
from tests.fake_ai import FakeEmbedder
from tests.test_crop_search_consistency import _make_catalog_sheet
from tests.test_fast_tile_crop import _make_room_like_photo


def test_resolve_query_origin_default_is_auto():
    assert resolve_query_origin("/catalog/xx.jpg.jpeg") is QueryOrigin.AUTO
    assert resolve_query_origin(None) is QueryOrigin.AUTO


def test_resolve_query_origin_from_path_and_hint(tmp_path):
    crop = tmp_path / "tilevision_crops" / "autocrop_xx.jpg_1.jpg"
    crop.parent.mkdir()
    crop.write_bytes(b"x")
    assert resolve_query_origin(crop) is QueryOrigin.CROP_TOOL
    copied = tmp_path / "autocrop_xx.jpg_1.jpg"
    copied.write_bytes(b"x")
    assert resolve_query_origin(copied) is QueryOrigin.CROP_TOOL
    eval_crop = tmp_path / "crop_600x600.jpg"
    eval_crop.write_bytes(b"x")
    assert resolve_query_origin(eval_crop) is QueryOrigin.AUTO
    assert (
        resolve_query_origin("/catalog/xx.jpg.jpeg", explicit="crop_tool")
        is QueryOrigin.CROP_TOOL
    )
    assert resolve_query_origin(crop, explicit="auto") is QueryOrigin.AUTO


def test_auto_crop_preserves_full_frame_clean_tile(tmp_path):
    sheet_path, crop_path = _make_catalog_sheet(tmp_path)
    analysis = analyze_query(Image.open(crop_path).convert("RGB"))
    assert analysis.kind.value == "clean_tile"

    out_path, result = save_auto_tile_crop(crop_path)
    src = Image.open(crop_path)
    assert result.method == "already_clean"
    assert result.image.size == src.size
    keep = out_path.parent / "last_autocrop.jpg"
    assert keep.is_file()
    with Image.open(out_path) as saved:
        assert saved.size == src.size


def test_auto_crop_still_isolates_room_photo(tmp_path):
    path = tmp_path / "room.jpg"
    _make_room_like_photo(path)
    _out, result = save_auto_tile_crop(path)
    src = Image.open(path)
    assert result.method != "already_clean"
    assert result.image.size[0] * result.image.size[1] < src.size[0] * src.size[1]


def test_drop_search_clean_tile_stays_single_view(tmp_path):
    _sheet, crop_path = _make_catalog_sheet(tmp_path)
    fx = FeatureExtractor(embedder=FakeEmbedder())
    _feat, embeddings = fx.extract_for_search(str(crop_path))
    assert len(embeddings) == 1


def test_crop_tool_path_uses_complementary_views(tmp_path):
    _sheet, crop_path = _make_catalog_sheet(tmp_path)
    crops = tmp_path / "tilevision_crops"
    crops.mkdir()
    query = crops / "autocrop_xx.jpg_99.jpg"
    query.write_bytes(crop_path.read_bytes())
    fx = FeatureExtractor(embedder=FakeEmbedder())
    _feat, embeddings = fx.extract_for_search(str(query))
    assert len(embeddings) == 2


def test_crop_tool_hint_works_outside_temp_folder(tmp_path):
    _sheet, crop_path = _make_catalog_sheet(tmp_path)
    copied = tmp_path / "diagnostic_copy.jpg"
    copied.write_bytes(crop_path.read_bytes())
    fx = FeatureExtractor(embedder=FakeEmbedder())
    _plain, plain_emb = fx.extract_for_search(str(copied))
    _hinted, hinted_emb = fx.extract_for_search(
        str(copied), query_origin="crop_tool"
    )
    assert len(plain_emb) == 1
    assert len(hinted_emb) == 2


def test_auto_crop_clean_tile_descriptors_match_fresh_drop(tmp_path):
    """Over-crop was the 32% floor_band keep; full-frame should track drop search."""
    sheet_path, crop_path = _make_catalog_sheet(tmp_path)
    index_pre = prepare_index_primary(sheet_path).primary
    fx = FeatureExtractor(embedder=FakeEmbedder())
    index = fx.extract_from_preprocessed(index_pre, for_query=False)

    drop, _ = fx.extract_for_search(str(crop_path))
    _out, auto = save_auto_tile_crop(crop_path)
    assert auto.method == "already_clean"
    crops = tmp_path / "tilevision_crops"
    crops.mkdir(exist_ok=True)
    auto_query = crops / "autocrop_xx.jpg_1.jpg"
    auto.image.save(auto_query, quality=95)
    crop_feat, crop_embs = fx.extract_for_search(str(auto_query))
    assert len(crop_embs) >= 1

    drop_tex = TextureDescriptor.similarity(drop.texture_histogram, index.texture_histogram)
    crop_tex = TextureDescriptor.similarity(
        crop_feat.texture_histogram, index.texture_histogram
    )
    drop_edge = EdgeDescriptor.similarity(drop.edge_histogram, index.edge_histogram)
    crop_edge = EdgeDescriptor.similarity(crop_feat.edge_histogram, index.edge_histogram)
    assert abs(crop_tex - drop_tex) < 0.08
    assert abs(crop_edge - drop_edge) < 0.08

    # Document the old bug class: isolate_tile_region still floor_bands a close-up.
    isolated = isolate_tile_region(Image.open(crop_path).convert("RGB"))
    src = Image.open(crop_path)
    isolated_ratio = (isolated.image.size[0] * isolated.image.size[1]) / (
        src.size[0] * src.size[1]
    )
    assert isolated.method == "floor_band"
    assert isolated_ratio < 0.5


def test_floor_band_overcrop_hurts_texture_vs_skip(tmp_path):
    sheet_path, crop_path = _make_catalog_sheet(tmp_path)
    index = FeatureExtractor(embedder=FakeEmbedder()).extract_from_preprocessed(
        prepare_index_primary(sheet_path).primary, for_query=False
    )
    src = Image.open(crop_path).convert("RGB")
    isolated = isolate_tile_region(src)
    fx = FeatureExtractor(embedder=FakeEmbedder())
    crops_dir = tmp_path / "tilevision_crops"
    crops_dir.mkdir()
    tight = crops_dir / "legacy_floor_band.jpg"
    isolated.image.save(tight, quality=95)
    full = crops_dir / "full.jpg"
    src.save(full, quality=95)
    tight_f, _ = fx.extract_for_search(str(tight))
    full_f, _ = fx.extract_for_search(str(full))
    tight_tex = TextureDescriptor.similarity(
        tight_f.texture_histogram, index.texture_histogram
    )
    full_tex = TextureDescriptor.similarity(
        full_f.texture_histogram, index.texture_histogram
    )
    assert full_tex >= tight_tex - 0.02


def test_resolve_crop_source_stem_auto_and_precise():
    auto = Path(
        r"C:\Users\HP\AppData\Local\Temp\tilevision_crops"
        r"\autocrop_xx.jpg_1954080693312.jpg"
    )
    precise = Path(
        r"C:\Users\HP\AppData\Local\Temp\tilevision_crops"
        r"\precise_xx.jpg_99.jpg"
    )
    assert SearchTilesUseCase._resolve_crop_source_stem(auto) == "xx.jpg"
    assert SearchTilesUseCase._resolve_crop_source_stem(precise) == "xx.jpg"


def test_crop_600_eval_files_are_not_crop_tool_origin(tmp_path):
    """eval crop_* queries live in the eval folder — drop routing must stay 1-view."""
    _sheet, crop_path = _make_catalog_sheet(tmp_path)
    eval_like = tmp_path / "real_queries" / "sheet001_crop_600.jpg"
    eval_like.parent.mkdir()
    eval_like.write_bytes(crop_path.read_bytes())
    assert resolve_query_origin(eval_like) is QueryOrigin.AUTO
    fx = FeatureExtractor(embedder=FakeEmbedder())
    _f, emb = fx.extract_for_search(str(eval_like))
    assert len(emb) == 1
