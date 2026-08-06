"""Tests for weak-result filtering and crop-source linking."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.core.models import TileImage
from src.core.use_cases.search_tiles import SearchTilesUseCase


def test_resolve_crop_source_stem_from_temp_file():
    path = Path(
        r"C:\Users\HP\AppData\Local\Temp\tilevision_crops"
        r"\crop_5mm-white-dotted-ceramic-floor-tile-500x500_1954080693312.jpg"
    )
    assert (
        SearchTilesUseCase._resolve_crop_source_stem(path)
        == "5mm-white-dotted-ceramic-floor-tile-500x500"
    )


def test_resolve_crop_source_stem_from_autocrop_file():
    path = Path(
        r"C:\Users\HP\AppData\Local\Temp\tilevision_crops"
        r"\autocrop_xx.jpg_1886531936448.jpg"
    )
    assert SearchTilesUseCase._resolve_crop_source_stem(path) == "xx.jpg"


def test_resolve_crop_source_stem_from_precise_file():
    path = Path(
        r"C:\Users\HP\AppData\Local\Temp\tilevision_crops"
        r"\precise_xx.jpg_1887012676096.jpg"
    )
    assert SearchTilesUseCase._resolve_crop_source_stem(path) == "xx.jpg"


def test_filter_weak_results_drops_low_scores():
    tiles = [
        TileImage(
            file_path=f"{i}.jpg",
            file_name=f"{i}.jpg",
            file_size=1,
            dimensions="1x1",
            id=i,
        )
        for i in range(5)
    ]
    reranked = [
        (0.75, tiles[0], False),
        (0.55, tiles[1], False),
        (0.40, tiles[2], False),
        (0.25, tiles[3], False),
        (0.20, tiles[4], False),
    ]
    kept = SearchTilesUseCase._filter_weak_results(reranked, top_k=10)
    names = [tile.file_name for _, tile, _ in kept]
    assert "0.jpg" in names
    assert "1.jpg" in names
    assert "3.jpg" not in names
    assert "4.jpg" not in names


def test_filter_weak_results_never_returns_empty_when_candidates_exist():
    """Reliability: low hybrid scores must still surface the best match."""
    tiles = [
        TileImage(
            file_path=f"{i}.jpg",
            file_name=f"{i}.jpg",
            file_size=1,
            dimensions="1x1",
            id=i,
        )
        for i in range(3)
    ]
    reranked = [
        (0.30, tiles[0], False),
        (0.22, tiles[1], False),
        (0.10, tiles[2], False),
    ]
    kept = SearchTilesUseCase._filter_weak_results(reranked, top_k=10)
    assert len(kept) >= 1
    assert kept[0][1].file_name == "0.jpg"

