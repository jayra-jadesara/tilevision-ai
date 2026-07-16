"""Tests for filename metadata parsing."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.core.use_cases.index_images import parse_filename_metadata


def test_underscore_format_parses_all_fields():
    brand, category, color, size, code = parse_filename_metadata(
        "Kajaria_Floor_Grey_60x60_ABC123"
    )
    assert brand == "Kajaria"
    assert category == "Floor"
    assert color == "Grey"
    assert size == "60x60"
    assert code == "ABC123"


def test_hyphenated_descriptive_name_extracts_color_and_category():
    brand, category, color, size, code = parse_filename_metadata(
        "5mm-white-dotted-ceramic-floor-tile-500x500"
    )
    assert brand == "Unknown"
    assert category == "Floor"
    assert color == "White"
    assert size == "500x500"
    assert code == "5mm-white-dotted-ceramic-floor-tile-500x500"


def test_cream_color_floor_tiles_parses_metadata():
    brand, category, color, size, code = parse_filename_metadata(
        "cream-color-floor-tiles"
    )
    assert brand == "Unknown"
    assert category == "Floor"
    assert color == "Cream"
    assert code == "cream-color-floor-tiles"


def test_single_token_filename_returns_unknown_brand():
    brand, category, color, size, code = parse_filename_metadata("images (9)")
    assert brand == "Unknown"
    assert category == "Unknown"
    assert color == "Unknown"
