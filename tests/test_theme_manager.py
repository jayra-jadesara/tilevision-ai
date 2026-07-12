"""Tests for the theme system (palette parity + live re-skinning)."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.theme.theme_manager import get_palette, get_app_stylesheet


def test_dark_and_light_palettes_have_identical_keys():
    dark = get_palette("dark")
    light = get_palette("light")
    assert set(dark.keys()) == set(light.keys())


def test_dark_and_light_palettes_have_different_values():
    dark = get_palette("dark")
    light = get_palette("light")
    # The two themes should genuinely differ in their core surface colors —
    # this is what "theme not worked properly" was ultimately about.
    assert dark["bg_app"] != light["bg_app"]
    assert dark["bg_panel"] != light["bg_panel"]
    assert dark["text_primary"] != light["text_primary"]


def test_unknown_theme_falls_back_to_dark():
    assert get_palette("nonexistent") == get_palette("dark")
    assert get_app_stylesheet("nonexistent") == get_app_stylesheet("dark")


def test_get_app_stylesheet_includes_combobox_popup_fix():
    # Regression guard for the original "dropdown text is unreadable" bug.
    for theme in ("dark", "light"):
        qss = get_app_stylesheet(theme)
        assert "QComboBox QAbstractItemView" in qss


def test_all_palette_values_are_valid_hex_colors():
    import re
    hex_pattern = re.compile(r"^#[0-9A-Fa-f]{6}$")
    for theme in ("dark", "light"):
        palette = get_palette(theme)
        for key, value in palette.items():
            assert hex_pattern.match(value), f"{theme}.{key} = {value!r} is not a valid hex color"
