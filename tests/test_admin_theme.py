"""Tests for vendor admin tool theme."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "admin_tool"))

from admin_theme import get_admin_qss


def test_dark_theme_styles_groupbox_background():
    qss = get_admin_qss("dark")
    assert "QGroupBox" in qss
    assert "#1E293B" in qss


def test_light_theme_styles_groupbox_background():
    qss = get_admin_qss("light")
    assert "QGroupBox" in qss
    assert "#FFFFFF" in qss
