"""Regression tests for PGYS2319 marketing-sheet panel isolation."""

from __future__ import annotations

import sys
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.ai.search_quality.image_analysis import analyze_image


def _legacy_text_region_score(gray: np.ndarray) -> float:
    """Pre-v11 detector (single Canny band on right 45%)."""
    h, w = gray.shape
    if w < 80 or h < 80:
        return 0.0
    right = gray[:, int(w * 0.55) :]
    edges = cv2.Canny(right, 60, 140)
    density = float(np.mean(edges > 0))
    return float(np.clip(density * 4.0, 0.0, 1.0))


def _legacy_left_panel_beneficial(analysis, aspect: float) -> bool:
    """Pre-v11 left_panel gate (aspect >= 1.12 hard floor)."""
    return (
        aspect >= 1.12
        and analysis.width >= 480
        and analysis.height >= 320
        and (
            analysis.text_region_score >= 0.12
            or analysis.has_preview_grid
            or analysis.white_border_ratio >= 0.25
        )
    )


def test_legacy_detector_would_fail_realistic_pgys2319_proportions():
    """
    Real PGYS2319.jpg (aspect 1.063) failed because:
    1. aspect 1.063 < 1.12 gate (even with has_preview_grid=True)
    2. legacy text_region_score ~0.024 on real image (sparse logo + small text)
    """
    from tests.test_crop_search_consistency import _make_catalog_sheet
    import tempfile

    tmp = Path(tempfile.mkdtemp())
    sheet_path, _ = _make_catalog_sheet(tmp)
    sheet = Image.open(sheet_path)
    gray = cv2.cvtColor(np.asarray(sheet), cv2.COLOR_RGB2GRAY)
    aspect = sheet.size[0] / sheet.size[1]

    legacy_text = _legacy_text_region_score(gray)
    assert aspect < 1.12
    assert legacy_text < 0.12

    analysis = analyze_image(sheet)
    assert _legacy_left_panel_beneficial(analysis, aspect) is False

    # Post-fix pipeline must flip both gates.
    assert analysis.text_region_score >= 0.12
    assert analysis.left_panel_beneficial is True
    assert analysis.center_crop_beneficial is False
