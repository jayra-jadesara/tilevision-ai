"""ORB explain detail + SearchResult match-point plumbing."""

from __future__ import annotations

import sys
from pathlib import Path

import cv2
import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.ai.verification.orb_verifier import OrbVerifier
from src.core.models import SearchResult, TileImage


def _textured_gray(seed: int = 0, size: int = 256) -> np.ndarray:
    rng = np.random.default_rng(seed)
    base = np.full((size, size), 200, dtype=np.uint8)
    for _ in range(80):
        x0, y0 = int(rng.integers(0, size)), int(rng.integers(0, size))
        x1, y1 = int(rng.integers(0, size)), int(rng.integers(0, size))
        c = int(rng.integers(40, 180))
        cv2.line(base, (x0, y0), (x1, y1), c, 1 + (seed % 3), cv2.LINE_AA)
    return base


def test_explain_returns_inlier_points_same_score_as_score():
    gray = _textured_gray(seed=7)
    verifier = OrbVerifier()
    detail = verifier.explain(gray, gray.copy())
    score = verifier.score(gray, gray.copy())
    assert detail.score == pytest.approx(score)
    assert detail.score >= 0.7
    assert len(detail.query_points) >= 4
    assert len(detail.query_points) == len(detail.candidate_points)
    assert len(detail.inlier_pairs) == len(detail.query_points)


def test_search_result_optional_orb_points_default_none():
    tile = TileImage(
        file_path="/tmp/a.jpg",
        file_name="a.jpg",
        file_size=1,
        dimensions="10x10",
    )
    result = SearchResult(tile=tile, similarity_score=90.0, thumbnail_path="/tmp/t.jpg")
    assert result.orb_match_points is None


def test_why_match_dialog_renders(qapp, tmp_path):
    from PIL import Image
    from src.presentation.views.why_match_dialog import WhyThisMatchDialog

    q = tmp_path / "q.png"
    c = tmp_path / "c.png"
    Image.fromarray(_textured_gray(1)).save(q)
    Image.fromarray(_textured_gray(1)).save(c)
    points = [((10.0, 12.0), (11.0, 13.0)), ((40.0, 50.0), (41.0, 52.0))]
    dlg = WhyThisMatchDialog(str(q), str(c), points, theme="light")
    dlg.show()
    qapp.processEvents()
    assert dlg.windowTitle() == "Why this match"
    dlg.close()


@pytest.fixture()
def qapp():
    from PySide6.QtWidgets import QApplication

    return QApplication.instance() or QApplication(sys.argv)
