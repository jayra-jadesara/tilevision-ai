"""Tests for query image hints and search confidence helpers."""

import sys
from pathlib import Path

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.core.models import SearchResult, TileImage
from src.utils.query_image_hints import (
    ConfidenceLevel,
    best_non_self_score,
    classify_confidence,
    confidence_message,
    should_suggest_crop,
)


def _write_image(path: Path, array: np.ndarray) -> None:
    Image.fromarray(array.astype(np.uint8)).save(path)


def _result(score: float, name: str = "tile.jpg") -> SearchResult:
    return SearchResult(
        tile=TileImage(file_path=f"/tmp/{name}", file_name=name, file_size=1, dimensions="1x1"),
        similarity_score=score,
        thumbnail_path="",
    )


def test_square_uniform_image_does_not_suggest_crop(tmp_path):
    square = np.full((400, 400, 3), 200, dtype=np.uint8)
    path = tmp_path / "square.png"
    _write_image(path, square)

    suggest, _ = should_suggest_crop(path)
    assert suggest is False


def test_wide_scene_image_suggests_crop(tmp_path):
    image = np.full((300, 700, 3), 210, dtype=np.uint8)
    image[:, :120] = (80, 60, 40)
    path = tmp_path / "scene.png"
    _write_image(path, image)

    suggest, reason = should_suggest_crop(path)
    assert suggest is True
    assert reason


def test_low_confidence_when_best_alternative_is_weak():
    results = [_result(100.0, "query.png"), _result(25.0, "other.png")]
    assert classify_confidence(results) == ConfidenceLevel.LOW
    assert best_non_self_score(results) == 25.0
    message = confidence_message(results)
    assert message is not None
    assert "Low confidence" in message


def test_high_confidence_when_alternative_is_strong():
    results = [_result(100.0), _result(62.0)]
    assert classify_confidence(results) == ConfidenceLevel.HIGH
    assert confidence_message(results) is None


def test_moderate_confidence_band():
    results = [_result(100.0), _result(45.0)]
    assert classify_confidence(results) == ConfidenceLevel.MODERATE
    message = confidence_message(results)
    assert message is not None
    assert "Moderate match" in message
