"""Opt-in multi-photo search: fuse 2–3 single-view embeddings via existing FAISS merge."""

from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import MagicMock

import numpy as np
import pytest
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.ai.models import TileFeatures
from src.core.use_cases.search_tiles import SearchTilesUseCase
from src.presentation.views.search_view import DropZone


def _write_png(path: Path, color=(80, 120, 40)) -> None:
    Image.new("RGB", (32, 32), color=color).save(path, format="PNG")


def _features(seed: float = 1.0) -> TileFeatures:
    emb = np.full(8, seed, dtype=np.float32)
    emb /= float(np.linalg.norm(emb))
    return TileFeatures(
        embedding=emb,
        color_histogram=np.zeros(8, dtype=np.float32),
        texture_histogram=np.zeros(8, dtype=np.float32),
        edge_histogram=np.zeros(8, dtype=np.float32),
        pattern_features=np.zeros(8, dtype=np.float32),
        dominant_color=(10, 20, 30),
        width=32,
        height=32,
    )


@pytest.fixture()
def qapp():
    from PySide6.QtWidgets import QApplication

    return QApplication.instance() or QApplication(sys.argv)


def test_execute_multi_photos_embeds_each_path_once_and_delegates(tmp_path):
    a = tmp_path / "a.png"
    b = tmp_path / "b.png"
    _write_png(a)
    _write_png(b)

    extractor = MagicMock()
    fa = _features(1.0)
    fb = _features(2.0)
    extractor.extract.side_effect = [fa, fb]

    use_case = SearchTilesUseCase.__new__(SearchTilesUseCase)
    use_case._feature_extractor = extractor
    captured: dict = {}

    def _fake_execute(path, **kwargs):
        captured["path"] = path
        captured["kwargs"] = kwargs
        return ["ok"]

    use_case.execute = _fake_execute  # type: ignore[method-assign]

    out = use_case.execute_multi_photos([str(a), str(b)], top_k=7, filters={"brand": "X"})

    assert out == ["ok"]
    assert extractor.extract.call_count == 2
    for call in extractor.extract.call_args_list:
        assert call.kwargs.get("for_query") is True
    assert captured["path"] == str(a)
    assert captured["kwargs"]["top_k"] == 7
    assert captured["kwargs"]["filters"] == {"brand": "X"}
    assert captured["kwargs"]["query_origin"] == "multi_photo"
    assert captured["kwargs"]["query_features_override"] is fa
    embs = captured["kwargs"]["query_embeddings_override"]
    assert len(embs) == 2
    np.testing.assert_allclose(embs[0], fa.embedding)
    np.testing.assert_allclose(embs[1], fb.embedding)


def test_execute_multi_photos_truncates_to_three(tmp_path):
    paths = []
    for i in range(5):
        p = tmp_path / f"{i}.png"
        _write_png(p)
        paths.append(str(p))

    extractor = MagicMock()
    extractor.extract.side_effect = [_features(float(i + 1)) for i in range(3)]

    use_case = SearchTilesUseCase.__new__(SearchTilesUseCase)
    use_case._feature_extractor = extractor
    captured: dict = {}
    use_case.execute = lambda path, **kwargs: captured.update(kwargs) or []  # type: ignore

    use_case.execute_multi_photos(paths)
    assert extractor.extract.call_count == 3
    assert len(captured["query_embeddings_override"]) == 3


def test_execute_multi_photos_requires_two_images():
    use_case = SearchTilesUseCase.__new__(SearchTilesUseCase)
    with pytest.raises(ValueError, match="at least 2"):
        use_case.execute_multi_photos(["/tmp/only.png"])


def test_drop_zone_single_mode_default_off(qapp):
    zone = DropZone(on_images_selected=lambda _p: None)
    assert zone._multi_photo_enabled is False
    assert "Drag a tile photo" in zone._title_label.text()


def test_drop_zone_multi_mode_updates_prompt(qapp):
    zone = DropZone(on_images_selected=lambda _p: None)
    zone.set_multi_photo_enabled(True)
    assert zone._multi_photo_enabled is True
    assert "2–3" in zone._title_label.text()


def test_search_viewmodel_multi_falls_back_to_single(qapp, tmp_path, monkeypatch):
    from src.presentation.viewmodels.search_viewmodel import SearchViewModel

    img = tmp_path / "one.png"
    _write_png(img)
    vm = SearchViewModel(use_case=MagicMock(), default_top_k=5)
    called: list[str] = []
    monkeypatch.setattr(vm, "search_by_image", lambda path, query_origin=None: called.append(path))
    vm.search_by_images([str(img)])
    assert called == [str(img)]
