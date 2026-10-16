"""
Multi-photo Auto/Precise/Manual crop must keep originals and crop every photo.

Regression for the Windows log where Auto Crop on 2 photos only cropped
photo 1, replaced the multi-photo query with a single search, and Precise
then ran on the auto-crop output instead of the user's photo.
"""

from __future__ import annotations

import sys
import types
from pathlib import Path
from unittest.mock import MagicMock

import pytest
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

pytest.importorskip("PySide6")

from PySide6.QtWidgets import QApplication

from src.presentation.views.search_view import SearchView


@pytest.fixture(scope="module")
def qapp():
    app = QApplication.instance() or QApplication(sys.argv)
    yield app


def _png(path: Path, color) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (64, 64), color=color).save(path)
    return str(path)


def _make_search_view(qapp, tmp_path):
    vm = MagicMock()
    vm.is_searching = False
    vm.search_by_image = MagicMock()
    vm.search_by_images = MagicMock()
    vm.clear_results = MagicMock()
    # Signal-like stubs used during SearchView init / connects.
    for name in (
        "state_changed",
        "results_ready",
        "status_message",
        "search_error",
        "query_image_selected",
        "filters_available",
        "search_history_updated",
        "search_stats_ready",
    ):
        sig = MagicMock()
        sig.connect = MagicMock()
        setattr(vm, name, sig)

    view = SearchView(viewmodel=vm, theme="light")
    view.show()
    qapp.processEvents()
    return view, vm


def test_auto_crop_two_photos_crops_both_and_searches_multi(qapp, tmp_path, monkeypatch):
    a = _png(tmp_path / "PGYS2319.jpg", (10, 20, 30))
    b = _png(tmp_path / "xx.jpg.jpeg", (40, 50, 60))
    crop_a = _png(tmp_path / "crop_a.jpg", (11, 22, 33))
    crop_b = _png(tmp_path / "crop_b.jpg", (44, 55, 66))

    view, vm = _make_search_view(qapp, tmp_path)
    view._on_images_chosen([a, b])
    assert view._original_query_paths == [a, b]
    assert view._drop_zone._multi_preview_labels[0].isVisible()
    assert view._drop_zone._multi_preview_labels[1].isVisible()

    cropped_sources = []

    class FakeWorker:
        def __init__(self, image_path, mode):
            self.image_path = image_path
            self.mode = mode
            self.crop_finished = MagicMock()
            self.crop_failed = MagicMock()
            self.finished = MagicMock()
            cropped_sources.append(image_path)

        def isRunning(self):
            return False

        def start(self):
            # Immediately deliver success for this photo.
            crop = types.SimpleNamespace(method="auto", confidence=0.9)
            path = crop_a if self.image_path == a else crop_b
            # Find which slot callback was connected.
            for call in self.crop_finished.connect.call_args_list:
                cb = call[0][0]
                cb(path, crop)
            for call in self.finished.connect.call_args_list:
                cb = call[0][0]
                cb(self)

        def deleteLater(self):
            pass

        def requestInterruption(self):
            pass

        def wait(self, *_a):
            return True

    monkeypatch.setattr(
        "src.presentation.views.search_view.TileCropWorker", FakeWorker
    )
    # Avoid QTimer delay — run next immediately.
    monkeypatch.setattr(
        "src.presentation.views.search_view.QTimer.singleShot",
        lambda _ms, fn: fn(),
    )

    view._on_auto_crop_clicked()

    assert cropped_sources == [a, b]
    vm.search_by_images.assert_called()
    called_paths = vm.search_by_images.call_args[0][0]
    assert called_paths == [crop_a, crop_b]
    # Multi-photo UI preserved.
    assert view._drop_zone._multi_preview_labels[0].isVisible()
    assert view._drop_zone._multi_preview_labels[1].isVisible()
    assert len(view._current_query_paths) == 2
    # Originals untouched for a later Precise crop.
    assert view._original_query_paths == [a, b]


def test_precise_after_auto_uses_original_path(qapp, tmp_path, monkeypatch):
    photo = _png(tmp_path / "tile.jpg", (1, 2, 3))
    auto_crop = _png(tmp_path / "autocrop_tile.jpg", (4, 5, 6))
    precise_crop = _png(tmp_path / "precise_tile.jpg", (7, 8, 9))

    view, vm = _make_search_view(qapp, tmp_path)
    view._on_image_chosen(photo)
    assert view._original_query_paths == [photo]

    sources = []

    class FakeWorker:
        def __init__(self, image_path, mode):
            self.image_path = image_path
            self.mode = mode
            self.crop_finished = MagicMock()
            self.crop_failed = MagicMock()
            self.finished = MagicMock()
            sources.append((mode, image_path))

        def isRunning(self):
            return False

        def start(self):
            crop = types.SimpleNamespace(method=self.mode, confidence=0.8, detail="")
            out = auto_crop if self.mode == "auto" else precise_crop
            for call in self.crop_finished.connect.call_args_list:
                call[0][0](out, crop)
            for call in self.finished.connect.call_args_list:
                call[0][0](self)

        def deleteLater(self):
            pass

        def requestInterruption(self):
            pass

        def wait(self, *_a):
            return True

    monkeypatch.setattr(
        "src.presentation.views.search_view.TileCropWorker", FakeWorker
    )

    view._on_auto_crop_clicked()
    assert sources[-1] == ("auto", photo)
    # Preview shows crop, but original is preserved.
    assert view._current_query_image_path == auto_crop
    assert view._original_query_paths == [photo]

    view._on_precise_crop_clicked()
    assert sources[-1] == ("precise", photo), (
        f"Precise must use original photo, got {sources[-1]}"
    )


def test_select_thumbnail_2_manual_crop_only_changes_photo_2(
    qapp, tmp_path, monkeypatch
):
    a = _png(tmp_path / "a.jpg", (10, 10, 10))
    b = _png(tmp_path / "b.jpg", (20, 20, 20))
    crop_b = _png(tmp_path / "crop_b.jpg", (30, 30, 30))

    view, vm = _make_search_view(qapp, tmp_path)
    view._on_images_chosen([a, b])
    view._on_query_thumbnail_selected(1)
    assert view._selected_query_index == 1

    class FakeDialog:
        DialogCode = types.SimpleNamespace(Accepted=1)

        def __init__(self, image_path, parent=None, theme="dark", window_title=None):
            self._path = image_path
            self.window_title = window_title
            assert image_path == b
            assert window_title and "2 of 2" in window_title

        def exec(self):
            return 1

        @property
        def cropped_image_path(self):
            return crop_b

    monkeypatch.setattr(
        "src.presentation.views.search_view.CropDialog", FakeDialog
    )
    vm.search_by_images.reset_mock()
    view._on_crop_clicked()
    paths = vm.search_by_images.call_args[0][0]
    assert paths == [a, crop_b]
    assert view._original_query_paths == [a, b]
    assert view._drop_zone._multi_preview_labels[0].isVisible()
    assert view._drop_zone._multi_preview_labels[1].isVisible()
