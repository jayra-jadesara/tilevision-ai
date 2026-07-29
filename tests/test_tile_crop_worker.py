"""Background Auto/Precise crop worker keeps Search UI responsive."""

from __future__ import annotations

import sys
import time
import types
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

PySide6 = pytest.importorskip("PySide6")

from PySide6.QtCore import QCoreApplication
from PySide6.QtWidgets import QApplication

from src.presentation.workers.tile_crop_worker import TileCropWorker


@pytest.fixture(scope="module")
def qapp():
    app = QApplication.instance() or QApplication(sys.argv)
    yield app


def _pump_until(condition, timeout=3.0):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        QCoreApplication.processEvents()
        if condition():
            return True
        time.sleep(0.01)
    return False


def test_auto_crop_worker_emits_finished(qapp, tmp_path, monkeypatch):
    image = tmp_path / "room.jpg"
    image.write_bytes(b"fake")

    crop_path = tmp_path / "crop.jpg"
    crop_path.write_bytes(b"crop")
    result = types.SimpleNamespace(method="opencv", confidence=0.8, detail="ok")

    monkeypatch.setattr(
        "src.ai.preprocess.fast_tile_crop.save_auto_tile_crop",
        lambda path: (crop_path, result),
    )

    finished = []
    failed = []
    worker = TileCropWorker(str(image), "auto")
    worker.crop_finished.connect(lambda path, crop: finished.append((path, crop)))
    worker.crop_failed.connect(failed.append)
    worker.start()

    assert _pump_until(lambda: bool(finished) or bool(failed))
    assert failed == []
    assert finished[0][0] == str(crop_path)
    assert finished[0][1].method == "opencv"
    assert _pump_until(lambda: worker.isFinished())


def test_precise_crop_worker_failure_emits_message(qapp, tmp_path, monkeypatch):
    image = tmp_path / "room.jpg"
    image.write_bytes(b"fake")

    def _fail(_path):
        raise RuntimeError("sam2 unavailable")

    monkeypatch.setattr(
        "src.ai.preprocess.precise_tile_crop.save_precise_tile_crop",
        _fail,
    )

    finished = []
    failed = []
    worker = TileCropWorker(str(image), "precise")
    worker.crop_finished.connect(lambda path, crop: finished.append((path, crop)))
    worker.crop_failed.connect(failed.append)
    worker.start()

    assert _pump_until(lambda: bool(finished) or bool(failed))
    assert finished == []
    assert failed and "sam2 unavailable" in failed[0]
    assert _pump_until(lambda: worker.isFinished())
