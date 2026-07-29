"""
Background Auto Crop / Precise Crop worker for Search view.

Keeps SAM2 / OpenCV crop work off the UI thread so Mac Intel clients never
freeze when clicking Precise Crop & Search.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Literal

from PySide6.QtCore import QThread, Signal

logger = logging.getLogger("tilevision.presentation.workers.tile_crop_worker")

CropMode = Literal["auto", "precise"]


class TileCropWorker(QThread):
    """Run OpenCV Auto Crop or ONNX SAM2 Precise Crop off the UI thread."""

    crop_finished = Signal(str, object)  # crop_path, PreciseCropResult/AutoCropResult
    crop_failed = Signal(str)

    def __init__(self, image_path: str, mode: CropMode) -> None:
        super().__init__()
        self._image_path = str(image_path)
        self._mode = mode

    def run(self) -> None:
        logger.info("Tile crop QThread started mode=%s path=%s", self._mode, self._image_path)
        try:
            if self.isInterruptionRequested():
                self.crop_failed.emit("Crop cancelled.")
                return

            if self._mode == "precise":
                from src.ai.preprocess.precise_tile_crop import save_precise_tile_crop

                crop_path, crop = save_precise_tile_crop(self._image_path)
            else:
                from src.ai.preprocess.fast_tile_crop import save_auto_tile_crop

                crop_path, crop = save_auto_tile_crop(self._image_path)

            if self.isInterruptionRequested():
                self.crop_failed.emit("Crop cancelled.")
                return

            logger.info(
                "Tile crop finished mode=%s method=%s path=%s",
                self._mode,
                getattr(crop, "method", "?"),
                crop_path,
            )
            self.crop_finished.emit(str(Path(crop_path)), crop)
        except Exception as exc:
            logger.error("Tile crop failed mode=%s: %s", self._mode, exc)
            self.crop_failed.emit(str(exc))
