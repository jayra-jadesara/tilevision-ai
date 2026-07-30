"""
In-memory LRU cache for search-result QPixmaps.

Keeps a bounded number of decoded thumbnails so repeat paints and
rapid re-searches do not re-decode the same JPEG from disk.
"""

from __future__ import annotations

import logging
import threading
from collections import OrderedDict
from pathlib import Path
from typing import Optional

from PySide6.QtCore import Qt
from PySide6.QtGui import QPixmap

logger = logging.getLogger("tilevision.presentation.thumbnail_cache")

_DEFAULT_CAPACITY = 96
_DEFAULT_MAX_EDGE = 128


class ThumbnailPixmapCache:
    """Thread-safe LRU of scaled thumbnail QPixmaps."""

    def __init__(
        self,
        capacity: int = _DEFAULT_CAPACITY,
        max_edge: int = _DEFAULT_MAX_EDGE,
    ) -> None:
        self._capacity = max(2, int(capacity))
        self._max_edge = max(32, int(max_edge))
        self._items: OrderedDict[str, QPixmap] = OrderedDict()
        self._lock = threading.Lock()

    def get(self, path: str | Path, *, fallback: str | Path | None = None) -> QPixmap:
        key = str(Path(path))
        with self._lock:
            hit = self._items.get(key)
            if hit is not None and not hit.isNull():
                self._items.move_to_end(key)
                return hit

        pixmap = QPixmap(key)
        if pixmap.isNull() and fallback is not None:
            pixmap = QPixmap(str(fallback))
        if not pixmap.isNull() and max(pixmap.width(), pixmap.height()) > self._max_edge:
            pixmap = pixmap.scaled(
                self._max_edge,
                self._max_edge,
                Qt.AspectRatioMode.KeepAspectRatio,
                Qt.TransformationMode.SmoothTransformation,
            )

        if not pixmap.isNull():
            with self._lock:
                self._items[key] = pixmap
                self._items.move_to_end(key)
                while len(self._items) > self._capacity:
                    self._items.popitem(last=False)
        return pixmap

    def invalidate(self, path: str | Path) -> None:
        key = str(Path(path))
        with self._lock:
            self._items.pop(key, None)

    def clear(self) -> None:
        with self._lock:
            self._items.clear()

    def __len__(self) -> int:
        with self._lock:
            return len(self._items)


# Shared by SearchView instances within the process.
THUMBNAIL_PIXMAP_CACHE = ThumbnailPixmapCache()
