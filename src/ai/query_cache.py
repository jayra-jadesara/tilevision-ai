"""
In-memory LRU cache for recent query embeddings.

Avoids re-running DINOv2 when the same query file is searched again
(same path + mtime + size). Offline-only — never hits the network.
"""

from __future__ import annotations

import logging
import threading
from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np

from src.ai.models import TileFeatures

logger = logging.getLogger("tilevision.ai.query_cache")

_DEFAULT_CAPACITY = 32


@dataclass(frozen=True, slots=True)
class QueryCacheKey:
    path: str
    mtime_ns: int
    size: int


@dataclass(slots=True)
class QueryCacheEntry:
    features: TileFeatures
    embeddings: list[np.ndarray]


class QueryEmbeddingCache:
    """Thread-safe LRU of recent drop-search embeddings."""

    def __init__(self, capacity: int = _DEFAULT_CAPACITY) -> None:
        self._capacity = max(4, int(capacity))
        self._items: OrderedDict[QueryCacheKey, QueryCacheEntry] = OrderedDict()
        self._lock = threading.Lock()

    @staticmethod
    def key_for_path(image_path: str | Path) -> Optional[QueryCacheKey]:
        path = Path(image_path)
        try:
            stat = path.stat()
        except OSError:
            return None
        return QueryCacheKey(
            path=str(path.resolve()),
            mtime_ns=int(stat.st_mtime_ns),
            size=int(stat.st_size),
        )

    def get(self, image_path: str | Path) -> Optional[QueryCacheEntry]:
        key = self.key_for_path(image_path)
        if key is None:
            return None
        with self._lock:
            entry = self._items.get(key)
            if entry is None:
                return None
            self._items.move_to_end(key)
            logger.info("Query embedding cache HIT: %s", Path(key.path).name)
            return entry

    def put(
        self,
        image_path: str | Path,
        features: TileFeatures,
        embeddings: list[np.ndarray],
    ) -> None:
        key = self.key_for_path(image_path)
        if key is None:
            return
        with self._lock:
            self._items[key] = QueryCacheEntry(
                features=features,
                embeddings=list(embeddings),
            )
            self._items.move_to_end(key)
            while len(self._items) > self._capacity:
                self._items.popitem(last=False)
            logger.debug("Query embedding cache STORE: %s", Path(key.path).name)

    def clear(self) -> None:
        with self._lock:
            self._items.clear()


# Process-wide cache shared by SearchTilesUseCase instances.
QUERY_EMBEDDING_CACHE = QueryEmbeddingCache()
