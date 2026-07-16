"""
Shared fake AI components for integration tests (no torch required).
"""

from __future__ import annotations

import numpy as np
from PIL import Image

from src.ai.descriptors.color_descriptor import ColorDescriptor
from src.ai.feature_extractor import ExtractTimings
from src.ai.models import TileFeatures


class FakeEmbedder:
    """Deterministic embedder based on average RGB of a small resize."""

    def __init__(self) -> None:
        self.calls = 0

    def load_model(self) -> None:
        pass

    def _rgb_embedding(self, img: Image.Image) -> np.ndarray:
        self.calls += 1
        img = img.convert("RGB").resize((8, 8))
        pixels = list(img.getdata())
        r = sum(p[0] for p in pixels) / (len(pixels) * 255.0)
        g = sum(p[1] for p in pixels) / (len(pixels) * 255.0)
        b = sum(p[2] for p in pixels) / (len(pixels) * 255.0)
        return np.array([r, g, b, 1.0], dtype=np.float32)

    def extract_from_preprocessed(self, processed) -> np.ndarray:
        return self._rgb_embedding(processed.pil)

    def extract_batch_from_preprocessed(self, processed_images) -> list:
        return [self.extract_from_preprocessed(p) for p in processed_images]

    def extract(self, image_path: str) -> np.ndarray:
        with Image.open(image_path) as img:
            return self._rgb_embedding(img)

    def get_embedding(self, image_path: str) -> list[float]:
        return self.extract(image_path).tolist()


def make_tile_features(embedding: list[float] | np.ndarray) -> TileFeatures:
    """Build a minimal valid TileFeatures object for repository tests."""
    embedding_arr = np.asarray(embedding, dtype=np.float32)
    return TileFeatures(
        embedding=embedding_arr,
        color_histogram=np.full(
            ColorDescriptor.vector_size(),
            1.0 / ColorDescriptor.vector_size(),
            dtype=np.float32,
        ),
        texture_histogram=np.full(54, 1.0 / 54, dtype=np.float32),
        edge_histogram=np.full(36, 1.0 / 36, dtype=np.float32),
        pattern_features=np.zeros(8, dtype=np.float32),
        dominant_color=(128, 128, 128),
        width=16,
        height=16,
    )


class FakeFeatureExtractor:
    """Lightweight FeatureExtractor stand-in for orchestration tests."""

    def __init__(self, embedder: FakeEmbedder | None = None) -> None:
        self._embedder = embedder or FakeEmbedder()
        self._last_timings = ExtractTimings(
            preprocessing=0.001,
            dinov2=0.002,
            descriptors=0.001,
            total=0.004,
        )

    @property
    def last_timings(self) -> ExtractTimings:
        return self._last_timings

    @property
    def embedder(self) -> FakeEmbedder:
        return self._embedder

    def load_model(self) -> None:
        self._embedder.load_model()

    def extract(self, image_path: str) -> TileFeatures:
        embedding = self._embedder.extract(image_path)
        return make_tile_features(embedding)

    def extract_batch(self, image_paths: list[str]) -> list[TileFeatures]:
        return [self.extract(path) for path in image_paths]
