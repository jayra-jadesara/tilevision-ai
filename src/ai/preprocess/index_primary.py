"""
Shared index-time primary crop + letterbox preparation.

Used by ``FeatureExtractor.extract_index_vectors`` (production indexing) and
``show_index_crops`` (debug). Keeping one implementation prevents silent
divergence between "what the debug PNGs show" and "what gets embedded."
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import cv2
import numpy as np
from PIL import Image

from src.ai.models import PreprocessedImage
from src.ai.preprocess.image_preprocessor import ImagePreprocessor
from src.ai.search_quality.image_analysis import ImageAnalysis, analyze_image
from src.ai.search_quality.views import IndexStrategy, IndexView, build_index_views

PrimarySource = Literal["panel", "full_sheet"]


@dataclass(frozen=True, slots=True)
class IndexPrimaryPreparation:
    """Exact primary view that feeds indexed TileFeatures descriptors."""

    source_path: str
    raw: Image.Image
    analysis: ImageAnalysis
    views: tuple[IndexView, ...]
    panel: Image.Image | None
    primary: PreprocessedImage
    primary_source: PrimarySource


def finalize_index_pil(
    view: Image.Image,
    *,
    original_size: tuple[int, int],
    match_pad_to_content: bool = False,
) -> PreprocessedImage:
    """
    Normalize + letterbox an index crop into a PreprocessedImage.

    This is the production finalize step formerly private on FeatureExtractor.
    """
    view = ImagePreprocessor.normalize_lighting(view)
    pad_color = None
    if match_pad_to_content:
        # Portrait panel letterboxes are ~45% pad; neutral gray PAD_COLOR
        # destroys LAB color/texture histograms (PGYS2319 color=0.07 class).
        mean = np.asarray(view.convert("RGB"), dtype=np.float32).mean(axis=(0, 1))
        pad_color = (
            int(np.clip(mean[0], 0, 255)),
            int(np.clip(mean[1], 0, 255)),
            int(np.clip(mean[2], 0, 255)),
        )
    view = ImagePreprocessor.resize_letterbox(view, pad_color=pad_color)
    rgb = ImagePreprocessor.to_numpy(view)
    bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    return PreprocessedImage(
        pil=view,
        rgb=rgb,
        bgr=bgr,
        gray=gray,
        width=original_size[0],
        height=original_size[1],
    )


def prepare_index_primary(image_path: str | Path) -> IndexPrimaryPreparation:
    """
    Build the exact primary PreprocessedImage used at index time.

    SAM2 / Precise Crop is intentionally not invoked here — it is UI-only
    (Precise Crop & Search). Catalog indexing uses heuristic panel isolation
    via ``primary_texture_panel`` when ``left_panel_beneficial``.
    """
    source = Path(image_path).expanduser().resolve()
    if not source.is_file():
        raise FileNotFoundError(f"Catalog image not found: {source}")

    raw = ImagePreprocessor.load(source)
    raw = ImagePreprocessor.to_rgb(raw)
    analysis = analyze_image(raw)
    views = tuple(
        build_index_views(
            raw,
            IndexStrategy.E_HEURISTIC_MULTIVIEW,
            analysis=analysis,
        )
    )

    panel: Image.Image | None = None
    if analysis.left_panel_beneficial:
        panel = ImagePreprocessor.primary_texture_panel(raw)

    if panel is not None:
        primary = finalize_index_pil(
            panel,
            original_size=raw.size,
            match_pad_to_content=True,
        )
        primary_source: PrimarySource = "panel"
    else:
        primary = ImagePreprocessor.preprocess(source)
        primary_source = "full_sheet"

    return IndexPrimaryPreparation(
        source_path=str(source),
        raw=raw,
        analysis=analysis,
        views=views,
        panel=panel,
        primary=primary,
        primary_source=primary_source,
    )
