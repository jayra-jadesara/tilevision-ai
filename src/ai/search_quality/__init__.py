"""
Production search-quality toolkit for TileVision AI.

Evidence-driven indexing / retrieval experiments. Production code imports
image analysis + multi-view selection only after bakeoff proves a win.
"""

from src.ai.search_quality.image_analysis import ImageAnalysis, analyze_image
from src.ai.search_quality.views import IndexView, IndexViewType, build_index_views

__all__ = [
    "ImageAnalysis",
    "analyze_image",
    "IndexView",
    "IndexViewType",
    "build_index_views",
]
