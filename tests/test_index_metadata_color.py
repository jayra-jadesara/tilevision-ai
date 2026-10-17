from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.ai.feature_versions import (
    CURRENT_EMBEDDING_DIMENSION,
    CURRENT_EMBEDDING_MODEL,
    CURRENT_FEATURE_VERSION,
    CURRENT_PATTERN_FEATURE_VERSION,
)
from src.core.models import TileImage
from src.core.use_cases.index_work_classify import IndexWorkKind, classify_index_work


def _indexed_tile(**kwargs) -> TileImage:
    base = {
        "id": 7,
        "file_path": "/tmp/a.jpg",
        "file_name": "a.jpg",
        "file_size": 1,
        "dimensions": "1x1",
        "is_indexed": True,
        "feature_version": CURRENT_FEATURE_VERSION,
        "pattern_feature_version": CURRENT_PATTERN_FEATURE_VERSION,
        "embedding_model": CURRENT_EMBEDDING_MODEL,
        "embedding_dimension": CURRENT_EMBEDDING_DIMENSION,
        "color": "Unknown",
    }
    base.update(kwargs)
    return TileImage(**base)


def test_metadata_only_when_clip_present_and_color_unknown(tmp_path):
    p = tmp_path / "tile.jpg"
    p.write_bytes(b"x" * 64)
    st = p.stat()
    rec = _indexed_tile(
        file_path=str(p),
        file_name=p.name,
        file_size=st.st_size,
        file_mtime=st.st_mtime,
    )
    work = classify_index_work(
        p,
        rec,
        force=False,
        clip_ids_present={7},
        clip_enabled=True,
    )
    assert work is IndexWorkKind.METADATA_ONLY
