"""Content-copy collapse for search results (sha256 / near-exact dHash)."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.ai.search_quality.result_dedupe import (
    collapse_scored_by_content,
    collapse_search_results,
)
from src.core.models import SearchResult, TileImage


def _tile(tid: int, path: str, *, sha: str = "", ph: str = "") -> TileImage:
    return TileImage(
        id=tid,
        file_path=path,
        file_name=Path(path).name,
        file_size=1,
        dimensions="1x1",
        sha256_hash=sha,
        perceptual_hash=ph,
    )


def test_three_identical_plus_two_different_yield_two_rows():
    a = _tile(1, r"C:\Tiles\New\New2\images (4).png", sha="abc")
    b = _tile(2, r"C:\Tiles\New\images (4).png", sha="abc")
    c = _tile(3, r"C:\Tiles\images (4).png", sha="abc")
    d = _tile(4, r"C:\Tiles\other.png", sha="def")
    e = _tile(5, r"C:\Tiles\third.png", sha="ghi")
    scored = [
        (a, 0.2982),
        (b, 0.2982),
        (c, 0.2982),
        (d, 0.27),
        (e, 0.25),
    ]
    collapsed = collapse_scored_by_content(scored)
    assert len(collapsed) == 3  # abc, def, ghi
    top_tile, top_score, paths = collapsed[0]
    assert top_score == 0.2982
    assert len(paths) == 3
    assert paths[0] == a.file_path
    assert {p for p in paths} == {a.file_path, b.file_path, c.file_path}


def test_collapse_search_results_preserves_score_order_and_top_k():
    results = [
        SearchResult(
            tile=_tile(1, "/a.png", sha="aaa"),
            similarity_score=90.0,
            thumbnail_path="/a.t",
        ),
        SearchResult(
            tile=_tile(2, "/a_copy.png", sha="aaa"),
            similarity_score=90.0,
            thumbnail_path="/a.t",
        ),
        SearchResult(
            tile=_tile(3, "/b.png", sha="bbb"),
            similarity_score=80.0,
            thumbnail_path="/b.t",
        ),
        SearchResult(
            tile=_tile(4, "/c.png", sha="ccc"),
            similarity_score=70.0,
            thumbnail_path="/c.t",
        ),
    ]
    out = collapse_search_results(results, top_k=2)
    assert len(out) == 2
    assert out[0].tile.sha256_hash == "aaa"
    assert out[0].extra_copy_count == 1
    assert out[1].tile.sha256_hash == "bbb"
    # Unique tiles fill top_k slots — ccc excluded.
    assert all(r.tile.sha256_hash != "ccc" for r in out)


def test_three_identical_two_different_as_search_results_two_rows_after_top_k():
    """Acceptance: 3 identical + 2 different → 2 unique rows when top_k=2."""
    results = []
    for i, (sha, score) in enumerate(
        [("same", 86.7), ("same", 86.7), ("same", 86.7), ("x", 70.0), ("y", 60.0)],
        start=1,
    ):
        results.append(
            SearchResult(
                tile=_tile(i, f"/p{i}.png", sha=sha),
                similarity_score=score,
                thumbnail_path=f"/t{i}",
            )
        )
    out = collapse_search_results(results, top_k=2)
    assert len(out) == 2
    assert out[0].extra_copy_count == 2
    assert out[1].tile.sha256_hash == "x"
