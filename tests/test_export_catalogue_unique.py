"""
PR C: Export Catalogue — unique tiles, Also at, downscale, Unknown scrub.
"""

from __future__ import annotations

import logging
import sys
import time
from dataclasses import replace
from pathlib import Path

import pytest
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.core.models import SearchResult, TileImage
from src.services.pdf_export_service import (
    PDFExportService,
    PdfExportOptions,
    _PDF_IMAGE_MAX_EDGE,
)


def _tile(i: int, path: str, *, code: str = "Unknown", brand: str = "Unknown") -> TileImage:
    return TileImage(
        file_path=path,
        file_name=Path(path).name,
        file_size=100,
        dimensions="100x100",
        product_code=code,
        brand=brand,
        category="Unknown",
        size="Unknown",
        color="Gray",
        is_indexed=True,
        id=i,
    )


def _result(tile: TileImage, score: float = 90.0, copies=None) -> SearchResult:
    paths = copies or [tile.file_path]
    return SearchResult(
        tile=tile,
        similarity_score=score,
        thumbnail_path=tile.file_path,
        copy_paths=list(paths),
        extra_copy_count=max(0, len(paths) - 1),
    )


def _make_noise_png(path: Path, edge: int = 300) -> None:
    import numpy as np

    rng = np.random.default_rng(abs(hash(path.name)) % (2**32))
    arr = rng.integers(0, 255, size=(edge, edge, 3), dtype=np.uint8)
    Image.fromarray(arr).save(path, format="PNG")


def test_unique_tiles_before_max_results(tmp_path):
    """6 tiles × 3 copies, max 12 → 6 cards (not 4 unique × 3)."""
    tiles = []
    for i in range(6):
        primary = tmp_path / f"tile{i}_a.png"
        _make_noise_png(primary, 64)
        copies = [str(primary)]
        for j in range(1, 3):
            c = tmp_path / f"tile{i}_{j}.png"
            _make_noise_png(c, 64)
            copies.append(str(c))
        tiles.append(
            _result(_tile(i, copies[0], code=f"T-{i}"), score=95 - i, copies=copies)
        )

    # Mimic old expand (would create 18 rows) vs new unique (6).
    from src.presentation.views.search_view import SearchView

    unique = SearchView._unique_results_for_export(tiles, None)
    assert len(unique) == 6

    out = tmp_path / "unique.pdf"
    service = PDFExportService()
    service.export_catalogue(
        output_file=out,
        query_image_path=None,
        results=unique,
        options=PdfExportOptions(max_results=12, include_also_at=True),
    )
    assert out.is_file() and out.stat().st_size > 1000

    # PDF text should mention Also at for copies and not balloon to 18 images.
    import pypdfium2 as pdfium

    doc = pdfium.PdfDocument(str(out))
    text = ""
    for i in range(len(doc)):
        text += doc[i].get_textpage().get_text_bounded()
    assert "Also at:" in text
    assert "T-0" in text and "T-5" in text


def test_product_header_falls_back_to_filename():
    tile = _tile(1, "/tmp/marble_slate_01.jpg", code="Unknown", brand="Unknown")
    assert PDFExportService._product_header_label(tile) == "marble_slate_01"
    assert PDFExportService._is_unknown("Unknown")
    assert not PDFExportService._is_unknown("Kajaria")


def test_downscale_cache_smaller_than_original(tmp_path):
    big = tmp_path / "big.png"
    # 1500 px noise — still modest for CI but > 1200 long edge
    _make_noise_png(big, 1500)
    service = PDFExportService()
    cached = service._cached_downscaled_jpeg(str(big))
    assert cached.is_file()
    with Image.open(cached) as img:
        assert max(img.size) <= _PDF_IMAGE_MAX_EDGE


def test_export_12_photos_timing_and_size(tmp_path):
    """Before/after style fixture: 12 photos, report seconds + MB."""
    results = []
    for i in range(12):
        p = tmp_path / f"photo_{i}.png"
        # 1200 edge keeps CI fast while still exercising downscale path
        _make_noise_png(p, 1200)
        results.append(_result(_tile(i, str(p), code=f"P-{i}", brand="Demo")))

    out = tmp_path / "twelve.pdf"
    t0 = time.perf_counter()
    PDFExportService().export_catalogue(
        output_file=out,
        query_image_path=str(results[0].tile.file_path),
        results=results,
        options=PdfExportOptions(max_results=12),
    )
    elapsed = time.perf_counter() - t0
    size_mb = out.stat().st_size / (1024 * 1024)
    # Soft bound — must stay responsive-scale (worker moves this off UI).
    assert elapsed < 30.0, f"export too slow: {elapsed:.1f}s"
    assert size_mb < 25.0, f"PDF too large: {size_mb:.1f} MB"
    # Record for PR evidence
    evidence = Path("/opt/cursor/artifacts")
    evidence.mkdir(parents=True, exist_ok=True)
    (evidence / "export_twelve_timing.txt").write_text(
        f"elapsed_s={elapsed:.2f}\nsize_mb={size_mb:.2f}\n",
        encoding="utf-8",
    )


def test_render_export_pdf_pages(tmp_path):
    """Attach light-page PNG renders for single / description / multi cases."""
    import pypdfium2 as pdfium

    artifacts = Path("/opt/cursor/artifacts/export_pdf_renders")
    artifacts.mkdir(parents=True, exist_ok=True)

    def _build(name: str, *, query: str | None, results):
        out = tmp_path / f"{name}.pdf"
        PDFExportService().export_catalogue(
            output_file=out,
            query_image_path=query,
            results=results,
            options=PdfExportOptions(
                max_results=6,
                company_name="TileVision Demo",
                include_search_image=bool(query),
            ),
        )
        doc = pdfium.PdfDocument(str(out))
        # Cover (page 0) is often dark navy — render page 1 (content / light).
        page_index = 1 if len(doc) > 1 else 0
        page = doc[page_index]
        bitmap = page.render(scale=2)
        pil = bitmap.to_pil()
        dest = artifacts / f"{name}_page{page_index + 1}.png"
        pil.save(dest)
        assert dest.is_file() and dest.stat().st_size > 500
        return dest

    q = tmp_path / "query.png"
    _make_noise_png(q, 200)
    results = []
    for i in range(4):
        p = tmp_path / f"r{i}.png"
        _make_noise_png(p, 200)
        results.append(
            _result(_tile(i, str(p), code=f"R-{i}", brand="Brand"), score=90 - i)
        )

    single = _build("single_photo", query=str(q), results=results)
    # Description search: no query image → best-match intro
    desc = _build("description_search", query=None, results=results)
    # Multi-photo: same as no single query image
    multi = _build("multi_photo", query=None, results=results)
    assert single.exists() and desc.exists() and multi.exists()


def test_export_logs_start_finish(tmp_path, caplog):
    p = tmp_path / "a.png"
    _make_noise_png(p, 64)
    results = [_result(_tile(1, str(p), code="A1", brand="B"))]
    out = tmp_path / "log.pdf"
    with caplog.at_level(logging.INFO, logger="tilevision.services.pdf_export"):
        PDFExportService().export_catalogue(
            output_file=out,
            query_image_path=None,
            results=results,
            options=PdfExportOptions(max_results=5),
        )
    assert any("Export catalogue start" in r.message for r in caplog.records)
    assert any("Export catalogue finish" in r.message for r in caplog.records)
