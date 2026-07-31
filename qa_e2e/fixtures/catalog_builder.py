"""
Build a small but distinctive showroom catalogue + query photos.

Images are real files on disk (PNG/JPG/WEBP/TIFF). Patterns are visually
distinct so DINOv2 + IndexFlatIP can rank the correct tile without mocks.
"""

from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Any, Dict, List, Tuple

from PIL import Image, ImageDraw


# Distinct product identities for expectation checks.
_TILE_SPECS: List[Tuple[str, str, str, Tuple[int, int, int], str]] = [
    ("A01", "MarbleCo", "Marble", (210, 200, 185), "veins"),
    ("A02", "MarbleCo", "Marble", (180, 175, 170), "veins"),
    ("B01", "WoodLux", "Wood", (140, 90, 45), "grain"),
    ("B02", "WoodLux", "Wood", (110, 70, 35), "grain"),
    ("C01", "GeoStone", "Geometric", (40, 90, 160), "grid"),
    ("C02", "GeoStone", "Geometric", (200, 60, 60), "grid"),
    ("D01", "Terra", "Terracotta", (180, 95, 55), "speckle"),
    ("D02", "Terra", "Terracotta", (160, 80, 50), "speckle"),
    ("E01", "SlatePro", "Slate", (70, 75, 80), "layers"),
    ("E02", "SlatePro", "Slate", (55, 60, 65), "layers"),
    ("F01", "Cemento", "Concrete", (130, 130, 125), "noise"),
    ("F02", "Cemento", "Concrete", (150, 148, 145), "noise"),
]


def _draw_pattern(draw: ImageDraw.ImageDraw, size: int, kind: str, base: Tuple[int, int, int], rng: random.Random) -> None:
    w = h = size
    if kind == "veins":
        for _ in range(18):
            x0 = rng.randint(0, w)
            y0 = rng.randint(0, h)
            x1 = x0 + rng.randint(-80, 80)
            y1 = y0 + rng.randint(40, 160)
            color = tuple(max(0, min(255, c + rng.randint(-40, 40))) for c in base)
            draw.line((x0, y0, x1, y1), fill=color, width=rng.randint(2, 5))
    elif kind == "grain":
        for y in range(0, h, 3):
            shade = tuple(max(0, min(255, c + rng.randint(-25, 25))) for c in base)
            draw.line((0, y, w, y + rng.randint(-2, 2)), fill=shade, width=2)
    elif kind == "grid":
        step = 28
        line = tuple(max(0, min(255, c - 50)) for c in base)
        for x in range(0, w, step):
            draw.line((x, 0, x, h), fill=line, width=3)
        for y in range(0, h, step):
            draw.line((0, y, w, y), fill=line, width=3)
    elif kind == "speckle":
        for _ in range(900):
            x = rng.randint(0, w - 1)
            y = rng.randint(0, h - 1)
            shade = tuple(max(0, min(255, c + rng.randint(-60, 60))) for c in base)
            draw.point((x, y), fill=shade)
    elif kind == "layers":
        for y in range(0, h, 10):
            shade = tuple(max(0, min(255, c + ((y // 10) % 2) * 20 - 10)) for c in base)
            draw.rectangle((0, y, w, y + 8), fill=shade)
    else:  # noise
        for _ in range(1200):
            x = rng.randint(0, w - 1)
            y = rng.randint(0, h - 1)
            r = rng.randint(1, 3)
            shade = tuple(max(0, min(255, c + rng.randint(-30, 30))) for c in base)
            draw.ellipse((x, y, x + r, y + r), fill=shade)


def _make_tile(path: Path, base: Tuple[int, int, int], kind: str, size: int, seed: int) -> None:
    rng = random.Random(seed)
    img = Image.new("RGB", (size, size), base)
    draw = ImageDraw.Draw(img)
    _draw_pattern(draw, size, kind, base, rng)
    # Soft border like a photographed tile edge
    draw.rectangle((2, 2, size - 3, size - 3), outline=tuple(max(0, c - 40) for c in base), width=3)
    path.parent.mkdir(parents=True, exist_ok=True)
    ext = path.suffix.lower()
    if ext in {".jpg", ".jpeg"}:
        img.save(path, quality=92)
    elif ext == ".webp":
        img.save(path, "WEBP", quality=90)
    elif ext in {".tif", ".tiff"}:
        img.save(path, "TIFF")
    else:
        img.save(path)


def _make_query_variant(src: Path, dest: Path, *, mode: str, seed: int) -> None:
    rng = random.Random(seed)
    img = Image.open(src).convert("RGB")
    w, h = img.size
    if mode == "crop":
        margin = int(min(w, h) * 0.12)
        img = img.crop((margin, margin, w - margin, h - margin))
    elif mode == "rotate":
        img = img.rotate(rng.choice([5, -7, 11]), expand=True, fillcolor=(30, 30, 30))
    elif mode == "dark":
        img = img.point(lambda p: max(0, int(p * 0.75)))
    elif mode == "large":
        img = img.resize((w * 2, h * 2), Image.Resampling.BICUBIC)
    elif mode == "small":
        img = img.resize((max(64, w // 3), max(64, h // 3)), Image.Resampling.BILINEAR)
    dest.parent.mkdir(parents=True, exist_ok=True)
    img.save(dest, quality=90)


def build_customer_catalog(root: Path, *, tile_count: int = 12) -> Dict[str, Any]:
    """
    Create:
      root/catalog/   — indexed showroom tiles
      root/queries/   — customer phone photos / crops
      root/corrupt/   — invalid files
      root/manifest.json
    """
    root = Path(root)
    catalog = root / "catalog"
    queries = root / "queries"
    corrupt = root / "corrupt"
    catalog.mkdir(parents=True, exist_ok=True)
    queries.mkdir(parents=True, exist_ok=True)
    corrupt.mkdir(parents=True, exist_ok=True)

    specs = _TILE_SPECS[: max(4, min(tile_count, len(_TILE_SPECS)))]
    tiles_meta = []
    for i, (code, brand, category, color, kind) in enumerate(specs):
        # Catalogue indexing accepts jpg/png/webp (not TIFF). Keep catalog as
        # JPEG so every file is indexed; query-format coverage lives in queries/.
        ext = ".jpg"
        name = f"TILE_{code}_{brand}_{category}{ext}"
        path = catalog / name
        # 256px is enough for DINOv2 and much faster on Mac Intel CPU CI.
        _make_tile(path, color, kind, size=256, seed=1000 + i)
        # Also write a second size variant for a couple tiles
        tiles_meta.append(
            {
                "product": code,
                "brand": brand,
                "category": category,
                "path": str(path),
                "file_name": name,
            }
        )

    query_meta = []
    # Primary happy-path queries (one per first 6 tiles)
    for i, tile in enumerate(tiles_meta[:6]):
        qid = f"q_match_{tile['product']}"
        dest = queries / f"{qid}.jpg"
        _make_query_variant(Path(tile["path"]), dest, mode="crop", seed=2000 + i)
        query_meta.append(
            {
                "id": qid,
                "path": str(dest),
                "expected_product": tile["product"],
                "max_rank": 3,
                "kind": "crop_match",
            }
        )

    # Large / small / rotated
    base = tiles_meta[0]
    for mode, qid in (("large", "q_large"), ("small", "q_small"), ("rotate", "q_rotate")):
        dest = queries / f"{qid}.jpg"
        _make_query_variant(Path(base["path"]), dest, mode=mode, seed=3000)
        query_meta.append(
            {
                "id": qid,
                "path": str(dest),
                "expected_product": base["product"],
                "max_rank": 5,
                "kind": mode,
            }
        )

    # Format-specific query copies
    fmt_src = Path(tiles_meta[1]["path"])
    for ext, qid in ((".png", "q_png"), (".webp", "q_webp"), (".tif", "q_tiff"), (".jpg", "q_jpg")):
        dest = queries / f"{qid}{ext}"
        Image.open(fmt_src).convert("RGB").save(dest)
        query_meta.append(
            {
                "id": qid,
                "path": str(dest),
                "expected_product": tiles_meta[1]["product"],
                "max_rank": 3,
                "kind": f"format{ext}",
            }
        )

    # Corrupt / unsupported
    (corrupt / "not_an_image.jpg").write_bytes(b"this is not an image payload")
    (corrupt / "empty.png").write_bytes(b"")
    (corrupt / "notes.txt").write_text("hello", encoding="utf-8")

    manifest: Dict[str, Any] = {
        "catalog_dir": str(catalog),
        "query_dir": str(queries),
        "corrupt_dir": str(corrupt),
        "tiles": tiles_meta,
        "queries": query_meta,
    }
    (root / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest
