"""
Golden search dataset — auto-generated queries mapped to source tile_id.

No manual labels. Every query is derived from the same catalog image.
"""

from __future__ import annotations

import json
import random
from dataclasses import asdict, dataclass
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageEnhance, ImageFont, ImageOps


@dataclass(frozen=True, slots=True)
class CatalogItem:
    tile_id: int
    kind: str  # "tile" | "sheet"
    path: Path


@dataclass(frozen=True, slots=True)
class GoldenQuery:
    tile_id: int
    variant: str
    path: Path
    kind: str


def _make_marble(
    h: int,
    w: int,
    seed: int,
    base: int = 230,
    tint: tuple[int, int, int] | None = None,
) -> Image.Image:
    """Production-like ceramic texture (veins + blotches), not flat noise."""
    rng = np.random.default_rng(seed)
    arr = np.full((h, w, 3), base, dtype=np.uint8)
    if tint is not None:
        arr = np.clip(arr.astype(np.int16) + np.asarray(tint, dtype=np.int16), 0, 255)
        arr = arr.astype(np.uint8)
    thickness = 1 + (seed % 4)
    for _ in range(60 + (seed % 40)):
        x0, y0 = int(rng.integers(0, w)), int(rng.integers(0, h))
        x1, y1 = int(rng.integers(0, w)), int(rng.integers(0, h))
        c = int(rng.integers(max(0, base - 55), max(1, base - 5)))
        color = (c, c, c)
        if tint is not None:
            color = tuple(int(np.clip(c + tint[i], 0, 255)) for i in range(3))
        cv2.line(arr, (x0, y0), (x1, y1), color, thickness, cv2.LINE_AA)
    for _ in range(10):
        cx, cy = int(rng.integers(0, w)), int(rng.integers(0, h))
        rad = int(rng.integers(18, 90))
        overlay = arr.copy()
        shade = int(rng.integers(max(0, base - 35), base + 1))
        shade_rgb = (shade, shade, shade)
        if tint is not None:
            shade_rgb = tuple(int(np.clip(shade + tint[i], 0, 255)) for i in range(3))
        cv2.circle(overlay, (cx, cy), rad, shade_rgb, -1)
        arr = cv2.addWeighted(overlay, 0.32, arr, 0.68, 0)
    return Image.fromarray(arr)


def _make_sheet(slab: Image.Image, name: str) -> Image.Image:
    sheet = Image.new("RGB", (1200, 900), (255, 255, 255))
    sheet.paste(slab.resize((500, 880)), (20, 10))
    draw = ImageDraw.Draw(sheet)
    try:
        font = ImageFont.truetype(
            "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 56
        )
        font_sm = ImageFont.truetype(
            "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 20
        )
    except Exception:
        font = ImageFont.load_default()
        font_sm = font
    draw.text((560, 36), "ELEGANT", fill=(180, 150, 40), font=font)
    for i, line in enumerate(
        ["Neutral ceramic series", "Soft light and shadow", f"SKU {name}", "750x1500mm"]
    ):
        draw.text((560, 120 + i * 32), line, fill=(15, 15, 15), font=font_sm)
    mini = slab.resize((90, 160))
    for r in range(2):
        for c in range(3):
            x, y = 560 + c * 110, 320 + r * 180
            sheet.paste(mini, (x, y))
            draw.rectangle((x, y, x + 90, y + 160), outline=(0, 0, 0), width=2)
    return sheet


def _center_crop(img: Image.Image, ratio: float) -> Image.Image:
    w, h = img.size
    cw, ch = max(1, int(w * ratio)), max(1, int(h * ratio))
    left, top = (w - cw) // 2, (h - ch) // 2
    return img.crop((left, top, left + cw, top + ch))


def _random_crop(img: Image.Image, ratio: float, rng: random.Random) -> Image.Image:
    w, h = img.size
    cw, ch = max(1, int(w * ratio)), max(1, int(h * ratio))
    left = rng.randint(0, max(0, w - cw))
    top = rng.randint(0, max(0, h - ch))
    return img.crop((left, top, left + cw, top + ch))


def _corner_crop(img: Image.Image, size: int = 600) -> Image.Image:
    return img.crop((0, 0, min(size, img.size[0]), min(size, img.size[1])))


def _phone_screenshot(img: Image.Image) -> Image.Image:
    """Simulate a phone screenshot: status bar + letterboxed content."""
    content = ImageOps.contain(img, (720, 1100), Image.Resampling.BICUBIC)
    canvas = Image.new("RGB", (780, 1280), (20, 20, 24))
    canvas.paste(content, ((780 - content.size[0]) // 2, 90))
    draw = ImageDraw.Draw(canvas)
    draw.rectangle((0, 0, 780, 70), fill=(10, 10, 12))
    draw.text((24, 22), "9:41  TileVision Camera", fill=(230, 230, 230))
    return canvas


def _catalogue_page_crop(img: Image.Image, kind: str) -> Image.Image:
    """Crop as if taken from a printed catalogue page."""
    if kind == "sheet":
        # Left slab region of a marketing sheet.
        return img.crop((40, 80, 520, 820)).resize((600, 900), Image.Resampling.BICUBIC)
    return _center_crop(img, 0.72)


def _room_scene(tile: Image.Image, seed: int) -> Image.Image:
    """Simple room-like photo with the tile on a floor band."""
    rng = np.random.default_rng(seed)
    room = np.full((900, 1400, 3), 55, dtype=np.uint8)
    room[:420, :] = (70, 78, 88)  # wall
    room[420:, :] = (110, 100, 90)  # floor
    floor_tile = tile.resize((480, 400), Image.Resampling.BICUBIC)
    arr = np.asarray(floor_tile)
    y0, x0 = 460, 460
    room[y0 : y0 + arr.shape[0], x0 : x0 + arr.shape[1]] = arr
    noise = rng.integers(0, 18, room.shape, dtype=np.uint8)
    room = np.clip(room.astype(np.int16) + noise, 0, 255).astype(np.uint8)
    return Image.fromarray(room)


VARIANT_SPECS = (
    "original",
    "crop_95",
    "crop_90",
    "crop_75",
    "crop_60",
    "crop_50",
    "center",
    "random_crop",
    "corner",
    "rotated",
    "brightness",
    "contrast",
    "jpeg30",
    "phone_screenshot",
    "catalogue_page",
    "room_scene",
)


def build_golden_catalog(
    root: Path,
    *,
    n_tiles: int = 24,
    n_sheets: int = 12,
    seed: int = 7,
) -> tuple[list[CatalogItem], list[GoldenQuery]]:
    """
    Build a golden catalog + auto-labeled queries.

    Tile IDs are assigned 1..N in creation order.
    """
    root = Path(root)
    catalog_dir = root / "catalog"
    query_dir = root / "queries"
    catalog_dir.mkdir(parents=True, exist_ok=True)
    query_dir.mkdir(parents=True, exist_ok=True)

    items: list[CatalogItem] = []
    queries: list[GoldenQuery] = []
    tile_id = 1
    rng = random.Random(seed)

    for i in range(n_sheets):
        tint = (
            (i * 37) % 50 - 25,
            (i * 53) % 40 - 20,
            (i * 19) % 45 - 22,
        )
        slab = _make_marble(900, 450, seed=1000 + i, base=235, tint=tint)
        sheet = _make_sheet(slab, name=f"PG{i:04d}")
        path = catalog_dir / f"sheet_{i:03d}.jpg"
        sheet.save(path, quality=95)
        items.append(CatalogItem(tile_id=tile_id, kind="sheet", path=path))
        qroot = query_dir / f"id_{tile_id:04d}"
        qroot.mkdir(exist_ok=True)
        queries.extend(_generate_queries(sheet, "sheet", tile_id, qroot, rng))
        tile_id += 1

    for i in range(n_tiles):
        tint = (
            (i * 41) % 60 - 30,
            (i * 29) % 50 - 25,
            (i * 17) % 55 - 27,
        )
        tile = _make_marble(1200, 1200, seed=2000 + i, base=220, tint=tint)
        path = catalog_dir / f"tile_{i:03d}.jpg"
        tile.save(path, quality=95)
        items.append(CatalogItem(tile_id=tile_id, kind="tile", path=path))
        qroot = query_dir / f"id_{tile_id:04d}"
        qroot.mkdir(exist_ok=True)
        queries.extend(_generate_queries(tile, "tile", tile_id, qroot, rng))
        tile_id += 1

    manifest = {
        "n_catalog": len(items),
        "n_queries": len(queries),
        "variants": list(VARIANT_SPECS),
        "items": [
            {"tile_id": it.tile_id, "kind": it.kind, "path": str(it.path)}
            for it in items
        ],
        "queries": [
            {
                "tile_id": q.tile_id,
                "variant": q.variant,
                "kind": q.kind,
                "path": str(q.path),
            }
            for q in queries
        ],
    }
    (root / "manifest.json").write_text(json.dumps(manifest, indent=2))
    return items, queries


def _generate_queries(
    image: Image.Image,
    kind: str,
    tile_id: int,
    qroot: Path,
    rng: random.Random,
) -> list[GoldenQuery]:
    out: list[GoldenQuery] = []
    builders = {
        "original": lambda: image.copy(),
        "crop_95": lambda: _center_crop(image, 0.95),
        "crop_90": lambda: _center_crop(image, 0.90),
        "crop_75": lambda: _center_crop(image, 0.75),
        "crop_60": lambda: _center_crop(image, 0.60),
        "crop_50": lambda: _center_crop(image, 0.50),
        "center": lambda: _center_crop(image, 0.66),
        "random_crop": lambda: _random_crop(image, 0.55, rng),
        "corner": lambda: _corner_crop(image, 600),
        "rotated": lambda: image.rotate(7, expand=True, fillcolor=(255, 255, 255)),
        "brightness": lambda: ImageEnhance.Brightness(image).enhance(1.22),
        "contrast": lambda: ImageEnhance.Contrast(image).enhance(1.28),
        "jpeg30": None,
        "phone_screenshot": lambda: _phone_screenshot(image),
        "catalogue_page": lambda: _catalogue_page_crop(image, kind),
        "room_scene": lambda: _room_scene(image if kind == "tile" else image.crop((20, 10, 520, 890)), tile_id),
    }
    for variant, fn in builders.items():
        path = qroot / f"{variant}.jpg"
        if variant == "jpeg30":
            image.save(path, quality=30)
        else:
            fn().save(path, quality=95)
        out.append(GoldenQuery(tile_id=tile_id, variant=variant, path=path, kind=kind))
    return out
