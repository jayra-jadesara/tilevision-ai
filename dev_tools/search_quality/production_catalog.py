"""
Production-representative catalog generator for search-optimization studies.

NOTE: Real customer catalog volumes are not mounted in this environment.
This generator creates 300+ distinctive ceramic-like masters covering the
material classes requested for production evaluation. Results are labeled
``catalog_source=synthetic_production_representative``.
"""

from __future__ import annotations

import json
import math
import random
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageEnhance, ImageFont, ImageOps


MATERIAL_CLASSES = (
    "marble",
    "onyx",
    "wood",
    "concrete",
    "terrazzo",
    "stone",
    "glossy",
    "matt",
    "bookmatch",
    "repetitive",
    "low_texture",
    "white",
    "dark",
    "near_duplicate",
)


@dataclass(frozen=True, slots=True)
class CatalogTile:
    tile_id: int
    path: Path
    material: str
    finish: str
    is_sheet: bool
    near_dup_of: int | None = None


def _clamp(v: int) -> int:
    return int(max(0, min(255, v)))


def _base_color(material: str, seed: int) -> tuple[int, int, int]:
    rng = random.Random(seed)
    tables = {
        "marble": (235, 230, 220),
        "onyx": (245, 240, 235),
        "wood": (150, 95, 50),
        "concrete": (145, 145, 140),
        "terrazzo": (220, 215, 205),
        "stone": (120, 115, 105),
        "glossy": (200, 205, 210),
        "matt": (160, 155, 150),
        "bookmatch": (230, 225, 215),
        "repetitive": (180, 100, 80),
        "low_texture": (240, 240, 238),
        "white": (248, 248, 246),
        "dark": (45, 48, 52),
        "near_duplicate": (232, 228, 218),
    }
    base = tables.get(material, (200, 200, 200))
    return tuple(_clamp(c + rng.randint(-18, 18)) for c in base)


def _render_face(
    h: int,
    w: int,
    material: str,
    seed: int,
    *,
    mirror: bool = False,
) -> Image.Image:
    rng = np.random.default_rng(seed)
    py_rng = random.Random(seed)
    base = _base_color(material, seed)
    arr = np.full((h, w, 3), base, dtype=np.uint8)

    if material in {"marble", "onyx", "bookmatch", "near_duplicate", "white"}:
        for _ in range(40 + seed % 40):
            x0, y0 = int(rng.integers(0, w)), int(rng.integers(0, h))
            x1, y1 = int(rng.integers(0, w)), int(rng.integers(0, h))
            c = _clamp(base[0] - int(rng.integers(10, 55)))
            color = (c, c, _clamp(c - 5))
            cv2.line(arr, (x0, y0), (x1, y1), color, 1 + seed % 3, cv2.LINE_AA)
        for _ in range(6):
            cx, cy = int(rng.integers(0, w)), int(rng.integers(0, h))
            rad = int(rng.integers(20, 90))
            overlay = arr.copy()
            shade = _clamp(base[0] - int(rng.integers(5, 30)))
            cv2.circle(overlay, (cx, cy), rad, (shade, shade, shade), -1)
            arr = cv2.addWeighted(overlay, 0.30, arr, 0.70, 0)

    elif material == "wood":
        for y in range(h):
            wave = int(12 * math.sin(y / 18.0 + seed))
            shade = tuple(_clamp(c + wave + int(rng.integers(-8, 8))) for c in base)
            arr[y, :] = shade
        for _ in range(30):
            x = int(rng.integers(0, w))
            cv2.line(arr, (x, 0), (x + int(rng.integers(-20, 20)), h), (80, 50, 25), 1)

    elif material == "concrete":
        noise = rng.integers(-25, 25, size=(h, w, 3), dtype=np.int16)
        arr = np.clip(arr.astype(np.int16) + noise, 0, 255).astype(np.uint8)
        for _ in range(200):
            x, y = int(rng.integers(0, w)), int(rng.integers(0, h))
            arr[y, x] = tuple(_clamp(c + int(rng.integers(-40, 40))) for c in base)

    elif material == "terrazzo":
        for _ in range(900):
            x, y = int(rng.integers(0, w)), int(rng.integers(0, h))
            r = int(rng.integers(2, 10))
            color = (
                int(rng.integers(40, 255)),
                int(rng.integers(40, 255)),
                int(rng.integers(40, 255)),
            )
            cv2.circle(arr, (x, y), r, color, -1)

    elif material == "stone":
        for y in range(0, h, 12):
            shade = tuple(_clamp(c + ((y // 12 + seed) % 3) * 10 - 10) for c in base)
            arr[y : y + 10, :] = shade
        noise = rng.integers(-15, 15, size=(h, w, 3), dtype=np.int16)
        arr = np.clip(arr.astype(np.int16) + noise, 0, 255).astype(np.uint8)

    elif material == "repetitive":
        step = 36 + (seed % 20)
        for x in range(0, w, step):
            cv2.line(arr, (x, 0), (x, h), (max(0, base[0] - 40),) * 3, 3)
        for y in range(0, h, step):
            cv2.line(arr, (0, y), (w, y), (max(0, base[0] - 40),) * 3, 3)

    elif material == "low_texture":
        noise = rng.normal(0, 2.0, size=arr.shape)
        arr = np.clip(arr.astype(np.float32) + noise, 0, 255).astype(np.uint8)

    elif material == "dark":
        for _ in range(50):
            x0, y0 = int(rng.integers(0, w)), int(rng.integers(0, h))
            x1, y1 = int(rng.integers(0, w)), int(rng.integers(0, h))
            c = _clamp(base[0] + int(rng.integers(10, 40)))
            cv2.line(arr, (x0, y0), (x1, y1), (c, c, c), 1, cv2.LINE_AA)

    elif material in {"glossy", "matt"}:
        noise = rng.integers(-20, 20, size=(h, w, 3), dtype=np.int16)
        arr = np.clip(arr.astype(np.int16) + noise, 0, 255).astype(np.uint8)
        if material == "glossy":
            # Specular streak
            cv2.ellipse(
                arr,
                (w // 3, h // 3),
                (w // 4, h // 10),
                25,
                0,
                360,
                (255, 255, 255),
                -1,
            )
            arr = cv2.addWeighted(arr, 0.85, np.full_like(arr, 255), 0.15, 0)

    img = Image.fromarray(arr)
    if mirror:
        img = ImageOps.mirror(img)
    # Finish cue
    if material == "matt":
        img = ImageEnhance.Contrast(img).enhance(0.85)
    if material == "glossy":
        img = ImageEnhance.Contrast(img).enhance(1.15)
    return img


def _make_sheet(slab: Image.Image, sku: str) -> Image.Image:
    sheet = Image.new("RGB", (1200, 900), (255, 255, 255))
    sheet.paste(slab.resize((500, 880)), (20, 10))
    draw = ImageDraw.Draw(sheet)
    try:
        font = ImageFont.truetype(
            "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 52
        )
        font_sm = ImageFont.truetype(
            "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 20
        )
    except Exception:
        font = ImageFont.load_default()
        font_sm = font
    draw.text((560, 40), "COLLECTION", fill=(170, 140, 40), font=font)
    for i, line in enumerate([sku, "Ceramic porcelain", "600x1200mm", "Indoor/Outdoor"]):
        draw.text((560, 120 + i * 34), line, fill=(20, 20, 20), font=font_sm)
    mini = slab.resize((90, 160))
    for r in range(2):
        for c in range(3):
            x, y = 560 + c * 110, 340 + r * 180
            sheet.paste(mini, (x, y))
            draw.rectangle((x, y, x + 90, y + 160), outline=(0, 0, 0), width=2)
    return sheet


def build_production_catalog(
    root: Path,
    *,
    n_tiles: int = 320,
    seed: int = 42,
) -> list[CatalogTile]:
    """
    Build >=300 distinctive catalog masters.

    Includes ~15% marketing sheets and paired near-duplicates.
    """
    root = Path(root)
    cat = root / "catalog"
    cat.mkdir(parents=True, exist_ok=True)
    tiles: list[CatalogTile] = []
    tile_id = 1
    materials = list(MATERIAL_CLASSES)
    # Scale composition to the requested size; hard-cap at n_tiles.
    n_near = min(n_tiles, max(2, (n_tiles // 12) * 2))  # even count
    if n_near % 2:
        n_near -= 1
    n_sheets = min(max(0, n_tiles - n_near), max(1, n_tiles // 8))

    # Near-duplicate pairs first (share material/seed family)
    for i in range(n_near // 2):
        if len(tiles) >= n_tiles:
            break
        mat = "near_duplicate" if i % 2 == 0 else "marble"
        face = _render_face(1000, 1000, mat, seed=10_000 + i)
        path_a = cat / f"tile_{tile_id:04d}_{mat}_a.jpg"
        face.save(path_a, quality=94)
        tiles.append(
            CatalogTile(tile_id, path_a, mat, "matt", False, near_dup_of=None)
        )
        id_a = tile_id
        tile_id += 1
        if len(tiles) >= n_tiles:
            break
        face_b = ImageEnhance.Brightness(face).enhance(1.04)
        face_b = ImageEnhance.Contrast(face_b).enhance(0.98)
        path_b = cat / f"tile_{tile_id:04d}_{mat}_b.jpg"
        face_b.save(path_b, quality=92)
        tiles.append(
            CatalogTile(tile_id, path_b, mat, "matt", False, near_dup_of=id_a)
        )
        tile_id += 1

    # Marketing sheets
    for i in range(n_sheets):
        if len(tiles) >= n_tiles:
            break
        mat = materials[i % len(materials)]
        if mat in {"repetitive", "low_texture"}:
            mat = "marble"
        slab = _render_face(900, 450, mat, seed=20_000 + i)
        if mat == "bookmatch":
            slab = _render_face(900, 450, "bookmatch", seed=20_000 + i, mirror=False)
        sheet = _make_sheet(slab, sku=f"SKU{tile_id:04d}")
        path = cat / f"sheet_{tile_id:04d}_{mat}.jpg"
        sheet.save(path, quality=94)
        tiles.append(CatalogTile(tile_id, path, mat, "sheet", True, None))
        tile_id += 1

    # Remaining square tiles
    while len(tiles) < n_tiles:
        mat = materials[len(tiles) % len(materials)]
        face = _render_face(
            1100,
            1100,
            mat,
            seed=30_000 + len(tiles),
            mirror=(mat == "bookmatch" and len(tiles) % 2 == 1),
        )
        finish = "glossy" if mat == "glossy" else ("matt" if mat == "matt" else "natural")
        path = cat / f"tile_{tile_id:04d}_{mat}.jpg"
        face.save(path, quality=94)
        tiles.append(CatalogTile(tile_id, path, mat, finish, False, None))
        tile_id += 1

    manifest = {
        "catalog_source": "synthetic_production_representative",
        "note": (
            "Real customer catalog volumes were not available in this environment. "
            "This catalog is generated to cover production material classes at "
            f"{len(tiles)} unique tiles."
        ),
        "n_tiles": len(tiles),
        "materials": {
            m: sum(1 for t in tiles if t.material == m) for m in MATERIAL_CLASSES
        },
        "tiles": [
            {
                "tile_id": t.tile_id,
                "path": str(t.path),
                "material": t.material,
                "finish": t.finish,
                "is_sheet": t.is_sheet,
                "near_dup_of": t.near_dup_of,
            }
            for t in tiles
        ],
    }
    (root / "catalog_manifest.json").write_text(json.dumps(manifest, indent=2))
    return tiles
