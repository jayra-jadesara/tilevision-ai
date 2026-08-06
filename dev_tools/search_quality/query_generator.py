"""
Query generation for production search-optimization studies.
"""

from __future__ import annotations

import json
import random
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image, ImageEnhance, ImageOps

from dev_tools.search_quality.production_catalog import CatalogTile


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
    "phone_screenshot",
    "jpeg30",
    "brightness_p30",
    "brightness_m30",
    "contrast_p30",
    "contrast_m30",
    "rotation_p10",
    "rotation_m10",
    "perspective",
    "catalogue_page",
    "room_scene",
    "whatsapp",
)


@dataclass(frozen=True, slots=True)
class QueryItem:
    tile_id: int
    variant: str
    path: Path
    material: str
    is_sheet: bool


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
    content = ImageOps.contain(img, (720, 1100), Image.Resampling.BICUBIC)
    canvas = Image.new("RGB", (780, 1280), (18, 18, 22))
    canvas.paste(content, ((780 - content.size[0]) // 2, 96))
    from PIL import ImageDraw

    draw = ImageDraw.Draw(canvas)
    draw.rectangle((0, 0, 780, 72), fill=(8, 8, 10))
    draw.text((24, 24), "9:41  Camera", fill=(230, 230, 230))
    return canvas


def _whatsapp_like(img: Image.Image) -> Image.Image:
    """Downscale + heavy JPEG + slight color shift (chat compression)."""
    small = ImageOps.contain(img, (800, 800), Image.Resampling.BICUBIC)
    arr = np.asarray(small, dtype=np.int16)
    arr = np.clip(arr + np.array([8, -4, 6], dtype=np.int16), 0, 255).astype(np.uint8)
    return Image.fromarray(arr)


def _perspective(img: Image.Image, seed: int) -> Image.Image:
    import cv2

    w, h = img.size
    rng = np.random.default_rng(seed)
    margin = int(min(w, h) * 0.08)
    src = np.float32([[0, 0], [w - 1, 0], [w - 1, h - 1], [0, h - 1]])
    dst = np.float32(
        [
            [rng.integers(0, margin), rng.integers(0, margin)],
            [w - 1 - rng.integers(0, margin), rng.integers(0, margin)],
            [w - 1 - rng.integers(0, margin), h - 1 - rng.integers(0, margin)],
            [rng.integers(0, margin), h - 1 - rng.integers(0, margin)],
        ]
    )
    mat = cv2.getPerspectiveTransform(src, dst)
    warped = cv2.warpPerspective(
        np.asarray(img.convert("RGB")), mat, (w, h), borderValue=(255, 255, 255)
    )
    return Image.fromarray(warped)


def _catalogue_page(img: Image.Image, is_sheet: bool) -> Image.Image:
    if is_sheet:
        return img.crop((40, 80, 520, 820)).resize((600, 900), Image.Resampling.BICUBIC)
    return _center_crop(img, 0.70)


def _room_scene(img: Image.Image, seed: int, is_sheet: bool) -> Image.Image:
    face = img
    if is_sheet:
        face = img.crop((20, 10, 520, 890))
    rng = np.random.default_rng(seed)
    room = np.full((900, 1400, 3), 55, dtype=np.uint8)
    room[:400, :] = (72, 80, 92)
    room[400:, :] = (118, 108, 96)
    tile = face.resize((480, 380), Image.Resampling.BICUBIC)
    arr = np.asarray(tile)
    y0, x0 = 470, 460
    room[y0 : y0 + arr.shape[0], x0 : x0 + arr.shape[1]] = arr
    noise = rng.integers(0, 16, room.shape, dtype=np.uint8)
    room = np.clip(room.astype(np.int16) + noise, 0, 255).astype(np.uint8)
    return Image.fromarray(room)


def generate_queries(
    tiles: list[CatalogTile],
    root: Path,
    *,
    seed: int = 7,
) -> list[QueryItem]:
    root = Path(root)
    qroot = root / "queries"
    qroot.mkdir(parents=True, exist_ok=True)
    rng = random.Random(seed)
    queries: list[QueryItem] = []

    for tile in tiles:
        img = Image.open(tile.path).convert("RGB")
        out_dir = qroot / f"id_{tile.tile_id:04d}"
        out_dir.mkdir(exist_ok=True)
        builders = {
            "original": lambda im=img: im.copy(),
            "crop_95": lambda im=img: _center_crop(im, 0.95),
            "crop_90": lambda im=img: _center_crop(im, 0.90),
            "crop_75": lambda im=img: _center_crop(im, 0.75),
            "crop_60": lambda im=img: _center_crop(im, 0.60),
            "crop_50": lambda im=img: _center_crop(im, 0.50),
            "center": lambda im=img: _center_crop(im, 0.66),
            "random_crop": lambda im=img: _random_crop(im, 0.55, rng),
            "corner": lambda im=img: _corner_crop(im, 600),
            "phone_screenshot": lambda im=img: _phone_screenshot(im),
            "jpeg30": None,
            "brightness_p30": lambda im=img: ImageEnhance.Brightness(im).enhance(1.30),
            "brightness_m30": lambda im=img: ImageEnhance.Brightness(im).enhance(0.70),
            "contrast_p30": lambda im=img: ImageEnhance.Contrast(im).enhance(1.30),
            "contrast_m30": lambda im=img: ImageEnhance.Contrast(im).enhance(0.70),
            "rotation_p10": lambda im=img: im.rotate(
                10, expand=True, fillcolor=(255, 255, 255)
            ),
            "rotation_m10": lambda im=img: im.rotate(
                -10, expand=True, fillcolor=(255, 255, 255)
            ),
            "perspective": lambda im=img: _perspective(im, tile.tile_id),
            "catalogue_page": lambda im=img: _catalogue_page(im, tile.is_sheet),
            "room_scene": lambda im=img: _room_scene(im, tile.tile_id, tile.is_sheet),
            "whatsapp": lambda im=img: _whatsapp_like(im),
        }
        for variant, fn in builders.items():
            path = out_dir / f"{variant}.jpg"
            if variant == "jpeg30":
                img.save(path, quality=30)
            elif variant == "whatsapp":
                # Extra compression pass
                tmp = fn()
                tmp.save(path, quality=28)
            else:
                fn().save(path, quality=92)
            queries.append(
                QueryItem(
                    tile_id=tile.tile_id,
                    variant=variant,
                    path=path,
                    material=tile.material,
                    is_sheet=tile.is_sheet,
                )
            )

    manifest = {
        "n_queries": len(queries),
        "variants": list(VARIANT_SPECS),
        "queries": [
            {
                "tile_id": q.tile_id,
                "variant": q.variant,
                "path": str(q.path),
                "material": q.material,
                "is_sheet": q.is_sheet,
            }
            for q in queries
        ],
    }
    (root / "query_manifest.json").write_text(json.dumps(manifest, indent=2))
    return queries
