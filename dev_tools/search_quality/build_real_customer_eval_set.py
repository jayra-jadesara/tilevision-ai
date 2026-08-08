#!/usr/bin/env python3
"""
Build a representative real-customer eval set for release gating.

Generates confusable white/marble catalog pairs plus query variants that mirror
showroom captures: partial crops, WhatsApp-style compression, phone screenshots,
and catalogue-page crops. Output lives under eval/real_customer_release/ and is
safe to commit (synthetic stand-ins — not customer PII).

Usage:
  python3 dev_tools/search_quality/build_real_customer_eval_set.py
  python3 dev_tools/search_quality/build_real_customer_eval_set.py --out eval/real_customer_release
"""

from __future__ import annotations

import argparse
import io
import json
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image, ImageEnhance, ImageOps

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from dev_tools.search_quality.golden_dataset import (
    _catalogue_page_crop,
    _center_crop,
    _make_marble,
    _make_sheet,
    _phone_screenshot,
)

# Four confusable pairs — same base/tint, different vein geometry (seed).
CONFUSABLE_PAIRS: tuple[tuple[int, int, tuple[int, int, int]], ...] = (
    (9101, 9102, (228, 228, 228)),
    (9201, 9202, (232, 230, 226)),
    (9301, 9302, (225, 228, 232)),
    (9401, 9402, (230, 226, 224)),
)


@dataclass(frozen=True, slots=True)
class QuerySpec:
    suffix: str
    query_kind: str
    builder: str  # key into _BUILDERS


def _whatsapp(img: Image.Image) -> Image.Image:
    """WhatsApp-style recompress + slight downscale."""
    small = img.copy()
    if max(small.size) > 1280:
        small.thumbnail((1280, 1280), Image.Resampling.LANCZOS)
    buf = io.BytesIO()
    small.save(buf, format="JPEG", quality=42, optimize=True)
    buf.seek(0)
    return Image.open(buf).convert("RGB")


def _low_quality_jpeg(img: Image.Image) -> Image.Image:
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=28, optimize=True)
    buf.seek(0)
    return Image.open(buf).convert("RGB")


def _crop_600(img: Image.Image) -> Image.Image:
    return _center_crop(img, 0.50).resize((600, 600), Image.Resampling.LANCZOS)


def _crop_600x1200(img: Image.Image) -> Image.Image:
    w, h = img.size
    cw = max(1, int(w * 0.55))
    ch = max(1, int(h * 0.85))
    left = (w - cw) // 2
    top = (h - ch) // 2
    return img.crop((left, top, left + cw, top + ch)).resize(
        (600, 1200), Image.Resampling.LANCZOS
    )


def _perspective(img: Image.Image) -> Image.Image:
    import cv2

    arr = np.asarray(img.convert("RGB"))
    h, w = arr.shape[:2]
    src = np.float32([[0, 0], [w - 1, 0], [w - 1, h - 1], [0, h - 1]])
    dst = np.float32([[18, 12], [w - 28, 8], [w - 12, h - 18], [8, h - 8]])
    m = cv2.getPerspectiveTransform(src, dst)
    warped = cv2.warpPerspective(arr, m, (w, h), borderValue=(255, 255, 255))
    return Image.fromarray(warped)


QUERY_SPECS: tuple[QuerySpec, ...] = (
    QuerySpec("original", "original", "original"),
    QuerySpec("crop_600", "crop_600x600", "crop_600"),
    QuerySpec("crop_75", "crop_600x1200", "crop_600x1200"),
    QuerySpec("whatsapp", "whatsapp", "whatsapp"),
    QuerySpec("phone", "phone_photo", "phone"),
    QuerySpec("jpeg_low", "low_quality_jpeg", "jpeg_low"),
    QuerySpec("perspective", "perspective_distortion", "perspective"),
)


def _apply_builder(img: Image.Image, key: str, *, kind: str) -> Image.Image:
    if key == "original":
        return img.copy()
    if key == "crop_600":
        return _crop_600(img)
    if key == "crop_600x1200":
        return _crop_600x1200(img)
    if key == "whatsapp":
        return _whatsapp(img)
    if key == "phone":
        return _phone_screenshot(img)
    if key == "jpeg_low":
        return _low_quality_jpeg(img)
    if key == "perspective":
        return _perspective(img)
    if key == "catalogue_page":
        return _catalogue_page_crop(img, kind)
    raise KeyError(key)


def build_eval_set(out_root: Path) -> Path:
    out_root = out_root.resolve()
    catalog_dir = out_root / "real_catalog"
    query_dir = out_root / "real_queries"
    catalog_dir.mkdir(parents=True, exist_ok=True)
    query_dir.mkdir(parents=True, exist_ok=True)

    manifest_path = out_root.parent / "real_customer_release.jsonl"
    lines: list[str] = []
    tile_id = 1

    # Two marketing sheets (catalogue_page queries).
    for i, (seed_a, _seed_b, tint) in enumerate(CONFUSABLE_PAIRS[:2]):
        slab = _make_marble(900, 450, seed=seed_a + 100, base=235, tint=tint)
        sheet = _make_sheet(slab, name=f"MARBLE-{i + 1:02d}")
        cat_path = catalog_dir / f"sheet_{tile_id:03d}.jpg"
        sheet.save(cat_path, quality=92)
        for spec in QUERY_SPECS:
            if spec.query_kind == "perspective_distortion":
                continue
            qimg = _apply_builder(sheet, spec.builder, kind="sheet")
            qpath = query_dir / f"sheet{tile_id:03d}_{spec.suffix}.jpg"
            qimg.save(qpath, quality=90)
            rec = {
                "query_path": str(qpath.relative_to(manifest_path.parent)),
                "true_tile_id": tile_id,
                "query_kind": spec.query_kind,
                "catalog_path": str(cat_path.relative_to(manifest_path.parent)),
            }
            lines.append(json.dumps(rec))
        # catalogue_page variant
        qimg = _apply_builder(sheet, "catalogue_page", kind="sheet")
        qpath = query_dir / f"sheet{tile_id:03d}_catalogue_page.jpg"
        qimg.save(qpath, quality=90)
        lines.append(
            json.dumps(
                {
                    "query_path": str(qpath.relative_to(manifest_path.parent)),
                    "true_tile_id": tile_id,
                    "query_kind": "catalogue_page",
                    "catalog_path": str(cat_path.relative_to(manifest_path.parent)),
                }
            )
        )
        tile_id += 1

    # Confusable marble tile pairs.
    for pair_idx, (seed_a, seed_b, tint) in enumerate(CONFUSABLE_PAIRS):
        for offset, seed in enumerate((seed_a, seed_b)):
            tile = _make_marble(1200, 1200, seed=seed, base=tint[0], tint=(0, 0, 0))
            # Slight per-tile tint shift so pairs are confusable but not identical.
            arr = np.asarray(tile, dtype=np.int16)
            arr += np.array([offset * 3 - 2, offset * 2 - 1, offset * -2 + 1], dtype=np.int16)
            tile = Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8))
            cat_path = catalog_dir / f"marble_p{pair_idx + 1}_{'a' if offset == 0 else 'b'}.jpg"
            tile.save(cat_path, quality=92)
            for spec in QUERY_SPECS:
                qimg = _apply_builder(tile, spec.builder, kind="tile")
                qpath = query_dir / f"tile{tile_id:03d}_{spec.suffix}.jpg"
                qimg.save(qpath, quality=90)
                lines.append(
                    json.dumps(
                        {
                            "query_path": str(qpath.relative_to(manifest_path.parent)),
                            "true_tile_id": tile_id,
                            "query_kind": spec.query_kind,
                            "catalog_path": str(cat_path.relative_to(manifest_path.parent)),
                        }
                    )
                )
            tile_id += 1

    manifest_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    meta = {
        "n_catalog_tiles": tile_id - 1,
        "n_queries": len(lines),
        "n_confusable_pairs": len(CONFUSABLE_PAIRS),
        "query_kinds": sorted({json.loads(line)["query_kind"] for line in lines}),
        "manifest": str(manifest_path),
    }
    (out_root / "manifest_meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(json.dumps(meta, indent=2))
    return manifest_path


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("eval/real_customer_release"),
        help="Output root (real_catalog/ + real_queries/ subdirs)",
    )
    args = parser.parse_args()
    build_eval_set(args.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
