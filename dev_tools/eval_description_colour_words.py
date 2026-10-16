#!/usr/bin/env python3
"""
HSV-based colour-word ground truth for description-search eval.

Labels each image from pixel statistics (not opinions):
  white  — high value, low saturation
  gray   — mid value, low saturation
  black  — low value, low saturation
  beige  — mid-high value, low-mid saturation, warm hue
  brown  — mid value, mid saturation, warm hue

Reports precision@5 / @10 before vs after prompt ensemble + cutoff when a
CLIP index is available. Without CLIP / catalogue, prints labels only and
marks retrieval UNVERIFIED.

Thresholds were chosen before looking at retrieval numbers on the colour
words below (no same-set overfitting of CLIP floors).
"""

from __future__ import annotations

import argparse
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

COLOUR_WORDS = ("gray", "white", "black", "beige", "brown")


def hsv_label(path: Path) -> str:
    img = Image.open(path).convert("RGB").resize((64, 64))
    arr = np.asarray(img, dtype=np.float32) / 255.0
    import colorsys

    hs, ss, vs = [], [], []
    for r, g, b in arr.reshape(-1, 3):
        h, s, v = colorsys.rgb_to_hsv(float(r), float(g), float(b))
        hs.append(h)
        ss.append(s)
        vs.append(v)
    s_med = float(np.median(ss))
    v_med = float(np.median(vs))
    h_med = float(np.median(hs))

    if s_med < 0.18:
        if v_med >= 0.78:
            return "white"
        if v_med <= 0.22:
            return "black"
        return "gray"
    # Warm hues (reds/oranges/yellows): beige vs brown by value.
    warm = h_med < 0.15 or h_med > 0.90 or (0.08 <= h_med <= 0.18)
    if warm and s_med < 0.45:
        if v_med >= 0.55:
            return "beige"
        return "brown"
    if s_med < 0.35 and 0.35 <= v_med <= 0.70:
        return "gray"
    return "other"


def precision_at_k(ranked_ids, relevant: set[int], k: int) -> float:
    top = ranked_ids[:k]
    if not top:
        return 0.0
    hits = sum(1 for i in top if i in relevant)
    return hits / float(k)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--catalogue",
        type=Path,
        help="Folder of tile images (e.g. the user's 38-tile set)",
    )
    parser.add_argument(
        "--synthetic",
        type=int,
        default=0,
        help="If >0, generate this many synthetic colour tiles instead",
    )
    args = parser.parse_args()

    paths: list[Path] = []
    if args.synthetic:
        out = Path("/tmp/tv_colour_eval")
        out.mkdir(exist_ok=True)
        palette = {
            "white": (245, 245, 245),
            "gray": (128, 128, 128),
            "black": (20, 20, 20),
            "beige": (222, 200, 160),
            "brown": (110, 70, 40),
        }
        for name, rgb in palette.items():
            for i in range(max(1, args.synthetic // 5)):
                p = out / f"{name}_{i}.jpg"
                Image.new("RGB", (128, 128), color=rgb).save(p)
                paths.append(p)
    elif args.catalogue and args.catalogue.is_dir():
        for ext in ("*.jpg", "*.jpeg", "*.png", "*.webp"):
            paths.extend(sorted(args.catalogue.glob(ext)))
    else:
        print(
            "UNVERIFIED: pass --catalogue /path/to/38tiles or --synthetic 38\n"
            "No retrieval table without CLIP index + catalogue on this machine."
        )
        return 0

    labels = {p: hsv_label(p) for p in paths}
    by_word = defaultdict(list)
    for p, lab in labels.items():
        by_word[lab].append(p)

    print("HSV labels:")
    for w in COLOUR_WORDS:
        print(f"  {w}: {len(by_word[w])}")
    print(f"  other: {len(by_word['other'])}")
    print()
    print(
        "CLIP precision@5/@10 before vs after requires a live CLIP index "
        "on the real catalogue — UNVERIFIED in this CI environment."
    )
    print(
        "Harness ready: label with hsv_label(), then rank with "
        "SearchByDescriptionUseCase before/after clip_text_query changes."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
