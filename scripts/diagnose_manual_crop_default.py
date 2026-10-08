#!/usr/bin/env python3
"""
Before/after evidence for Manual Crop UX (Auto-aligned default).

Reproduces the client failure class without needing ``xx.jpg.jpeg``:

  - **Before (old UX bias):** empty canvas → user draws a tight center box
    (~35% area). Edge/pattern vs the catalog panel collapse — same shape as
    the real Manual Crop row (edge≈0.276, pat≈0.266).
  - **After (new default):** opening Manual Crop pre-fills Auto Crop's box
    (full frame on clean / over-crop-guarded close-ups). Descriptors track
    Auto Crop.

Usage::

  python scripts/diagnose_manual_crop_default.py --out /tmp/manual_crop_diag

On a real client machine, also re-run ``diagnose_crop_tool_search.py`` after
opening Manual Crop once (accept the suggested box or expand a tight one)
and compare ``last_manual.jpg`` to Auto/Precise.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.ai.descriptors.color_descriptor import ColorDescriptor
from src.ai.descriptors.edge_descriptor import EdgeDescriptor
from src.ai.descriptors.pattern_descriptor import PatternDescriptor
from src.ai.descriptors.texture_descriptor import TextureDescriptor
from src.ai.feature_extractor import FeatureExtractor
from src.ai.preprocess.fast_tile_crop import save_auto_tile_crop
from src.ai.preprocess.index_primary import prepare_index_primary
from src.presentation.views.manual_crop_ux import suggest_manual_default_box
from tests.fake_ai import FakeEmbedder
from tests.test_crop_search_consistency import _make_catalog_sheet


def _components(feat, index) -> dict[str, float]:
    return {
        "color": ColorDescriptor.similarity(feat.color_histogram, index.color_histogram),
        "tex": TextureDescriptor.similarity(
            feat.texture_histogram, index.texture_histogram
        ),
        "edge": EdgeDescriptor.similarity(feat.edge_histogram, index.edge_histogram),
        "pat": PatternDescriptor.similarity(
            feat.pattern_features, index.pattern_features
        ),
    }


def _make_gradient_closeup(base: Image.Image) -> Image.Image:
    arr = np.asarray(base.convert("RGB")).astype(np.float32)
    height = arr.shape[0]
    for y in range(height):
        arr[y] *= 0.50 + 0.50 * (y / max(1, height - 1))
    return Image.fromarray(arr.astype(np.uint8))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=Path("manual_crop_diag_out"))
    parser.add_argument(
        "--tight-keep",
        type=float,
        default=0.35,
        help="Area fraction for the simulated old-UX tight center crop",
    )
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    fixture_dir = args.out / "fixture"
    fixture_dir.mkdir(parents=True, exist_ok=True)

    sheet_path, crop_path = _make_catalog_sheet(fixture_dir)
    base = Image.open(crop_path).convert("RGB")
    # Lighting-gradient close-up matches the client over-crop class.
    source = _make_gradient_closeup(base)
    source_path = args.out / "xx_gradient.jpg"
    source.save(source_path, quality=95)

    sheet_copy = args.out / "sheet_PGYS2319.jpg"
    sheet_copy.write_bytes(Path(sheet_path).read_bytes())

    index_pre = prepare_index_primary(sheet_copy).primary
    fx = FeatureExtractor(embedder=FakeEmbedder())
    index = fx.extract_from_preprocessed(index_pre, for_query=False)

    # BEFORE: floor_band over-crop (same keep≈32% damage class as a tight
    # Manual selection / pre-PR-#68 Auto). Geometric center crops on the
    # synthetic fixture do not reproduce client edge/pat collapse.
    from src.ai.preprocess.fast_tile_crop import isolate_tile_region

    isolated = isolate_tile_region(source)
    tight = isolated.image
    tight_path = args.out / "manual_tight_before.jpg"
    tight.save(tight_path, quality=95)

    suggestion = suggest_manual_default_box(source)
    default = source.crop(suggestion.box)
    default_path = args.out / "manual_default_after.jpg"
    default.save(default_path, quality=95)

    auto_path, auto = save_auto_tile_crop(source_path)
    auto_keep = args.out / "last_autocrop_ref.jpg"
    auto.image.save(auto_keep, quality=95)

    rows = []
    for tag, path in [
        ("manual_tight_before", tight_path),
        ("manual_default_after", default_path),
        ("auto_crop_ref", auto_keep),
    ]:
        feat, _ = fx.extract_for_search(str(path))
        comps = _components(feat, index)
        rows.append({"tag": tag, "size": list(Image.open(path).size), **comps})

    report = {
        "source": str(source_path),
        "suggestion": {
            "method": suggestion.method,
            "box": list(suggestion.box),
            "keep_ratio": suggestion.keep_ratio,
        },
        "auto_method": auto.method,
        "rows": rows,
        "note": (
            "FakeEmbedder: emb column omitted (not comparable to client DINOv2). "
            "color/tex/edge/pat use production descriptors vs index panel."
        ),
    }
    (args.out / "manual_crop_default_report.json").write_text(json.dumps(report, indent=2))

    print("=== Manual Crop default before/after (fixture gradient close-up) ===")
    print(
        f"tight(before) method={isolated.method} "
        f"size={tight.size[0]}x{tight.size[1]}"
    )
    print(f"suggestion method={suggestion.method} keep={suggestion.keep_ratio:.2f}")
    print(f"auto method={auto.method}")
    print(f"{'tag':<24} {'size':>12} {'color':>7} {'tex':>7} {'edge':>7} {'pat':>7}")
    for r in rows:
        print(
            f"{r['tag']:<24} {r['size'][0]}x{r['size'][1]:>4} "
            f"{r['color']:7.3f} {r['tex']:7.3f} {r['edge']:7.3f} {r['pat']:7.3f}"
        )
    after = rows[1]
    auto_row = rows[2]
    print(
        "\nPrimary success metric on this fixture: Manual default ≡ Auto Crop "
        f"(color Δ={after['color']-auto_row['color']:+.3f} "
        f"edge Δ={after['edge']-auto_row['edge']:+.3f} "
        f"pat Δ={after['pat']-auto_row['pat']:+.3f})."
    )
    print(
        "Note: synthetic sheet panels can make a tight crop look *better* vs the "
        "index panel than full-frame; client marble xx.jpg showed the opposite "
        "(Manual edge/pat collapse). Use diagnose_crop_tool_search.py on the "
        "client for the real emb/final table."
    )
    print(f"Wrote {args.out / 'manual_crop_default_report.json'}")
    print(
        "\nClient follow-up: open Manual Crop on xx.jpg.jpeg (pre-filled box), "
        "Search, then run diagnose_crop_tool_search.py and compare last_manual "
        "to Auto/Precise."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
