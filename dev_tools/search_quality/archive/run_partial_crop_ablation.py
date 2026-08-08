#!/usr/bin/env python3
"""
ARCHIVED — PARTIAL_CROP view-strategy ablation (PR #41 investigation).

Production now hardcodes content + 82%% tighten only; see docs/PARTIAL_CROP_FIX.md.
This script is kept as a reusable ablation pattern but requires the removed
PARTIAL_CROP mode API to run — do not invoke from CI or regular tool paths.

Original usage:
  python dev_tools/search_quality/archive/run_partial_crop_ablation.py \\
      --out /tmp/partial_crop_ablation
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from src.ai.embedder import DINOv2Embedder
from src.ai.feature_extractor import FeatureExtractor
from src.ai.search_quality.fusion import FusionMethod
from src.ai.search_quality.query_views import (
    PARTIAL_CROP_MODES,
    set_partial_crop_mode,
)
from src.ai.search_quality.views import IndexStrategy

from dev_tools.search_quality.golden_dataset import build_golden_catalog
from dev_tools.search_quality.run_bakeoff import BakeoffEngine, metrics_to_dict

CROP_VARIANTS = ("crop_50", "crop_60", "crop_75", "crop_90", "crop_95")
MODES = tuple(sorted(PARTIAL_CROP_MODES))


def _crop_slice(payload: dict) -> dict[str, dict[str, float]]:
    out: dict[str, dict[str, float]] = {}
    for variant in CROP_VARIANTS:
        stats = payload.get("by_variant", {}).get(variant)
        if stats:
            out[variant] = {
                "recall@1": float(stats["recall@1"]),
                "recall@5": float(stats["recall@5"]),
                "n": int(stats["n"]),
            }
    return out


def _aggregate(rows: dict[str, dict[str, float]]) -> dict[str, float]:
    n = sum(v["n"] for v in rows.values()) or 1
    r1 = sum(v["recall@1"] * v["n"] for v in rows.values()) / n
    r5 = sum(v["recall@5"] * v["n"] for v in rows.values()) / n
    return {"recall@1": r1, "recall@5": r5, "n": n}


def run_ablation(
    out_dir: Path,
    *,
    n_tiles: int = 24,
    n_sheets: int = 12,
    modes: tuple[str, ...] = MODES,
) -> dict:
    out_dir.mkdir(parents=True, exist_ok=True)
    weights = Path("model_weights/dinov2-large/config.json")
    if not weights.is_file():
        raise FileNotFoundError("DINOv2 weights missing — run scripts/download_dinov2_model.py")

    items, queries = build_golden_catalog(out_dir / "golden", n_tiles=n_tiles, n_sheets=n_sheets)
    crop_queries = [q for q in queries if q.variant.startswith("crop_")]
    print(f"Golden catalog={len(items)} crop queries={len(crop_queries)}")

    emb = DINOv2Embedder()
    emb.load_model()
    fx = FeatureExtractor(embedder=emb)
    engine = BakeoffEngine(fx)
    index_path = out_dir / "idx_A_primary_only.index"
    mgr, index_s, _ = engine.index_strategy(items, IndexStrategy.A_PRIMARY_ONLY, index_path)
    print(f"Indexed {mgr.get_total_count()} vectors in {index_s:.1f}s")

    report: dict[str, object] = {
        "catalog_size": len(items),
        "n_crop_queries": len(crop_queries),
        "modes": {},
    }

    print("\nmode                      crop_50 R@1  crop_50 R@5  agg R@1  agg R@5")
    print("-" * 72)

    for mode in modes:
        set_partial_crop_mode(mode)
        metrics = engine.evaluate(
            mgr,
            crop_queries,
            fusion=FusionMethod.MAX,
            orb_verification=False,
        )
        payload = metrics_to_dict(metrics)
        crops = _crop_slice(payload)
        agg = _aggregate(crops)
        report["modes"][mode] = {"by_variant": crops, "aggregate_crop": agg}
        (out_dir / f"mode_{mode}.json").write_text(json.dumps(payload, indent=2))

        c50 = crops.get("crop_50", {"recall@1": 0.0, "recall@5": 0.0})
        print(
            f"{mode:<25} {c50['recall@1']:.4f}       {c50['recall@5']:.4f}       "
            f"{agg['recall@1']:.4f}   {agg['recall@5']:.4f}"
        )

    (out_dir / "ablation_report.json").write_text(json.dumps(report, indent=2))
    print(f"\nWrote {out_dir / 'ablation_report.json'}")
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=Path("/tmp/partial_crop_ablation"))
    parser.add_argument("--tiles", type=int, default=24)
    parser.add_argument("--sheets", type=int, default=12)
    parser.add_argument(
        "--mode",
        choices=MODES,
        action="append",
        help="Run one mode only (default: all four)",
    )
    args = parser.parse_args()
    modes = tuple(args.mode) if args.mode else MODES
    run_ablation(args.out, n_tiles=args.tiles, n_sheets=args.sheets, modes=modes)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
