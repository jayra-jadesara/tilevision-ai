#!/usr/bin/env python3
"""Standalone historical ablation — does not depend on production mode-switch API (removed after PR #42). Kept runnable for future similar investigations."""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import replace
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from src.ai.embedder import DINOv2Embedder
from src.ai.feature_extractor import FeatureExtractor
from src.ai.preprocess.fast_tile_crop import isolate_tile_region
from src.ai.preprocess.image_preprocessor import ImagePreprocessor
from src.ai.search_quality.fusion import FusionMethod
from src.ai.search_quality.query_analyzer import QueryAnalysis, QueryKind, analyze_query
from src.ai.search_quality.views import IndexStrategy

from dev_tools.search_quality.golden_dataset import build_golden_catalog
from dev_tools.search_quality.run_bakeoff import BakeoffEngine, metrics_to_dict

if TYPE_CHECKING:
    from collections.abc import Callable

# ---------------------------------------------------------------------------
# Historical PARTIAL_CROP mode API (removed from production query_views.py)
# ---------------------------------------------------------------------------

PARTIAL_CROP_MODES = frozenset(
    {
        "old_single",
        "new_primary_only",
        "new_primary_plus_tighten",
        "new_primary_plus_all",
    }
)

_partial_crop_mode: str = "new_primary_plus_tighten"


def get_partial_crop_mode() -> str:
    return _partial_crop_mode


def set_partial_crop_mode(mode: str) -> None:
    global _partial_crop_mode
    if mode not in PARTIAL_CROP_MODES:
        raise ValueError(
            f"Invalid partial_crop_mode {mode!r}; expected one of "
            f"{sorted(PARTIAL_CROP_MODES)}"
        )
    _partial_crop_mode = mode


def _partial_crop_max_views(mode: str, cap: int) -> int:
    if mode in {"old_single", "new_primary_only"}:
        return 1
    if mode == "new_primary_plus_tighten":
        return min(2, cap)
    return min(3, cap)


def _center_crop(image: Image.Image, ratio: float) -> Image.Image:
    w, h = image.size
    cw, ch = max(1, int(w * ratio)), max(1, int(h * ratio))
    left, top = (w - cw) // 2, (h - ch) // 2
    return image.crop((left, top, left + cw, top + ch))


def _pad_crop_border(image: Image.Image, ratio: float = 0.10) -> Image.Image:
    """Loosen a tight crop with edge-reflected padding (helps slightly-wide captures)."""
    import cv2

    rgb = np.asarray(image.convert("RGB"))
    h, w = rgb.shape[:2]
    pad_x = max(2, int(w * ratio))
    pad_y = max(2, int(h * ratio))
    padded = cv2.copyMakeBorder(
        rgb,
        pad_y,
        pad_y,
        pad_x,
        pad_x,
        cv2.BORDER_REFLECT_101,
    )
    return Image.fromarray(padded)


def _partial_crop_crops_for_mode(
    mode: str,
    working: Image.Image,
    analysis: QueryAnalysis,
    *,
    max_views: int,
) -> list[Image.Image]:
    """Build PARTIAL_CROP view list for one historical ablation mode."""
    crops: list[Image.Image] = []
    if mode == "old_single":
        high_frame = analysis.white_border_ratio >= 0.25
        if high_frame or ImagePreprocessor._looks_like_scene_photo(working):
            crops.append(isolate_tile_region(working).image)
        else:
            crops.append(
                ImagePreprocessor.crop_to_content_region(
                    working,
                    min_margin_ratio=0.05,
                )
            )
        return crops

    content = ImagePreprocessor.crop_to_content_region(
        working,
        min_margin_ratio=0.02,
    )
    crops.append(content)
    if mode == "new_primary_only":
        return crops
    if mode == "new_primary_plus_tighten" and max_views >= 2:
        crops.append(_center_crop(content, 0.82))
    elif mode == "new_primary_plus_all":
        if max_views >= 2:
            crops.append(_pad_crop_border(content))
        if max_views >= 3:
            crops.append(_center_crop(content, 0.82))
    return crops


def _dedupe_crops(
    crops: list[Image.Image],
    *,
    max_views: int,
    fallback: Image.Image,
) -> list[Image.Image]:
    unique: list[Image.Image] = []
    seen: list[tuple[int, int]] = []
    for crop in crops:
        key = crop.size
        if key in seen and len(unique) > 0:
            prev = np.asarray(unique[-1].resize(key), dtype=np.int16)
            cur = np.asarray(crop.resize(key), dtype=np.int16)
            if float(np.mean(np.abs(prev - cur))) < 4.0:
                continue
        seen.append(key)
        unique.append(crop)
        if len(unique) >= max_views:
            break
    return unique or [fallback]


# ---------------------------------------------------------------------------
# Monkeypatch helpers — historical PARTIAL_CROP without touching production
# ---------------------------------------------------------------------------

_PATCH_INSTALLED = False
_ORIG_PLAN: Callable[..., object] | None = None
_ORIG_COLLECT: Callable[..., tuple[QueryAnalysis, list[Image.Image]]] | None = None
_ORIG_EXTRACT_FOR_SEARCH: Callable[..., tuple[object, list[np.ndarray]]] | None = None


def _plan_query_views_ablation(
    analysis: QueryAnalysis,
    *,
    max_views_cap: int = 3,
):
    import src.ai.search_quality.query_views as qv

    assert _ORIG_PLAN is not None
    if analysis.kind != QueryKind.PARTIAL_CROP:
        return _ORIG_PLAN(analysis, max_views_cap=max_views_cap)
    base = _ORIG_PLAN(analysis, max_views_cap=max_views_cap)
    mode = get_partial_crop_mode()
    return replace(
        base,
        max_views=_partial_crop_max_views(mode, max_views_cap),
    )


def _collect_query_crop_pils_ablation(
    image: Image.Image,
    *,
    analysis: QueryAnalysis | None = None,
    max_views_cap: int = 3,
) -> tuple[QueryAnalysis, list[Image.Image]]:
    import src.ai.search_quality.query_views as qv

    assert _ORIG_COLLECT is not None
    rgb = ImagePreprocessor.to_rgb(image)
    rgb = ImagePreprocessor.trim_uniform_borders(rgb)
    analysis = analysis or analyze_query(rgb)
    if analysis.kind != QueryKind.PARTIAL_CROP:
        return _ORIG_COLLECT(image, analysis=analysis, max_views_cap=max_views_cap)

    plan = _plan_query_views_ablation(analysis, max_views_cap=max_views_cap)
    working = rgb
    mode = get_partial_crop_mode()
    crops = _partial_crop_crops_for_mode(
        mode,
        working,
        analysis,
        max_views=plan.max_views,
    )
    unique = _dedupe_crops(crops, max_views=plan.max_views, fallback=working)
    return analysis, unique


def _extract_for_search_ablation(
    self: FeatureExtractor,
    image_path: str,
    *,
    preloaded: Image.Image | None = None,
):
    import logging
    import time
    from pathlib import Path as PathLib

    from src.ai.feature_extractor import ExtractTimings
    from src.ai.search_quality.query_views import collect_query_crop_pils

    logger = logging.getLogger("tilevision.ai.feature_extractor")

    total_start = time.perf_counter()
    t0 = time.perf_counter()
    path = PathLib(image_path)
    image = ImagePreprocessor.to_rgb(
        preloaded if preloaded is not None else ImagePreprocessor.load(path)
    )

    analysis = analyze_query(image)
    partial_mode = get_partial_crop_mode()
    use_multi = (
        analysis.kind
        in {
            QueryKind.ROOM_SCENE,
            QueryKind.PHONE_SCREENSHOT,
        }
        or (
            analysis.kind == QueryKind.PARTIAL_CROP
            and partial_mode != "old_single"
        )
    ) and "tilevision_crops" not in path.as_posix().lower()

    if use_multi:
        max_cap = ImagePreprocessor._capped_query_max_views(3)
        _, crop_pils = collect_query_crop_pils(
            image,
            analysis=analysis,
            max_views_cap=max_cap,
        )
        original_width, original_height = image.size
        views = [
            ImagePreprocessor._finalize_query_pil(
                crop,
                original_width=original_width,
                original_height=original_height,
            )
            for crop in crop_pils
        ]
    else:
        views = ImagePreprocessor.prepare_query_views(
            path,
            max_views=1,
            preloaded=image,
        )

    preprocess_elapsed = time.perf_counter() - t0

    embeddings: list[np.ndarray] = []
    dinov2_elapsed = 0.0
    for view in views:
        t1 = time.perf_counter()
        emb = np.asarray(
            self._embedder.extract_from_preprocessed(view, for_query=True),
            dtype=np.float32,
        )
        dinov2_elapsed += time.perf_counter() - t1
        embeddings.append(emb)

    features = self._fuse_query_embeddings(
        views[0], [embeddings[0]], dinov2_elapsed
    )
    self._last_timings = ExtractTimings(
        preprocessing=preprocess_elapsed,
        dinov2=dinov2_elapsed,
        descriptors=self._last_timings.descriptors,
        total=time.perf_counter() - total_start,
    )
    logger.info(
        "Search extract (adaptive query): kind=%s views=%d "
        "preprocess=%.2fs dinov2=%.2fs total=%.2fs",
        analysis.kind.value,
        len(embeddings),
        preprocess_elapsed,
        dinov2_elapsed,
        self._last_timings.total,
    )
    return features, embeddings


def _install_ablation_patches() -> None:
    global _PATCH_INSTALLED, _ORIG_PLAN, _ORIG_COLLECT, _ORIG_EXTRACT_FOR_SEARCH
    if _PATCH_INSTALLED:
        return

    import src.ai.search_quality.query_views as qv

    _ORIG_PLAN = qv.plan_query_views
    _ORIG_COLLECT = qv.collect_query_crop_pils
    _ORIG_EXTRACT_FOR_SEARCH = FeatureExtractor.extract_for_search

    qv.plan_query_views = _plan_query_views_ablation
    qv.collect_query_crop_pils = _collect_query_crop_pils_ablation
    FeatureExtractor.extract_for_search = _extract_for_search_ablation
    _PATCH_INSTALLED = True


# ---------------------------------------------------------------------------
# Ablation runner (unchanged methodology)
# ---------------------------------------------------------------------------

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
    _install_ablation_patches()

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
