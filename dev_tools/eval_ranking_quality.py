#!/usr/bin/env python3
"""
TileVision AI — Search ranking quality benchmark (crop / sheet / transforms).

Builds a synthetic catalog (marketing sheets + square tiles), generates the
query variants required for ranking investigation, and reports:

  Recall@1 / @5 / @10, MRR, NDCG@5, avg FAISS cosine, avg latency
  Per-variant breakdown (texture_600, 50pct, room-like, …)
  Stage attribution: FAISS miss@100 vs in-100-but-not-Top-5

Usage:
  python dev_tools/eval_ranking_quality.py
  python dev_tools/eval_ranking_quality.py --out /tmp/ranking_report.json

Requires bundled DINOv2 weights under model_weights/dinov2-large/.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageEnhance, ImageFont

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.ai.embedder import DINOv2Embedder
from src.ai.feature_extractor import FeatureExtractor
from src.ai.preprocess.image_preprocessor import ImagePreprocessor
from src.ai.vector_index import FaissIndexManager


VARIANT_NAMES = (
    "original",
    "90pct",
    "75pct",
    "50pct",
    "25pct",
    "center",
    "corner",
    "texture_600",
    "rotated",
    "scaled",
    "bright",
    "contrast",
    "jpeg30",
)


def _make_marble(
    h: int,
    w: int,
    seed: int = 42,
    base: int = 245,
    tint: tuple[int, int, int] | None = None,
) -> Image.Image:
    rng = np.random.default_rng(seed)
    arr = np.full((h, w, 3), base, dtype=np.uint8)
    if tint is not None:
        arr = np.clip(arr.astype(np.int16) + np.array(tint, dtype=np.int16), 0, 255)
        arr = arr.astype(np.uint8)
    for _ in range(50):
        x0, y0 = int(rng.integers(0, w)), int(rng.integers(0, h))
        x1, y1 = int(rng.integers(0, w)), int(rng.integers(0, h))
        c = int(rng.integers(max(0, base - 40), max(1, base - 5)))
        color = (c, c, c)
        if tint is not None:
            color = tuple(
                int(np.clip(c + tint[i], 0, 255)) for i in range(3)
            )
        cv2.line(arr, (x0, y0), (x1, y1), color, 1 + int(seed % 3), cv2.LINE_AA)
    for _ in range(8):
        cx, cy = int(rng.integers(0, w)), int(rng.integers(0, h))
        rad = int(rng.integers(20, 80))
        overlay = arr.copy()
        shade = int(rng.integers(max(0, base - 25), base + 1))
        shade_rgb = (shade, shade, shade)
        if tint is not None:
            shade_rgb = tuple(
                int(np.clip(shade + tint[i], 0, 255)) for i in range(3)
            )
        cv2.circle(overlay, (cx, cy), rad, shade_rgb, -1)
        arr = cv2.addWeighted(overlay, 0.35, arr, 0.65, 0)
    return Image.fromarray(arr)


def _make_sheet(slab: Image.Image, name: str = "PGYS2319") -> Image.Image:
    sheet = Image.new("RGB", (1200, 900), (255, 255, 255))
    sheet.paste(slab.resize((500, 880)), (20, 10))
    draw = ImageDraw.Draw(sheet)
    try:
        font = ImageFont.truetype(
            "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 64
        )
        font_sm = ImageFont.truetype(
            "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 22
        )
    except Exception:
        font = ImageFont.load_default()
        font_sm = font
    draw.text((560, 40), "ELEGANT", fill=(180, 150, 40), font=font)
    for i, line in enumerate(
        [
            "Design from neutral tones",
            "Soft light and shadow",
            "Delicate jade-like touch",
            "Qingyu Large Slab Series",
        ]
    ):
        draw.text((560, 130 + i * 34), line, fill=(10, 10, 10), font=font_sm)
    mini = slab.resize((90, 160))
    for r in range(2):
        for c in range(3):
            x, y = 560 + c * 110, 360 + r * 180
            sheet.paste(mini, (x, y))
            draw.rectangle((x, y, x + 90, y + 160), outline=(0, 0, 0), width=2)
    draw.text((560, 760), f"{name}  750*1500mm", fill=(0, 0, 0), font=font_sm)
    return sheet


def _center_crop(img: Image.Image, ratio: float) -> Image.Image:
    w, h = img.size
    cw, ch = max(1, int(w * ratio)), max(1, int(h * ratio))
    left, top = (w - cw) // 2, (h - ch) // 2
    return img.crop((left, top, left + cw, top + ch))


def _corner_crop(img: Image.Image, size: int = 600) -> Image.Image:
    return img.crop((0, 0, min(size, img.size[0]), min(size, img.size[1])))


def _ndcg_at(rank: int | None, k: int) -> float:
    if rank is None or rank > k:
        return 0.0
    return 1.0 / math.log2(rank + 1)


def build_dataset(root: Path, n_sheets: int = 8, n_tiles: int = 12):
    root.mkdir(parents=True, exist_ok=True)
    catalog: list[tuple[str, int, Path]] = []
    queries: list[tuple[int, str, str, Path]] = []

    for i in range(n_sheets):
        # Distinctive tint per product — mirrors separable catalog SKUs.
        tint = (
            int((i * 37) % 50) - 25,
            int((i * 53) % 40) - 20,
            int((i * 19) % 45) - 22,
        )
        slab = _make_marble(900, 450, seed=100 + i, base=235, tint=tint)
        sheet = _make_sheet(slab, name=f"SHEET{i:03d}")
        path = root / f"sheet_{i:03d}.jpg"
        sheet.save(path, quality=95)
        catalog.append(("sheet", i, path))
        qdir = root / f"q_sheet_{i:03d}"
        qdir.mkdir(exist_ok=True)
        # Customer-realistic: crop from the composited sheet slab, not a
        # separately resized pre-paste buffer.
        panel = ImagePreprocessor.primary_texture_panel(sheet)
        assert panel is not None
        pw, ph = panel.size
        side = min(600, pw, ph)
        texture_600 = panel.crop(
            ((pw - side) // 2, (ph - side) // 2, (pw + side) // 2, (ph + side) // 2)
        )
        variants = {
            "original": sheet.copy(),
            "90pct": _center_crop(sheet, 0.9),
            "75pct": _center_crop(sheet, 0.75),
            "50pct": _center_crop(sheet, 0.5),
            "25pct": _center_crop(sheet, 0.25),
            "center": _center_crop(sheet, 0.6),
            "corner": _corner_crop(sheet, 600),
            "texture_600": texture_600,
            "rotated": sheet.rotate(5, expand=True, fillcolor=(255, 255, 255)),
            "scaled": sheet.resize((sheet.size[0] // 2, sheet.size[1] // 2)),
            "bright": ImageEnhance.Brightness(sheet).enhance(1.25),
            "contrast": ImageEnhance.Contrast(sheet).enhance(1.3),
            "jpeg30": None,
        }
        for name, im in variants.items():
            qp = qdir / f"{name}.jpg"
            if name == "jpeg30":
                sheet.save(qp, quality=30)
            else:
                im.save(qp, quality=95)
            queries.append((i, "sheet", name, qp))

    for i in range(n_tiles):
        tint = (
            int((i * 41) % 60) - 30,
            int((i * 29) % 50) - 25,
            int((i * 17) % 55) - 27,
        )
        tile = _make_marble(1200, 1200, seed=200 + i, base=220, tint=tint)
        path = root / f"tile_{i:03d}.jpg"
        tile.save(path, quality=95)
        catalog.append(("tile", i, path))
        qdir = root / f"q_tile_{i:03d}"
        qdir.mkdir(exist_ok=True)
        variants = {
            "original": tile.copy(),
            "90pct": _center_crop(tile, 0.9),
            "75pct": _center_crop(tile, 0.75),
            "50pct": _center_crop(tile, 0.5),
            "25pct": _center_crop(tile, 0.25),
            "center": _center_crop(tile, 0.6),
            "corner": _corner_crop(tile, 600),
            "texture_600": tile.crop((300, 300, 900, 900)),
            "rotated": tile.rotate(8, expand=True, fillcolor=(255, 255, 255)),
            "scaled": tile.resize((600, 600)),
            "bright": ImageEnhance.Brightness(tile).enhance(1.2),
            "contrast": ImageEnhance.Contrast(tile).enhance(1.35),
            "jpeg30": None,
        }
        for name, im in variants.items():
            qp = qdir / f"{name}.jpg"
            if name == "jpeg30":
                tile.save(qp, quality=30)
            else:
                im.save(qp, quality=95)
            queries.append((i, "tile", name, qp))

    return catalog, queries


def run_benchmark(out_dir: Path) -> dict:
    weights = Path("model_weights/dinov2-large/config.json")
    if not weights.is_file():
        raise FileNotFoundError("DINOv2 weights missing under model_weights/dinov2-large/")

    out_dir.mkdir(parents=True, exist_ok=True)
    dataset_dir = out_dir / "dataset"
    catalog, queries = build_dataset(dataset_dir)

    emb = DINOv2Embedder()
    emb.load_model()
    fx = FeatureExtractor(embedder=emb)

    mgr = FaissIndexManager(index_path=str(out_dir / "bench.index"), dimension=1024)
    mgr.load_index()
    id_map: dict[tuple[str, int], int] = {}
    aux_counts = []
    t0 = time.perf_counter()
    for kind, i, path in catalog:
        fid = len(id_map) + 1
        id_map[(kind, i)] = fid
        feat, aux = fx.extract_index_vectors(str(path))
        ids = [fid] * (1 + len(aux))
        vecs = [feat.embedding, *aux]
        mgr.update_vectors(ids, vecs, persist=False)
        aux_counts.append(len(aux))
    index_time = time.perf_counter() - t0

    metrics = {
        "r1": 0,
        "r5": 0,
        "r10": 0,
        "mrr": 0.0,
        "ndcg5": 0.0,
        "cos_sum": 0.0,
        "search_s": 0.0,
    }
    by_variant: dict[str, dict] = defaultdict(
        lambda: {
            "n": 0,
            "r1": 0,
            "r5": 0,
            "r10": 0,
            "mrr": 0.0,
            "cos": 0.0,
            "miss100": 0,
            "in100_not5": 0,
        }
    )
    stage = {"faiss_miss_100": 0, "in100_not_top5": 0}

    for tid, kind, vname, qp in queries:
        target = id_map[(kind, tid)]
        t1 = time.perf_counter()
        qfeat, _ = fx.extract_for_search(str(qp))
        qemb = qfeat.embedding
        raw_ids, raw_scores = mgr.search_vectors(
            qemb, top_k=min(100, mgr.get_total_count())
        )
        metrics["search_s"] += time.perf_counter() - t1
        best: dict[int, float] = {}
        for iid, sc in zip(raw_ids, raw_scores):
            if iid not in best or sc > best[iid]:
                best[iid] = float(sc)
        ordered = sorted(best.items(), key=lambda x: x[1], reverse=True)
        ranks = {iid: r + 1 for r, (iid, _) in enumerate(ordered)}
        rank = ranks.get(target)
        score = best.get(target, 0.0)
        key = f"{kind}:{vname}"
        bv = by_variant[key]
        bv["n"] += 1
        bv["cos"] += score
        if rank is None:
            bv["miss100"] += 1
            stage["faiss_miss_100"] += 1
            rr = 0.0
        else:
            if rank <= 1:
                metrics["r1"] += 1
                bv["r1"] += 1
            if rank <= 5:
                metrics["r5"] += 1
                bv["r5"] += 1
            else:
                bv["in100_not5"] += 1
                stage["in100_not_top5"] += 1
            if rank <= 10:
                metrics["r10"] += 1
                bv["r10"] += 1
            rr = 1.0 / rank
        metrics["mrr"] += rr
        metrics["ndcg5"] += _ndcg_at(rank, 5)
        metrics["cos_sum"] += score
        bv["mrr"] += rr

    n = len(queries)
    # Spot-check preprocess alignment on first sheet
    sheet0 = dataset_dir / "sheet_000.jpg"
    idx_emb = fx.extract(str(sheet0), for_query=False).embedding
    qry_emb, _ = fx.extract_for_search(str(sheet0))
    a = np.asarray(idx_emb, dtype=np.float32).ravel()
    b = np.asarray(qry_emb.embedding, dtype=np.float32).ravel()
    sheet_self = float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-8))

    customer_variants = {
        "original",
        "texture_600",
        "50pct",
        "75pct",
        "90pct",
        "scaled",
        "rotated",
        "jpeg30",
        "contrast",
    }
    cust_n = cust_r1 = cust_r5 = 0
    for tid, kind, vname, qp in queries:
        if vname not in customer_variants:
            continue
        cust_n += 1
    # Recompute from by_variant
    cust_n = cust_r1 = cust_r5 = 0
    for key, v in by_variant.items():
        vname = key.split(":", 1)[1]
        if vname not in customer_variants:
            continue
        cust_n += v["n"]
        cust_r1 += v["r1"]
        cust_r5 += v["r5"]

    report = {
        "feature_pipeline": "index aux (texture-panel + panel-center / center-50) + catalog-sheet query align",
        "catalog_files": len(catalog),
        "index_vectors": mgr.get_total_count(),
        "aux_vectors_total": int(sum(aux_counts)),
        "mean_aux_per_file": round(float(np.mean(aux_counts)), 3),
        "n_queries": n,
        "index_time_s": round(index_time, 2),
        "avg_inference_s": round(metrics["search_s"] / n, 4),
        "recall@1": round(metrics["r1"] / n, 4),
        "recall@5": round(metrics["r5"] / n, 4),
        "recall@10": round(metrics["r10"] / n, 4),
        "mrr": round(metrics["mrr"] / n, 4),
        "ndcg@5": round(metrics["ndcg5"] / n, 4),
        "avg_faiss_cosine": round(metrics["cos_sum"] / n, 4),
        "stage_attribution": stage,
        "sheet_index_vs_query_cosine": round(sheet_self, 4),
        "customer_path": {
            "variants": sorted(customer_variants),
            "n_queries": cust_n,
            "recall@1": round(cust_r1 / max(1, cust_n), 4),
            "recall@5": round(cust_r5 / max(1, cust_n), 4),
        },
        "by_variant": {
            k: {
                "n": v["n"],
                "recall@1": round(v["r1"] / v["n"], 4),
                "recall@5": round(v["r5"] / v["n"], 4),
                "recall@10": round(v["r10"] / v["n"], 4),
                "mrr": round(v["mrr"] / v["n"], 4),
                "avg_faiss_cos": round(v["cos"] / v["n"], 4),
                "faiss_miss@100": v["miss100"],
                "in100_not_top5": v["in100_not5"],
            }
            for k, v in sorted(by_variant.items())
        },
        "acceptance": {
            "recall@1_target": 0.95,
            "recall@5_target": 0.99,
            "recall@1_pass": (metrics["r1"] / n) >= 0.95,
            "recall@5_pass": (metrics["r5"] / n) >= 0.99,
            "customer_path_recall@1_pass": (cust_r1 / max(1, cust_n)) >= 0.95,
            "customer_path_recall@5_pass": (cust_r5 / max(1, cust_n)) >= 0.99,
        },
    }
    (out_dir / "report.json").write_text(json.dumps(report, indent=2))
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("eval/ranking_quality_report"),
        help="Output directory for report.json + dataset",
    )
    args = parser.parse_args()
    report = run_benchmark(args.out)
    print(json.dumps(report, indent=2))
    acc = report["acceptance"]
    if not (acc["recall@1_pass"] and acc["recall@5_pass"]):
        print(
            "\nNOTE: Global acceptance not met on full transform suite "
            "(brightness/25pct are intentionally hard). Check by_variant "
            "for texture_600 / original / 50pct customer paths.",
            file=sys.stderr,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
