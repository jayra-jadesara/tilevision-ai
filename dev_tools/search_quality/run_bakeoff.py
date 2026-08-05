#!/usr/bin/env python3
"""
Production search-accuracy bakeoff.

Phases:
  1) Build golden dataset (auto-labeled queries)
  2) Measure Strategy A (primary-only) baseline with stage timings
  3) Benchmark Strategies B–E + production_v8 independently
  4) Benchmark score fusion methods on the winning index strategy
  5) Embedding drift + failure-stage attribution
  6) Write reports under --out

Usage:
  python dev_tools/search_quality/run_bakeoff.py --out /tmp/search_accuracy
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.ai.embedder import DINOv2Embedder
from src.ai.feature_extractor import FeatureExtractor
from src.ai.preprocess.image_preprocessor import ImagePreprocessor, PreprocessedImage
from src.ai.search_quality.fusion import FusionMethod, ScoredHit, fuse_hits, tune_weighted_max
from src.ai.search_quality.image_analysis import analyze_image
from src.ai.search_quality.views import IndexStrategy, IndexViewType, build_index_views
from src.ai.vector_index import FaissIndexManager

from dev_tools.search_quality.golden_dataset import (
    VARIANT_SPECS,
    CatalogItem,
    GoldenQuery,
    build_golden_catalog,
)


def _cos(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a, dtype=np.float32).ravel()
    b = np.asarray(b, dtype=np.float32).ravel()
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-8))


def _ndcg(rank: int | None, k: int) -> float:
    if rank is None or rank > k:
        return 0.0
    return 1.0 / math.log2(rank + 1)


@dataclass
class StageTimings:
    embed_s: float = 0.0
    faiss_s: float = 0.0
    fuse_s: float = 0.0
    total_s: float = 0.0


@dataclass
class Metrics:
    n: int = 0
    r1: int = 0
    r5: int = 0
    r10: int = 0
    mrr: float = 0.0
    ndcg5: float = 0.0
    timings: StageTimings = field(default_factory=StageTimings)
    by_variant: dict = field(default_factory=lambda: defaultdict(lambda: {
        "n": 0, "r1": 0, "r5": 0, "r10": 0, "mrr": 0.0
    }))
    stage_fail: dict = field(default_factory=lambda: {
        "faiss_miss_100": 0,
        "in100_not_top5": 0,
        "in5_not_top1": 0,
    })
    vectors: int = 0
    mean_views: float = 0.0


class BakeoffEngine:
    def __init__(self, fx: FeatureExtractor, dim: int = 1024):
        self.fx = fx
        self.dim = dim
        self._embedder = fx._embedder

    def embed_pil_index(self, pil: Image.Image) -> np.ndarray:
        view = ImagePreprocessor.normalize_lighting(pil.convert("RGB"))
        view = ImagePreprocessor.resize_letterbox(view)
        rgb = ImagePreprocessor.to_numpy(view)
        import cv2

        bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
        gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
        pre = PreprocessedImage(
            pil=view, rgb=rgb, bgr=bgr, gray=gray, width=pil.size[0], height=pil.size[1]
        )
        return np.asarray(
            self._embedder.extract_from_preprocessed(pre, for_query=False),
            dtype=np.float32,
        )

    def embed_query(self, path: Path) -> tuple[np.ndarray, float]:
        t0 = time.perf_counter()
        feats, _ = self.fx.extract_for_search(str(path))
        return np.asarray(feats.embedding, dtype=np.float32), time.perf_counter() - t0

    def index_strategy(
        self,
        items: list[CatalogItem],
        strategy: IndexStrategy,
        index_path: Path,
    ) -> tuple[FaissIndexManager, float, Metrics]:
        mgr = FaissIndexManager(index_path=str(index_path), dimension=self.dim)
        mgr.load_index()
        meta = Metrics()
        view_counts = []
        # Deduplicate by cosine near-dup against primary like production.
        t0 = time.perf_counter()
        for item in items:
            raw = ImagePreprocessor.to_rgb(ImagePreprocessor.load(item.path))
            views = build_index_views(raw, strategy)
            ids: list[int] = []
            vecs: list[np.ndarray] = []
            primary = None
            kept = 0
            for view in views:
                emb = self.embed_pil_index(view.image)
                if view.view_type == IndexViewType.PRIMARY:
                    primary = emb
                    ids.append(item.tile_id)
                    vecs.append(emb)
                    kept += 1
                    continue
                assert primary is not None
                if _cos(primary, emb) >= 0.985:
                    continue
                if any(_cos(emb, v) >= 0.99 for v in vecs[1:]):
                    continue
                ids.append(item.tile_id)
                vecs.append(emb)
                kept += 1
            view_counts.append(kept)
            mgr.update_vectors(ids, vecs, persist=False)
        meta.vectors = mgr.get_total_count()
        meta.mean_views = float(np.mean(view_counts)) if view_counts else 0.0
        index_s = time.perf_counter() - t0
        return mgr, index_s, meta

    def evaluate(
        self,
        mgr: FaissIndexManager,
        queries: list[GoldenQuery],
        *,
        fusion: FusionMethod = FusionMethod.MAX,
        aux_weight: float = 1.0,
        search_k: int = 100,
    ) -> Metrics:
        metrics = Metrics(vectors=mgr.get_total_count())
        total_embed = total_faiss = total_fuse = 0.0
        ntotal = mgr.get_total_count()
        k = min(search_k, max(1, ntotal))

        for q in queries:
            qemb, embed_s = self.embed_query(q.path)
            total_embed += embed_s
            t1 = time.perf_counter()
            raw_ids, raw_scores = mgr.search_vectors(qemb, top_k=k)
            faiss_s = time.perf_counter() - t1
            total_faiss += faiss_s

            hits: list[ScoredHit] = []
            for rank, (tid, sc) in enumerate(zip(raw_ids, raw_scores), start=1):
                # Without per-vector metadata in FlatIP IDMap, treat non-first
                # duplicate ranks as aux-ish; weight applied uniformly via tune.
                hits.append(
                    ScoredHit(
                        tile_id=int(tid),
                        score=float(sc),
                        view_weight=aux_weight,
                        rank_in_list=rank,
                    )
                )
            t2 = time.perf_counter()
            fused = fuse_hits(hits, fusion)
            fuse_s = time.perf_counter() - t2
            total_fuse += fuse_s

            ranks = {tid: i + 1 for i, (tid, _) in enumerate(fused)}
            rank = ranks.get(q.tile_id)
            score = next((s for tid, s in fused if tid == q.tile_id), 0.0)

            metrics.n += 1
            bv = metrics.by_variant[q.variant]
            bv["n"] += 1
            if rank is None:
                metrics.stage_fail["faiss_miss_100"] += 1
                rr = 0.0
            else:
                if rank <= 1:
                    metrics.r1 += 1
                    bv["r1"] += 1
                elif rank <= 5:
                    metrics.stage_fail["in5_not_top1"] += 1
                if rank <= 5:
                    metrics.r5 += 1
                    bv["r5"] += 1
                else:
                    metrics.stage_fail["in100_not_top5"] += 1
                if rank <= 10:
                    metrics.r10 += 1
                    bv["r10"] += 1
                rr = 1.0 / rank
            metrics.mrr += rr
            metrics.ndcg5 += _ndcg(rank, 5)
            bv["mrr"] += rr

        n = max(1, metrics.n)
        metrics.timings = StageTimings(
            embed_s=total_embed / n,
            faiss_s=total_faiss / n,
            fuse_s=total_fuse / n,
            total_s=(total_embed + total_faiss + total_fuse) / n,
        )
        return metrics


def metrics_to_dict(m: Metrics) -> dict:
    n = max(1, m.n)
    return {
        "n_queries": m.n,
        "recall@1": round(m.r1 / n, 4),
        "recall@5": round(m.r5 / n, 4),
        "recall@10": round(m.r10 / n, 4),
        "mrr": round(m.mrr / n, 4),
        "ndcg@5": round(m.ndcg5 / n, 4),
        "index_vectors": m.vectors,
        "mean_views_per_tile": round(m.mean_views, 3),
        "latency": {
            "embed_s": round(m.timings.embed_s, 4),
            "faiss_s": round(m.timings.faiss_s, 5),
            "fuse_s": round(m.timings.fuse_s, 5),
            "total_s": round(m.timings.total_s, 4),
        },
        "stage_fail": m.stage_fail,
        "by_variant": {
            k: {
                "n": v["n"],
                "recall@1": round(v["r1"] / max(1, v["n"]), 4),
                "recall@5": round(v["r5"] / max(1, v["n"]), 4),
                "recall@10": round(v["r10"] / max(1, v["n"]), 4),
                "mrr": round(v["mrr"] / max(1, v["n"]), 4),
            }
            for k, v in sorted(m.by_variant.items())
        },
    }


CUSTOMER_VARIANTS = {
    "original",
    "crop_95",
    "crop_90",
    "crop_75",
    "crop_60",
    "crop_50",
    "phone_screenshot",
    "catalogue_page",
    "room_scene",
    "jpeg30",
    "rotated",
    "contrast",
}


def customer_slice(d: dict) -> dict:
    r1 = r5 = n = 0
    for variant, stats in d["by_variant"].items():
        if variant not in CUSTOMER_VARIANTS:
            continue
        n += stats["n"]
        r1 += int(round(stats["recall@1"] * stats["n"]))
        r5 += int(round(stats["recall@5"] * stats["n"]))
    return {
        "n_queries": n,
        "recall@1": round(r1 / max(1, n), 4),
        "recall@5": round(r5 / max(1, n), 4),
    }


def embedding_drift_report(engine: BakeoffEngine, items: list[CatalogItem], out: Path) -> dict:
    """Cosine(original_index_emb, query_variant_emb) for first tile + sheet."""
    report = {}
    samples = [it for it in items if it.kind == "sheet"][:1] + [
        it for it in items if it.kind == "tile"
    ][:1]
    for item in samples:
        primary = engine.embed_pil_index(
            ImagePreprocessor.to_rgb(ImagePreprocessor.load(item.path))
        )
        qdir = item.path.parent.parent / "queries" / f"id_{item.tile_id:04d}"
        row = {}
        for variant in (
            "original",
            "crop_95",
            "crop_90",
            "crop_75",
            "crop_60",
            "crop_50",
            "center",
            "random_crop",
            "catalogue_page",
        ):
            qp = qdir / f"{variant}.jpg"
            if not qp.exists():
                continue
            qemb, _ = engine.embed_query(qp)
            row[variant] = round(_cos(primary, qemb), 4)
        # Panel aux drift for sheets
        analysis = analyze_image(Image.open(item.path))
        row["analysis"] = {
            "kind": analysis.kind.value,
            "left_panel_beneficial": analysis.left_panel_beneficial,
            "center_crop_beneficial": analysis.center_crop_beneficial,
            "texture_richness": round(analysis.texture_richness, 3),
        }
        report[f"{item.kind}_{item.tile_id}"] = row
    out.write_text(json.dumps(report, indent=2))
    return report


def run(out_dir: Path, n_tiles: int, n_sheets: int) -> dict:
    out_dir.mkdir(parents=True, exist_ok=True)
    weights = Path("model_weights/dinov2-large/config.json")
    if not weights.is_file():
        raise FileNotFoundError("DINOv2 weights missing")

    print("Building golden dataset...")
    items, queries = build_golden_catalog(
        out_dir / "golden", n_tiles=n_tiles, n_sheets=n_sheets
    )
    print(f"Catalog={len(items)} queries={len(queries)}")

    emb = DINOv2Embedder()
    emb.load_model()
    fx = FeatureExtractor(embedder=emb)
    engine = BakeoffEngine(fx)

    drift = embedding_drift_report(engine, items, out_dir / "embedding_drift.json")
    print("Embedding drift sample:", json.dumps(drift, indent=2)[:800])

    strategy_results = {}
    for strategy in [
        IndexStrategy.A_PRIMARY_ONLY,
        IndexStrategy.B_FULL_CENTER,
        IndexStrategy.C_FULL_ADAPTIVE,
        IndexStrategy.D_FULL_TEXTURE,
        IndexStrategy.E_HEURISTIC_MULTIVIEW,
        IndexStrategy.PRODUCTION_V8,
    ]:
        print(f"\n=== Strategy {strategy.value} ===")
        mgr, index_s, meta = engine.index_strategy(
            items, strategy, out_dir / f"idx_{strategy.value}.index"
        )
        metrics = engine.evaluate(mgr, queries, fusion=FusionMethod.MAX)
        metrics.mean_views = meta.mean_views
        metrics.vectors = meta.vectors
        payload = metrics_to_dict(metrics)
        payload["index_build_s"] = round(index_s, 2)
        payload["customer_path"] = customer_slice(payload)
        payload["memory_proxy"] = {
            "vectors": metrics.vectors,
            "bytes_approx": metrics.vectors * 1024 * 4,
            "vs_primary_only_ratio": round(
                metrics.vectors / max(1, len(items)), 3
            ),
            "projected_50k_vectors": int(
                round(50_000 * (metrics.vectors / max(1, len(items))))
            ),
            "projected_50k_mb": round(
                50_000
                * (metrics.vectors / max(1, len(items)))
                * 1024
                * 4
                / (1024 * 1024),
                1,
            ),
        }
        strategy_results[strategy.value] = payload
        print(
            f"  R@1={payload['recall@1']} R@5={payload['recall@5']} "
            f"custR@5={payload['customer_path']['recall@5']} "
            f"vectors={payload['index_vectors']} views={payload['mean_views_per_tile']}"
        )
        (out_dir / f"strategy_{strategy.value}.json").write_text(
            json.dumps(payload, indent=2)
        )

    # Pick best by customer-path Recall@5 then Recall@1 then overall R@1.
    def key(name_payload):
        name, p = name_payload
        c = p["customer_path"]
        return (c["recall@5"], c["recall@1"], p["recall@5"], p["recall@1"])

    best_name, best_payload = max(strategy_results.items(), key=key)
    print(f"\nWinning strategy (customer-path first): {best_name}")

    # Fusion bakeoff on winning strategy index
    print("\n=== Fusion bakeoff ===")
    mgr, _, meta = engine.index_strategy(
        items, IndexStrategy(best_name), out_dir / f"idx_fusion_{best_name}.index"
    )
    fusion_results = {}
    for method in FusionMethod:
        m = engine.evaluate(mgr, queries, fusion=method, aux_weight=1.0)
        m.vectors = meta.vectors
        m.mean_views = meta.mean_views
        payload = metrics_to_dict(m)
        payload["customer_path"] = customer_slice(payload)
        fusion_results[method.value] = payload
        print(
            f"  {method.value}: R@1={payload['recall@1']} R@5={payload['recall@5']} "
            f"custR@5={payload['customer_path']['recall@5']}"
        )

    # Tune weighted_max aux weight on a validation half
    half = queries[::2]
    # Build trials from FAISS raw hits
    trials = []
    for q in half:
        qemb, _ = engine.embed_query(q.path)
        ids, scores = mgr.search_vectors(qemb, top_k=min(50, mgr.get_total_count()))
        hits = [
            ScoredHit(tile_id=int(t), score=float(s), view_weight=0.9, rank_in_list=i + 1)
            for i, (t, s) in enumerate(zip(ids, scores))
        ]
        trials.append((hits, q.tile_id))
    best_w, tuned_r1 = tune_weighted_max(trials)
    fusion_results["weighted_max_tuned"] = {
        "aux_weight": best_w,
        "val_recall@1": round(tuned_r1, 4),
    }
    m = engine.evaluate(
        mgr, queries, fusion=FusionMethod.WEIGHTED_MAX, aux_weight=best_w
    )
    m.vectors = meta.vectors
    tuned_payload = metrics_to_dict(m)
    tuned_payload["customer_path"] = customer_slice(tuned_payload)
    tuned_payload["aux_weight"] = best_w
    fusion_results["weighted_max_tuned_full"] = tuned_payload

    best_fusion, best_fusion_payload = max(
        ((k, v) for k, v in fusion_results.items() if "by_variant" in v),
        key=lambda kv: (
            kv[1]["customer_path"]["recall@5"],
            kv[1]["customer_path"]["recall@1"],
            kv[1]["recall@5"],
            kv[1]["recall@1"],
        ),
    )

    baseline = strategy_results[IndexStrategy.A_PRIMARY_ONLY.value]
    winner = strategy_results[best_name]
    report = {
        "goal": "production Recall@1 / Recall@5 on golden auto-labeled queries",
        "catalog_size": len(items),
        "n_queries": len(queries),
        "variants": list(VARIANT_SPECS),
        "embedding_drift_sample": drift,
        "strategies": strategy_results,
        "winning_strategy": best_name,
        "fusion": fusion_results,
        "winning_fusion": best_fusion,
        "delta_vs_primary_only": {
            "recall@1": round(winner["recall@1"] - baseline["recall@1"], 4),
            "recall@5": round(winner["recall@5"] - baseline["recall@5"], 4),
            "customer_recall@1": round(
                winner["customer_path"]["recall@1"]
                - baseline["customer_path"]["recall@1"],
                4,
            ),
            "customer_recall@5": round(
                winner["customer_path"]["recall@5"]
                - baseline["customer_path"]["recall@5"],
                4,
            ),
            "vector_ratio": winner["memory_proxy"]["vs_primary_only_ratio"],
        },
        "reject_if_customer_r1_regresses": winner["customer_path"]["recall@1"]
        < baseline["customer_path"]["recall@1"],
        "acceptance": {
            "customer_path_r5": winner["customer_path"]["recall@5"],
            "customer_path_r1": winner["customer_path"]["recall@1"],
            "beats_primary_r5": winner["customer_path"]["recall@5"]
            >= baseline["customer_path"]["recall@5"],
            "beats_primary_r1": winner["customer_path"]["recall@1"]
            >= baseline["customer_path"]["recall@1"],
        },
    }
    (out_dir / "bakeoff_report.json").write_text(json.dumps(report, indent=2))
    print("\nWrote", out_dir / "bakeoff_report.json")
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("/opt/cursor/artifacts/search_accuracy"),
    )
    parser.add_argument("--tiles", type=int, default=24)
    parser.add_argument("--sheets", type=int, default=12)
    args = parser.parse_args()
    report = run(args.out, n_tiles=args.tiles, n_sheets=args.sheets)
    print(json.dumps({
        "winning_strategy": report["winning_strategy"],
        "winning_fusion": report["winning_fusion"],
        "delta_vs_primary_only": report["delta_vs_primary_only"],
        "acceptance": report["acceptance"],
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
