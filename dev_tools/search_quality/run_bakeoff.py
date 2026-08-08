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
from dev_tools.search_quality.real_customer import (
    CATALOG_SOURCE_REAL,
    CATALOG_SOURCE_SYNTHETIC,
    catalog_items_from_records,
    format_query_kind_table,
    load_real_customer_manifest,
    low_sample_warning,
    query_kind_breakdown,
    records_to_golden_queries,
    validate_ground_truth_ids,
)
from src.ai.search_quality.query_views import PARTIAL_CROP_MODES, set_partial_crop_mode


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
    rerank_s: float = 0.0
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

    def embed_query(self, path: Path) -> tuple[list[np.ndarray], float]:
        t0 = time.perf_counter()
        _feats, embs = self.fx.extract_for_search(str(path))
        vectors = embs if embs else [_feats.embedding]
        return (
            [np.asarray(v, dtype=np.float32) for v in vectors],
            time.perf_counter() - t0,
        )

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
        orb_verification: bool = False,
        catalog_by_id: dict[int, CatalogItem] | None = None,
    ) -> Metrics:
        metrics = Metrics(vectors=mgr.get_total_count())
        total_embed = total_faiss = total_fuse = total_rerank = 0.0
        ntotal = mgr.get_total_count()
        k = min(search_k, max(1, ntotal))
        catalog_by_id = catalog_by_id or {}

        orb = None
        if orb_verification:
            from src.ai.verification.orb_verifier import OrbVerifier
            from src.core.use_cases.search_tiles import (
                ORB_BOOST_MAX,
                ORB_MAX_CANDIDATES,
                ORB_VERIFICATION_BAND,
            )

            orb = OrbVerifier()

        for q in queries:
            qembs, embed_s = self.embed_query(q.path)
            total_embed += embed_s
            t1 = time.perf_counter()
            best_scores: dict[int, float] = {}
            for qemb in qembs:
                raw_ids, raw_scores = mgr.search_vectors(qemb, top_k=k)
                for tid, sc in zip(raw_ids, raw_scores):
                    tile_id = int(tid)
                    score = float(sc)
                    prev = best_scores.get(tile_id)
                    if prev is None or score > prev:
                        best_scores[tile_id] = score
            faiss_s = time.perf_counter() - t1
            total_faiss += faiss_s

            hits: list[ScoredHit] = []
            fused_sorted = sorted(best_scores.items(), key=lambda item: item[1], reverse=True)
            for rank, (tid, sc) in enumerate(fused_sorted, start=1):
                hits.append(
                    ScoredHit(
                        tile_id=tid,
                        score=sc,
                        view_weight=aux_weight,
                        rank_in_list=rank,
                    )
                )
            t2 = time.perf_counter()
            fused = fuse_hits(hits, fusion)
            fuse_s = time.perf_counter() - t2
            total_fuse += fuse_s

            t3 = time.perf_counter()
            if orb is not None and fused:
                fused = self._orb_nudge_fused(
                    q.path,
                    fused,
                    catalog_by_id,
                    orb,
                    ORB_VERIFICATION_BAND,
                    ORB_MAX_CANDIDATES,
                    ORB_BOOST_MAX,
                )
            total_rerank += time.perf_counter() - t3

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
            rerank_s=total_rerank / n,
            total_s=(total_embed + total_faiss + total_fuse + total_rerank) / n,
        )
        return metrics

    @staticmethod
    def _orb_nudge_fused(
        query_path: Path,
        fused: list[tuple[int, float]],
        catalog_by_id: dict[int, CatalogItem],
        orb,
        band: float,
        max_candidates: int,
        boost_max: float,
    ) -> list[tuple[int, float]]:
        """Apply ORB boost to near-tie fused scores (bakeoff mirror of production)."""
        if len(fused) < 2:
            return fused
        try:
            q_pil = ImagePreprocessor.load(query_path)
            q_rgb = ImagePreprocessor.to_rgb(q_pil)
            import cv2

            query_gray = cv2.cvtColor(np.asarray(q_rgb), cv2.COLOR_RGB2GRAY)
        except Exception:
            return fused

        top = float(fused[0][1])
        band_idxs: list[int] = []
        for i, (_tid, sc) in enumerate(fused):
            if top - float(sc) > band:
                break
            band_idxs.append(i)
            if len(band_idxs) >= max_candidates:
                break
        if len(band_idxs) < 2:
            return fused

        updated = list(fused)
        changed = False
        for i in band_idxs:
            tid, sc = updated[i]
            item = catalog_by_id.get(int(tid))
            if item is None:
                continue
            try:
                c_pil = ImagePreprocessor.load(item.path)
                c_rgb = ImagePreprocessor.to_rgb(c_pil)
                import cv2

                cand_gray = cv2.cvtColor(np.asarray(c_rgb), cv2.COLOR_RGB2GRAY)
            except Exception:
                continue
            orb_score = float(orb.score(query_gray, cand_gray))
            if orb_score <= 0.0:
                continue
            updated[i] = (tid, min(1.0, float(sc) + boost_max * orb_score))
            changed = True
        if not changed:
            return fused
        updated.sort(key=lambda item: item[1], reverse=True)
        return updated


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
            "rerank_ms": round(m.timings.rerank_s * 1000.0, 3),
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
    # Real-customer manifests use free-text query_kind tags (whatsapp, …)
    # that are not in CUSTOMER_VARIANTS — fall back to overall metrics.
    if n == 0 and d.get("catalog_source") == CATALOG_SOURCE_REAL:
        return {
            "n_queries": d.get("n_queries", 0),
            "recall@1": d.get("recall@1", 0.0),
            "recall@5": d.get("recall@5", 0.0),
        }
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
            qembs, _ = engine.embed_query(qp)
            if len(qembs) == 1:
                qemb = qembs[0]
            else:
                qemb = np.mean(np.vstack(qembs), axis=0)
                qemb = qemb / (np.linalg.norm(qemb) + 1e-8)
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


def run(
    out_dir: Path,
    n_tiles: int,
    n_sheets: int,
    *,
    orb_verification: bool = True,
    real_queries: Path | None = None,
    pooling: str = "cls",
    partial_crop_mode: str | None = None,
) -> dict:
    out_dir.mkdir(parents=True, exist_ok=True)
    if partial_crop_mode is not None:
        set_partial_crop_mode(partial_crop_mode)
        print(f"partial_crop_mode: {partial_crop_mode}")
    weights = Path("model_weights/dinov2-large/config.json")
    if not weights.is_file():
        raise FileNotFoundError("DINOv2 weights missing")

    catalog_source = CATALOG_SOURCE_SYNTHETIC
    real_breakdown_payload: dict | None = None

    if real_queries is not None:
        print(f"Loading real-customer manifest: {real_queries}")
        records = load_real_customer_manifest(Path(real_queries))
        warn = low_sample_warning(len(records))
        if warn:
            print(f"\n*** {warn} ***\n")

        catalog_from_manifest = catalog_items_from_records(records)
        if catalog_from_manifest is not None:
            items = catalog_from_manifest
            print(
                f"Catalog from manifest catalog_path fields: {len(items)} tile(s)"
            )
        else:
            print(
                "Manifest has no complete catalog_path coverage — "
                "building synthetic catalog for ID validation / indexing."
            )
            items, _synthetic_queries = build_golden_catalog(
                out_dir / "golden", n_tiles=n_tiles, n_sheets=n_sheets
            )

        catalog_by_id = {item.tile_id: item for item in items}
        validate_ground_truth_ids(records, set(catalog_by_id))
        queries = records_to_golden_queries(records)
        catalog_source = CATALOG_SOURCE_REAL
        print(
            f"Real-customer mode: catalog={len(items)} queries={len(queries)} "
            f"source={catalog_source}"
        )
    else:
        print("Building golden dataset...")
        items, queries = build_golden_catalog(
            out_dir / "golden", n_tiles=n_tiles, n_sheets=n_sheets
        )
        print(f"Catalog={len(items)} queries={len(queries)}")
        catalog_by_id = {item.tile_id: item for item in items}

    print(f"ORB verification: {'on' if orb_verification else 'off'}")
    print(f"catalog_source: {catalog_source}")

    emb = DINOv2Embedder(pooling=pooling)
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
        metrics = engine.evaluate(
            mgr,
            queries,
            fusion=FusionMethod.MAX,
            orb_verification=orb_verification,
            catalog_by_id=catalog_by_id,
        )
        metrics.mean_views = meta.mean_views
        metrics.vectors = meta.vectors
        payload = metrics_to_dict(metrics)
        payload["index_build_s"] = round(index_s, 2)
        payload["catalog_source"] = catalog_source
        if catalog_source == CATALOG_SOURCE_REAL:
            payload["by_query_kind"] = query_kind_breakdown(payload)
            real_breakdown_payload = payload["by_query_kind"]
            print("\nPer-query_kind breakdown:")
            print(format_query_kind_table(payload["by_query_kind"]))
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
        m = engine.evaluate(
            mgr,
            queries,
            fusion=method,
            aux_weight=1.0,
            orb_verification=orb_verification,
            catalog_by_id=catalog_by_id,
        )
        m.vectors = meta.vectors
        m.mean_views = meta.mean_views
        payload = metrics_to_dict(m)
        payload["catalog_source"] = catalog_source
        if catalog_source == CATALOG_SOURCE_REAL:
            payload["by_query_kind"] = query_kind_breakdown(payload)
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
        qembs, _ = engine.embed_query(q.path)
        best_scores: dict[int, float] = {}
        for qemb in qembs:
            ids, scores = mgr.search_vectors(qemb, top_k=min(50, mgr.get_total_count()))
            for tile_id, score in zip(ids, scores):
                tid = int(tile_id)
                sc = float(score)
                prev = best_scores.get(tid)
                if prev is None or sc > prev:
                    best_scores[tid] = sc
        hits = [
            ScoredHit(tile_id=tid, score=sc, view_weight=0.9, rank_in_list=i + 1)
            for i, (tid, sc) in enumerate(
                sorted(best_scores.items(), key=lambda item: item[1], reverse=True),
                start=1,
            )
        ]
        trials.append((hits, q.tile_id))
    best_w, tuned_r1 = tune_weighted_max(trials)
    fusion_results["weighted_max_tuned"] = {
        "aux_weight": best_w,
        "val_recall@1": round(tuned_r1, 4),
    }
    m = engine.evaluate(
        mgr,
        queries,
        fusion=FusionMethod.WEIGHTED_MAX,
        aux_weight=best_w,
        orb_verification=orb_verification,
        catalog_by_id=catalog_by_id,
    )
    m.vectors = meta.vectors
    tuned_payload = metrics_to_dict(m)
    tuned_payload["catalog_source"] = catalog_source
    if catalog_source == CATALOG_SOURCE_REAL:
        tuned_payload["by_query_kind"] = query_kind_breakdown(tuned_payload)
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
        "catalog_source": catalog_source,
        "orb_verification": bool(orb_verification),
        "pooling": pooling,
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
    if catalog_source == CATALOG_SOURCE_REAL:
        report["by_query_kind"] = real_breakdown_payload or query_kind_breakdown(
            winner
        )
        warn = low_sample_warning(len(queries))
        report["low_sample_warning"] = warn
        if warn:
            print(f"\n*** {warn} ***\n")
        print("\nFinal per-query_kind breakdown (winning strategy):")
        print(format_query_kind_table(report["by_query_kind"]))
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
    parser.add_argument(
        "--orb-verification",
        choices=("on", "off"),
        default="on",
        help="Enable ORB near-tie geometric verification (default: on).",
    )
    parser.add_argument(
        "--real-queries",
        type=Path,
        default=None,
        help=(
            "JSONL ground-truth manifest of real customer photos. "
            "Skips synthetic query generation. See docs/REAL_CUSTOMER_BENCHMARK.md."
        ),
    )
    parser.add_argument(
        "--pooling",
        choices=("cls", "mean_patch"),
        default="cls",
        help="DINOv2 token pooling for bakeoff A/B (default: cls, production).",
    )
    parser.add_argument(
        "--partial-crop-mode",
        choices=sorted(PARTIAL_CROP_MODES),
        default=None,
        help="Diagnostic: PARTIAL_CROP view strategy (ablation only).",
    )
    args = parser.parse_args()
    orb_on = args.orb_verification == "on"
    report = run(
        args.out,
        n_tiles=args.tiles,
        n_sheets=args.sheets,
        orb_verification=orb_on,
        real_queries=args.real_queries,
        pooling=args.pooling,
        partial_crop_mode=args.partial_crop_mode,
    )
    summary = {
        "winning_strategy": report["winning_strategy"],
        "winning_fusion": report["winning_fusion"],
        "catalog_source": report.get("catalog_source"),
        "orb_verification": report["orb_verification"],
        "delta_vs_primary_only": report["delta_vs_primary_only"],
        "acceptance": report["acceptance"],
    }
    if report.get("by_query_kind") is not None:
        summary["by_query_kind"] = report["by_query_kind"]
    if report.get("low_sample_warning"):
        summary["low_sample_warning"] = report["low_sample_warning"]
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
