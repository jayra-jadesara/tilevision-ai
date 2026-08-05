#!/usr/bin/env python3
"""
Production search-quality optimization study (measurement-first).

Phases:
  1) Build 300+ production-representative catalog + query set
  2) Cache embeddings (resume-safe)
  3) View ablation / memory tradeoff
  4) Room-scene root-cause analysis (no code changes)
  5) Ranking failure export
  6) Fusion bakeoff
  7) Embedding drift matrix
  8) Emit HTML/JSON/CSV deliverables

Usage:
  python dev_tools/search_quality/run_optimization_study.py \\
      --out /opt/cursor/artifacts/search_optimization --tiles 320

IMPORTANT: Does not modify production ranking code. Commit only after
a measured architecture win is proven in the report.
"""

from __future__ import annotations

import argparse
import csv
import html
import json
import math
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.ai.embedder import DINOv2Embedder
from src.ai.feature_extractor import FeatureExtractor
from src.ai.preprocess.image_preprocessor import ImagePreprocessor
from src.ai.search_quality.fusion import FusionMethod, ScoredHit, fuse_hits, tune_weighted_max
from src.ai.search_quality.views import (
    _adaptive_content_crop,
    _center_box,
    _texture_rich_crop,
)
from src.ai.vector_index import FaissIndexManager
from src.ai.preprocess.fast_tile_crop import isolate_tile_region

from dev_tools.search_quality.production_catalog import (
    CatalogTile,
    build_production_catalog,
)
from dev_tools.search_quality.query_generator import (
    VARIANT_SPECS,
    QueryItem,
    generate_queries,
)


VIEW_TYPES = (
    "primary",
    "center",
    "texture",
    "adaptive",
    "panel",
    "panel_center",
)


def _cos(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a, dtype=np.float32).ravel()
    b = np.asarray(b, dtype=np.float32).ravel()
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-8))


def _ndcg(rank: int | None, k: int) -> float:
    if rank is None or rank > k:
        return 0.0
    return 1.0 / math.log2(rank + 1)


def _embed_pil(fx: FeatureExtractor, pil: Image.Image) -> np.ndarray:
    return fx._embed_index_view(pil.convert("RGB"), original_size=pil.size)


def _write_csv(path: Path, rows: list[dict], fieldnames: list[str] | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("")
        return
    fields = fieldnames or list(rows[0].keys())
    with path.open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=fields)
        w.writeheader()
        for row in rows:
            w.writerow({k: row.get(k, "") for k in fields})


class OptimizationStudy:
    def __init__(self, out: Path, n_tiles: int):
        self.out = Path(out)
        self.out.mkdir(parents=True, exist_ok=True)
        self.n_tiles = n_tiles
        self.cache_dir = self.out / "cache"
        self.cache_dir.mkdir(exist_ok=True)
        self.reports = self.out / "reports"
        self.reports.mkdir(exist_ok=True)

        print("Loading DINOv2...", flush=True)
        self.emb = DINOv2Embedder()
        self.emb.load_model()
        self.fx = FeatureExtractor(embedder=self.emb)

        self.tiles = []
        self.queries = []
        self.catalog_views: dict[int, dict[str, np.ndarray]] = {}
        self.query_embs: dict[str, np.ndarray] = {}
        self.meta_by_id: dict[int, dict] = {}

    # ── Phase 1–2: dataset ─────────────────────────────────────────────
    def build_dataset(self) -> None:
        manifest = self.out / "catalog_manifest.json"
        if manifest.exists():
            meta = json.loads(manifest.read_text())
            if meta.get("n_tiles") == self.n_tiles and meta.get("tiles"):
                print(f"Reusing catalog n={self.n_tiles}...", flush=True)
                self.tiles = [
                    CatalogTile(
                        tile_id=int(t["tile_id"]),
                        path=Path(t["path"]),
                        material=t["material"],
                        finish=t["finish"],
                        is_sheet=bool(t["is_sheet"]),
                        near_dup_of=t.get("near_dup_of"),
                    )
                    for t in meta["tiles"]
                ]
            else:
                self.tiles = []
        else:
            self.tiles = []

        if not self.tiles:
            print(f"Building catalog n={self.n_tiles}...", flush=True)
            t0 = time.perf_counter()
            self.tiles = build_production_catalog(self.out, n_tiles=self.n_tiles)
            print(f"  catalog={len(self.tiles)} in {time.perf_counter()-t0:.1f}s", flush=True)

        qman = self.out / "query_manifest.json"
        expected_q = len(self.tiles) * len(VARIANT_SPECS)
        if qman.exists():
            qmeta = json.loads(qman.read_text())
            if qmeta.get("n_queries") == expected_q and qmeta.get("queries"):
                print(f"Reusing queries n={expected_q}...", flush=True)
                self.queries = [
                    QueryItem(
                        tile_id=int(q["tile_id"]),
                        variant=q["variant"],
                        path=Path(q["path"]),
                        material=q["material"],
                        is_sheet=bool(q["is_sheet"]),
                    )
                    for q in qmeta["queries"]
                    if Path(q["path"]).exists()
                ]
                if len(self.queries) != expected_q:
                    self.queries = []
            else:
                self.queries = []
        else:
            self.queries = []

        if not self.queries:
            print("Generating queries...", flush=True)
            t0 = time.perf_counter()
            self.queries = generate_queries(self.tiles, self.out)
            print(f"  queries={len(self.queries)} in {time.perf_counter()-t0:.1f}s", flush=True)
        else:
            print(f"  queries={len(self.queries)}", flush=True)

        self.meta_by_id = {
            t.tile_id: {
                "material": t.material,
                "finish": t.finish,
                "is_sheet": t.is_sheet,
                "path": str(t.path),
                "near_dup_of": t.near_dup_of,
            }
            for t in self.tiles
        }

    # ── Embedding cache ────────────────────────────────────────────────
    def cache_catalog_views(self) -> None:
        path = self.cache_dir / "catalog_views.npz"
        meta_path = self.cache_dir / "catalog_views_meta.json"
        if path.exists() and meta_path.exists():
            meta = json.loads(meta_path.read_text())
            if meta.get("n_tiles") == len(self.tiles):
                print("Loading cached catalog views...", flush=True)
                data = np.load(path)
                for key in data.files:
                    tid_s, name = key.split("__", 1)
                    tid = int(tid_s)
                    self.catalog_views.setdefault(tid, {})[name] = np.asarray(
                        data[key], dtype=np.float32
                    )
                if len(self.catalog_views) == len(self.tiles):
                    print(f"  loaded {len(self.catalog_views)} tiles", flush=True)
                    return
                print(
                    f"Resuming catalog embeddings "
                    f"({len(self.catalog_views)}/{len(self.tiles)})...",
                    flush=True,
                )
            else:
                self.catalog_views = {}
                print("Embedding catalog views (all candidates)...", flush=True)
        else:
            print("Embedding catalog views (all candidates)...", flush=True)

        t0 = time.perf_counter()
        for i, tile in enumerate(self.tiles, 1):
            if tile.tile_id in self.catalog_views and "primary" in self.catalog_views[tile.tile_id]:
                continue
            raw = ImagePreprocessor.to_rgb(ImagePreprocessor.load(tile.path))
            views: dict[str, np.ndarray] = {}
            # Production-faithful primary (multi-scale index extract)
            primary = np.asarray(
                self.fx.extract(str(tile.path), for_query=False).embedding,
                dtype=np.float32,
            )
            views["primary"] = primary

            cbox = _center_box(raw, 0.50)
            views["center"] = _embed_pil(self.fx, raw.crop(cbox))
            t_img, _ = _texture_rich_crop(raw)
            views["texture"] = _embed_pil(self.fx, t_img)
            a_img, _ = _adaptive_content_crop(raw)
            views["adaptive"] = _embed_pil(self.fx, a_img)

            panel = ImagePreprocessor.primary_texture_panel(raw)
            if panel is not None:
                views["panel"] = _embed_pil(self.fx, panel)
                if min(panel.size) >= 200:
                    pbox = _center_box(panel, 0.72)
                    views["panel_center"] = _embed_pil(self.fx, panel.crop(pbox))

            cleaned = {"primary": primary}
            for name, vec in views.items():
                if name == "primary":
                    continue
                if _cos(primary, vec) >= 0.985:
                    continue
                cleaned[name] = vec
            self.catalog_views[tile.tile_id] = cleaned
            if i % 10 == 0 or i == len(self.tiles):
                packed = {
                    f"{tid}__{name}": np.asarray(vec, dtype=np.float32)
                    for tid, vmap in self.catalog_views.items()
                    for name, vec in vmap.items()
                }
                np.savez_compressed(path, **packed)
                meta_path.write_text(json.dumps({"n_tiles": len(self.tiles)}))
                print(
                    f"  catalog {len(self.catalog_views)}/{len(self.tiles)} "
                    f"({time.perf_counter()-t0:.0f}s) [checkpoint]",
                    flush=True,
                )

        packed = {
            f"{tid}__{name}": np.asarray(vec, dtype=np.float32)
            for tid, vmap in self.catalog_views.items()
            for name, vec in vmap.items()
        }
        np.savez_compressed(path, **packed)
        meta_path.write_text(json.dumps({"n_tiles": len(self.tiles)}))
        print(f"  cached catalog views in {time.perf_counter()-t0:.1f}s", flush=True)

    def _persist_query_cache(self, arrays: dict[str, np.ndarray]) -> None:
        path = self.cache_dir / "query_embs.npz"
        meta_path = self.cache_dir / "query_embs_meta.json"
        key_map = {}
        packed = {}
        for i, (k, v) in enumerate(arrays.items()):
            kid = f"q{i}"
            key_map[kid] = k
            packed[kid] = np.asarray(v, dtype=np.float32)
        np.savez_compressed(path, **packed)
        meta_path.write_text(
            json.dumps({"n_queries": len(self.queries), "key_map": key_map})
        )

    def cache_queries(self) -> None:
        path = self.cache_dir / "query_embs.npz"
        meta_path = self.cache_dir / "query_embs_meta.json"
        arrays: dict[str, np.ndarray] = {}
        if path.exists() and meta_path.exists():
            meta = json.loads(meta_path.read_text())
            if meta.get("key_map"):
                print("Loading cached query embeddings...", flush=True)
                data = np.load(path)
                key_map = meta["key_map"]
                arrays = {
                    key_map[k]: np.asarray(data[k], dtype=np.float32)
                    for k in data.files
                    if k in key_map
                }
                self.query_embs = dict(arrays)
                print(f"  loaded {len(self.query_embs)}", flush=True)
                if len(arrays) == len(self.queries) and meta.get("n_queries") == len(
                    self.queries
                ):
                    return
                print(
                    f"Resuming query embeddings ({len(arrays)}/{len(self.queries)})...",
                    flush=True,
                )

        print("Embedding queries (production extract_for_search)...", flush=True)
        t0 = time.perf_counter()
        pending = [q for q in self.queries if str(q.path) not in arrays]
        for i, q in enumerate(pending, 1):
            feats, _ = self.fx.extract_for_search(str(q.path))
            key = str(q.path)
            arrays[key] = np.asarray(feats.embedding, dtype=np.float32)
            self.query_embs[key] = arrays[key]
            done = len(arrays)
            if i % 50 == 0 or done == len(self.queries):
                self._persist_query_cache(arrays)
                print(
                    f"  queries {done}/{len(self.queries)} "
                    f"({time.perf_counter()-t0:.0f}s) [checkpoint]",
                    flush=True,
                )
        self._persist_query_cache(arrays)
        print(f"  cached queries in {time.perf_counter()-t0:.1f}s", flush=True)

    # ── Index helpers ──────────────────────────────────────────────────
    def build_index(self, view_set: set[str]) -> tuple[FaissIndexManager, float]:
        t0 = time.perf_counter()
        mgr = FaissIndexManager(
            index_path=str(self.cache_dir / f"tmp_{'_'.join(sorted(view_set))}.index"),
            dimension=1024,
        )
        mgr.load_index()
        for tid, views in self.catalog_views.items():
            ids = []
            vecs = []
            for name in view_set:
                if name in views:
                    ids.append(tid)
                    vecs.append(views[name])
            if not ids and "primary" in views:
                ids = [tid]
                vecs = [views["primary"]]
            if ids:
                mgr.update_vectors(ids, vecs, persist=False)
        return mgr, time.perf_counter() - t0

    def evaluate(
        self,
        mgr: FaissIndexManager,
        *,
        fusion: FusionMethod = FusionMethod.MAX,
        aux_weight: float = 1.0,
        queries=None,
    ) -> dict:
        return self._evaluate_clean(
            mgr, fusion=fusion, aux_weight=aux_weight, queries=queries
        )

    def _evaluate_clean(
        self,
        mgr: FaissIndexManager,
        *,
        fusion: FusionMethod = FusionMethod.MAX,
        aux_weight: float = 1.0,
        queries=None,
    ) -> dict:
        queries = queries or self.queries
        n = r1 = r5 = r10 = 0
        mrr = ndcg5 = 0.0
        by_variant = defaultdict(lambda: {"n": 0, "r1": 0, "r5": 0, "r10": 0, "mrr": 0.0})
        by_material = defaultdict(lambda: {"n": 0, "r1": 0, "r5": 0})
        failures = []
        t_faiss = 0.0
        k = min(100, max(1, mgr.get_total_count()))

        for q in queries:
            emb = self.query_embs[str(q.path)]
            t0 = time.perf_counter()
            ids, scores = mgr.search_vectors(emb, top_k=k)
            t_faiss += time.perf_counter() - t0
            hits = [
                ScoredHit(int(t), float(s), aux_weight, i + 1)
                for i, (t, s) in enumerate(zip(ids, scores))
            ]
            fused = fuse_hits(hits, fusion)
            ranks = {tid: i + 1 for i, (tid, _) in enumerate(fused)}
            rank = ranks.get(q.tile_id)
            score = next((s for tid, s in fused if tid == q.tile_id), 0.0)
            top1_id = fused[0][0] if fused else None
            top1_score = fused[0][1] if fused else 0.0

            n += 1
            bv = by_variant[q.variant]
            bm = by_material[q.material]
            bv["n"] += 1
            bm["n"] += 1
            if rank is None:
                rr = 0.0
                stage = "faiss_miss_100"
            else:
                rr = 1.0 / rank
                if rank <= 1:
                    r1 += 1
                    bv["r1"] += 1
                    bm["r1"] += 1
                if rank <= 5:
                    r5 += 1
                    bv["r5"] += 1
                    bm["r5"] += 1
                if rank <= 10:
                    r10 += 1
                    bv["r10"] += 1
                if rank == 1:
                    stage = "ok"
                elif rank <= 5:
                    stage = "in5_not_top1"
                else:
                    stage = "in100_not_top5"
            mrr += rr
            ndcg5 += _ndcg(rank, 5)
            bv["mrr"] += rr

            if rank is None or rank > 5:
                failures.append(
                    {
                        "query_path": str(q.path),
                        "variant": q.variant,
                        "material": q.material,
                        "expected_tile_id": q.tile_id,
                        "returned_tile_id": top1_id if top1_id is not None else "",
                        "returned_material": self.meta_by_id.get(top1_id or -1, {}).get(
                            "material", ""
                        ),
                        "expected_similarity": round(float(score), 4),
                        "returned_similarity": round(float(top1_score), 4),
                        "embedding_distance": round(1.0 - float(score), 4),
                        "rank": rank if rank is not None else "",
                        "failure_stage": stage,
                        "category": q.material,
                        "reason": self._failure_reason(q, stage, score, top1_score),
                    }
                )

        return {
            "n_queries": n,
            "recall@1": round(r1 / max(1, n), 4),
            "recall@5": round(r5 / max(1, n), 4),
            "recall@10": round(r10 / max(1, n), 4),
            "mrr": round(mrr / max(1, n), 4),
            "ndcg@5": round(ndcg5 / max(1, n), 4),
            "avg_faiss_s": round(t_faiss / max(1, n), 6),
            "index_vectors": mgr.get_total_count(),
            "by_variant": {
                k: {
                    "n": v["n"],
                    "recall@1": round(v["r1"] / max(1, v["n"]), 4),
                    "recall@5": round(v["r5"] / max(1, v["n"]), 4),
                    "recall@10": round(v["r10"] / max(1, v["n"]), 4),
                    "mrr": round(v["mrr"] / max(1, v["n"]), 4),
                }
                for k, v in sorted(by_variant.items())
            },
            "by_material": {
                k: {
                    "n": v["n"],
                    "recall@1": round(v["r1"] / max(1, v["n"]), 4),
                    "recall@5": round(v["r5"] / max(1, v["n"]), 4),
                }
                for k, v in sorted(by_material.items())
            },
            "failures": failures,
        }

    def _failure_reason(self, q, stage: str, score: float, top1_score: float) -> str:
        if q.variant == "room_scene":
            return "room_scene_query_side_suspect"
        if q.variant == "catalogue_page" and stage != "ok":
            return "catalogue_layout_vs_texture_mismatch"
        if stage == "faiss_miss_100":
            return "parent_not_in_faiss_top100"
        if stage == "in100_not_top5":
            return f"parent_mid_pack_score_{score:.3f}_vs_top1_{top1_score:.3f}"
        return stage

    # ── Ablation ───────────────────────────────────────────────────────
    def run_ablation(self) -> list[dict]:
        print("\n=== VIEW ABLATION ===", flush=True)
        rows = []
        mgr_b, index_s = self.build_index({"primary"})
        base = self._evaluate_clean(mgr_b)
        base_r5 = base["recall@5"]
        base_r1 = base["recall@1"]
        rows.append(
            {
                "config": "primary_only",
                "views": "primary",
                "recall@1": base_r1,
                "recall@5": base_r5,
                "recall@1_gain_pp": 0.0,
                "recall@5_gain_pp": 0.0,
                "index_vectors": base["index_vectors"],
                "vectors_per_tile": round(base["index_vectors"] / max(1, len(self.tiles)), 3),
                "memory_mb": round(base["index_vectors"] * 1024 * 4 / (1024 * 1024), 2),
                "index_time_s": round(index_s, 4),
                "keep": "baseline",
            }
        )
        print(f"  primary_only R@5={base_r5}", flush=True)

        # Single-aux additions
        for aux in ("center", "texture", "adaptive", "panel", "panel_center"):
            views = {"primary", aux}
            # Skip if almost no tile has this view
            present = sum(1 for v in self.catalog_views.values() if aux in v)
            if present < max(5, len(self.tiles) * 0.05):
                rows.append(
                    {
                        "config": f"primary+{aux}",
                        "views": f"primary,{aux}",
                        "recall@1": base_r1,
                        "recall@5": base_r5,
                        "recall@1_gain_pp": 0.0,
                        "recall@5_gain_pp": 0.0,
                        "index_vectors": base["index_vectors"],
                        "vectors_per_tile": 1.0,
                        "memory_mb": rows[0]["memory_mb"],
                        "index_time_s": 0.0,
                        "keep": f"rejected_rare_view_present={present}",
                    }
                )
                continue
            mgr_v, index_s = self.build_index(views)
            metrics = self._evaluate_clean(mgr_v)
            g5 = round((metrics["recall@5"] - base_r5) * 100, 2)
            g1 = round((metrics["recall@1"] - base_r1) * 100, 2)
            mem = round(metrics["index_vectors"] * 1024 * 4 / (1024 * 1024), 2)
            keep = "keep" if g5 >= 1.0 or g1 >= 1.0 else "reject_<1pp_gain"
            rows.append(
                {
                    "config": f"primary+{aux}",
                    "views": f"primary,{aux}",
                    "recall@1": metrics["recall@1"],
                    "recall@5": metrics["recall@5"],
                    "recall@1_gain_pp": g1,
                    "recall@5_gain_pp": g5,
                    "index_vectors": metrics["index_vectors"],
                    "vectors_per_tile": round(
                        metrics["index_vectors"] / max(1, len(self.tiles)), 3
                    ),
                    "memory_mb": mem,
                    "index_time_s": round(index_s, 4),
                    "keep": keep,
                    "tiles_with_view": present,
                }
            )
            print(
                f"  primary+{aux}: R@5={metrics['recall@5']} gain={g5}pp "
                f"vecs={metrics['index_vectors']} -> {keep}",
                flush=True,
            )

        # Current production E = heuristic available views
        e_views = set()
        for views in self.catalog_views.values():
            e_views.update(views.keys())
        # Production uses whatever extract_index_vectors keeps — approximate as
        # primary + all non-dup cached views (heuristic filtered at cache time)
        mgr_p, index_s = self.build_index(set(VIEW_TYPES))
        prod_metrics = self._evaluate_clean(mgr_p)
        # Actually only views that exist per tile — build_index already handles
        g5 = round((prod_metrics["recall@5"] - base_r5) * 100, 2)
        g1 = round((prod_metrics["recall@1"] - base_r1) * 100, 2)
        rows.append(
            {
                "config": "all_cached_views",
                "views": ",".join(VIEW_TYPES),
                "recall@1": prod_metrics["recall@1"],
                "recall@5": prod_metrics["recall@5"],
                "recall@1_gain_pp": g1,
                "recall@5_gain_pp": g5,
                "index_vectors": prod_metrics["index_vectors"],
                "vectors_per_tile": round(
                    prod_metrics["index_vectors"] / max(1, len(self.tiles)), 3
                ),
                "memory_mb": round(
                    prod_metrics["index_vectors"] * 1024 * 4 / (1024 * 1024), 2
                ),
                "index_time_s": round(index_s, 4),
                "keep": "reference_full",
            }
        )
        print(
            f"  all_views: R@5={prod_metrics['recall@5']} vecs={prod_metrics['index_vectors']}",
            flush=True,
        )

        # Greedy memory trim: start from useful singles
        useful = [
            r["config"].split("+", 1)[-1]
            for r in rows
            if r["config"].startswith("primary+") and r["keep"] == "keep"
        ]
        if useful:
            combo = {"primary", *useful}
            mgr_c, index_s = self.build_index(combo)
            m = self._evaluate_clean(mgr_c)
            rows.append(
                {
                    "config": "greedy_" + "+".join(["primary", *useful]),
                    "views": ",".join(sorted(combo)),
                    "recall@1": m["recall@1"],
                    "recall@5": m["recall@5"],
                    "recall@1_gain_pp": round((m["recall@1"] - base_r1) * 100, 2),
                    "recall@5_gain_pp": round((m["recall@5"] - base_r5) * 100, 2),
                    "index_vectors": m["index_vectors"],
                    "vectors_per_tile": round(m["index_vectors"] / max(1, len(self.tiles)), 3),
                    "memory_mb": round(m["index_vectors"] * 1024 * 4 / (1024 * 1024), 2),
                    "index_time_s": round(index_s, 4),
                    "keep": "candidate_production",
                }
            )
            print(
                f"  greedy combo {combo}: R@5={m['recall@5']} vecs={m['index_vectors']}",
                flush=True,
            )

        _write_csv(self.reports / "ablation_study.csv", rows)
        return rows

    # ── Room RCA ───────────────────────────────────────────────────────
    def room_scene_rca(self, mgr: FaissIndexManager) -> list[dict]:
        print("\n=== ROOM SCENE ROOT CAUSE (no code changes) ===", flush=True)
        rows = []
        rooms = [q for q in self.queries if q.variant == "room_scene"]
        # Instrument every room-scene query (preprocess is cheap vs DINOv2)
        sample = rooms
        for q in sample:
            raw = ImagePreprocessor.to_rgb(ImagePreprocessor.load(q.path))
            looks = ImagePreprocessor._looks_like_scene_photo(raw)
            panel = ImagePreprocessor.primary_texture_panel(raw)
            iso = None
            iso_method = ""
            iso_conf = 0.0
            iso_size = ""
            if looks and panel is None:
                iso = isolate_tile_region(raw)
                iso_method = iso.method
                iso_conf = float(iso.confidence)
                iso_size = f"{iso.image.size[0]}x{iso.image.size[1]}"
            qemb = self.query_embs[str(q.path)]
            primary = self.catalog_views[q.tile_id]["primary"]
            best_aux = max(
                (
                    _cos(qemb, v)
                    for name, v in self.catalog_views[q.tile_id].items()
                    if name != "primary"
                ),
                default=0.0,
            )
            cos_primary = _cos(qemb, primary)
            ids, scores = mgr.search_vectors(qemb, top_k=min(100, mgr.get_total_count()))
            best = {}
            for tid, sc in zip(ids, scores):
                if tid not in best or sc > best[tid]:
                    best[int(tid)] = float(sc)
            ordered = sorted(best.items(), key=lambda x: x[1], reverse=True)
            rank = next(
                (i + 1 for i, (tid, _) in enumerate(ordered) if tid == q.tile_id),
                None,
            )
            if cos_primary < 0.55 and best_aux < 0.55:
                reason = "embedding_drift_query_far_from_all_views"
            elif looks and panel is None and iso_method in {"center_fallback", "floor_band"}:
                reason = f"auto_crop_weak_method={iso_method}"
            elif rank is not None and rank <= 5:
                reason = "ok_in_top5"
            elif best_aux >= 0.75 and (rank is None or rank > 5):
                reason = "aux_aligned_but_distractors_win"
            else:
                reason = "mixed_query_preprocess_or_ambiguity"

            rows.append(
                {
                    "tile_id": q.tile_id,
                    "material": q.material,
                    "looks_like_scene": looks,
                    "catalog_sheet_panel_on_query": panel is not None,
                    "isolate_method": iso_method,
                    "isolate_confidence": round(iso_conf, 3),
                    "isolate_size": iso_size,
                    "source_size": f"{raw.size[0]}x{raw.size[1]}",
                    "cos_vs_primary": round(cos_primary, 4),
                    "cos_vs_best_aux": round(best_aux, 4),
                    "faiss_rank": rank if rank is not None else "",
                    "root_cause_hypothesis": reason,
                }
            )
        _write_csv(self.reports / "room_scene_rca.csv", rows)
        # Summary counts
        summary = defaultdict(int)
        for r in rows:
            summary[r["root_cause_hypothesis"]] += 1
        (self.reports / "room_scene_rca_summary.json").write_text(
            json.dumps({"n": len(rows), "hypotheses": dict(summary)}, indent=2)
        )
        print("  hypotheses:", dict(summary), flush=True)
        return rows

    # ── Fusion ─────────────────────────────────────────────────────────
    def run_fusion(self, view_set: set[str]) -> dict:
        print("\n=== FUSION BAKEOFF ===", flush=True)
        mgr, _ = self.build_index(view_set)
        results = {}
        for method in FusionMethod:
            m = self._evaluate_clean(mgr, fusion=method)
            results[method.value] = {
                "recall@1": m["recall@1"],
                "recall@5": m["recall@5"],
                "mrr": m["mrr"],
                "ndcg@5": m["ndcg@5"],
            }
            print(
                f"  {method.value}: R@1={m['recall@1']} R@5={m['recall@5']}",
                flush=True,
            )
        # Tuned weighted max
        trials = []
        for q in self.queries[::3]:
            emb = self.query_embs[str(q.path)]
            ids, scores = mgr.search_vectors(emb, top_k=min(50, mgr.get_total_count()))
            hits = [
                ScoredHit(int(t), float(s), 0.9, i + 1)
                for i, (t, s) in enumerate(zip(ids, scores))
            ]
            trials.append((hits, q.tile_id))
        best_w, val_r1 = tune_weighted_max(trials)
        tuned = self._evaluate_clean(
            mgr, fusion=FusionMethod.WEIGHTED_MAX, aux_weight=best_w
        )
        results["weighted_max_tuned"] = {
            "aux_weight": best_w,
            "val_recall@1": round(val_r1, 4),
            "recall@1": tuned["recall@1"],
            "recall@5": tuned["recall@5"],
            "mrr": tuned["mrr"],
            "ndcg@5": tuned["ndcg@5"],
        }
        print(
            f"  weighted_max_tuned w={best_w}: R@1={tuned['recall@1']} R@5={tuned['recall@5']}",
            flush=True,
        )
        # Prefer Recall@5, then Recall@1. Reject candidates that lose >1pp R@1
        # vs MAX (production default) even if R@5 is slightly higher.
        max_r1 = results.get("max", {}).get("recall@1", 0.0)
        eligible = []
        for k, v in results.items():
            if not isinstance(v, dict) or "recall@5" not in v:
                continue
            if k != "max" and v["recall@1"] + 0.01 < max_r1:
                continue
            eligible.append((k, v))
        if not eligible:
            eligible = [("max", results["max"])]
        winner = max(
            eligible,
            key=lambda kv: (kv[1]["recall@5"], kv[1]["recall@1"]),
        )
        results["winner"] = winner[0]
        results["winner_rule"] = (
            "maximize Recall@5 then Recall@1; reject methods losing >1pp R@1 vs MAX"
        )
        (self.reports / "fusion_bakeoff.json").write_text(json.dumps(results, indent=2))
        return results

    # ── Embedding drift ────────────────────────────────────────────────
    def embedding_similarity(self) -> None:
        print("\n=== EMBEDDING DRIFT ===", flush=True)
        rows = []
        # Sample one tile per material
        seen = set()
        samples = []
        for t in self.tiles:
            if t.material in seen:
                continue
            seen.add(t.material)
            samples.append(t)
            if len(samples) >= 14:
                break
        variants = [
            "original",
            "crop_95",
            "crop_90",
            "crop_75",
            "crop_60",
            "crop_50",
            "center",
            "catalogue_page",
            "room_scene",
            "phone_screenshot",
            "whatsapp",
        ]
        for t in samples:
            primary = self.catalog_views[t.tile_id]["primary"]
            for variant in variants:
                qp = self.out / "queries" / f"id_{t.tile_id:04d}" / f"{variant}.jpg"
                if not qp.exists():
                    continue
                qemb = self.query_embs[str(qp)]
                rows.append(
                    {
                        "tile_id": t.tile_id,
                        "material": t.material,
                        "variant": variant,
                        "cosine_vs_primary": round(_cos(primary, qemb), 4),
                        "best_aux_cosine": round(
                            max(
                                (
                                    _cos(qemb, v)
                                    for name, v in self.catalog_views[t.tile_id].items()
                                    if name != "primary"
                                ),
                                default=0.0,
                            ),
                            4,
                        ),
                    }
                )
        _write_csv(self.reports / "embedding_similarity.csv", rows)

    # ── Reports ────────────────────────────────────────────────────────
    def write_reports(
        self,
        ablation: list[dict],
        baseline: dict,
        production: dict,
        fusion: dict,
        recommended_views: set[str],
    ) -> None:
        # ranking failures + confusion from production eval
        _write_csv(self.reports / "ranking_failures.csv", production["failures"])
        confusion = defaultdict(lambda: defaultdict(int))
        for f in production["failures"]:
            confusion[f["material"]][f["returned_material"] or "NONE"] += 1
        conf_rows = []
        for exp, ret_map in sorted(confusion.items()):
            for ret, cnt in sorted(ret_map.items()):
                conf_rows.append(
                    {
                        "expected_material": exp,
                        "returned_material": ret,
                        "count": cnt,
                    }
                )
        _write_csv(self.reports / "confusion_matrix.csv", conf_rows)

        mem_rows = []
        for r in ablation:
            mem_rows.append(
                {
                    "config": r["config"],
                    "vectors": r["index_vectors"],
                    "vectors_per_tile": r["vectors_per_tile"],
                    "memory_mb": r["memory_mb"],
                    "projected_50k_mb": round(
                        float(r["vectors_per_tile"]) * 50000 * 1024 * 4 / (1024 * 1024),
                        1,
                    ),
                    "recall@5": r["recall@5"],
                }
            )
        _write_csv(self.reports / "memory_report.csv", mem_rows)

        perf_rows = [
            {
                "config": "primary_only",
                "recall@1": baseline["recall@1"],
                "recall@5": baseline["recall@5"],
                "avg_faiss_s": baseline["avg_faiss_s"],
                "index_vectors": baseline["index_vectors"],
            },
            {
                "config": "recommended",
                "recall@1": production["recall@1"],
                "recall@5": production["recall@5"],
                "avg_faiss_s": production["avg_faiss_s"],
                "index_vectors": production["index_vectors"],
                "views": ",".join(sorted(recommended_views)),
            },
        ]
        _write_csv(self.reports / "performance_report.csv", perf_rows)

        # Room summary
        room = baseline["by_variant"].get("room_scene", {})
        room_prod = production["by_variant"].get("room_scene", {})

        report = {
            "catalog_source": "synthetic_production_representative",
            "catalog_note": (
                "Real customer catalog volumes were not available in this "
                "cloud environment. Benchmark uses a 300+ tile production-"
                "representative synthetic catalog covering requested materials."
            ),
            "platform": {
                "os": sys.platform,
                "cuda": False,
                "note": (
                    "Executed on Linux CPU in cloud agent. Windows / macOS Intel / "
                    "Apple Silicon parity must be validated via CI workflows; "
                    "this run does not fabricate cross-platform numbers."
                ),
            },
            "n_tiles": len(self.tiles),
            "n_queries": len(self.queries),
            "variants": list(VARIANT_SPECS),
            "baseline_primary_only": {
                k: baseline[k]
                for k in (
                    "recall@1",
                    "recall@5",
                    "recall@10",
                    "mrr",
                    "ndcg@5",
                    "index_vectors",
                    "avg_faiss_s",
                    "by_variant",
                    "by_material",
                )
            },
            "recommended_config": {
                "views": sorted(recommended_views),
                "fusion": fusion.get("winner", "max"),
                "metrics": {
                    k: production[k]
                    for k in (
                        "recall@1",
                        "recall@5",
                        "recall@10",
                        "mrr",
                        "ndcg@5",
                        "index_vectors",
                        "avg_faiss_s",
                        "by_variant",
                        "by_material",
                    )
                },
            },
            "delta_vs_primary": {
                "recall@1": round(production["recall@1"] - baseline["recall@1"], 4),
                "recall@5": round(production["recall@5"] - baseline["recall@5"], 4),
            },
            "ablation": ablation,
            "fusion": fusion,
            "room_scene": {
                "baseline_recall@5": room.get("recall@5"),
                "recommended_recall@5": room_prod.get("recall@5"),
                "note": "See room_scene_rca.csv for query-side root-cause hypotheses.",
            },
            "architecture_decision": self._decision(
                baseline, production, ablation, recommended_views
            ),
        }
        (self.reports / "search_benchmark.json").write_text(json.dumps(report, indent=2))
        self._write_html(report)
        print("Wrote reports to", self.reports, flush=True)

    def _decision(self, baseline, production, ablation, views) -> dict:
        improve = production["recall@5"] >= baseline["recall@5"] + 0.01
        return {
            "commit_architecture_change": bool(improve),
            "reason": (
                "Recommended view set improves Recall@5 by >=1pp vs primary-only"
                if improve
                else "No sufficient Recall improvement to justify architecture change"
            ),
            "recommended_views": sorted(views),
            "vectors_per_tile": round(
                production["index_vectors"] / max(1, len(self.tiles)), 3
            ),
        }

    def _write_html(self, report: dict) -> None:
        rows = "".join(
            f"<tr><td>{html.escape(r['config'])}</td>"
            f"<td>{r['recall@1']}</td><td>{r['recall@5']}</td>"
            f"<td>{r['recall@5_gain_pp']}</td>"
            f"<td>{r['vectors_per_tile']}</td>"
            f"<td>{r['memory_mb']}</td>"
            f"<td>{html.escape(str(r['keep']))}</td></tr>"
            for r in report["ablation"]
        )
        fusion_rows = "".join(
            f"<tr><td>{html.escape(k)}</td><td>{v.get('recall@1','')}</td>"
            f"<td>{v.get('recall@5','')}</td></tr>"
            for k, v in report["fusion"].items()
            if isinstance(v, dict) and "recall@5" in v
        )
        body = f"""<!DOCTYPE html>
<html><head><meta charset="utf-8"><title>TileVision Search Benchmark</title>
<style>
body{{font-family:ui-sans-serif,system-ui,sans-serif;margin:32px;color:#122}}
table{{border-collapse:collapse;width:100%;margin:16px 0}}
th,td{{border:1px solid #ccd;padding:8px;text-align:left}}
th{{background:#eef3f8}}
.badge{{display:inline-block;padding:4px 8px;background:#e8f5e9;border-radius:4px}}
.warn{{background:#fff3e0}}
</style></head><body>
<h1>TileVision AI — Search Quality Optimization Benchmark</h1>
<p class="badge">Catalog: {report['n_tiles']} tiles · Queries: {report['n_queries']}</p>
<p class="warn"><strong>Catalog source:</strong> {html.escape(report['catalog_note'])}</p>
<p><strong>Platform:</strong> {html.escape(report['platform']['note'])}</p>
<h2>Headline</h2>
<ul>
<li>Primary-only Recall@5: <b>{report['baseline_primary_only']['recall@5']}</b></li>
<li>Recommended Recall@5: <b>{report['recommended_config']['metrics']['recall@5']}</b>
 (Δ {report['delta_vs_primary']['recall@5']:+})</li>
<li>Recommended views: <b>{', '.join(report['recommended_config']['views'])}</b></li>
<li>Fusion winner: <b>{report['recommended_config']['fusion']}</b></li>
<li>Room-scene Recall@5 (recommended): <b>{report['room_scene']['recommended_recall@5']}</b></li>
<li>Architecture decision: <b>{html.escape(report['architecture_decision']['reason'])}</b></li>
</ul>
<h2>Ablation</h2>
<table><tr><th>Config</th><th>R@1</th><th>R@5</th><th>R@5 gain pp</th>
<th>Vec/tile</th><th>MB</th><th>Decision</th></tr>{rows}</table>
<h2>Fusion</h2>
<table><tr><th>Method</th><th>R@1</th><th>R@5</th></tr>{fusion_rows}</table>
<h2>Room scene</h2>
<p>{html.escape(report['room_scene']['note'])}</p>
</body></html>"""
        (self.reports / "search_benchmark.html").write_text(body)

    def run(self) -> dict:
        self.build_dataset()
        self.cache_catalog_views()
        self.cache_queries()

        ablation = self.run_ablation()
        mgr_base, _ = self.build_index({"primary"})
        baseline = self._evaluate_clean(mgr_base)

        # Recommended = greedy keep views, else all that help
        useful = [
            r["config"].split("+", 1)[-1]
            for r in ablation
            if r["config"].startswith("primary+") and r["keep"] == "keep"
        ]
        recommended = {"primary", *useful} if useful else {"primary"}
        # Always compare to full cached views too
        mgr_full, _ = self.build_index(set(VIEW_TYPES))
        full = self._evaluate_clean(mgr_full)
        mgr_g, _ = self.build_index(recommended)
        greedy = self._evaluate_clean(mgr_g)
        # Pick better of greedy vs full on R@5 then fewer vectors
        if (
            full["recall@5"] > greedy["recall@5"] + 0.005
            or (
                abs(full["recall@5"] - greedy["recall@5"]) <= 0.005
                and full["recall@1"] > greedy["recall@1"]
            )
        ):
            production = full
            recommended = set(VIEW_TYPES)
        else:
            production = greedy

        print(
            f"\nRecommended views={sorted(recommended)} "
            f"R@5={production['recall@5']} vecs={production['index_vectors']}",
            flush=True,
        )

        mgr, _ = self.build_index(recommended)
        self.room_scene_rca(mgr)
        fusion = self.run_fusion(recommended)
        # Re-eval production with winning fusion if not max
        win = fusion.get("winner", "max")
        if win != "max" and win in FusionMethod._value2member_map_:
            production = self._evaluate_clean(mgr, fusion=FusionMethod(win))
        self.embedding_similarity()
        self.write_reports(ablation, baseline, production, fusion, recommended)

        decision = json.loads((self.reports / "search_benchmark.json").read_text())[
            "architecture_decision"
        ]
        return decision


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("/opt/cursor/artifacts/search_optimization"),
    )
    parser.add_argument("--tiles", type=int, default=320)
    args = parser.parse_args()
    study = OptimizationStudy(args.out, args.tiles)
    decision = study.run()
    print(json.dumps(decision, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
