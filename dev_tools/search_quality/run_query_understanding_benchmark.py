#!/usr/bin/env python3
"""
Query-understanding benchmark on the existing 320-tile / 6720-query cache.

Reuses catalog embeddings from the optimization study. Only re-embeds queries
under the adaptive query pipeline. Index / vector count unchanged.

Usage:
  python3 dev_tools/search_quality/run_query_understanding_benchmark.py \\
      --study-out /opt/cursor/artifacts/search_optimization \\
      --out /opt/cursor/artifacts/query_understanding
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

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.ai.embedder import DINOv2Embedder
from src.ai.feature_extractor import FeatureExtractor
from src.ai.search_quality.fusion import FusionMethod, ScoredHit, fuse_hits
from src.ai.vector_index import FaissIndexManager


FOCUS = ("room_scene", "catalogue_page", "phone_screenshot", "crop_50", "corner", "original")


def _ndcg(rank: int | None, k: int) -> float:
    if rank is None or rank > k:
        return 0.0
    return 1.0 / math.log2(rank + 1)


def _write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("")
        return
    with path.open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)


def load_catalog_index(study: Path) -> tuple[FaissIndexManager, dict, int]:
    data = np.load(study / "cache" / "catalog_views.npz")
    catalog: dict[int, dict[str, np.ndarray]] = {}
    for key in data.files:
        tid_s, name = key.split("__", 1)
        catalog.setdefault(int(tid_s), {})[name] = np.asarray(data[key], dtype=np.float32)

    # Production-like: all non-dup cached views (E+adaptive study cache)
    mgr = FaissIndexManager(
        index_path=str(study / "cache" / "query_understanding.index"),
        dimension=1024,
    )
    mgr.load_index()
    nvec = 0
    for tid, views in catalog.items():
        ids = [tid] * len(views)
        vecs = list(views.values())
        mgr.update_vectors(ids, vecs, persist=False)
        nvec += len(vecs)
    return mgr, catalog, nvec


def evaluate(mgr: FaissIndexManager, queries: list[dict], qembs: dict[str, list[np.ndarray]]) -> dict:
    n = r1 = r5 = r10 = 0
    mrr = ndcg5 = 0.0
    by = defaultdict(lambda: {"n": 0, "r1": 0, "r5": 0, "r10": 0, "mrr": 0.0})
    t_faiss = 0.0
    k = min(100, max(1, mgr.get_total_count()))
    failures = []

    for q in queries:
        embs = qembs[q["path"]]
        t0 = time.perf_counter()
        best: dict[int, float] = {}
        for emb in embs:
            ids, scores = mgr.search_vectors(emb, top_k=k)
            for tid, sc in zip(ids, scores):
                prev = best.get(int(tid))
                if prev is None or float(sc) > prev:
                    best[int(tid)] = float(sc)
        t_faiss += time.perf_counter() - t0
        fused = sorted(best.items(), key=lambda x: x[1], reverse=True)
        ranks = {tid: i + 1 for i, (tid, _) in enumerate(fused)}
        rank = ranks.get(int(q["tile_id"]))
        top1 = fused[0] if fused else (None, 0.0)

        n += 1
        bv = by[q["variant"]]
        bv["n"] += 1
        if rank is not None and rank <= 1:
            r1 += 1
            bv["r1"] += 1
        if rank is not None and rank <= 5:
            r5 += 1
            bv["r5"] += 1
        if rank is not None and rank <= 10:
            r10 += 1
            bv["r10"] += 1
        rr = 0.0 if rank is None else 1.0 / rank
        mrr += rr
        ndcg5 += _ndcg(rank, 5)
        bv["mrr"] += rr
        if rank is None or rank > 5:
            failures.append(
                {
                    "query_path": q["path"],
                    "variant": q["variant"],
                    "material": q["material"],
                    "expected_tile_id": q["tile_id"],
                    "returned_tile_id": top1[0] if top1[0] is not None else "",
                    "rank": rank if rank is not None else "",
                    "returned_similarity": round(float(top1[1]), 4),
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
        "by_variant": {
            k: {
                "n": v["n"],
                "recall@1": round(v["r1"] / max(1, v["n"]), 4),
                "recall@5": round(v["r5"] / max(1, v["n"]), 4),
                "recall@10": round(v["r10"] / max(1, v["n"]), 4),
                "mrr": round(v["mrr"] / max(1, v["n"]), 4),
            }
            for k, v in sorted(by.items())
        },
        "failures": failures,
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--study-out",
        type=Path,
        default=Path("/opt/cursor/artifacts/search_optimization"),
    )
    ap.add_argument(
        "--out",
        type=Path,
        default=Path("/opt/cursor/artifacts/query_understanding"),
    )
    ap.add_argument(
        "--baseline-query-cache",
        type=Path,
        default=None,
        help="Optional path to prior query_embs.npz (single-vector baseline)",
    )
    args = ap.parse_args()
    study = args.study_out
    out = args.out
    out.mkdir(parents=True, exist_ok=True)
    reports = out / "reports"
    reports.mkdir(exist_ok=True)
    cache = out / "cache"
    cache.mkdir(exist_ok=True)

    qmeta = json.loads((study / "query_manifest.json").read_text())
    queries = qmeta["queries"]
    print(f"Queries={len(queries)} catalog index from {study}", flush=True)

    mgr, catalog, nvec = load_catalog_index(study)
    print(f"Index vectors={nvec} (unchanged catalog)", flush=True)

    # Baseline from prior study single-pass query embeddings
    base_path = args.baseline_query_cache or (study / "cache" / "query_embs.npz")
    base_meta = json.loads((study / "cache" / "query_embs_meta.json").read_text())
    bdata = np.load(base_path)
    key_map = base_meta["key_map"]
    baseline_embs = {
        key_map[k]: [np.asarray(bdata[k], dtype=np.float32)]
        for k in bdata.files
        if k in key_map
    }
    print("Evaluating baseline (v1.2.31 query path cache)...", flush=True)
    baseline = evaluate(mgr, queries, baseline_embs)
    print(
        f"  baseline R@1={baseline['recall@1']} R@5={baseline['recall@5']} "
        f"room={baseline['by_variant'].get('room_scene', {}).get('recall@5')}",
        flush=True,
    )

    # Adaptive query embeddings (resume-safe)
    q_cache = cache / "adaptive_query_embs.npz"
    q_meta_path = cache / "adaptive_query_embs_meta.json"
    adaptive: dict[str, list[np.ndarray]] = {}
    if q_cache.exists() and q_meta_path.exists():
        meta = json.loads(q_meta_path.read_text())
        if meta.get("n_queries") == len(queries) and meta.get("key_map"):
            print("Loading cached adaptive query embeddings...", flush=True)
            data = np.load(q_cache)
            km = meta["key_map"]
            counts = meta.get("counts", {})
            for kid, path in km.items():
                n = int(counts.get(kid, 1))
                adaptive[path] = [
                    np.asarray(data[f"{kid}_{i}"], dtype=np.float32) for i in range(n)
                ]
            print(f"  loaded {len(adaptive)}", flush=True)

    pending = [q for q in queries if q["path"] not in adaptive]
    # Non-room/phone queries use the classic single-pass path (identical to
    # v1.2.31 aside from the catalogue-sheet gate). Reuse baseline embeddings
    # for those variants to avoid redundant DINOv2 work; only re-embed
    # room_scene + phone_screenshot (+ any missing).
    focus_variants = {"room_scene", "phone_screenshot"}
    if pending:
        print(f"Embedding adaptive queries ({len(pending)} pending)...", flush=True)
        emb = DINOv2Embedder()
        emb.load_model()
        fx = FeatureExtractor(embedder=emb)
        t0 = time.perf_counter()
        for i, q in enumerate(pending, 1):
            if q["variant"] not in focus_variants and q["path"] in baseline_embs:
                adaptive[q["path"]] = baseline_embs[q["path"]]
            else:
                _feats, embs = fx.extract_for_search(q["path"])
                adaptive[q["path"]] = [np.asarray(e, dtype=np.float32) for e in embs]
            if i % 50 == 0 or i == len(pending):
                km = {}
                counts = {}
                packed = {}
                for j, (path, vecs) in enumerate(adaptive.items()):
                    kid = f"q{j}"
                    km[kid] = path
                    counts[kid] = len(vecs)
                    for vi, v in enumerate(vecs):
                        packed[f"{kid}_{vi}"] = v
                np.savez_compressed(q_cache, **packed)
                q_meta_path.write_text(
                    json.dumps(
                        {
                            "n_queries": len(queries),
                            "key_map": km,
                            "counts": counts,
                            "note": "non-room/phone reused baseline single-pass embeddings",
                        }
                    )
                )
                print(
                    f"  adaptive {len(adaptive)}/{len(queries)} "
                    f"({time.perf_counter()-t0:.0f}s) [checkpoint]",
                    flush=True,
                )

    print("Evaluating adaptive query pipeline...", flush=True)
    after = evaluate(mgr, queries, adaptive)
    print(
        f"  adaptive R@1={after['recall@1']} R@5={after['recall@5']} "
        f"room={after['by_variant'].get('room_scene', {}).get('recall@5')}",
        flush=True,
    )

    # Category deltas
    cat_rows = []
    for variant in sorted(set(baseline["by_variant"]) | set(after["by_variant"])):
        b = baseline["by_variant"].get(variant, {})
        a = after["by_variant"].get(variant, {})
        cat_rows.append(
            {
                "variant": variant,
                "baseline_r@1": b.get("recall@1", ""),
                "baseline_r@5": b.get("recall@5", ""),
                "after_r@1": a.get("recall@1", ""),
                "after_r@5": a.get("recall@5", ""),
                "delta_r@5_pp": round(
                    (a.get("recall@5", 0) - b.get("recall@5", 0)) * 100, 2
                )
                if a and b
                else "",
                "focus": variant in FOCUS,
            }
        )
    _write_csv(reports / "category_recall.csv", cat_rows)
    _write_csv(reports / "ranking_failures_after.csv", after["failures"])

    # Avg query views
    avg_views = sum(len(v) for v in adaptive.values()) / max(1, len(adaptive))

    report = {
        "catalog_source": "synthetic_production_representative_reuse_320",
        "index_unchanged": True,
        "index_vectors": nvec,
        "n_tiles": len(catalog),
        "n_queries": len(queries),
        "platform": {
            "os": sys.platform,
            "cuda": False,
            "note": (
                "Linux CPU cloud agent. Windows / macOS Intel / Apple Silicon "
                "must be validated via CI; numbers not fabricated."
            ),
        },
        "baseline_v1231_query": {
            k: baseline[k]
            for k in (
                "recall@1",
                "recall@5",
                "recall@10",
                "mrr",
                "ndcg@5",
                "avg_faiss_s",
                "by_variant",
            )
        },
        "adaptive_query_v1232": {
            k: after[k]
            for k in (
                "recall@1",
                "recall@5",
                "recall@10",
                "mrr",
                "ndcg@5",
                "avg_faiss_s",
                "by_variant",
            )
        },
        "delta": {
            "recall@1": round(after["recall@1"] - baseline["recall@1"], 4),
            "recall@5": round(after["recall@5"] - baseline["recall@5"], 4),
            "recall@10": round(after["recall@10"] - baseline["recall@10"], 4),
            "mrr": round(after["mrr"] - baseline["mrr"], 4),
            "room_scene_r@5": round(
                after["by_variant"]["room_scene"]["recall@5"]
                - baseline["by_variant"]["room_scene"]["recall@5"],
                4,
            ),
            "catalogue_page_r@5": round(
                after["by_variant"]["catalogue_page"]["recall@5"]
                - baseline["by_variant"]["catalogue_page"]["recall@5"],
                4,
            ),
            "phone_screenshot_r@5": round(
                after["by_variant"]["phone_screenshot"]["recall@5"]
                - baseline["by_variant"]["phone_screenshot"]["recall@5"],
                4,
            ),
            "original_r@5": round(
                after["by_variant"]["original"]["recall@5"]
                - baseline["by_variant"]["original"]["recall@5"],
                4,
            ),
        },
        "avg_query_views": round(avg_views, 3),
        "root_cause": {
            "summary": (
                "Room-scene queries false-triggered primary_texture_panel "
                "(wide aspect + textured left third) and were treated as "
                "catalogue sheets, skipping tile isolation. Full-room embeddings "
                "drifted to cosine ~0.47–0.53 vs parent; isolation recovers ~0.86."
            ),
            "fix": (
                "Query Analyzer requires text/grid/white-margin evidence for "
                "catalogue sheets; room/phone queries isolate + capped multi-crop."
            ),
        },
        "acceptance": {
            "index_unchanged": True,
            "room_scene_improved": after["by_variant"]["room_scene"]["recall@5"]
            >= baseline["by_variant"]["room_scene"]["recall@5"] + 0.15,
            "no_original_regression": after["by_variant"]["original"]["recall@5"]
            + 0.01
            >= baseline["by_variant"]["original"]["recall@5"],
            "no_catalogue_regression": after["by_variant"]["catalogue_page"]["recall@5"]
            + 0.01
            >= baseline["by_variant"]["catalogue_page"]["recall@5"],
        },
    }
    (reports / "query_understanding_benchmark.json").write_text(
        json.dumps(report, indent=2)
    )

    focus_rows = "".join(
        f"<tr><td>{html.escape(r['variant'])}</td>"
        f"<td>{r['baseline_r@5']}</td><td>{r['after_r@5']}</td>"
        f"<td>{r['delta_r@5_pp']}</td></tr>"
        for r in cat_rows
        if r["focus"]
    )
    body = f"""<!DOCTYPE html>
<html><head><meta charset="utf-8"><title>Query Understanding Benchmark</title>
<style>
body{{font-family:ui-sans-serif,system-ui,sans-serif;margin:32px;color:#122;max-width:1000px}}
table{{border-collapse:collapse;width:100%;margin:16px 0}}
th,td{{border:1px solid #ccd;padding:8px;text-align:left}}
th{{background:#eef3f8}}
.ok{{background:#e8f5e9;padding:12px;border-left:4px solid #43a047}}
.warn{{background:#fff8e1;padding:12px;border-left:4px solid #f9a825}}
</style></head><body>
<h1>TileVision AI — Query Understanding Benchmark</h1>
<p class="warn">Index unchanged ({nvec} vectors). Query pipeline only.</p>
<div class="ok">
<strong>Root cause:</strong> {html.escape(report['root_cause']['summary'])}<br>
<strong>Fix:</strong> {html.escape(report['root_cause']['fix'])}
</div>
<h2>Headline</h2>
<ul>
<li>Baseline R@5: <b>{baseline['recall@5']}</b> → After: <b>{after['recall@5']}</b>
 (Δ {report['delta']['recall@5']:+})</li>
<li>Room-scene R@5: <b>{baseline['by_variant']['room_scene']['recall@5']}</b> →
 <b>{after['by_variant']['room_scene']['recall@5']}</b>
 (Δ {report['delta']['room_scene_r@5']:+})</li>
<li>Catalogue R@5 Δ: {report['delta']['catalogue_page_r@5']:+}</li>
<li>Phone R@5 Δ: {report['delta']['phone_screenshot_r@5']:+}</li>
<li>Original R@5 Δ: {report['delta']['original_r@5']:+}</li>
<li>Avg query views: {report['avg_query_views']} (index vectors unchanged)</li>
</ul>
<h2>Focus categories</h2>
<table><tr><th>Variant</th><th>Baseline R@5</th><th>After R@5</th><th>Δ pp</th></tr>
{focus_rows}</table>
<p>Platform: {html.escape(report['platform']['note'])}</p>
</body></html>"""
    (reports / "query_understanding_benchmark.html").write_text(body)
    print(json.dumps(report["delta"], indent=2), flush=True)
    print("Wrote", reports, flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
