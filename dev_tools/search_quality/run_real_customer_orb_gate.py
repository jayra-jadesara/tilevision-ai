#!/usr/bin/env python3
"""
Run real-customer release gate: bakeoff harness (PR #39) + production ORB A/B.

Compares enable_orb_verification on/off on the eval/real_customer_release set
using the full SearchTilesUseCase path (hybrid rerank + optional ORB).

Usage:
  python3 dev_tools/search_quality/build_real_customer_eval_set.py
  python3 dev_tools/search_quality/run_real_customer_orb_gate.py \\
      --manifest eval/real_customer_release.jsonl \\
      --out /tmp/real_customer_orb_gate
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import dataclass
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.ai.embedder import DINOv2Embedder
from src.ai.feature_extractor import FeatureExtractor
from src.ai.vector_index import FaissIndexManager
from src.core.models import TileImage
from src.core.use_cases.search_tiles import SearchTilesUseCase
from src.data.db_context import DatabaseContext
from src.data.sqlite_repository import SQLiteImageRepository
from src.utils.image_utils import compute_sha256

from dev_tools.search_quality.real_customer import load_real_customer_manifest


@dataclass
class GateResult:
    orb_on: bool
    n: int
    r1: int
    r5: int
    mrr_sum: float
    by_kind: dict[str, dict[str, float]]
    confusable_pair_r1: int
    confusable_pair_n: int
    latency_s: float


def _bootstrap_catalog(
    records,
    work_dir: Path,
) -> tuple[SearchTilesUseCase, SearchTilesUseCase, dict[int, int]]:
    """Index manifest catalog into SQLite+FAISS; return use cases orb on/off."""
    db_path = work_dir / "database" / "tiles.db"
    index_path = work_dir / "index" / "tiles.index"
    thumb_dir = work_dir / "thumbnails"
    for p in (db_path.parent, index_path.parent, thumb_dir):
        p.mkdir(parents=True, exist_ok=True)

    db = DatabaseContext(str(db_path))
    repo = SQLiteImageRepository(db)
    fx = FeatureExtractor(embedder=DINOv2Embedder())
    fx.load_model()
    index = FaissIndexManager(index_path=str(index_path), dimension=1024)
    index.load_index()

    catalog_by_path: dict[Path, int] = {}
    for rec in records:
        if rec.catalog_path is not None:
            catalog_by_path.setdefault(rec.catalog_path, rec.true_tile_id)

    manifest_to_sqlite: dict[int, int] = {}
    for cat, manifest_tid in sorted(catalog_by_path.items(), key=lambda item: item[1]):
        features = fx.extract(str(cat))
        from PIL import Image

        img = Image.open(cat)
        w, h = img.size
        tile = TileImage(
            file_path=str(cat.resolve()),
            file_name=cat.name,
            file_size=cat.stat().st_size,
            dimensions=f"{w}x{h}",
            is_indexed=False,
            features=features,
            sha256_hash=compute_sha256(cat),
        )
        assigned = repo.add(tile)
        index.add_vectors([assigned], [features.embedding.tolist()], persist=False)
        repo.mark_as_indexed(assigned, True)
        manifest_to_sqlite[manifest_tid] = assigned

    index.save_index()

    uc_on = SearchTilesUseCase(repo, fx, index, str(thumb_dir), enable_orb_verification=True)
    uc_off = SearchTilesUseCase(repo, fx, index, str(thumb_dir), enable_orb_verification=False)
    return uc_on, uc_off, manifest_to_sqlite


def _evaluate_production(
    use_case: SearchTilesUseCase,
    records,
    *,
    orb_on: bool,
    confusable_names: set[str],
    manifest_to_sqlite: dict[int, int],
) -> GateResult:
    by_kind: dict[str, dict[str, float]] = {}
    r1 = r5 = 0
    mrr_sum = 0.0
    conf_r1 = conf_n = 0
    t0 = time.perf_counter()

    for rec in records:
        qpath = str(rec.query_path)
        truth = manifest_to_sqlite.get(rec.true_tile_id, rec.true_tile_id)
        kind = rec.query_kind
        bucket = by_kind.setdefault(kind, {"n": 0, "r1": 0, "r5": 0, "mrr": 0.0})
        bucket["n"] += 1

        results = use_case.execute(qpath, top_k=5)
        rank = next(
            (i + 1 for i, r in enumerate(results) if r.tile.id == truth),
            None,
        )
        if rank == 1:
            r1 += 1
            bucket["r1"] += 1
        if rank is not None and rank <= 5:
            r5 += 1
            bucket["r5"] += 1
        if rank is not None:
            mrr_sum += 1.0 / rank
            bucket["mrr"] += 1.0 / rank

        if rec.catalog_path is not None and Path(rec.catalog_path).name in confusable_names:
            conf_n += 1
            if rank == 1:
                conf_r1 += 1

    elapsed = time.perf_counter() - t0
    n = len(records)
    return GateResult(
        orb_on=orb_on,
        n=n,
        r1=r1,
        r5=r5,
        mrr_sum=mrr_sum,
        by_kind=by_kind,
        confusable_pair_r1=conf_r1,
        confusable_pair_n=conf_n,
        latency_s=elapsed,
    )


def _summarize(result: GateResult) -> dict:
    n = max(1, result.n)
    out = {
        "orb_verification": result.orb_on,
        "n_queries": result.n,
        "recall@1": round(result.r1 / n, 4),
        "recall@5": round(result.r5 / n, 4),
        "mrr": round(result.mrr_sum / n, 4),
        "confusable_marble_pairs": {
            "n": result.confusable_pair_n,
            "recall@1": round(result.confusable_pair_r1 / max(1, result.confusable_pair_n), 4),
        },
        "latency_total_s": round(result.latency_s, 2),
        "by_query_kind": {
            k: {
                "n": int(v["n"]),
                "recall@1": round(v["r1"] / max(1, v["n"]), 4),
                "recall@5": round(v["r5"] / max(1, v["n"]), 4),
                "mrr": round(v["mrr"] / max(1, v["n"]), 4),
            }
            for k, v in sorted(result.by_kind.items())
        },
    }
    return out


def run_gate(manifest: Path, out_dir: Path) -> dict:
    manifest = manifest.resolve()
    records = load_real_customer_manifest(manifest)
    out_dir.mkdir(parents=True, exist_ok=True)

    confusable_names = {
        p.name
        for p in (manifest.parent / "real_customer_release" / "real_catalog").glob("marble_p*.jpg")
    }

    print(f"Loaded {len(records)} real-customer queries from {manifest}")
    uc_on, uc_off, id_map = _bootstrap_catalog(records, out_dir / "catalog")

    res_on = _evaluate_production(
        uc_on, records, orb_on=True, confusable_names=confusable_names, manifest_to_sqlite=id_map
    )
    res_off = _evaluate_production(
        uc_off, records, orb_on=False, confusable_names=confusable_names, manifest_to_sqlite=id_map
    )

    summary = {
        "manifest": str(manifest),
        "production_path": True,
        "orb_on": _summarize(res_on),
        "orb_off": _summarize(res_off),
        "delta_r1": round(res_on.r1 / max(1, res_on.n) - res_off.r1 / max(1, res_off.n), 4),
        "delta_confusable_r1": round(
            res_on.confusable_pair_r1 / max(1, res_on.confusable_pair_n)
            - res_off.confusable_pair_r1 / max(1, res_off.confusable_pair_n),
            4,
        ),
        "recommendation": "",
    }

    # Decision rule: enable ORB only if it improves overall R@1 or confusable-pair R@1
    # without lowering the other metric.
    on_r1 = res_on.r1 / max(1, res_on.n)
    off_r1 = res_off.r1 / max(1, res_off.n)
    on_conf = res_on.confusable_pair_r1 / max(1, res_on.confusable_pair_n)
    off_conf = res_off.confusable_pair_r1 / max(1, res_off.confusable_pair_n)

    if on_r1 > off_r1 or (on_conf > off_conf and on_r1 >= off_r1 - 0.02):
        summary["recommendation"] = "enable_orb_verification=true"
    else:
        summary["recommendation"] = "enable_orb_verification=false"

    (out_dir / "orb_gate_report.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print("\n=== Production SearchTilesUseCase ORB gate ===")
    print(f"ORB off: R@1={summary['orb_off']['recall@1']} confusable R@1={summary['orb_off']['confusable_marble_pairs']['recall@1']}")
    print(f"ORB on:  R@1={summary['orb_on']['recall@1']} confusable R@1={summary['orb_on']['confusable_marble_pairs']['recall@1']}")
    print(f"Recommendation: {summary['recommendation']}")

    # Also run PR #39 bakeoff harness for both ORB settings.
    from dev_tools.search_quality.run_bakeoff import run as run_bakeoff

    for orb_flag, sub in ((True, "bakeoff_orb_on"), (False, "bakeoff_orb_off")):
        print(f"\nRunning bakeoff harness (orb={'on' if orb_flag else 'off'})…")
        bake = run_bakeoff(
            out_dir / sub,
            n_tiles=24,
            n_sheets=12,
            orb_verification=orb_flag,
            real_queries=manifest,
        )
        summary[sub] = {
            "orb_verification": orb_flag,
            "n_queries": bake.get("n_queries"),
            "by_query_kind": bake.get("by_query_kind"),
            "strategies": {
                k: {
                    "recall@1": v.get("recall@1"),
                    "recall@5": v.get("recall@5"),
                    "latency": v.get("latency"),
                }
                for k, v in bake.get("strategies", {}).items()
                if k == "A_primary_only"
            },
        }

    (out_dir / "orb_gate_report.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path("eval/real_customer_release.jsonl"),
    )
    parser.add_argument("--out", type=Path, default=Path("/tmp/real_customer_orb_gate"))
    args = parser.parse_args()
    run_gate(args.manifest, args.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
