#!/usr/bin/env python3
"""
Compare production MAX vs true multi-view RRF on real-customer bakeoff queries.

Indexes once (default: production_v8 + winning B_full_center), then evaluates
MAX and RRF for several ``rrf_k`` values with ORB on. Reports overall and
per-query_kind R@1/R@5 deltas (RRF − MAX).

Usage::

  PYTHONUNBUFFERED=1 QT_QPA_PLATFORM=offscreen TILEVISION_OFFLINE_MODEL=1 \\
    python -u dev_tools/search_quality/compare_fusion_max_rrf.py \\
      --real-queries eval/real_customer_release.jsonl \\
      --out /opt/cursor/artifacts/fusion_max_vs_rrf
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.ai.search_quality.fusion import FusionMethod
from src.ai.search_quality.views import IndexStrategy
from dev_tools.search_quality.run_bakeoff import (
    BakeoffEngine,
    metrics_to_dict,
)
from dev_tools.search_quality.real_customer import (
    CATALOG_SOURCE_REAL,
    customer_slice,
    format_query_kind_table,
    load_real_customer_manifest,
    query_kind_breakdown,
    records_to_catalog_and_queries,
)


def _payload(m, catalog_source: str) -> dict:
    payload = metrics_to_dict(m)
    payload["catalog_source"] = catalog_source
    if catalog_source == CATALOG_SOURCE_REAL:
        payload["by_query_kind"] = query_kind_breakdown(payload)
    payload["customer_path"] = customer_slice(payload)
    return payload


def _delta_table(rrf: dict, mx: dict) -> list[dict]:
    kinds = sorted(
        set((rrf.get("by_query_kind") or {})) | set((mx.get("by_query_kind") or {}))
    )
    rows = []
    for kind in kinds:
        a = (rrf.get("by_query_kind") or {}).get(kind, {})
        b = (mx.get("by_query_kind") or {}).get(kind, {})
        rows.append(
            {
                "query_kind": kind,
                "n": a.get("n") or b.get("n") or 0,
                "rrf_r1": a.get("recall@1", 0.0),
                "max_r1": b.get("recall@1", 0.0),
                "d_r1": round(a.get("recall@1", 0.0) - b.get("recall@1", 0.0), 4),
                "rrf_r5": a.get("recall@5", 0.0),
                "max_r5": b.get("recall@5", 0.0),
                "d_r5": round(a.get("recall@5", 0.0) - b.get("recall@5", 0.0), 4),
            }
        )
    return rows


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--real-queries", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument(
        "--strategies",
        default="production_v8,B_full_center",
        help="Comma-separated IndexStrategy values to index+evaluate",
    )
    parser.add_argument(
        "--rrf-k",
        default="30,60,90",
        help="Comma-separated RRF k values to try",
    )
    parser.add_argument("--orb-verification", choices=("on", "off"), default="on")
    parser.add_argument("--pooling", default="cls")
    args = parser.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)
    records = load_real_customer_manifest(args.real_queries)
    items, queries, catalog_source = records_to_catalog_and_queries(records)
    catalog_by_id = {item.tile_id: item for item in items}
    orb = args.orb_verification == "on"
    k_values = [max(1, int(x)) for x in args.rrf_k.split(",") if x.strip()]
    strategies = [s.strip() for s in args.strategies.split(",") if s.strip()]

    print(
        f"Real-customer fusion compare: catalog={len(items)} queries={len(queries)} "
        f"orb={orb} strategies={strategies} rrf_k={k_values}"
    )
    engine = BakeoffEngine(pooling=args.pooling)

    report: dict = {
        "n_queries": len(queries),
        "catalog_size": len(items),
        "orb_verification": orb,
        "strategies": {},
        "note": (
            "True multi-view RRF (per-view ranks). ORB banding uses MAX cosine "
            "so score-scale artifacts from the old bakeoff path are avoided. "
            "DEFAULT production remains max until explicitly enabled."
        ),
    }

    for strat_name in strategies:
        strategy = IndexStrategy(strat_name)
        print(f"\n=== Index {strategy.value} ===")
        mgr, index_s, meta = engine.index_strategy(
            items, strategy, args.out / f"idx_{strategy.value}.index"
        )
        print(
            f"  index_s={index_s:.1f} vectors={meta.vectors} "
            f"mean_views={meta.mean_views:.2f}"
        )

        max_m = engine.evaluate(
            mgr,
            queries,
            fusion=FusionMethod.MAX,
            orb_verification=orb,
            catalog_by_id=catalog_by_id,
        )
        max_m.vectors = meta.vectors
        max_m.mean_views = meta.mean_views
        max_payload = _payload(max_m, catalog_source)
        print(
            f"  MAX  R@1={max_payload['recall@1']} R@5={max_payload['recall@5']} "
            f"custR@1={max_payload['customer_path']['recall@1']}"
        )
        print(format_query_kind_table(max_payload.get("by_query_kind") or {}))

        strat_report: dict = {
            "index_build_s": round(index_s, 2),
            "vectors": meta.vectors,
            "mean_views": meta.mean_views,
            "max": max_payload,
            "rrf": {},
        }

        best_k = None
        best_key = None
        for k in k_values:
            rrf_m = engine.evaluate(
                mgr,
                queries,
                fusion=FusionMethod.RRF,
                rrf_k=k,
                orb_verification=orb,
                catalog_by_id=catalog_by_id,
            )
            rrf_m.vectors = meta.vectors
            rrf_m.mean_views = meta.mean_views
            rrf_payload = _payload(rrf_m, catalog_source)
            deltas = _delta_table(rrf_payload, max_payload)
            regressions = [
                d
                for d in deltas
                if d["d_r1"] < -1e-9 or d["d_r5"] < -1e-9
            ]
            focus = [
                d
                for d in deltas
                if d["query_kind"].startswith("crop_")
                or d["query_kind"] in {"original", "clean_tile"}
            ]
            focus_reg = [
                d for d in focus if d["d_r1"] < -1e-9 or d["d_r5"] < -1e-9
            ]
            entry = {
                "rrf_k": k,
                "metrics": rrf_payload,
                "delta_vs_max": {
                    "recall@1": round(
                        rrf_payload["recall@1"] - max_payload["recall@1"], 4
                    ),
                    "recall@5": round(
                        rrf_payload["recall@5"] - max_payload["recall@5"], 4
                    ),
                    "customer_recall@1": round(
                        rrf_payload["customer_path"]["recall@1"]
                        - max_payload["customer_path"]["recall@1"],
                        4,
                    ),
                    "customer_recall@5": round(
                        rrf_payload["customer_path"]["recall@5"]
                        - max_payload["customer_path"]["recall@5"],
                        4,
                    ),
                },
                "by_query_kind_delta": deltas,
                "any_kind_regression": bool(regressions),
                "crop_original_regression": bool(focus_reg),
            }
            strat_report["rrf"][str(k)] = entry
            print(
                f"  RRF k={k} R@1={rrf_payload['recall@1']} "
                f"R@5={rrf_payload['recall@5']} "
                f"dR@1={entry['delta_vs_max']['recall@1']:+.4f} "
                f"dR@5={entry['delta_vs_max']['recall@5']:+.4f} "
                f"kind_reg={len(regressions)} crop/orig_reg={len(focus_reg)}"
            )
            key = (
                entry["delta_vs_max"]["customer_recall@5"],
                entry["delta_vs_max"]["customer_recall@1"],
                entry["delta_vs_max"]["recall@5"],
                entry["delta_vs_max"]["recall@1"],
                -int(entry["crop_original_regression"]),
                -int(entry["any_kind_regression"]),
            )
            if best_key is None or key > best_key:
                best_key = key
                best_k = k

        strat_report["best_rrf_k"] = best_k
        report["strategies"][strategy.value] = strat_report

        if best_k is not None:
            best = strat_report["rrf"][str(best_k)]
            print(f"\n  Best k for {strategy.value}: {best_k}")
            print(
                f"  {'query_kind':<28} {'n':>4} {'dR@1':>7} {'dR@5':>7} "
                f"{'maxR@1':>7} {'rrfR@1':>7}"
            )
            print("-" * 70)
            for d in best["by_query_kind_delta"]:
                print(
                    f"{d['query_kind']:<28} {d['n']:4d} {d['d_r1']:+7.4f} "
                    f"{d['d_r5']:+7.4f} {d['max_r1']:7.4f} {d['rrf_r1']:7.4f}"
                )

    out_path = args.out / "fusion_max_vs_rrf_report.json"
    out_path.write_text(json.dumps(report, indent=2))
    print(f"\nWrote {out_path}")

    # Recommendation stub from production_v8 if present
    pv = report["strategies"].get("production_v8") or next(
        iter(report["strategies"].values()), None
    )
    if pv and pv.get("best_rrf_k") is not None:
        best = pv["rrf"][str(pv["best_rrf_k"])]
        d = best["delta_vs_max"]
        print("\n=== Recommendation (from primary strategy) ===")
        if (
            d["recall@1"] > 1e-9
            and not best["crop_original_regression"]
            and not best["any_kind_regression"]
        ):
            print(
                f"RRF k={pv['best_rrf_k']} improves overall R@1 by "
                f"{d['recall@1']:+.4f} with no kind regressions — "
                "safe to keep opt-in; consider default-on after client confirm."
            )
        elif d["recall@1"] > 1e-9 and best["crop_original_regression"]:
            print(
                f"RRF k={pv['best_rrf_k']} improves overall but regresses "
                "crop_*/original — keep DEFAULT max; do not flip."
            )
        else:
            print(
                f"No clean R@1 gain for RRF on this catalog (best dR@1="
                f"{d['recall@1']:+.4f}). Keep DEFAULT max; leave RRF opt-in."
            )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
