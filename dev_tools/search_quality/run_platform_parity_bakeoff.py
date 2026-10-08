#!/usr/bin/env python3
"""
Cross-platform parity bakeoff: one locked production config, one JSON report.

Uses the same BakeoffEngine / real-customer path as ``run_bakeoff.py``, but
locks the knobs that must stay identical across CI runners:

  - index strategy: production_v8
  - fusion: max (shipped default; RRF stays off)
  - ORB near-tie verification: on
  - pooling: cls

Writes ``platform_bakeoff_report.json`` (fusion-compare shape + platform
metadata) for later diffing by ``compare_platform_parity.py``.

Usage::

  PYTHONUNBUFFERED=1 QT_QPA_PLATFORM=offscreen TILEVISION_OFFLINE_MODEL=1 \\
    python -u dev_tools/search_quality/run_platform_parity_bakeoff.py \\
      --real-queries eval/real_customer_release.jsonl \\
      --out /tmp/platform_bakeoff \\
      --platform-label macos-15-intel
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.ai.embedder import DINOv2Embedder
from src.ai.feature_extractor import FeatureExtractor
from src.ai.search_quality.fusion import FusionMethod
from src.ai.search_quality.views import IndexStrategy
from dev_tools.search_quality.run_bakeoff import (
    BakeoffEngine,
    customer_slice,
    metrics_to_dict,
)
from dev_tools.search_quality.real_customer import (
    CATALOG_SOURCE_REAL,
    catalog_items_from_records,
    format_query_kind_table,
    load_real_customer_manifest,
    query_kind_breakdown,
    records_to_golden_queries,
    validate_ground_truth_ids,
)


def _runtime_meta() -> dict:
    import torch

    device = "cpu"
    if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
        device = "mps"
    elif torch.cuda.is_available():
        device = "cuda"
    return {
        "python": sys.version.split()[0],
        "platform_system": platform.system(),
        "platform_release": platform.release(),
        "machine": platform.machine(),
        "processor": platform.processor(),
        "torch": getattr(torch, "__version__", "?"),
        "torch_device": device,
        "git_sha": os.environ.get("GITHUB_SHA")
        or os.environ.get("TILEVISION_GIT_SHA")
        or "",
        "github_runner_os": os.environ.get("RUNNER_OS", ""),
        "github_runner_arch": os.environ.get("RUNNER_ARCH", ""),
    }


def build_report(
    *,
    real_queries: Path,
    out_dir: Path,
    platform_label: str,
    orb_verification: bool = True,
    pooling: str = "cls",
) -> dict:
    out_dir.mkdir(parents=True, exist_ok=True)
    weights = Path("model_weights/dinov2-large/config.json")
    if not weights.is_file():
        raise FileNotFoundError(
            "DINOv2 weights missing at model_weights/dinov2-large/config.json"
        )

    records = load_real_customer_manifest(real_queries)
    items = catalog_items_from_records(records)
    if items is None:
        raise SystemExit(
            "Manifest has no complete catalog_path coverage — cannot run parity bakeoff."
        )
    catalog_by_id = {item.tile_id: item for item in items}
    validate_ground_truth_ids(records, set(catalog_by_id))
    queries = records_to_golden_queries(records)

    strategy = IndexStrategy.PRODUCTION_V8
    fusion = FusionMethod.MAX
    print(
        f"Platform parity bakeoff: platform={platform_label} "
        f"catalog={len(items)} queries={len(queries)} "
        f"strategy={strategy.value} fusion={fusion.value} "
        f"orb={'on' if orb_verification else 'off'}"
    )

    emb = DINOv2Embedder(pooling=pooling)
    emb.load_model()
    engine = BakeoffEngine(FeatureExtractor(embedder=emb))

    mgr, index_s, meta = engine.index_strategy(
        items, strategy, out_dir / f"idx_{strategy.value}.index"
    )
    print(
        f"  index_s={index_s:.1f} vectors={meta.vectors} "
        f"mean_views={meta.mean_views:.2f}"
    )

    metrics = engine.evaluate(
        mgr,
        queries,
        fusion=fusion,
        orb_verification=orb_verification,
        catalog_by_id=catalog_by_id,
    )
    metrics.vectors = meta.vectors
    metrics.mean_views = meta.mean_views

    payload = metrics_to_dict(metrics)
    payload["catalog_source"] = CATALOG_SOURCE_REAL
    payload["by_query_kind"] = query_kind_breakdown(payload)
    payload["customer_path"] = customer_slice(payload)
    payload["index_build_s"] = round(index_s, 2)
    payload["index_strategy"] = strategy.value
    payload["fusion"] = fusion.value
    payload["orb_verification"] = orb_verification
    payload["pooling"] = pooling
    payload["platform"] = platform_label
    payload["runtime"] = _runtime_meta()
    payload["manifest"] = str(real_queries)
    payload["n_catalog_tiles"] = len(items)

    print(
        f"  overall R@1={payload['recall@1']} R@5={payload['recall@5']} "
        f"MRR={payload['mrr']}"
    )
    print("\nPer-query_kind:")
    print(format_query_kind_table(payload["by_query_kind"]))

    report_path = out_dir / "platform_bakeoff_report.json"
    report_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"\nWrote {report_path}")
    return payload


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--real-queries",
        type=Path,
        default=Path("eval/real_customer_release.jsonl"),
    )
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument(
        "--platform-label",
        required=True,
        help="CI matrix label, e.g. macos-15-intel / windows-latest",
    )
    parser.add_argument("--orb-verification", choices=("on", "off"), default="on")
    parser.add_argument("--pooling", default="cls")
    args = parser.parse_args()

    build_report(
        real_queries=args.real_queries,
        out_dir=args.out,
        platform_label=args.platform_label,
        orb_verification=args.orb_verification == "on",
        pooling=args.pooling,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
