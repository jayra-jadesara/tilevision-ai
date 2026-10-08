#!/usr/bin/env python3
"""
Diff platform bakeoff JSON reports against a baseline (default: macos-15-intel).

Prints one row per query_kind (+ overall), one column per platform, and flags
any R@1 cell whose absolute delta vs baseline exceeds ``--tolerance``
(default 0.02 = 2 percentage points).

Usage::

  python dev_tools/search_quality/compare_platform_parity.py \\
    --reports artifacts/macos-15-intel/platform_bakeoff_report.json \\
              artifacts/macos-15/platform_bakeoff_report.json \\
              artifacts/windows-latest/platform_bakeoff_report.json \\
              artifacts/ubuntu-latest/platform_bakeoff_report.json \\
    --baseline macos-15-intel \\
    --out /tmp/platform_parity
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


DEFAULT_TOLERANCE = 0.02  # 2 percentage points on R@1


def _load_report(path: Path) -> dict:
    data = json.loads(path.read_text(encoding="utf-8"))
    if "platform" not in data:
        raise ValueError(f"{path}: missing 'platform' field")
    if "by_query_kind" not in data:
        raise ValueError(f"{path}: missing 'by_query_kind'")
    return data


def _r1(report: dict, kind: str | None) -> float:
    if kind is None:
        return float(report.get("recall@1", 0.0))
    stats = (report.get("by_query_kind") or {}).get(kind) or {}
    return float(stats.get("recall@1", 0.0))


def _r5(report: dict, kind: str | None) -> float:
    if kind is None:
        return float(report.get("recall@5", 0.0))
    stats = (report.get("by_query_kind") or {}).get(kind) or {}
    return float(stats.get("recall@5", 0.0))


def _n(report: dict, kind: str | None) -> int:
    if kind is None:
        return int(report.get("n_queries", 0))
    stats = (report.get("by_query_kind") or {}).get(kind) or {}
    return int(stats.get("n", 0))


def compare_reports(
    reports: list[dict],
    *,
    baseline_label: str,
    tolerance: float = DEFAULT_TOLERANCE,
) -> dict:
    by_platform = {r["platform"]: r for r in reports}
    if baseline_label not in by_platform:
        raise SystemExit(
            f"Baseline platform {baseline_label!r} not found in reports: "
            f"{sorted(by_platform)}"
        )
    baseline = by_platform[baseline_label]
    platforms = [baseline_label] + sorted(
        p for p in by_platform if p != baseline_label
    )

    kinds = sorted(
        {
            kind
            for report in reports
            for kind in (report.get("by_query_kind") or {})
        }
    )
    rows: list[dict] = []
    violations: list[dict] = []

    def _row(kind: str | None, label: str) -> dict:
        cells = {}
        base_r1 = _r1(baseline, kind)
        base_r5 = _r5(baseline, kind)
        for plat in platforms:
            report = by_platform[plat]
            r1 = _r1(report, kind)
            r5 = _r5(report, kind)
            d_r1 = round(r1 - base_r1, 4)
            d_r5 = round(r5 - base_r5, 4)
            flagged = abs(d_r1) > tolerance + 1e-12
            cells[plat] = {
                "recall@1": r1,
                "recall@5": r5,
                "d_r1_vs_baseline": d_r1,
                "d_r5_vs_baseline": d_r5,
                "flagged": flagged,
            }
            if flagged and plat != baseline_label:
                violations.append(
                    {
                        "query_kind": label,
                        "platform": plat,
                        "baseline": baseline_label,
                        "r1": r1,
                        "baseline_r1": base_r1,
                        "d_r1": d_r1,
                        "tolerance": tolerance,
                    }
                )
        return {
            "query_kind": label,
            "n": _n(baseline, kind),
            "cells": cells,
        }

    rows.append(_row(None, "OVERALL"))
    for kind in kinds:
        rows.append(_row(kind, kind))

    verdict = "PASS" if not violations else "FAIL"
    return {
        "baseline": baseline_label,
        "tolerance_r1": tolerance,
        "platforms": platforms,
        "rows": rows,
        "violations": violations,
        "verdict": verdict,
        "runtime_by_platform": {
            p: by_platform[p].get("runtime") or {} for p in platforms
        },
        "config_by_platform": {
            p: {
                "fusion": by_platform[p].get("fusion"),
                "index_strategy": by_platform[p].get("index_strategy"),
                "orb_verification": by_platform[p].get("orb_verification"),
                "n_queries": by_platform[p].get("n_queries"),
                "n_catalog_tiles": by_platform[p].get("n_catalog_tiles"),
                "recall@1": by_platform[p].get("recall@1"),
                "recall@5": by_platform[p].get("recall@5"),
            }
            for p in platforms
        },
    }


def format_table(result: dict) -> str:
    platforms = result["platforms"]
    baseline = result["baseline"]
    tol = result["tolerance_r1"]
    header_parts = [f"{'query_kind':<28}", f"{'n':>4}"]
    for plat in platforms:
        header_parts.append(f"{plat:>16}")
    lines = [
        f"Platform parity vs baseline={baseline} (R@1 tolerance={tol:.0%})",
        " ".join(header_parts),
        "-" * (28 + 5 + 17 * len(platforms)),
    ]

    for row in result["rows"]:
        parts = [f"{row['query_kind']:<28}", f"{row['n']:>4}"]
        for plat in platforms:
            cell = row["cells"][plat]
            r1 = cell["recall@1"]
            if plat == baseline:
                parts.append(f"{r1:>16.4f}")
            else:
                mark = " *" if cell["flagged"] else "  "
                parts.append(
                    f"{r1:>7.4f}{cell['d_r1_vs_baseline']:+.3f}{mark}"
                )
        lines.append(" ".join(parts))

    lines.append("")
    lines.append(
        f"Verdict: {result['verdict']} "
        f"({len(result['violations'])} R@1 cell(s) beyond ±{tol:.0%})"
    )
    if result["violations"]:
        lines.append("Violations:")
        for v in result["violations"]:
            lines.append(
                f"  - {v['query_kind']} @ {v['platform']}: "
                f"R@1={v['r1']:.4f} vs baseline {v['baseline_r1']:.4f} "
                f"(Δ={v['d_r1']:+.4f})"
            )
    else:
        lines.append(
            "All platforms match the baseline within tolerance on every "
            "query_kind and overall."
        )
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--reports",
        type=Path,
        nargs="+",
        required=True,
        help="Paths to platform_bakeoff_report.json files",
    )
    parser.add_argument(
        "--baseline",
        default="macos-15-intel",
        help="Platform label used as the parity baseline",
    )
    parser.add_argument(
        "--tolerance",
        type=float,
        default=DEFAULT_TOLERANCE,
        help=(
            "Max absolute R@1 delta vs baseline before flagging "
            f"(default {DEFAULT_TOLERANCE} = 2pp). Chosen as a small buffer "
            "above single-query flip noise on n≈10 per kind (1/10=0.10 would "
            "hide real gaps; 0.00 is too tight for float/ORB nondeterminism)."
        ),
    )
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()

    reports = [_load_report(p) for p in args.reports]
    result = compare_reports(
        reports, baseline_label=args.baseline, tolerance=args.tolerance
    )
    table = format_table(result)
    print(table)

    if args.out is not None:
        args.out.mkdir(parents=True, exist_ok=True)
        (args.out / "platform_parity_report.json").write_text(
            json.dumps(result, indent=2), encoding="utf-8"
        )
        (args.out / "platform_parity_table.txt").write_text(
            table + "\n", encoding="utf-8"
        )
        print(f"\nWrote {args.out / 'platform_parity_report.json'}")
        print(f"Wrote {args.out / 'platform_parity_table.txt'}")

    return 0 if result["verdict"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
