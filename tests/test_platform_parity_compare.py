"""Unit tests for cross-platform bakeoff compare (no DINOv2)."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from dev_tools.search_quality.compare_platform_parity import (
    compare_reports,
    format_table,
)


def _kind(n: int, r1: float, r5: float) -> dict:
    return {
        "n": n,
        "recall@1": r1,
        "recall@5": r5,
        "recall@10": r5,
        "mrr": r1,
    }


def _report(platform: str, overall_r1: float, kinds: dict[str, float]) -> dict:
    by_kind = {
        name: _kind(10, r1, min(1.0, r1 + 0.1)) for name, r1 in kinds.items()
    }
    return {
        "platform": platform,
        "n_queries": sum(v["n"] for v in by_kind.values()),
        "recall@1": overall_r1,
        "recall@5": min(1.0, overall_r1 + 0.05),
        "by_query_kind": by_kind,
        "fusion": "max",
        "index_strategy": "production_v8",
        "orb_verification": True,
        "n_catalog_tiles": 10,
        "runtime": {"machine": "test"},
    }


def test_compare_pass_within_tolerance():
    reports = [
        _report(
            "macos-15-intel",
            0.90,
            {"original": 0.90, "crop_600x600": 0.90, "whatsapp": 0.80},
        ),
        _report(
            "macos-15",
            0.91,
            {"original": 0.90, "crop_600x600": 0.91, "whatsapp": 0.80},
        ),
        _report(
            "windows-latest",
            0.90,
            {"original": 0.90, "crop_600x600": 0.90, "whatsapp": 0.81},
        ),
        _report(
            "ubuntu-latest",
            0.90,
            {"original": 0.90, "crop_600x600": 0.90, "whatsapp": 0.80},
        ),
    ]
    result = compare_reports(
        reports, baseline_label="macos-15-intel", tolerance=0.02
    )
    assert result["verdict"] == "PASS"
    assert result["violations"] == []
    table = format_table(result)
    assert "Verdict: PASS" in table
    assert "macos-15-intel" in table


def test_compare_fail_when_kind_diverges():
    reports = [
        _report(
            "macos-15-intel",
            0.90,
            {"original": 0.90, "crop_600x600": 1.00},
        ),
        _report(
            "windows-latest",
            0.85,
            {"original": 0.90, "crop_600x600": 0.70},  # −0.30 vs baseline
        ),
    ]
    result = compare_reports(
        reports, baseline_label="macos-15-intel", tolerance=0.02
    )
    assert result["verdict"] == "FAIL"
    kinds = {v["query_kind"] for v in result["violations"]}
    assert "crop_600x600" in kinds
    assert "OVERALL" in kinds
    table = format_table(result)
    assert "*" in table
    assert "Verdict: FAIL" in table


def test_compare_requires_baseline(tmp_path: Path):
    reports = [_report("ubuntu-latest", 0.9, {"original": 0.9})]
    with pytest.raises(SystemExit, match="Baseline platform"):
        compare_reports(reports, baseline_label="macos-15-intel")


def test_cli_writes_artifacts(tmp_path: Path, monkeypatch):
    from dev_tools.search_quality import compare_platform_parity as mod

    paths = []
    for plat, r1 in (("macos-15-intel", 0.9), ("ubuntu-latest", 0.9)):
        p = tmp_path / f"{plat}.json"
        p.write_text(
            json.dumps(_report(plat, r1, {"original": 0.9})), encoding="utf-8"
        )
        paths.append(p)
    out = tmp_path / "out"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "compare_platform_parity.py",
            "--reports",
            str(paths[0]),
            str(paths[1]),
            "--baseline",
            "macos-15-intel",
            "--out",
            str(out),
        ],
    )
    assert mod.main() == 0
    assert (out / "platform_parity_report.json").is_file()
    assert (out / "platform_parity_table.txt").is_file()
