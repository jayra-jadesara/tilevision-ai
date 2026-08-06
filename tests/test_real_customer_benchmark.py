"""Tests for real-customer bakeoff manifest loading and reporting."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from dev_tools.search_quality.real_customer import (
    CATALOG_SOURCE_REAL,
    MissingGroundTruthError,
    catalog_items_from_records,
    load_real_customer_manifest,
    low_sample_warning,
    query_kind_breakdown,
    records_to_golden_queries,
    validate_ground_truth_ids,
)
from dev_tools.search_quality.run_bakeoff import metrics_to_dict
from dev_tools.search_quality.run_bakeoff import Metrics


def _write_rgb(path: Path, color: tuple[int, int, int], size: int = 64) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(
        np.full((size, size, 3), color, dtype=np.uint8)
    ).save(path)


def _make_fixture_tree(tmp_path: Path) -> tuple[Path, list]:
    """2 catalog tiles + 3 queries (two kinds) with complete catalog_path."""
    cat = tmp_path / "real_catalog"
    queries = tmp_path / "real_queries"
    _write_rgb(cat / "tile_0001.jpg", (240, 240, 240))
    _write_rgb(cat / "tile_0002.jpg", (40, 80, 200))
    _write_rgb(queries / "whatsapp_001.jpg", (235, 235, 235))
    _write_rgb(queries / "crop_a.jpg", (45, 85, 195))
    _write_rgb(queries / "whatsapp_002.jpg", (230, 230, 230))

    manifest = tmp_path / "real_customer_queries.jsonl"
    rows = [
        {
            "query_path": "real_queries/whatsapp_001.jpg",
            "relevant_ids": [1],
            "query_kind": "whatsapp",
            "catalog_path": "real_catalog/tile_0001.jpg",
        },
        {
            "query_path": "real_queries/crop_a.jpg",
            "true_tile_id": "TILE_0002",
            "query_kind": "crop_600x600",
            "catalog_path": "real_catalog/tile_0002.jpg",
        },
        {
            "query_image": "real_queries/whatsapp_002.jpg",
            "query_id": 1,
            "query_kind": "whatsapp",
            "catalog_path": "real_catalog/tile_0001.jpg",
        },
    ]
    with manifest.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row) + "\n")
    return manifest, rows


def test_missing_true_tile_id_hard_fails_with_clear_message(tmp_path):
    manifest, _ = _make_fixture_tree(tmp_path)
    records = load_real_customer_manifest(manifest)
    # Catalog only has tile 1 — id 2 is missing.
    with pytest.raises(MissingGroundTruthError) as excinfo:
        validate_ground_truth_ids(records, catalog_ids={1})
    msg = str(excinfo.value)
    assert "2" in msg
    assert "missing" in msg.lower()
    assert "deflate" in msg.lower() or "Recall" in msg


def test_query_kind_breakdown_groups_correctly(tmp_path):
    manifest, _ = _make_fixture_tree(tmp_path)
    records = load_real_customer_manifest(manifest)
    items = catalog_items_from_records(records)
    assert items is not None
    assert {i.tile_id for i in items} == {1, 2}
    validate_ground_truth_ids(records, {i.tile_id for i in items})

    golden = records_to_golden_queries(records)
    assert [q.variant for q in golden] == ["whatsapp", "crop_600x600", "whatsapp"]
    assert golden[1].tile_id == 2

    # Simulate metrics_to_dict by_variant from evaluate().
    metrics = Metrics()
    metrics.n = 3
    metrics.r1 = 2
    metrics.r5 = 3
    metrics.mrr = 2.5
    metrics.by_variant["whatsapp"] = {"n": 2, "r1": 1, "r5": 2, "r10": 2, "mrr": 1.5}
    metrics.by_variant["crop_600x600"] = {"n": 1, "r1": 1, "r5": 1, "r10": 1, "mrr": 1.0}
    payload = metrics_to_dict(metrics)
    payload["catalog_source"] = CATALOG_SOURCE_REAL
    breakdown = query_kind_breakdown(payload)

    assert set(breakdown) == {"crop_600x600", "whatsapp"}
    assert breakdown["whatsapp"]["n"] == 2
    assert breakdown["crop_600x600"]["n"] == 1
    assert breakdown["crop_600x600"]["recall@1"] == 1.0
    assert payload["catalog_source"] == "real_customer"


def test_catalog_source_tagged_real_customer_in_report_payload(tmp_path):
    """Report payloads must carry catalog_source=real_customer (not synthetic)."""
    manifest, _ = _make_fixture_tree(tmp_path)
    records = load_real_customer_manifest(manifest)
    items = catalog_items_from_records(records)
    assert items is not None

    metrics = Metrics(vectors=len(items))
    metrics.n = len(records)
    payload = metrics_to_dict(metrics)
    payload["catalog_source"] = CATALOG_SOURCE_REAL
    payload["by_query_kind"] = query_kind_breakdown(payload)

    report = {
        "catalog_source": payload["catalog_source"],
        "n_queries": payload["n_queries"],
        "by_query_kind": payload["by_query_kind"],
        "low_sample_warning": low_sample_warning(payload["n_queries"]),
    }
    assert report["catalog_source"] == "real_customer"
    assert report["catalog_source"] != "synthetic_production_representative"
    assert report["low_sample_warning"] is not None
    assert "low-confidence" in report["low_sample_warning"]


def test_load_accepts_eval_compatible_field_names(tmp_path):
    _write_rgb(tmp_path / "q.jpg", (10, 10, 10))
    _write_rgb(tmp_path / "c.jpg", (20, 20, 20))
    manifest = tmp_path / "m.jsonl"
    manifest.write_text(
        json.dumps(
            {
                "query_path": "q.jpg",
                "relevant_ids": [9],
                "category": "phone_photo",
                "catalog_path": "c.jpg",
            }
        )
        + "\n",
        encoding="utf-8",
    )
    records = load_real_customer_manifest(manifest)
    assert records[0].true_tile_id == 9
    assert records[0].query_kind == "phone_photo"
