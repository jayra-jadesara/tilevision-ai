"""
Real-customer query manifest loading for the search bakeoff harness.

Compatible with eval/queries.example.jsonl field names:
  - query_path (required; query_image accepted as alias)
  - relevant_ids (preferred ground truth) OR true_tile_id / query_id
  - query_kind (preferred category tag; category accepted as alias)

Does not load or store customer photo bytes — only paths + labels.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

from dev_tools.search_quality.golden_dataset import CatalogItem, GoldenQuery

# Suggested tags (free-text; unknown values still report fine).
KNOWN_QUERY_KINDS = frozenset(
    {
        "original",
        "crop_600x600",
        "crop_600x1200",
        "catalogue_page",
        "phone_photo",
        "room_photo",
        "whatsapp",
        "low_quality_jpeg",
        "perspective_distortion",
    }
)

CATALOG_SOURCE_REAL = "real_customer"
CATALOG_SOURCE_SYNTHETIC = "synthetic_production_representative"

# Below this, Recall numbers are low-confidence — warn loudly.
LOW_SAMPLE_WARNING_THRESHOLD = 30


@dataclass(frozen=True, slots=True)
class RealCustomerRecord:
    query_path: Path
    true_tile_id: int
    query_kind: str
    catalog_path: Path | None = None


class MissingGroundTruthError(ValueError):
    """Raised when one or more true_tile_id values are absent from the catalog."""


def parse_tile_id(raw) -> int:
    """Accept int, numeric string, or TILE_00231-style labels."""
    if isinstance(raw, bool):
        raise ValueError(f"Invalid tile id: {raw!r}")
    if isinstance(raw, int):
        return int(raw)
    text = str(raw).strip()
    if not text:
        raise ValueError("Empty tile id")
    if text.isdigit() or (text[0] == "-" and text[1:].isdigit()):
        return int(text)
    match = re.search(r"(\d+)$", text)
    if match:
        return int(match.group(1))
    raise ValueError(f"Cannot parse tile id from {raw!r}")


def _resolve_path(raw: str, base_dir: Path) -> Path:
    path = Path(raw)
    if path.is_absolute():
        return path
    # Prefer relative to manifest dir, then cwd.
    candidate = (base_dir / path).resolve()
    if candidate.exists():
        return candidate
    cwd_candidate = (Path.cwd() / path).resolve()
    if cwd_candidate.exists():
        return cwd_candidate
    return candidate


def load_real_customer_manifest(manifest_path: Path) -> list[RealCustomerRecord]:
    """
    Load a JSONL ground-truth manifest for real customer photos.

    Field conventions (compatible with eval/*.jsonl):
      query_path | query_image
      relevant_ids (list) | true_tile_id | query_id
      query_kind | category
      catalog_path (optional — catalog image for this true_tile_id)
    """
    path = Path(manifest_path)
    if not path.is_file():
        raise FileNotFoundError(f"Real-customer manifest not found: {path}")

    base_dir = path.parent
    records: list[RealCustomerRecord] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, start=1):
            text = line.strip()
            if not text or text.startswith("#"):
                continue
            try:
                obj = json.loads(text)
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"{path}:{line_no}: invalid JSON — {exc}"
                ) from exc
            if not isinstance(obj, dict):
                raise ValueError(f"{path}:{line_no}: expected a JSON object")

            raw_query = obj.get("query_path") or obj.get("query_image")
            if not raw_query:
                raise ValueError(
                    f"{path}:{line_no}: missing query_path (or query_image alias)"
                )

            true_id: int | None = None
            if "true_tile_id" in obj and obj["true_tile_id"] is not None:
                true_id = parse_tile_id(obj["true_tile_id"])
            elif obj.get("relevant_ids"):
                true_id = parse_tile_id(obj["relevant_ids"][0])
            elif "query_id" in obj and obj["query_id"] is not None:
                true_id = parse_tile_id(obj["query_id"])
            if true_id is None:
                raise ValueError(
                    f"{path}:{line_no}: missing ground truth "
                    "(true_tile_id, relevant_ids, or query_id)"
                )

            kind = (
                str(obj.get("query_kind") or obj.get("category") or "unknown")
                .strip()
                .lower()
            )
            if not kind:
                kind = "unknown"

            catalog_raw = obj.get("catalog_path")
            catalog_path = (
                _resolve_path(str(catalog_raw), base_dir) if catalog_raw else None
            )

            records.append(
                RealCustomerRecord(
                    query_path=_resolve_path(str(raw_query), base_dir),
                    true_tile_id=true_id,
                    query_kind=kind,
                    catalog_path=catalog_path,
                )
            )

    if not records:
        raise ValueError(f"No queries found in manifest: {path}")
    return records


def catalog_items_from_records(
    records: Iterable[RealCustomerRecord],
) -> list[CatalogItem] | None:
    """
    Build CatalogItem list from per-row catalog_path when every unique
    true_tile_id has at least one path. Returns None if incomplete.
    """
    by_id: dict[int, Path] = {}
    for rec in records:
        if rec.catalog_path is None:
            continue
        by_id.setdefault(rec.true_tile_id, rec.catalog_path)

    needed = {rec.true_tile_id for rec in records}
    if not needed or needed - set(by_id):
        return None

    items: list[CatalogItem] = []
    for tile_id in sorted(by_id):
        path = by_id[tile_id]
        if not path.is_file():
            raise FileNotFoundError(
                f"Catalog image for tile_id={tile_id} not found: {path}"
            )
        items.append(CatalogItem(tile_id=tile_id, kind="tile", path=path))
    return items


def validate_ground_truth_ids(
    records: Iterable[RealCustomerRecord],
    catalog_ids: set[int],
) -> None:
    """Hard-fail listing every missing true_tile_id (never silently skip)."""
    missing = sorted({rec.true_tile_id for rec in records} - set(catalog_ids))
    if missing:
        raise MissingGroundTruthError(
            "Real-customer manifest references tile id(s) missing from the "
            f"indexed catalog: {missing}. Fix the labels or index those tiles "
            "before running the bakeoff — silent skips would deflate Recall."
        )


def records_to_golden_queries(
    records: Iterable[RealCustomerRecord],
) -> list[GoldenQuery]:
    """Map manifest rows onto GoldenQuery (variant = query_kind)."""
    queries: list[GoldenQuery] = []
    for rec in records:
        if not rec.query_path.is_file():
            raise FileNotFoundError(f"Query image not found: {rec.query_path}")
        queries.append(
            GoldenQuery(
                tile_id=rec.true_tile_id,
                variant=rec.query_kind,
                path=rec.query_path,
                kind="tile",
            )
        )
    return queries


def low_sample_warning(n_queries: int) -> str | None:
    if n_queries < LOW_SAMPLE_WARNING_THRESHOLD:
        return (
            f"WARNING: only {n_queries} real-customer queries — Recall@1/@5 "
            f"and MRR are low-confidence below ~{LOW_SAMPLE_WARNING_THRESHOLD} "
            "samples. Do not treat these as headline production numbers."
        )
    return None


def query_kind_breakdown(metrics_payload: dict) -> dict:
    """
    Rename by_variant → by_query_kind for real-customer reports.

    Reuses the existing metrics_to_dict breakdown without duplicating scoring.
    """
    by_variant = metrics_payload.get("by_variant") or {}
    return {
        kind: {
            "n": stats.get("n", 0),
            "recall@1": stats.get("recall@1", 0.0),
            "recall@5": stats.get("recall@5", 0.0),
            "recall@10": stats.get("recall@10", 0.0),
            "mrr": stats.get("mrr", 0.0),
        }
        for kind, stats in sorted(by_variant.items())
    }


def format_query_kind_table(breakdown: dict) -> str:
    """Human-readable per-query_kind table for console / docs paste."""
    lines = [
        f"{'query_kind':<28} {'n':>4} {'R@1':>7} {'R@5':>7} {'MRR':>7}",
        "-" * 56,
    ]
    for kind, stats in breakdown.items():
        lines.append(
            f"{kind:<28} {stats['n']:>4} "
            f"{stats['recall@1']:>7.4f} {stats['recall@5']:>7.4f} "
            f"{stats['mrr']:>7.4f}"
        )
    return "\n".join(lines)
