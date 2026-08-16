# TileVision AI — Real Customer Image Benchmark

**Scope:** bakeoff harness / reporting only. No search-logic changes.  
**catalog_source tag:** `real_customer` (never mix with
`synthetic_production_representative` tables).

## Why

Every accuracy number in this repo so far comes from synthetic /
auto-generated query variants. Real customer photos (WhatsApp compression,
phone photos, perspective distortion, room installs) need a human-labeled
ground-truth manifest — auto-labeling cannot invent truth for photos the
catalog never transformed.

## Collecting a manifest

1. Index the **real** catalogue once (any platform — Windows / macOS Intel /
   Apple Silicon are equivalent for indexing).
2. For each anonymized customer photo, record which catalog tile is the
   correct match (`true_tile_id` = the tile's integer FAISS / SQLite id, or a
   `TILE_00231`-style label whose trailing digits match that id).
3. Tag the photo with a `query_kind` from (at least):
   `original`, `crop_600x600`, `crop_600x1200`, `catalogue_page`,
   `phone_photo`, `room_photo`, `whatsapp`, `low_quality_jpeg`,
   `perspective_distortion`.
4. Store photos under `eval/real_queries/` (gitignored — never commit).
5. Optionally set `catalog_path` per row so the bakeoff can build the index
   from those files without a separate synthetic catalog.

### Schema (compatible with `eval/queries.example.jsonl`)

```json
{"query_path": "real_queries/whatsapp_001.jpg", "relevant_ids": [231], "query_kind": "whatsapp", "catalog_path": "real_catalog/tile_0231.jpg"}
{"query_path": "real_queries/crop_014.jpg", "true_tile_id": "TILE_00987", "query_kind": "crop_600x600", "catalog_path": "real_catalog/tile_0987.jpg"}
```

| Field | Required | Notes |
|-------|----------|-------|
| `query_path` | yes | Alias: `query_image` |
| `relevant_ids` / `true_tile_id` / `query_id` | one of | Primary truth = first `relevant_ids` entry or `true_tile_id` |
| `query_kind` | yes | Alias: `category` (free-text tag for breakdown) |
| `catalog_path` | recommended | Catalog image for that `true_tile_id` |

See `eval/real_customer_queries.example.jsonl`.

**Missing ground-truth IDs hard-fail** with a clear list — they are never
silently skipped (that would deflate Recall).

## Release eval set (v1.2.34)

A committed, reproducible stand-in set lives at:

- Manifest: `eval/real_customer_release.jsonl`
- Catalog/queries: `eval/real_customer_release/` (synthetic marble pairs +
  customer-like variants — safe to commit, not showroom PII)
- Builder: `python3 dev_tools/search_quality/build_real_customer_eval_set.py`
- ORB gate: `python3 dev_tools/search_quality/run_real_customer_orb_gate.py`

Regenerate after editing pair seeds or query specs.

## Command

```bash
python3 dev_tools/search_quality/run_bakeoff.py \
  --real-queries eval/real_customer_release.jsonl \
  --out /opt/cursor/artifacts/real_customer \
  --orb-verification on
```

For anonymized showroom photos, use a private manifest under
`eval/real_queries/` (gitignored) with the same schema.

Uses the same bakeoff evaluate path and `--orb-verification` flag as the
synthetic run. Report field `catalog_source` is set to `real_customer`.

If the manifest has fewer than ~30 queries, the harness prints a visible
low-confidence warning — do not treat tiny runs as headline numbers.

## Measured — post PR #44-47 (panel isolation + batch-indexing + crop-tool fixes), 2026-08-14

Same manifest (`eval/real_customer_release.jsonl`, n=70), same command, run
after: catalog-sheet panel isolation wired into primary extraction (v14-16),
batch-indexing descriptor divergence fix (v17), Auto/Precise/Manual crop
over-crop fix (PR #46), crop-tool multi-view latency fix (PR #47).

| query_kind | n | R@1 (ORB off) | R@1 (ORB on) | R@5 (both) |
|------------|---|---------------|--------------|------------|
| original | 10 | 0.9000 | 0.9000 | 1.0000 |
| crop_600x600 | 10 | 0.9000 | 0.9000 | 1.0000 |
| crop_600x1200 | 10 | 0.8000 | 0.9000 | 1.0000 |
| whatsapp | 10 | 0.8000 | 1.0000 | 1.0000 |
| phone_photo | 10 | 0.7000 | 0.8000 | 1.0000 |
| low_quality_jpeg | 10 | 0.9000 | 1.0000 | 1.0000 |
| perspective_distortion | 8 | 0.6250 | 1.0000 | 1.0000 |
| catalogue_page | 2 | 0.5000 | 0.5000 | 1.0000 |
| **overall** | **70** | **0.7714** | **0.9143** | **1.0000** |

Winning index strategy: `B_full_center`. `perspective_distortion` — the
weakest category at the original ORB-on decision (0.375) — improved to a
perfect 1.0, the single largest gain in this run. R@5 is 1.0 across every
category with or without ORB.

**Caveats, not yet resolved:**
- Manifest catalog is 10 tiles (`catalog_path` entries in the manifest),
  far smaller than the ~328-tile production catalog — recall typically gets
  harder with more visually-similar competitors in the pool. Treat this as
  strong directional evidence, not production-scale proof.
- `catalogue_page` n=2 is too small to read anything into the flat 0.5.
- Full JSON reports: `bakeoff_report.json` under the `--out` paths used for
  this run (ORB on / off), not committed to the repo.
  
## Measured — production path (release gate, n=70)

Full `SearchTilesUseCase` (hybrid rerank + optional ORB), confusable marble
catalog. Source: `run_real_customer_orb_gate.py` on `eval/real_customer_release.jsonl`.

| query_kind | n | R@1 (ORB off) | R@1 (ORB on) | R@5 (both) |
|------------|---|---------------|--------------|------------|
| original | 10 | 0.7000 | 0.8000 | 0.9000 |
| crop_600x600 | 10 | 0.6000 | 0.7000 | 0.9000 |
| crop_600x1200 | 10 | 0.6000 | 0.7000 | 0.9000 |
| whatsapp | 10 | 0.4000 | 0.6000 | 0.8000 |
| phone_photo | 10 | 0.4000 | 0.6000 | 1.0000 |
| low_quality_jpeg | 10 | 0.5000 | 0.6000 | 0.8000 |
| perspective_distortion | 8 | 0.3750 | 0.3750 | 0.7500 |
| catalogue_page | 2 | 0.5000 | 0.5000 | 1.0000 |
| **overall** | **70** | **0.5143** | **0.6286** | **0.8714** |
| **confusable pairs** | **56** | **0.5179** | **0.6607** | — |

Replace rows above when re-running the gate after manifest updates.

## Platforms

The harness is pure Python / OpenCV / NumPy and must run identically on
Windows, macOS Intel, and Apple Silicon. **Photo collection does not need to
be repeated per platform** — index the customer catalog once, run the same
manifest on each OS for parity checks. Do not maintain three separate
customer photo sets.
