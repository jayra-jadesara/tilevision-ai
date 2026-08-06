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

## Command

```bash
python3 dev_tools/search_quality/run_bakeoff.py \
  --real-queries eval/real_customer_queries.jsonl \
  --out /opt/cursor/artifacts/real_customer \
  --orb-verification on
```

Uses the same bakeoff evaluate path and `--orb-verification` flag as the
synthetic run. Report field `catalog_source` is set to `real_customer`.

If the manifest has fewer than ~30 queries, the harness prints a visible
low-confidence warning — do not treat tiny runs as headline numbers.

## Measured (fill after a real run)

| query_kind | n | Recall@1 | Recall@5 | MRR |
|------------|---|----------|----------|-----|
| original | | | | |
| crop_600x600 | | | | |
| crop_600x1200 | | | | |
| catalogue_page | | | | |
| phone_photo | | | | |
| room_photo | | | | |
| whatsapp | | | | |
| low_quality_jpeg | | | | |
| perspective_distortion | | | | |
| **overall** | | | | |

*(Leave blank until a real anonymized customer set is evaluated — do not
fabricate numbers.)*

## Platforms

The harness is pure Python / OpenCV / NumPy and must run identically on
Windows, macOS Intel, and Apple Silicon. **Photo collection does not need to
be repeated per platform** — index the customer catalog once, run the same
manifest on each OS for parity checks. Do not maintain three separate
customer photo sets.
