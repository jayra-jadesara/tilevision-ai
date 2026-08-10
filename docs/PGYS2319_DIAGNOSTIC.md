# PGYS2319 missing-result diagnostic (client playbook)

Run on the **client machine** (or any host with the re-indexed catalog under
`/Volumes/Siscon Base/DEEP 09-03-26/NEW MARBLE JPEG` and query `xx.jpg`).

Settled production log (do not re-derive):

```
kind=clean_tile views=1
search_k=100, query_crops=1, unique_ids=100
Candidates for reranking: 100
Weak-result filter: kept 10 of 100 (min_raw=0.430)
```

## Prerequisites

- TileVision AI v1.2.34+ (includes `scripts/explain_search.py`)
- DINOv2 weights: `python scripts/download_dinov2_model.py`
- Catalog parent directory containing `database/tiles.db` and `index/tiles.index`

Adjust paths below to match the installed catalog profile.

```bash
export CATALOG="/path/to/catalog/profile"   # parent of database/ + index/
export QUERY="/Users/apple/Desktop/xx.jpg"
export PGYS_SHEET="/Volumes/Siscon Base/DEEP 09-03-26/NEW MARBLE JPEG/PGYS2319.jpg"
```

## Task 1 — Where did PGYS2319 drop out?

```bash
python scripts/explain_search.py "$QUERY" \
  --catalog "$CATALOG" \
  --top 30 \
  --pool-size 100 \
  --find-tile PGYS2319
```

Interpret `--find-tile` output:

| Finding | Meaning | Next step |
|---------|---------|-----------|
| `FAISS pool: ABSENT` | Never in top-100 unique ids | Task 2 (index-side) |
| In pool, rerank rank > 10 | Retrieved but hybrid-scored low | Check component scores (`emb`, `color`, …) |
| In pool, `Weak filter: DROPPED` | Below `min_raw=0.430` | Compare `final` vs floor |
| In pool, rank ≤ 10, kept | Should appear in UI | Look elsewhere (dedup, display cap) |

Widen pool if absent at 100:

```bash
python scripts/explain_search.py "$QUERY" --catalog "$CATALOG" \
  --find-tile PGYS2319 --pool-size 500 --top 30
```

## Task 2 — Index crop inspection

```bash
python scripts/explain_search.py \
  --show-index-crop "$PGYS_SHEET" \
  --output-dir /tmp/index_crop_debug
```

Or resolve by tile stem from the indexed catalog:

```bash
python scripts/explain_search.py \
  --catalog "$CATALOG" \
  --show-index-crop PGYS2319 \
  --output-dir /tmp/index_crop_debug
```

Inspect PNGs under `/tmp/index_crop_debug/`:

- `*_view0_primary.png` — full sheet (primary FAISS vector)
- `*_view*_panel*.png` — left-panel aux vector (critical for texture crops)
- `*_primary_texture_panel.png` — direct `primary_texture_panel()` output
- `*_primary_preprocess_letterbox.png` — letterboxed primary embed input

**Verdict:** panel PNG must show **only the marble slab** (left ~35–45%).
If logo, Chinese text, or photo-grid appear in panel/aux views, the indexed
vector is polluted → index-side fix required (Task 3), not query-side.

Printed metrics to capture:

- `left_panel_beneficial` / `center_crop_beneficial`
- `text_region_score`, `has_preview_grid`
- `primary_texture_panel std` and `mean_abs_delta_full`

## Task 3 — Fix (only after Task 2 confirms bad crops)

Do **not** reindex speculatively. If panel crops include marketing content,
file an issue with:

1. Saved PNGs from Task 2
2. `Image analysis` block from `--show-index-crop`
3. Measured `split_x` / panel std from logs

Likely area: `primary_texture_panel()` left split (~45%) or
`left_panel_beneficial` gate in `image_analysis.py` for asymmetric left-third layouts.

## Task 4 — Windows vs macOS Intel parity

Run **identical** commands on both platforms against the **same** re-indexed
catalog copy. Write JSON snapshots and diff:

**macOS Intel:**

```bash
python scripts/explain_search.py "$QUERY" \
  --catalog "$CATALOG" \
  --top 30 --pool-size 100 --find-tile PGYS2319 \
  --parity-out /tmp/pgys2319_mac.json
```

**Windows:**

```cmd
python scripts\explain_search.py "%QUERY%" ^
  --catalog "%CATALOG%" ^
  --top 30 --pool-size 100 --find-tile PGYS2319 ^
  --parity-out C:\Temp\pgys2319_win.json
```

Compare `report.candidates` ranks/scores and `tile_lookup`. Pure CPU inference
should match within float noise (~1e-4). Large divergences indicate a platform
bug and must be resolved before trusting any index-side fix.

```bash
diff <(jq -S . /tmp/pgys2319_mac.json) <(jq -S . /tmp/pgys2319_win.json)
```

## Combined one-liner (Tasks 1 + 2)

```bash
python scripts/explain_search.py "$QUERY" \
  --catalog "$CATALOG" \
  --top 30 --pool-size 100 --find-tile PGYS2319 \
  --show-index-crop "$PGYS_SHEET" \
  --output-dir /tmp/index_crop_debug \
  --parity-out /tmp/pgys2319_$(uname -s).json
```
