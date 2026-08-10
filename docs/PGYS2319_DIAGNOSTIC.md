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

## Fix (v11, feature_version bump → Rebuild Search Index required)

Root cause on real `PGYS2319.jpg` was **two independent gate failures**:

1. **`text_region_score` undercounting (~0.024):** the legacy detector averaged
   single-threshold Canny edge density over the entire right 45%. Sparse gold
   logo + small caption blocks in the top band were diluted by white margin and
   photo-grid cells. Fixed with multi-threshold top-band Canny, horizontal Sobel
   row activity, and slab-vs-marketing column contrast gating.

2. **`aspect >= 1.12` hard floor:** real sheet is aspect **1.063** — below the
   old threshold even with `has_preview_grid=True`. Preview-grid sheets now use
   a **1.03** aspect floor; `primary_texture_panel()` shares the same gate via
   `marketing_sheet_panel_eligible()`.

Additional safeguard: center-crop aux is blocked on wide preview-grid sheets
(center region includes grid photos).

After upgrading, customers must **Rebuild Search Index** (feature_version **13**).

`panel` top-caption band: **13%** of panel height shaved after content crop
(v12 used 10%, which left a partial clipped caption line on real PGYS2319 at
2x zoom). Accept only after zoomed top-strip inspection — see below.

## Which index views become live FAISS vectors

For `catalog_sheet` tiles indexed via Strategy E (`extract_index_vectors` in
`src/ai/feature_extractor.py`):

| View | Embedded? | Notes |
|------|-----------|-------|
| `primary` | **Yes** (always) | Full sheet via `extract()` — stored in SQLite features + FAISS |
| `panel` | **Yes** (if not near-dup) | Aux vector from `primary_texture_panel()` — **live FAISS id** |
| `panel_center` | **Yes** (if not near-dup) | 72% center of panel — separate aux vector |
| `adaptive` | **Yes** (if not near-dup) | Content-region crop |
| `center` | Only when `center_crop_beneficial` | Blocked on preview-grid marketing sheets |

`panel` is **not** a debug-only artifact — it is embedded and passed to
`vector_index.update_vectors()` unless `_maybe_append_aux` drops it as a
near-duplicate of primary or another aux (cosine ≥ 0.985 / 0.99). Because
caption-contaminated `panel` and clean `panel_center` differ enough, **both**
typically become live vectors. v12+ shaves a top/left caption band from
`primary_texture_panel()` so the `panel` aux is marble-only (v13: top band
13%, verified at 2x zoom on real PGYS2319).

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

**Zoom acceptance check (required — do not skip):** a full-size glance misses
residual caption bleed. After `show_index_crop.py`, extract and 2x-zoom the
top strip of `*_view1_panel.png`:

```python
from PIL import Image
img = Image.open("/tmp/index_crop_debug/PGYS2319_view1_panel.png")
strip = img.crop((0, 0, img.width, 80)).resize((img.width * 2, 160))
strip.save("/tmp/top_strip_check.png")
```

Inspect `/tmp/top_strip_check.png` directly — zero text strokes before closing.
Repeat for the bottom 80px (`top = img.height - 80`) to confirm the larger
top cut has not eaten usable marble texture at the bottom edge.

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

## Client Confirmation Steps (required before closing this issue)

A clean index-time crop proves **what gets embedded** for `PGYS2319.jpg`. It
does **not** prove the client's original failed search (`xx.jpg` → missing
`PGYS2319`) is fixed — that requires their real FAISS index and query on
their machine. These steps are the **closing criteria** for this issue:

1. **Upgrade** to the build containing feature_version **13** (panel top band
   13% + prior v11 panel-isolation fixes).

2. Run **Rebuild Search Index** on the catalog that contains `PGYS2319.jpg`.
   Index-time vectors changed; existing v10/v11/v12 vectors do not self-heal.

3. **Zoom-verify** panel crops on the real sheet (see zoom acceptance check
   above) — confirm zero caption text at 2x in the top 80px and intact marble
   at the bottom 80px of `*_view1_panel.png`.

4. Re-run the **exact search that failed**: drop `xx.jpg`, confirm `PGYS2319`
   appears in the UI results.

5. Run for a definitive rank/score (not just present/absent):

   ```bash
   python scripts/explain_search.py "/Users/apple/Desktop/xx.jpg" \
     --catalog "<real catalog path>" \
     --find-tile PGYS2319 --top 30 --pool-size 100
   ```

   Report: FAISS pool rank, hybrid rerank rank, `final` score, weak-filter
   kept/dropped, and component scores (`emb`, `color`, `tex`, `edge`, `pat`).

6. **Platform parity** (original concern): if feasible, run step 5 on **Windows
   and macOS Intel** against the **same rebuilt index** and confirm ranks/scores
   agree within float noise (~1e-4). Use `--parity-out` JSON files and diff.

Until steps 2, 4, and 5 pass on the client machine, this issue remains open
even if index-time crops look clean in isolation.
