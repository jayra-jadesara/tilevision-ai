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

After upgrading, customers must **Rebuild Search Index** (feature_version **16**).

`panel` top-caption band: **13%** of panel height shaved after content crop
(v12 used 10%, which left a partial clipped caption line on real PGYS2319 at
2x zoom). Accept only after zoomed top-strip inspection — see below.

**v14 (critical):** when `left_panel_beneficial`, the tile's stored
`TileFeatures` row (embedding **and** color/texture/edge/pattern/dominant)
is extracted from the isolated panel, not the full marketing sheet. Full-sheet
remains a FAISS aux for sheet self-hit. Portrait panels letterbox with
**content-matched pad** (not neutral gray) so pad pixels do not destroy LAB
histograms. This is the fix for `color=0.075` on xx.jpg vs PGYS2319 — the
near-white softening from Issue B remains as a general improvement but is not
what resolved this pair.

**v15:** `normalize_lighting()` no longer stretches high-key, low-chroma
frames (cream/white marble). The old heuristic stretched any L-channel
2nd–98th percentile span < 40, which posterized genuine subtle marble once
v14 routed isolated panels through the primary path. Stretch still runs for
underexposed/crushed photos (dark mean / highlights well below white). Same
function is on the query path — cream marble queries are skipped too.

**v16:** `EdgeDescriptor` uses adaptive Canny thresholds from the image's
own Sobel magnitude (legacy fixed 80/180 found zero edges on cream marble
→ all-zero histogram → cosine similarity exactly `0.0`). Both-empty
histograms now score `1.0` (equally unstructured) instead of `0.0`.

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

1. **Upgrade** to the build containing feature_version **16** (panel primary
   TileFeatures + lighting heuristic + adaptive edge descriptor + prior
   panel-isolation / self-hit / near-white fixes).

2. Run **Rebuild Search Index** on the catalog that contains `PGYS2319.jpg`.
   Index-time primary vectors **and** stored descriptors changed; existing
   v10–v15 vectors do not self-heal.

3. **Zoom-verify** panel crops on the real sheet (see zoom acceptance check
   above) — confirm zero caption text at 2x in the top 80px and intact marble
   at the bottom 80px of `*_view1_panel.png`. Also open
   `*_primary_preprocess_letterbox.png` and confirm soft natural marble
   (not harsh posterized B&W from the old L-channel stretch).

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

## Issue A — UI shows 1 result while CLI keeps 9 (same catalog)

**Root cause (confirmed):** not a different catalog path and not a mid-rebuild
race. `xx.jpg` is itself an indexed catalog tile. Production
`SearchTilesUseCase.execute()` demoted same-file exact matches to score
`0.97` **and cleared `exact_match`**, so the weak-result filter used
`reference=0.97` → `min_raw=0.582`. Every similar marble (including
PGYS2319 at `final=0.436`) was dropped → UI **"Found 1 similar tile(s)"** at
~96% (display of 0.97).

`explain_search.py` did **not** demote self-hits the same way, so its floor
stayed at the absolute `0.381` and reported 9 kept — CLI/UI mismatch.

**Fix:** keep `exact_match=True` when demoting the same-file self-hit score to
0.97 (parent-sheet aux at 1.0 still ranks above). Weak-filter reference then
comes from the next non-exact peer. `explain_search.py` mirrors this path.
Search logs now print the resolved FAISS `index=` path for support diffs.

## Issue B — `color=0.075` on white marble pairs

**True root cause (exact reproduction):**

```text
ColorDescriptor.similarity(
  extract(xx_primary_preprocess_letterbox.png),      # clean query
  extract(PGYS2319_primary_preprocess_letterbox.png) # was FULL SHEET
) == 0.075
```

Panel isolation from Tasks 1–3 only fed an **aux FAISS embedding**. The single
stored `TileFeatures` row (hybrid color/texture/edge/pattern +
`candidate.embedding`) still came from the uncropped marketing sheet.

**v14 fix:** primary extraction uses `primary_texture_panel()` when
`left_panel_beneficial`, with content-matched letterbox pad. Near-white
softening (Issue B query-time path) is kept for genuine WB pairs but is
**not** what fixed this tile — inputs must be matched first.

Synthetic before/after (same slab crop vs sheet primary):

| Path | LAB hist CORREL | color_sim | edge | pattern |
|------|-----------------|-----------|------|---------|
| Legacy full-sheet primary | 0.21 | 0.79* | 0.41 | 0.47 |
| Panel + gray pad (broken) | 0.63 | 0.66 | 0.76 | 0.62 |
| Panel + content pad (v14) | 0.65 | **0.99** | **0.77** | **0.67** |

\\*near-white soft path masks hist weakness on synthetic; real PGYS2319
sheet sat fails the near-white gate → similarity stays at hist ≈ 0.075.

## Issue C — posterized primary letterbox after v14

**Symptom:** real `PGYS2319_primary_preprocess_letterbox.png` looked harsh
B&W while `panel` / `panel_center` / legacy full-sheet looked natural.

**Cause:** `normalize_lighting()` stretched any narrow L-range (< 40). Cream
marble is high-key with intrinsic low L-span; stretch manufactured false
contrast. Dormant until v14 fed isolated panels into primary preprocess.

**v15 fix:** skip stretch when `mean_L` high + highlights near white + low
chroma (or already bright well-exposed). Underexposed/crushed frames still
stretch. Query-side `preprocess_for_query` uses the same gate — cream
marble queries are no longer silently posterized either.

Synthetic exact reproduction (`xx` letterbox vs panel primary letterbox):

| Lighting path | color_sim |
|---------------|-----------|
| Old stretch on primary (query natural) | ~0.65 |
| Both sides old-stretched | ~0.62 |
| **v15 natural primary + query** | **~0.87** |

Re-check on real files after rebuild:

```bash
python scripts/show_index_crop.py PGYS2319.jpg --output-dir /tmp/index_crop_debug
# open PGYS2319_primary_preprocess_letterbox.png — expect soft marble
```

## Issue D — `edge=0.0` exactly on clean marble panels (v16)

**Symptom:** after v15 lighting fix, real
`xx_primary_preprocess_letterbox` vs `PGYS2319_primary_preprocess_letterbox`
gave texture/pattern sensible scores but `EdgeDescriptor.similarity == 0.0`
exactly. Synthetic panel-primary test showed `edge_before=edge_after=0.000`.

**Cause:** fixed Canny `80/180` found **zero** edge pixels on high-key
subtle marble → all-zero orientation histogram. Cosine of (zero, anything)
or the `1e-8` denom path collapsed to exact `0.0`. Detection failure, not
genuine dissimilarity.

**v16 fix:** adaptive Canny from Sobel-magnitude percentile (floored so
subtle veins still register), mag-mask fallback if Canny density < 0.2%,
and both-empty similarity → `1.0`. Granite vs solid stays low.

Synthetic stand-in (content-matched letterboxes): edge ≈ **0.79**; with
query gray-pad letterbox ≈ **0.51** (pad borders inflate query edges —
still far from exact 0). Re-run the real-file `EdgeDescriptor.similarity`
after rebuild and report the number.
