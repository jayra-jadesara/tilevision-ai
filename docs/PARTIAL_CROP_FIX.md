# TileVision AI — PARTIAL_CROP Search Fix

**Scope:** query-time view generation for `QueryKind.PARTIAL_CROP` only. Index /
FAISS / `TileFeatures` unchanged.  
**feature_version:** still **10** (no Rebuild Search Index required).

## Root cause

`PARTIAL_CROP` queries were classified correctly but, before PR #41, received
only a single embedded view while `ROOM_SCENE` / `PHONE_SCREENSHOT` queries
used multi-view FAISS MAX-merge. Adding a third mirror-padded view
(`_pad_crop_border`, `cv2.BORDER_REFLECT_101`) in the original PR #41 fix
regressed `crop_50` Recall@1 without helping retrieval.

## Ablation (golden 36-tile catalog, 180 `crop_*` queries, FAISS-only bakeoff)

| Mode | crop_50 R@1 | crop_50 R@5 | agg `crop_*` R@1 | agg `crop_*` R@5 |
|------|-------------|-------------|------------------|------------------|
| `old_single` (pre-PR-41) | 0.7222 | 0.9444 | 0.8833 | 0.9444 |
| `new_primary_only` | 0.6667 | 0.8333 | 0.8722 | 0.9222 |
| **`new_primary_plus_tighten`** | **0.7500** | **0.9444** | **0.8889** | **0.9444** |
| `new_primary_plus_all` (PR #41 broken) | 0.6667 | 0.8333 | 0.8722 | 0.9222 |

`new_primary_plus_all` and `new_primary_only` scored **exactly** identical on
aggregate R@1/R@5 across all 180 queries.

## Confirmed mechanism (`explain_search.py` + per-view FAISS)

Investigated the three golden `crop_50` queries where
`new_primary_plus_tighten` recovered R@1 but `new_primary_plus_all` /
`new_primary_only` missed (tile ids 2, 5, 11 — pulled automatically from
per-query bakeoff rank comparison, not hand-picked).

**The `_WEAK_RESULT_RELATIVE_FLOOR` / weak-filter hypothesis is refuted.**

1. The bakeoff ablation path is **FAISS-only** (no hybrid rerank, no
   `_filter_weak_results`), so weak-filter math cannot explain the ablation
   deltas.
2. `explain_search.py` in `new_primary_plus_all` mode on the same three
   queries shows the true tile **is retrieved into the FAISS candidate pool**
   (`true_in_pool=True` for all three). Hybrid rerank scores for the true tile
   are well above the computed weak floor (`min_raw` 0.43–0.51); none were
   `weak_filter_dropped=True`.
3. Failures are **FAISS MAX-merge ranking**, not post-rerank filtering.

Per-view FAISS evidence (representative query, tile id 5, `crop_50`):

| View | Content | True-tile FAISS | True-tile rank | Top-1 tile | Top-1 FAISS |
|------|---------|-----------------|----------------|------------|-------------|
| 0 — `crop_to_content_region` | primary | 0.5827 | 35 | 27 | 0.9022 |
| 1 — mirror pad (`plus_all`) | pad | 0.8038 | 25 | 5 (self in top-5) | — |
| 1 — 82% tighten (`plus_tighten`) | tighten | **0.9151** | **1** | **5 (self)** | **0.9151** |

Across all three disagreeing queries, the **82% center tighten view (index 1)**
supplies the winning MAX-merge score (~0.91+). The primary content crop alone
ranks the true tile at positions 14–35. The mirror-pad view does not beat the
primary view's MAX score for tiles 2 and 11; for tile 5 it improves the true
tile (0.58 → 0.80) but not enough for R@1. Spurious top-1 candidates trace to
**view 0 (primary)**, not the pad view.

**Why `plus_all` ≡ `plus_only` on aggregate metrics:** on CPU-capped query
paths (`_capped_query_max_views` → 2), `plus_all` becomes content + pad only
(no third tighten slot). The pad view never raises any tile's MAX score above
the primary view's top-1 winner enough to flip a query from R@1 miss to hit, so
aggregate R@1/R@5 match `new_primary_only` exactly.

## Production configuration

`PARTIAL_CROP` queries embed **two views**, best-first:

1. `ImagePreprocessor.crop_to_content_region` (`min_margin_ratio=0.02`)
2. 82% center tighten of that crop (`_center_crop(content, 0.82)`)

Both views participate in `_search_faiss_multi_crop()` MAX-merge. No
`_pad_crop_border` / mirror reflection.

`ROOM_SCENE` and `PHONE_SCREENSHOT` view generation are unchanged.

## Verification commands

```bash
# Targeted unit tests
pytest tests/ -k "partial_crop or search_pipeline or explain_search or multi_crop_faiss or query_analyzer" -q

# Post-fix vs pre-fix crop_* slice (golden bakeoff, FAISS-only)
python3 dev_tools/search_quality/run_bakeoff.py --out /tmp/crop_fix_check
```

Expected post-fix vs `old_single` control: `crop_50` R@1 0.7222→0.7500, R@5
0.9444→0.9444, aggregate `crop_*` R@1 +0.0056.

Diagnostic replay (optional, requires DINOv2 weights + golden catalog):

```bash
python3 scripts/explain_search.py path/to/crop_50.jpg --catalog /path/to/catalog --top 10
```
