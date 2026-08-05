# TileVision AI — Production Search Accuracy Architecture

**Status:** Evidence-driven (golden-dataset bakeoff)  
**Release target:** follows measured winner (see bakeoff report)  
**Owners:** Search / CV

## 1. Problem

Ceramic manufacturers run catalogs of 10k–50k+ images. Queries are rarely the
indexed master file. They are crops, phone shots, catalogue clips, and room
photos. The product requirement is:

> If the upload belongs to a catalog tile, that tile must appear in Top-5
> (ideally Top-1).

## 2. Measurement first

No algorithm change ships without the golden bakeoff:

```bash
python dev_tools/search_quality/run_bakeoff.py \
  --out /tmp/search_accuracy --tiles 16 --sheets 8
```

For every query the harness records:

| Metric | Meaning |
|--------|---------|
| Recall@1 / @5 / @10 | Parent tile_id in top-K after fusion |
| MRR / NDCG@5 | Rank quality |
| embed_s / faiss_s / fuse_s | Stage latency |
| stage_fail | miss@100 vs in-100-not-Top-5 vs in-5-not-Top-1 |

Queries are **auto-generated** from each catalog image (no manual labels):
original, 95/90/75/60/50% crops, center/random/corner, rotate, brightness,
contrast, JPEG, phone screenshot, catalogue page, room scene.

## 3. Failure stages (production evidence)

| Stage | Symptom | Measured cause |
|-------|---------|----------------|
| Preprocess | Same-file query ≠ index embedding | Scene auto-crop / straighten on marketing sheets |
| Embedding | Crop cosine ≪ 0.7 vs primary | Layout/text dominates DINOv2 primary |
| FAISS | Parent rank ≫ 5 (e.g. ~54/328) | Single primary vector only |
| Rerank | Display ~27% despite strong FAISS aux | Hybrid uses SQLite primary only |
| Filter | Weak-result floor | Secondary; not root cause when FAISS rank is mid-pack |

## 4. Image analysis (before extra views)

`src/ai/search_quality/image_analysis.py` classifies each indexed image with
lightweight OpenCV/NumPy heuristics:

- clean tile vs catalog sheet vs bordered tile
- white border ratio, texture richness, text-column score, preview grid
- **left_panel_beneficial** / **center_crop_beneficial**

Extra views are created **only** when beneficial. Blind quadrant crops are
forbidden.

## 5. Index strategies (bakeoff)

| ID | Views |
|----|-------|
| A | Primary only (legacy 1:1) |
| B | Full + center 50% |
| C | Full + adaptive content crop |
| D | Full + texture-rich window |
| E | Heuristic multi-view (panel / panel-center / center) |
| production_v8 | Shipped v1.2.29 path (panel + center) |

Each aux vector shares `tile_id` with the primary. Near-duplicates
(cos ≥ 0.985 vs primary) are dropped.

## 6. Score fusion (bakeoff)

After FAISS returns per-vector hits, collapse by `tile_id` using:

- MAX (production default)
- Weighted MAX (aux weight tuned on validation half)
- Average / Weighted Average
- Reciprocal Rank Fusion (RRF)

**Only the fusion with best measured customer-path Recall@1/@5 is selected.**

## 7. Memory / index size

Approximate FlatIP float32 storage:

```
bytes ≈ n_vectors × 1024 × 4
```

Bakeoff reports `vs_primary_only_ratio` and projects 50k-tile memory.
Multi-view is accepted only when Recall gains justify the ratio.

## 8. Search path (production)

```
query image
  → preprocess_for_query (sheet-aware: no scene crop / straighten)
  → DINOv2 embedding
  → FAISS FlatIP (over-fetch for multi-id vectors)
  → fuse by tile_id (winning method)
  → hybrid rerank + FAISS aux boost
  → weak-result filter → Top-K UI
```

## 9. Regression

- Unit: `tests/search_quality/test_search_quality_core.py`
- Crop/sheet DINOv2: `tests/test_crop_search_consistency.py`
- Bakeoff CI job (optional / scheduled): strategy A vs winning strategy must not
  regress customer-path Recall@5

Platforms: Windows, macOS Intel, macOS Apple Silicon (existing QA workflows).

## 10. Customer action after feature_version bump

**Settings → Rebuild Search Index** so aux vectors materialize in FAISS.

## 11. Bakeoff results (v1.2.30 / feature_v9)

Golden set: 24 images × 16 auto-labeled variants = 384 queries.

| Strategy | Cust R@5 | Vectors/tile |
|----------|----------|--------------|
| A primary-only | 0.837 | 1.00 |
| B full+center | 0.851 | 2.00 |
| C adaptive | 0.837 | 1.00 |
| D texture | 0.868 | 2.00 |
| **E heuristic** | **0.903** | **2.33** |

Fusion winner: **MAX** (Average/RRF rejected — Recall@1 regression).

Full report: bakeoff artifacts / `SEARCH_ACCURACY_REPORT.md`.

