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

## 12. Optimization study (v1.2.31 / feature_v10)

Harness: `dev_tools/search_quality/run_optimization_study.py`

Catalog: **320** synthetic production-representative tiles × **21** query
variants = **6720** queries (real customer volumes were not mounted in the
cloud study environment; reports label
`catalog_source=synthetic_production_representative`).

| Config | R@1 | R@5 | Cat R@5 | Vec/tile |
|--------|-----|-----|---------|----------|
| Primary only | 0.467 | 0.692 | 0.756 | 1.00 |
| Strategy E (v9) | 0.491 | 0.705 | 0.838 | 2.08 |
| **E + force adaptive** | **0.503** | **0.721** | **0.847** | **2.22** |
| All cached views | 0.505 | 0.725 | 0.847 | 3.32 |

Rejected (measured):

- Always-on **texture** alone: +0.73pp R@5 (<1pp keep threshold)
- Always-on texture with E+adaptive: +3.19 vec/tile for only +0.38pp vs E+adaptive
- Fusion **Average / WAvg / RRF / Softmax**: Recall@1 regressions vs **MAX**
- Index changes for **room scene**: R@5 ≈ 0.07 under every index config;
  RCA shows **70%** `embedding_drift_query_far_from_all_views` (query-side;
  single-pass `extract_for_search`, weak scene crop). No index change until a
  measured query-path fix.

Shipped change: Strategy E always attempts adaptive content crop (near-dup
aux still dropped). Customer must **Rebuild Search Index**.

Cross-platform (Windows / macOS Intel / Apple Silicon): not fabricated here —
validate via existing CI/search gates.

