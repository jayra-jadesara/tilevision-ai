# TileVision AI — Query Understanding (v1.2.32)

**Scope:** query pipeline only. Index / FAISS vector count unchanged.  
**feature_version:** still **10** (no Rebuild Search Index required for this release).

## Root cause (measured)

On the 320-tile / 6720-query study set, room-scene Recall@5 stayed ≈ **0.07**
under every index configuration. RCA + cosine probes showed:

1. Room photos are wide (`1400×900`) and trip `primary_texture_panel`
   (left ~45% has texture).
2. `preprocess_for_query` treated that as a **catalogue sheet** and
   **skipped tile isolation**.
3. The full room was embedded → cosine ≈ **0.47–0.53** vs parent tile.
4. The same image with OpenCV isolation → cosine ≈ **0.86–0.90**.

So the failure was **query misclassification**, not missing index views.

## Architecture

```
query image
  → QueryAnalyzer (OpenCV/PIL/NumPy)
       CLEAN_TILE / PARTIAL     → single crop
       CATALOG_SHEET            → left panel (+ optional panel-center)
       ROOM_SCENE               → isolate + capped multi-crop
       PHONE_SCREENSHOT         → strip UI chrome → isolate (+ multi-crop)
  → DINOv2 per crop (platform cap ≤2 on Mac / Windows-CPU)
  → FAISS MAX merge per tile_id (existing SearchTilesUseCase)
```

Catalogue sheet detection requires **white-margin / text-column** evidence and
low ceiling/floor color delta. Grid Hough alone is rejected (rooms produce
line segments).

## Benchmark

```bash
python3 dev_tools/search_quality/run_query_understanding_benchmark.py \
  --study-out /opt/cursor/artifacts/search_optimization \
  --out /opt/cursor/artifacts/query_understanding
```

Reuses the existing 320-tile catalog embedding cache; only room/phone query
embeddings are recomputed (other variants keep the classic single-pass path).

### Measured (Linux CPU, index unchanged, 1062 vectors)

| Metric | v1.2.31 query | v1.2.32 query | Δ |
|--------|---------------|---------------|---|
| Recall@1 | 0.5045 | 0.5144 | +0.99pp |
| Recall@5 | 0.7253 | 0.7421 | +1.68pp |
| Room-scene R@5 | 0.0781 | 0.3844 | **+30.63pp** |
| Phone R@5 | 0.7219 | 0.7688 | +4.69pp |
| Catalogue R@5 | 0.8469 | 0.8469 | 0 |
| Original R@5 | 0.8719 | 0.8719 | 0 |

Production path: multi-crop only for `room_scene` / `phone_screenshot`; all
other query kinds keep the v1.2.31 single-pass preprocess (with the fixed
catalogue-sheet gate).

## Platforms

Analyzer + crop heuristics are pure CPU OpenCV/NumPy — identical on Windows,
macOS Intel, and Apple Silicon. DINOv2 view count is capped via
`ImagePreprocessor._capped_query_max_views` (existing Mac/Windows CPU guard).
Validate end-to-end via CI search gates; do not fabricate per-OS scores.
