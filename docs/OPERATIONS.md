# TileVision AI — Operations Guide

## When to Rebuild the Index

Rebuild or re-scan when:

- The app shows the **stale feature banner** at the top
- You upgraded TileVision AI after a pipeline change
- `feature_version` changed (currently **5**)
- Search results look wrong after an update

### Full clean rebuild

1. Open **Settings → Rebuild FAISS Index**
2. Or re-scan all folders from the **Index** page with **Force re-index** enabled

### Manual clean slate (advanced)

Delete data under `%USERPROFILE%\.tilevision_ai\`:

- `tiles.db` — SQLite catalog
- `tiles.index` — FAISS vectors
- `thumbnails\` — cached thumbnails (optional)

Then re-scan your tile folders.

---

## Filename Convention (for filters)

Use this pattern for best metadata and filter support:

```
Brand_Category_Color_Size_ProductCode.jpg
```

Example: `Kajaria_Floor_Grey_60x60_ABC123.jpg`

Hyphenated descriptive names also work:

```
5mm-white-dotted-ceramic-floor-tile-500x500.jpg
```

After renaming files, **re-index** the folder.

---

## Benchmark Indexing Speed

1. Open TileVision AI from a terminal to see logs
2. Index a folder of known size (e.g. 20–50 tiles)
3. Look for log lines:

```
INDEX BATCH TIMING
INDEX TIMING
```

Key stages: `image_loading`, `preprocessing`, `dinov2`, `descriptors`, `database`, `faiss`

**Expected on CPU (DINOv2-large):** ~8–10 seconds per image  
**Expected on GPU:** significantly faster

---

## Benchmark Search Speed

1. Run a search from the **Search** page
2. Check logs for:

```
SEARCH TIMING
```

Key stages: `preprocessing`, `dinov2`, `faiss`, `database`, `reranking`

**Expected:** under 0.5s when query is already indexed; 5–15s on CPU for new query images.

---

## Accuracy Evaluation

### Auto mode (quick sanity check)

```powershell
cd C:\Users\HP\Projects\tilevision-ai
python dev_tools/eval_recall_precision.py --catalog-auto --max-queries 30
```

### Manifest mode (ground truth — recommended)

1. Generate or edit the manifest:
```powershell
python dev_tools/generate_eval_manifest.py
# creates eval/my_queries.jsonl
```

2. Run evaluation with 90% accuracy target:
```powershell
python dev_tools/eval_recall_precision.py --manifest eval/my_queries.jsonl --target-k 5 --target-recall 0.90
```

Expected output ends with:
```
PASS: Recall@5 = 100.0%
```

Metrics reported: **Recall@K**, **Precision@K**, **MRR**, **nDCG@K**

---

## Testing Search Manually

1. Index a folder with varied tile types (speckled, marble, wood, plain)
2. Use **catalogue-quality crop images** when possible (not room photos)
3. For room photos, use the **crop dialog** before searching
4. Apply filters (Brand/Category/Color/Size) after metadata is populated
5. Verify self-match: search an indexed tile → top result should be **100%**

---

## GPU Setup (optional, recommended)

Install CUDA-compatible PyTorch, then restart TileVision AI. The app auto-detects CUDA and uses mixed precision.

---

## Troubleshooting

| Symptom | Fix |
|---------|-----|
| Filters show only "Unknown" | Rename files + re-index |
| Stale feature banner | Re-scan or Rebuild FAISS Index |
| Low accuracy on room photos | Crop to tile surface |
| Slow indexing | Use GPU or smaller catalog batches |
| Rebuild FAISS not visible | Update to latest build; check Settings |
