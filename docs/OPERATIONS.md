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

### Large catalogs (70–200 MB, 2000+ images)

TileVision now applies three optimizations automatically:

1. **Early downscale** — source images decode to max **2048 px** before AI (config: `max_decode_edge` in `config.json`)
2. **Fast re-scan skip** — unchanged files skip via **file size + mtime** (no full SHA256 read)
3. **Adaptive batching** — files ≥10 MB use batch 4; ≥50 MB use batch 2 and 1 worker

Tunable keys in `%USERPROFILE%\.tilevision_ai\config.json`:

```json
{
  "index_batch_size": 12,
  "index_checkpoint_interval": 25,
  "max_decode_edge": 2048,
  "preprocess_workers": 4,
  "large_file_mb_threshold": 10,
  "huge_file_mb_threshold": 50
}
```

**Production recommendation:** keep 70–200 MB originals for archive, but index a **preview folder** (1500×1500 JPG, ~1–2 MB each) for fastest first-time indexing. Use **GPU** for bulk loads.

| Setup | 2000 images (estimate) |
|-------|------------------------|
| 200 MB + CPU | 8–15+ hours, OOM risk |
| 200 MB + early downscale + CPU | 5–8 hours |
| 2 MB previews + GPU | **30–90 minutes** |

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

## GPU Setup (recommended for production)

Your logs show `torch 2.13.0+cpu` — **CPU-only PyTorch**. The app already supports GPU; you need the CUDA wheel.

### Step 1 — Check NVIDIA driver

```powershell
nvidia-smi
```

If this fails, install the latest driver from NVIDIA first.

### Step 2 — Install CUDA PyTorch (Windows)

From the project folder:

```powershell
cd C:\Users\HP\Projects\tilevision-ai
powershell -ExecutionPolicy Bypass -File scripts\install_pytorch_cuda.ps1
```

Or manually:

```powershell
pip uninstall -y torch torchvision torchaudio
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124
```

### Step 3 — Verify

```powershell
python dev_tools/check_gpu.py
```

Expected: `Active: CUDA` and your GPU name.

### Step 4 — Restart TileVision AI

Settings → Overview → **AI Device** should show your GPU name and VRAM.

Startup log should show:

```
CUDA GPU: NVIDIA ... (X.X GB VRAM)
Indexing performance: device=... batch=24 ...
```

### GPU tuning (`config.json`)

```json
{
  "inference_device": "auto",
  "gpu_index_batch_size": 24,
  "index_batch_size": 12
}
```

- `auto` — use GPU when available (default)
- `cuda` — force GPU (falls back to CPU with warning if missing)
- `cpu` — force CPU
- `gpu_index_batch_size` — larger batches on GPU (24 default vs 12 on CPU)

**Accuracy:** GPU uses mixed precision (`autocast`) — same DINOv2-large model, no accuracy pipeline change. FAISS stays on CPU (fine for 2000–50K tiles).

---

## Troubleshooting

| Symptom | Fix |
|---------|-----|
| Filters show only "Unknown" | Rename files + re-index |
| Stale feature banner | Re-scan or Rebuild FAISS Index |
| Low accuracy on room photos | Crop to tile surface |
| Slow indexing | Use GPU or smaller catalog batches |
| Rebuild FAISS not visible | Update to latest build; check Settings |
