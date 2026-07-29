# SAM 2 — Experimental Precise Crop (cross-platform)

Status: **feature branch / lab only**. Do **not** ship in customer DMG/EXE yet.

## Goal

**Precise Crop & Search** uses accurate SAM 2 on:

| Platform | Primary backend | Fallback |
|----------|-----------------|----------|
| **Windows** | ONNX SAM2 (CPU) → Transformers SAM2 if experimental stack | GrabCut |
| **Mac Intel** | **ONNX SAM2 (CPU)** — works with torch 2.2 / transformers&lt;5 | GrabCut |
| **Mac Apple Silicon** | Transformers SAM2 if available → else ONNX | GrabCut |

Default search / Auto Crop stay on fast OpenCV (no SAM, no slowdown).

## Enable SAM 2 (all platforms)

### In the app (preferred — default ON)
Settings → Preferences → **Use SAM 2 for Precise Crop** (defaults to ON)

Then Search → **Precise Crop & Search**.

### Download weights

**Mac Intel + Windows (recommended — ONNX):**
```bash
pip install onnxruntime   # already in requirements.txt
python scripts/download_sam2_onnx_model.py
# → model_weights/sam2.1-hiera-tiny-onnx/   (~126 MB)
```

**Optional Transformers path (Windows / Apple Silicon lab stacks):**
```bash
pip install -r requirements-sam2-experimental.txt
python scripts/download_sam2_model.py
```

### Or via environment (lab / CI)
```bash
export TILEVISION_ENABLE_SAM2=1
python main.py
```

If neither SAM2 backend can load, GrabCut still runs so the button never breaks.

### Mac Intel — solved via ONNX

Official PyTorch wheels for Mac x86_64 stop at torch 2.2.x, so Transformers
`Sam2Model` is unavailable. **ONNX Runtime + SAM2.1 tiny encoder/decoder** is
the production Accurate path for Mac Intel (and Windows CPU without experimental
deps).

## Buttons

| Button | Backend |
|--------|---------|
| (default drop image) | Fast OpenCV scene focus |
| Auto Crop & Search | Fast OpenCV |
| Precise Crop & Search | Transformers SAM2 → ONNX SAM2 → GrabCut → fast |
| Crop and Search | Manual |

## Before production

- [x] Same code path for Windows / Mac Intel / Mac Silicon
- [x] GrabCut fallback never fails the button
- [x] Settings toggle for experimental SAM2 (default **ON**)
- [x] Download helper: `scripts/download_sam2_model.py`
- [x] **ONNX SAM2 for Mac Intel + Windows** (`scripts/download_sam2_onnx_model.py`)
- [x] Bundle optional SAM2 weights into installers (per-platform)
  - `TILEVISION_BUNDLE_SAM2=auto` → **ONNX on Windows + Mac Intel + Silicon**
  - Transformers safetensors on Windows + Silicon (skipped on Mac Intel)
  - `TILEVISION_BUNDLE_SAM2=0` → production DINOv2-only when unset
- [ ] Optional: install experimental Transformers deps on Win/Silicon for non-ONNX path
- [ ] Then version bump + release

### Lab installer build

```bash
# Mac Intel + Apple Silicon (ONNX on both; Transformers on Silicon)
export TILEVISION_BUNDLE_SAM2=auto
bash scripts/build_mac.sh

# Windows (ONNX + optional Transformers weights)
$env:TILEVISION_BUNDLE_SAM2 = "auto"
powershell -ExecutionPolicy Bypass -File scripts/build_windows.ps1
```

Production customer builds leave `TILEVISION_BUNDLE_SAM2` unset (or `0`) until approved.
