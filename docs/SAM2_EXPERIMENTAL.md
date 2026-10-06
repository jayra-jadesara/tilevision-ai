# SAM 2 — Precise Crop (cross-platform)

Status: **shipped in customer installers** when `TILEVISION_BUNDLE_SAM2=auto`
(the default in `scripts/build_mac.sh`, `scripts/build_windows.ps1`, and the
GitHub Actions Build workflow). ONNX weights are bundled on Windows, Mac Intel,
and Mac Apple Silicon.

## Goal

**Same Precise Crop behavior on every OS:**

| Platform | Primary backend | Fallback |
|----------|-----------------|----------|
| **Windows** | **ONNX SAM2** | GrabCut |
| **Mac Intel** | **ONNX SAM2** | GrabCut |
| **Mac Apple Silicon** | **ONNX SAM2** | GrabCut |

Identical weights + onnxruntime → identical crop quality. Transformers SAM2 is
an optional lab fallback only (not used when ONNX is available).

**Mac Intel note:** production Intel builds pin torch 2.2.2 (last x64 wheel),
so Transformers SAM2 (needs torch≥2.5) is unavailable there. Customer builds
always ship ONNX, so GrabCut is only the last-resort safety net.

Default search / Auto Crop stay on fast OpenCV (no SAM, no slowdown).

## Enable

### In the app (preferred — default ON)
Settings → **Use SAM 2 for Precise Crop** → Search → **Precise Crop & Search**

### Download weights (required once)
```bash
pip install onnxruntime   # in requirements.txt
python scripts/download_sam2_onnx_model.py
# → model_weights/sam2.1-hiera-tiny-onnx/   (~126 MB)
```

### Installer build (same package contents on Mac + Windows)
```bash
# Default in build scripts is already TILEVISION_BUNDLE_SAM2=auto
bash scripts/build_mac.sh          # Intel + Silicon get the same ONNX bundle
# Windows:
powershell -ExecutionPolicy Bypass -File scripts/build_windows.ps1
```

Set `TILEVISION_BUNDLE_SAM2=0` only for DINOv2-only lab builds without Precise Crop.

## Buttons

| Button | Backend |
|--------|---------|
| (default drop image) | Fast OpenCV |
| Auto Crop & Search | Fast OpenCV |
| Precise Crop & Search | ONNX SAM2 → (optional Transformers) → GrabCut → fast |
| Crop and Search | Manual |

## Before production

- [x] Same code path for Windows / Mac Intel / Mac Silicon
- [x] ONNX SAM2 primary on **all** platforms (Mac == Windows)
- [x] GrabCut fallback never fails the button
- [x] Settings toggle (default **ON**)
- [x] `scripts/download_sam2_onnx_model.py`
- [x] Installer bundling via `TILEVISION_BUNDLE_SAM2=auto`
- [x] CI Build workflow caches/downloads ONNX on Windows + Mac Intel + Silicon
- [x] Version bump to **1.0.13** (Precise Crop ONNX on all platforms)
- [ ] Tag `v1.0.13` after merge to `main` → GitHub Actions publishes installers

### v1.0.13 customer notes

- **Precise Crop & Search** uses the **same ONNX SAM2** path on Windows, Mac Intel, and Mac Apple Silicon
- Default search / Auto Crop unchanged (fast OpenCV)
- Settings toggle defaults ON; GrabCut remains the safety fallback
