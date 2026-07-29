# SAM 2 — Experimental Precise Crop (cross-platform)

Status: **feature branch / lab only**. Do **not** ship in customer DMG/EXE yet.

## Goal

**Precise Crop & Search** uses the **same SAM2 path** on:

| Platform | When SAM2 runs | Fallback |
|----------|----------------|----------|
| **Windows** | `TILEVISION_ENABLE_SAM2=1` + experimental deps | GrabCut |
| **Mac Intel** | Same (no OS blacklist) | GrabCut |
| **Mac Apple Silicon** | Same | GrabCut |

Default search / Auto Crop stay on fast OpenCV (no SAM, no slowdown).

## Enable SAM 2 (all platforms)

### In the app (preferred — default ON)
Settings → Preferences → **Use SAM 2 for Precise Crop** (defaults to ON)

Then Search → **Precise Crop & Search**.

### Download local weights (optional, faster offline)
```bash
python scripts/download_sam2_model.py
```

### Or via environment (lab / CI)
```bash
pip install -r requirements-sam2-experimental.txt
export TILEVISION_ENABLE_SAM2=1
python main.py
```

Then: Search → **Precise Crop & Search**.

If SAM2 cannot load (old torch / missing Sam2Model), GrabCut still runs so the
button never breaks on Windows or Mac Intel.

### Mac Intel note

Official PyTorch wheels for Mac x86_64 stop at torch 2.2.x, while SAM2 usually
needs newer torch. Options:

1. Use GrabCut (works today on Intel)
2. Lab machine with a custom stack + `TILEVISION_SAM2_FORCE=1`
3. Later: optional ONNX SAM package for Intel CPU

Windows and Apple Silicon can use the experimental requirements file normally.

## Buttons

| Button | Backend |
|--------|---------|
| (default drop image) | Fast OpenCV scene focus |
| Auto Crop & Search | Fast OpenCV |
| Precise Crop & Search | SAM2 (if allowed) → else GrabCut → else fast |
| Crop and Search | Manual |

## Before production

- [x] Same code path for Windows / Mac Intel / Mac Silicon
- [x] GrabCut fallback never fails the button
- [x] Settings toggle for experimental SAM2 (default **ON**)
- [x] Download helper: `scripts/download_sam2_model.py`
- [x] Bundle optional SAM2 tiny weights into installers (per-platform)
  - `TILEVISION_BUNDLE_SAM2=auto` → **Windows + Mac Apple Silicon** yes; **Mac Intel** skip
  - `TILEVISION_BUNDLE_SAM2=0` → production DINOv2-only (default when unset)
  - Build scripts download ~150 MB safetensors when bundling is on
- [ ] Install experimental deps (`requirements-sam2-experimental.txt`) on Win/Silicon lab builds so frozen apps can *run* SAM2 (weights alone are not enough on production transformers&lt;5)
- [ ] Intel ONNX path (if full SAM2 cannot ship on x86_64 Mac)
- [ ] Then version bump + release

### Lab installer build

```bash
# Mac Apple Silicon (bundles SAM2); Mac Intel still GrabCut-only
export TILEVISION_BUNDLE_SAM2=auto
bash scripts/build_mac.sh

# Windows
# PowerShell:
$env:TILEVISION_BUNDLE_SAM2 = "auto"
powershell -ExecutionPolicy Bypass -File scripts/build_windows.ps1
```

Production customer builds leave `TILEVISION_BUNDLE_SAM2` unset (or `0`).
