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

```bash
pip install -r requirements-sam2-experimental.txt
export TILEVISION_ENABLE_SAM2=1
# optional local weights:
# export TILEVISION_SAM2_MODEL_DIR=/path/to/sam2.1-hiera-tiny
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
- [ ] Settings toggle instead of env flag
- [ ] Bundle optional SAM2 tiny weights
- [ ] Intel ONNX path (if full SAM2 cannot ship on x86_64 Mac)
- [ ] Then version bump + release
