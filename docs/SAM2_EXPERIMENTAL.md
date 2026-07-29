# SAM 2 — Experimental Precise Crop (cross-platform)

Status: **feature branch / lab only**. Do **not** ship in customer DMG/EXE yet.

## Goal

**Precise Crop & Search** must work on:

| Platform | Precise Crop backend |
|----------|----------------------|
| **Windows** | GrabCut always; SAM2 optional if enabled + deps |
| **Mac Intel** | **GrabCut only** (SAM2 intentionally skipped) |
| **Mac Apple Silicon** | GrabCut always; SAM2 optional if enabled + deps |

Default search / Auto Crop stay on fast OpenCV (no SAM, no slowdown).

## Why Mac Intel never uses SAM2

Production Mac Intel builds pin `transformers<5` / torch 2.2.x. SAM2 needs a
newer stack. Precise Crop still works there via **OpenCV GrabCut**.

## Enable SAM 2 (Apple Silicon / Windows GPU lab)

```bash
pip install -r requirements-sam2-experimental.txt
export TILEVISION_ENABLE_SAM2=1
python main.py
```

On Mac Intel the flag is ignored for SAM2; GrabCut still runs.

## Buttons

| Button | Backend |
|--------|---------|
| (default drop image) | Fast OpenCV scene focus |
| Auto Crop & Search | Fast OpenCV |
| Precise Crop & Search | SAM2 (if allowed) → else GrabCut → else fast |
| Crop and Search | Manual |

## Before production

- [x] GrabCut path works on Windows / Mac Intel / Mac Silicon
- [x] SAM2 gated off on Mac Intel
- [ ] Settings toggle instead of env flag
- [ ] Bundle optional SAM2 tiny weights for Silicon/Windows only
- [ ] Separate installer flavor or optional download
- [ ] Then version bump + release
