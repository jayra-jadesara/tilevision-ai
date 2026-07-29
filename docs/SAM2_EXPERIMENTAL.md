# SAM 2 — Experimental Precise Crop (not production)

Status: **feature branch / lab only**. Do **not** ship in customer DMG/EXE until
Mac Intel compatibility and model packaging are decided.

## Goal

Room photos → isolate tile surface more accurately than fast OpenCV → then DINOv2 search.

## What shipped in this branch

| Path | When | Backend |
|------|------|---------|
| Default search | Always | Fast OpenCV auto-focus (production v1.0.12) |
| Auto Crop & Search | Button | Fast OpenCV |
| Precise Crop & Search | Button | SAM 2 if enabled, else GrabCut |
| Crop and Search | Button | Manual |

Default search **never** loads SAM 2 (keeps Mac Intel snappy).

## Enable SAM 2 (dev)

```bash
pip install -r requirements-sam2-experimental.txt
export TILEVISION_ENABLE_SAM2=1
# optional:
# export TILEVISION_SAM2_MODEL_DIR=/path/to/local/sam2.1-hiera-tiny
python main.py
```

Then on Search: choose a room photo → **Precise Crop & Search**.

Without the flag, Precise Crop still works via **OpenCV GrabCut** (no SAM download).

## Why not production yet

1. Production `requirements.txt` pins `transformers<5` for **Mac Intel** (torch 2.2.x).
2. SAM 2 weights are large; not bundled in current installers.
3. First load can be slow; must stay opt-in only.
4. Need offline packaging plan before customer release.

## Model

Default hub id: `facebook/sam2.1-hiera-tiny` (smallest practical checkpoint).

## Next before production

- [ ] Decide Mac Intel strategy (CPU GrabCut only vs drop SAM on Intel builds)
- [ ] Bundle tiny weights under `model_weights/sam2.1-hiera-tiny/`
- [ ] Gate button behind Settings → Experimental
- [ ] Measure Precise Crop latency on M-series and Intel
- [ ] Then version bump + release (separate from normal search fixes)
