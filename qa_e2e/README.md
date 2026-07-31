# TileVision AI — Human-like End-to-End QA

This suite drives the **real** TileVision AI desktop app the way a showroom
customer does: launch → wait for readiness → index a catalogue → drag photos →
crop → search → cancel → search again → verify results.

It uses:

- Real **PySide6** UI (`MainWindow`, `DropZone`, buttons, results table)
- Real **DINOv2** embeddings
- Real **FAISS IndexFlatIP**
- Real **SQLite** catalogue metadata
- Real **hybrid reranking**
- Real **Auto Crop** / **Precise Crop** workers

**Production code is not modified.** The harness isolates `HOME` so the QA
session writes to a temporary `~/.tilevision_ai` only.

## Target platform

Primary: **macOS Intel** showroom machines (including **macOS 13** and later).

CI uses the native Intel runner `macos-15-intel` (same as customer Intel DMG
builds). Run locally on any Intel Mac with macOS 13+ via `qa_e2e/run_qa.py`.

Also runnable on Apple Silicon / Linux with `QT_QPA_PLATFORM=offscreen` for
headless execution (still real DINOv2 / FAISS / SQLite / UI).

## Quick start (local Mac Intel)

```bash
# 1) Dependencies
python3.12 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt pytest psutil

# 2) DINOv2 weights (once)
python scripts/download_dinov2_model.py

# 3) Run the customer QA suite
export TILEVISION_DEV_MODE=1
export TILEVISION_QA_OUT=./qa_e2e/artifacts/$(date +%Y%m%d_%H%M%S)
python qa_e2e/run_qa.py
```

Interactive windows (not offscreen):

```bash
python qa_e2e/run_qa.py --interactive
```

Optional frozen `.app` smoke after a local build:

```bash
export TILEVISION_QA_APP_PATH=./dist/TileVisionAI.app
python qa_e2e/run_qa.py -k frozen_app
```

## What is tested

| Customer action | Scenario |
|---|---|
| Launch + UI ready | `test_00_startup_readiness` |
| DINOv2 / FAISS / SQLite ready | `test_00_startup_readiness` |
| Index Folder | `test_01_index_folder` |
| Drag & Drop search + full pipeline stages | `test_02_drag_drop_search` |
| Open Image (browse path) | `test_03_open_dialog_search` |
| Auto Crop / Precise Crop | `test_04_auto_precise_crop` |
| Cancel → search again | `test_05_cancel_and_research` |
| PNG / JPG / WEBP / TIFF / large / small | `test_06_formats_and_corrupt` |
| Corrupt / unsupported drops | `test_06_formats_and_corrupt` |
| Multi-step human session | `test_07_human_session` |
| Frozen `.app` smoke | `test_08_frozen_app_smoke` |

Pipeline stages asserted from live logs (not mocks):

- Embedding generated / cache hit
- Embedding normalized
- FAISS search complete
- SQLite metadata loaded
- Hybrid rerank complete
- Results ready for UI

## Artifacts

Each run writes to `TILEVISION_QA_OUT`:

- `qa_report.html` — actions, screenshots, logs, timings, PASS/FAIL
- `session.json` — machine-readable action log
- `summary.json` — verdict
- `screenshots/` — UI grabs after each major action
- `junit.xml` — CI junit

## GitHub Actions

Workflow: `.github/workflows/qa_e2e_macos.yml`

- Runner: `macos-13` (Intel)
- Downloads DINOv2 weights
- Runs `python qa_e2e/run_qa.py`
- Uploads the HTML report + screenshots as an artifact

## Design notes

- **Human simulator**: random delays, pointer wander, jittered clicks
- **Failure detector**: search never starts, worker exit, empty FAISS/SQLite,
  missing thumbnails, spinner stuck, UI freeze
- **Expectations**: query photos are crops/variants of catalog tiles; top ranks
  must contain the expected product code from the filename
- Normal `tests/` CI does **not** run this suite (marker `qa_e2e` only)
