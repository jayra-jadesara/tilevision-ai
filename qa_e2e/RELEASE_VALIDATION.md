# TileVision AI — Release Validation System

Ship gate for production builds. Extends the existing `qa_e2e` harness —
**not** a second framework. Uses the real customer stack only:

PySide6 · DINOv2 · SAM2 · SQLite · FAISS IndexFlatIP · Hybrid rerank · real images

## Policy

- Overall **PASS** only if **every** gate and **every** scenario passes.
- One failure ⇒ overall **FAIL**.
- Mocks are forbidden.
- Artifacts must let you inspect every failure before shipping.

## Ordered pipeline

| Gate | What it proves |
|------|----------------|
| G1 Launch | App starts, MainWindow visible, no crash |
| G2 Environment | Python / Torch / Torchvision / FAISS / SQLite / OpenCV / PySide6 / DINOv2 / SAM2 / CPU / RAM / OS / arch |
| G3 AI startup | DINOv2 loaded, FAISS ready, SQLite ready, UI ready, license valid |
| G4 Index catalog | Image count, embedding/FAISS vectors, SQLite rows, no failures |
| G5 Scenarios | Customer journeys S01–S30 |

## Scenarios (S01–S30)

Open/search · drag-drop · open-file · auto crop · precise crop · search · cancel ·
re-search · index new folder · reopen · large/tiny · PNG/JPEG/WEBP/TIFF · corrupt ·
unicode/long filename · multi-search · memory stress · 100 consecutive searches ·
idle 30 minutes · export PDF · preview · zoom · scroll · resize · dark/light mode

Each scenario checks: no crash, no exception, no freeze, spinner stops, search
completes, results/thumbnails/metadata, ranking where applicable.

## Reports

Every run writes under `TILEVISION_QA_OUT`:

- `release_report.html`
- `release_report.json`
- `release_report.pdf`
- `release_summary.json`
- `session.json` · `screenshots/` · `failures/<id>/` · logs

## Local run

```bash
export TILEVISION_DEV_MODE=1
export TILEVISION_QA_OUT=./qa_e2e/artifacts/release_$(date +%Y%m%d_%H%M%S)
python qa_e2e/run_release_validation.py --profile full
```

Profiles:

| Profile | S22 searches | S23 idle | Use |
|---------|--------------|----------|-----|
| `full`  | 100          | 1800s    | Release tags / manual ship gate |
| `pr`    | 10           | 60s      | Pull requests |

## GitHub Actions

Workflow: `.github/workflows/release_validation.yml`

Matrix:

- `windows-latest`
- `macos-15-intel`
- `macos-15` (Apple Silicon)

Triggers: PRs touching `qa_e2e/**` / `src/**` (profile `pr`), `v*` tags
(profile `full`), and `workflow_dispatch`.

The job **fails** if any scenario fails. Artifacts upload HTML / PDF / JSON /
screenshots / logs even on failure.
