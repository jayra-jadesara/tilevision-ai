# TileVision AI — macOS Release Packaging

## Deliverables

| File | Arch | Notes |
|------|------|-------|
| `TileVision-AI-Intel.dmg` | x86_64 | Native Intel Mac |
| `TileVision-AI-AppleSilicon.dmg` | arm64 | Native Apple Silicon |
| `TileVision-AI-macOS-both.zip` | both DMGs | Vendor convenience zip — **not** Universal2 |

Volume name inside each DMG: **TileVision AI**

DMG layout:
- `TileVision AI.app`
- `Applications` → `/Applications`
- `READ ME FIRST.txt`

## Universal2 — not supported

**Do not claim Universal2.**

Reasons:
1. Mac Intel production builds pin **`torch==2.2.2` / `torchvision==0.17.2`** (last official x86_64 macOS wheels).
2. Apple Silicon builds install **current torch/torchvision** from `requirements.txt`.
3. Those native extension modules are **different ABIs** and cannot be reliably merged with `lipo` into one fat `.app`.
4. Cross-building Intel on an arm64 host is rejected (`install_mac_deps.sh`) because compiled wheels would still be arm64.

Ship **two** DMGs. Customers pick by About This Mac.

## Code signing / notarization

**Unsigned build** unless Apple Developer credentials are configured.

This repository’s CI does **not** run `codesign` / `notarytool` / staple.
Do not fabricate notarization. Customers use **Right-click → Open** once.

When credentials become available, add a dedicated signing step; do not fake it.

## Bundled runtime

Customer machines need **no Python**. The `.app` includes:

- PySide6 (+ Qt platform plugins via PyInstaller)
- DINOv2 weights (`model_weights/dinov2-large`)
- SAM2 ONNX (`model_weights/sam2.1-hiera-tiny-onnx` when `TILEVISION_BUNDLE_SAM2=auto`)
- FAISS, OpenCV, Pillow, ReportLab, ONNX Runtime
- Icons / resources / default config
- certifi CA bundle

Offline: `TILEVISION_OFFLINE_MODEL=1` at build time; models load from the bundle.

## macOS paths (runtime)

| Data | Location |
|------|----------|
| Config, SQLite, FAISS, thumbnails, logs | `~/.tilevision_ai/` |
| License crypto store | `~/Library/Application Support/TileVisionAI/.lic/` |
| PDF export | User-chosen path |
| Windows paths | Not used on Darwin |

## How to build

```bash
# On macos-15-intel runner / Intel Mac:
MACOS_ARCH=x64 bash scripts/build_mac.sh

# On Apple Silicon Mac:
MACOS_ARCH=arm64 bash scripts/build_mac.sh
```

CI: `.github/workflows/build.yml` (`workflow_dispatch` or tag `v*`).

## Validation

### Packaged application (required for ship)
GitHub Actions workflow **`Packaged App Release Validation`**:

1. Build `.app` + professional DMG on `macos-15-intel` and `macos-15`
2. Mount DMG → install to `/Applications/TileVision AI.app`
3. Run installed binary `--verify-bundle`
4. Run installed binary `--release-validation` (S01–S30)
5. FAIL unless `release_summary.json` verdict is **PASS**

Drivers (`qa_e2e`) are loaded from the CI checkout via `TILEVISION_QA_SUITE_DIR`.
Product modules and models come from the **installed** `.app` (`sys.frozen`).

```bash
# Local (on a Mac, after building a DMG):
bash scripts/validate_installed_mac_dmg.sh \
  dist/TileVision-AI-AppleSilicon.dmg \
  "$PWD" \
  /tmp/packaged_rv_out
```

### Static / smoke (packaging CI)
1. `scripts/verify_frozen_mac_app.sh` — arch, models, torch, deps
2. `scripts/smoke_launch_mac_app.sh` — short first-launch smoke

### Source-tree Release Validation
Still available for day-to-day PR gates (`release_validation.yml`) but does **not**
prove the DMG-installed customer binary.

## Reports

`scripts/generate_macos_packaging_report.py` writes:

- `packaging_report.json` / `.md`
- `dependency_report.json`
- `pyinstaller_report.json`
- `known_limitations.md`
