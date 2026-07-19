# Packaging TileVision AI

PyInstaller builds must run **on the target OS** (no cross-compiling).

| Platform | Spec file | Output |
|----------|-----------|--------|
| Windows | `packaging/tilevision.spec` | `dist/TileVisionAI/TileVisionAI.exe` |
| macOS | `packaging/tilevision_mac.spec` | `dist/TileVisionAI.app` |
| Linux | `packaging/tilevision_linux.spec` | `dist/TileVisionAI/TileVisionAI` |

See also [docs/CROSS_PLATFORM.md](../docs/CROSS_PLATFORM.md) for run-from-source setup.

---

## Prerequisites

```bash
pip install -r requirements.txt
pip install pyinstaller
```

Run preflight on the build machine:

```bash
python scripts/preflight_check.py
```

---

## 1. Pre-download DINOv2 weights (required for offline installs)

The app uses **DINOv2** (`facebook/dinov2-large`), not CLIP. Weights must be
downloaded on your build machine **before** packaging — never on a customer's
offline PC.

```bash
python scripts/download_dinov2_model.py
```

This creates `model_weights/dinov2-large/` (~1 GB). PyInstaller specs bundle
this folder automatically when present.

For strict offline runtime, set on customer builds:

```bash
export TILEVISION_OFFLINE_MODEL=1   # macOS / Linux
set TILEVISION_OFFLINE_MODEL=1      # Windows
```

---

## 2. Set up licensing

Follow `admin_tool/README.md` to generate a real signing keypair and embed
the public key into `src/licensing/validator.py` (`EMBEDDED_PUBLIC_KEY_PEM`).

Confirm `TILEVISION_DEV_MODE` is **not** set in your build environment.

---

## 3. Build

**Windows:**

```powershell
pyinstaller packaging/tilevision.spec --clean
```

**macOS:**

```bash
pyinstaller packaging/tilevision_mac.spec --clean
```

**Linux:**

```bash
pyinstaller packaging/tilevision_linux.spec --clean
```

Use a **one-folder** build (not `--onefile`) — single-file mode re-extracts
~1–2 GB of PyTorch + model weights on every launch.

---

## 4. Test the built app

On a clean VM **with no internet**, confirm:

- App launches and shows activation screen
- Folder indexing completes
- Search returns results (confirms bundled DINOv2 weights)
- License activation works with a key from the Admin tool

---

## 5. Installers

### Windows — Inno Setup (automated)

Install [Inno Setup 6](https://jrsoftware.org/isinfo.php), then run the full release script:

```powershell
powershell -ExecutionPolicy Bypass -File scripts/build_windows.ps1
```

Output:
- App folder: `dist/TileVisionAI/`
- Installer: `dist/installer/TileVisionAI-Setup-1.0.0.exe`

Or build the installer manually after PyInstaller:

```powershell
iscc packaging\tilevision_setup.iss
```

The installer:
- Installs to `Program Files\TileVision AI`
- Creates Start Menu shortcut (optional desktop shortcut)
- Creates `%PROGRAMDATA%\TileVisionAI\.lic\` for encrypted license storage

### macOS / Linux

| Platform | Tool |
|----------|------|
| macOS | `hdiutil` DMG + notarization (Apple Developer account) |
| Linux | `.deb` / AppImage via `fpm` or `appimagetool` |

Not automated beyond PyInstaller — validate the raw build first.

### GitHub Actions (all platforms)

Push a version tag to build release artifacts on GitHub:

```bash
git tag v1.0.0
git push origin v1.0.0
```

Or trigger manually: **Actions → Build → Run workflow**.

Artifacts (14-day retention):
- `tilevision-windows` — app folder + `TileVisionAI-Setup-*.exe`
- `tilevision-macos` — `TileVisionAI-macOS.zip` (`.app` bundle)
- `tilevision-linux` — `TileVisionAI-Linux.tar.gz`

---

## Known gaps

- No automated release pipeline (manual build per OS)
- No code signing (SmartScreen / Gatekeeper warnings on first run)
- macOS notarization and Linux `.desktop` integration are manual steps
