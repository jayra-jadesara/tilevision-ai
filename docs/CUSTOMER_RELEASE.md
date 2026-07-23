# Customer Release Guide — TileVision AI

For vendors selling to **ceramic showrooms** on **Windows and Mac**.

Same product on both platforms: index tiles, visual search, PDF export, offline licensing.  
The only difference is the installer file — not the features or search results.

---

## Your release workflow (build from home)

```
┌─────────────────────────────────────────────────────────────┐
│  1. Develop & test on your PC (python main.py)              │
│  2. Bump version in packaging/tilevision_setup.iss          │
│  3. Build Windows + Mac (see below)                         │
│  4. Test both on clean machines (offline)                   │
│  5. Send installer to customer + license key                 │
└─────────────────────────────────────────────────────────────┘
```

### Option A — Build on your Windows PC (Windows only)

```powershell
powershell -ExecutionPolicy Bypass -File scripts/build_windows.ps1
```

Ship: `dist/installer/TileVisionAI-Setup-1.0.0.exe`

### Option B — Build everything via GitHub (no Mac at home)

```bash
git tag v1.0.1
git push origin v1.0.1
```

Open **GitHub → Actions → Build**. When finished, download:

| Artifact | For |
|----------|-----|
| `tilevision-windows` | Windows showrooms |
| `tilevision-macos` | Mac showrooms |

### Option C — Build Mac on a Mac

```bash
bash scripts/build_mac.sh
```

Ship: `dist/TileVisionAI-macOS.dmg` or `dist/TileVisionAI.app`

---

## Per-customer licensing (important)

Each computer has its **own Machine ID**. Mac and Windows IDs are different.

1. Customer installs app → opens **License Activation**
2. Customer copies **Machine ID** and sends it to you
3. You generate a key in `admin_tool/main.py` for that exact ID
4. Customer pastes key → app works **fully offline**

Issue a **separate key** for each Mac and each Windows PC.

---

## What to send each customer

### Windows showroom

- `TileVisionAI-Setup-1.0.0.exe`
- License key (trial or full)
- Short note: *If Windows SmartScreen appears, click “More info” → “Run anyway”* (until you add code signing)

### Mac showroom (Intel AND Apple Silicon)

Send **one zip** — works for every Mac:

**`TileVisionAI-macOS-1.0.0.zip`**

Inside the zip:
| File | For which Mac |
|------|----------------|
| `TileVisionAI-macOS-Intel.dmg` | Intel iMac, Intel MacBook (Core i5/i7/i9) |
| `TileVisionAI-macOS-AppleSilicon.dmg` | M1, M2, M3, M4 Macs |
| `READ ME FIRST.txt` | Simple guide — customer picks the right file |

Also include: **license key** (trial or full)

**Client with Intel iMac 2020:** use **Intel** `.dmg`  
**Client with M1/M2/M3 Mac:** use **Apple Silicon** `.dmg`

First time: Right-click app → **Open** → **Open**

---

## Same features on Mac and Windows

| Feature | Windows | Mac |
|---------|---------|-----|
| Folder indexing | Yes | Yes |
| Visual similarity search | Yes | Yes |
| PDF catalogue export | Yes | Yes |
| Offline license | Yes | Yes |
| Folder auto-watch | Yes | Yes |
| iPhone HEIC photos | Yes* | Yes* |
| NVIDIA GPU (CUDA) | Yes | — |
| Apple Silicon GPU (MPS) | — | Yes |

\*Requires `pillow-heif` (included in installer builds).

Search uses the **same AI pipeline** (DINOv2 + descriptors + FAISS) on both platforms.  
Results may differ by a fraction of a percent between GPU types — same tiles, same ranking logic.

---

## New version checklist

Before each release:

1. Run tests: `python -m pytest tests/ -q -m "not slow"`
2. Unset `TILEVISION_DEV_MODE`
3. Update revoked license IDs if any refunds (`src/licensing/revocation.py`)
4. Build Windows + Mac artifacts
5. Test on **one Windows PC** and **one Mac** without internet:
   - Launch → activate with trial key
   - Index a sample folder
   - Search with a photo
   - Export PDF
6. Upload installers to Google Drive / USB / your website
7. Email customers: new installer + note that old license keys still work on the same Machine ID

---

## Automatic update notifications

Packaged apps (Windows `.exe` installer, Mac `.app`) check for updates on startup
when the PC has internet — **once per day**. Customers see a dialog with a
**Download Update** button that opens the correct installer for their platform.

### How it works

1. You push a version tag → GitHub Actions builds Windows + Mac installers
2. CI publishes a **GitHub Release** with `update_manifest.json`
3. Customer app reads the manifest and compares versions
4. Customer clicks **Download Update** → browser opens the Windows `.exe` or Mac `.dmg`
5. They run the installer — license and tile catalogue stay on the PC

### Release a version with update links

```bash
# 1. Bump APP_VERSION in src/version.py and packaging/tilevision_setup.iss
# 2. Commit and tag:
git tag v1.0.1
git push origin v1.0.1
```

After the **Build** workflow finishes, the release includes:

| File | Used by |
|------|---------|
| `TileVisionAI-Setup-1.0.1.exe` | Windows update link |
| `TileVisionAI-macOS-Intel-1.0.1.dmg` | Intel Mac update link (optional) |
| `TileVisionAI-macOS-AppleSilicon-1.0.1.dmg` | Apple Silicon update link (optional) |
| `TileVisionAI-macOS-1.0.1.zip` | **All Macs** — both DMGs + install guide |
| `update_manifest.json` | In-app update checker |

Customers can also use **Settings → Check for Updates** anytime.

**Note:** This notifies and downloads — it does not silently auto-install (safer for
showroom PCs). Full silent auto-update would require code signing on both platforms.

---

| Platform | Cost | Benefit |
|----------|------|---------|
| Windows Authenticode | ~$200–400/year | Fewer SmartScreen warnings |
| Apple Developer | $99/year | Gatekeeper accepts app without Right-click Open |

Until then, unsigned builds work — include the first-run instructions above.
