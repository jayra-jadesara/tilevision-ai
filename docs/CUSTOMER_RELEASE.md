# Customer Release Guide — TileVision AI

For vendors selling to **ceramic showrooms** on **Windows and Mac**.

Same product on both platforms: index tiles, visual search, PDF export, offline licensing.  
The only difference is the installer file — not the features or search results.

---

## Release workflow (current)

```
┌──────────────────────────────────────────────────────────────────────┐
│  1. Develop & test on main (CI green on the HEAD commit)             │
│  2. Branch release/X.Y.Z from main                                   │
│  3. Bump version: python scripts/bump_version.py X.Y.Z               │
│     (updates src/version.py + both Inno setup .iss files)            │
│  4. Write docs/release_notes/X.Y.Z.md (customer-facing What's new)   │
│  5. Open draft PR "Release X.Y.Z" → wait for CI green → merge        │
│  6. Owner tags vX.Y.Z on main and pushes the tag (do NOT auto-tag)   │
│  7. Build workflow publishes installers + update_manifest.json       │
│  8. Smoke-test on clean Windows and Mac machines (checklist below)   │
│  9. Send installer + licence key to the customer                      │
└──────────────────────────────────────────────────────────────────────┘
```

`CURRENT_FEATURE_VERSION` is **not** bumped for ordinary releases (only when
the on-disk AI feature schema changes).

### Why the bump script

> Release `v1.2.33` bumped `src/version.py` and `packaging/tilevision_setup.iss`
> by hand but missed `packaging/tilevision_admin_setup.iss`. CI failed on all
> four platforms in under 30 seconds. Use the bump script so all three files
> stay in sync.

```bash
git checkout main && git pull
git checkout -b release/X.Y.Z
python scripts/bump_version.py X.Y.Z
# edit docs/release_notes/X.Y.Z.md
# update this guide if the process changed
git add -A
git commit -m "Bump version to X.Y.Z"
# open draft PR, wait for green CI, merge
# OWNER only:
git checkout main && git pull
git tag vX.Y.Z
git push origin vX.Y.Z
```

### Build artifacts (after the tag)

The **Build** workflow on tag `vX.Y.Z` publishes a GitHub Release with:

| File | For |
|------|-----|
| `TileVisionAI-Setup-X.Y.Z.exe` | Windows showrooms |
| `TileVisionAI-macOS-Intel-X.Y.Z.dmg` | Intel Macs |
| `TileVisionAI-macOS-AppleSilicon-X.Y.Z.dmg` | Apple Silicon (M1–M4) |
| `TileVisionAI-macOS-READ-ME-FIRST.txt` | Which Mac DMG to pick |
| `update_manifest.json` | In-app update checker (`release_notes` from `docs/release_notes/X.Y.Z.md`) |
| `TileVisionAI-Admin-VENDOR-ONLY-*.exe` | **Vendor only** — never send to customers |

Customer-facing “What's new” text is **one file**:
`docs/release_notes/X.Y.Z.md`. The Build job loads it for both
`update_manifest.json` and the GitHub Release body. Do not hard-code notes in
`.github/workflows/build.yml`.

---

## Smoke test (clean Windows and clean Mac)

Unset `TILEVISION_DEV_MODE`. Use a machine that does not already have a
development catalogue.

1. Install the new build and **activate a licence**.
2. **Add a folder** and run **Update Search Index** (not Full rebuild unless you intend to).
3. **Search by one photo**.
4. **Multi-photo search** (2–3 photos of the same tile).
5. **Auto Crop / Precise Crop / Manual Crop** on a query photo.
6. **Description search** (needs internet **once** for the model download).
7. **Export Catalogue** PDF with **several selected** results; confirm the PDF is small and the UI stayed responsive.
8. Open **Help** — all **12 steps**, screenshots match the text (light and dark if you switch theme).
9. **Quit** the app. Confirm there is **no loose `tiles.db`** next to the encrypted database (expect `tiles.db.enc` / sealed store; no plain working `tiles.db` left behind after a clean quit).
10. Open `tilevision.log` and confirm a line like **“Catalogue database encrypted at rest”**.

Optional: **Settings → Check for Updates** after the GitHub Release is published, to confirm the new `update_manifest.json` notes appear.

---

## Per-customer licensing (important)

Each computer has its **own Machine ID**. Mac and Windows IDs are different.

1. Customer installs app → opens **Licence Activation**
2. Customer copies **Machine ID** and sends it to you
3. You generate a key in `admin_tool/main.py` for that exact ID
4. Customer pastes key → app works **fully offline**

Issue a **separate key** for each Mac and each Windows PC.

---

## What to send each customer

### Windows showroom

- `TileVisionAI-Setup-X.Y.Z.exe`
- Licence key (trial or full)
- Short note: *If Windows SmartScreen appears, click “More info” → “Run anyway”* (until you add code signing)

### Mac showroom

Send the **correct .dmg** (or both, with the READ-ME):

| File | For which Mac |
|------|----------------|
| `TileVisionAI-macOS-Intel-X.Y.Z.dmg` | Intel iMac, Intel MacBook (Core i5/i7/i9) |
| `TileVisionAI-macOS-AppleSilicon-X.Y.Z.dmg` | M1, M2, M3, M4 Macs |
| `TileVisionAI-macOS-READ-ME-FIRST.txt` | Customer picks the right file |

Also include: **licence key** (trial or full).

First time: Right-click app → **Open** → **Open**.

Prefer **Settings → Check for Updates** for upgrades (parallel CDN download). Browser downloads from GitHub can be very slow.

---

## Same features on Mac and Windows

| Feature | Windows | Mac |
|---------|---------|-----|
| Folder indexing | Yes | Yes |
| Visual similarity search | Yes | Yes |
| PDF catalogue export | Yes | Yes |
| Offline licence | Yes | Yes |
| Folder auto-watch | Yes | Yes |
| iPhone HEIC photos | Yes* | Yes* |
| NVIDIA GPU (CUDA) | Yes | — |
| Apple Silicon GPU (MPS) | — | Yes |

\*Requires `pillow-heif` (included in installer builds).

Search uses the **same AI pipeline** (DINOv2 + descriptors + FAISS) on both platforms.

---

## Local builds (optional)

### Windows PC (Windows installer only)

```powershell
powershell -ExecutionPolicy Bypass -File scripts/build_windows.ps1
```

### Mac

```bash
bash scripts/build_mac.sh
```

### Before every release

1. `python -m pytest tests/ -q -m "not slow"`
2. Confirm CI on `main` HEAD is green
3. Confirm `CURRENT_FEATURE_VERSION` unchanged unless schema work requires it
4. Update revoked licence IDs if any refunds (`src/licensing/revocation.py`)

---

| Platform | Cost | Benefit |
|----------|------|---------|
| Windows Authenticode | ~$200–400/year | Fewer SmartScreen warnings |
| Apple Developer | $99/year | Gatekeeper accepts app without Right-click Open |

Until then, unsigned builds work — include the first-run instructions above.
