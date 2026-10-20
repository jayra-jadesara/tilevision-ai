# TileVision AI — Hardware smoke test checklist

Printable checklist for a **clean Windows** PC and a **clean Mac**.  
Cursor / CI cannot run real installers — the owner (or QA) runs this after each candidate build.

Unset `TILEVISION_DEV_MODE`. Prefer a machine without a leftover development catalogue (`~/.tilevision_ai` / `%USERPROFILE%\.tilevision_ai`).

After the run: **Settings → Copy Diagnostics** and paste the clipboard text back to support.

---

## Windows — clean install

| # | Step | Pass? |
|---|------|-------|
| 1 | Install the Windows Setup EXE (silent or UI). Launch **TileVision AI**. | ☐ |
| 2 | **Activate licence** (paste key for this machine’s Machine ID). | ☐ |
| 3 | **Add folder** + **Update Search Index** (not Full rebuild unless intentional). | ☐ |
| 4 | **Search by one photo**. | ☐ |
| 5 | **Multi-photo** search (2–3 photos of the same tile). | ☐ |
| 6 | **Auto / Precise / Manual crop** on a query photo. | ☐ |
| 7 | **Description search** (internet **once** for model download). | ☐ |
| 8 | **Export Catalogue** PDF with **several selected** results; UI stayed responsive. | ☐ |
| 9 | Open **Help** — all **12 steps**; screenshots match the text (try light and dark). | ☐ |
| 10 | **Duplicates** page — Open / Folder / Reveal / Delete buttons fully readable. | ☐ |
| 11 | **Quit** the app. | ☐ |
| 12 | Confirm **no loose `tiles.db`** next to `tiles.db.enc` under `%USERPROFILE%\.tilevision_ai\database\`. | ☐ |
| 13 | Open `%USERPROFILE%\.tilevision_ai\logs\tilevision.log` — last quit has **“Catalogue database encrypted at rest”**. | ☐ |
| 14 | **Settings → Copy Diagnostics** → paste into your reply. | ☐ |

Build / version: _______________  Tester: _______________  Date: _______________

---

## Mac — clean install

| # | Step | Pass? |
|---|------|-------|
| 1 | Install from the DMG (drag to Applications). Launch **TileVision AI**. | ☐ |
| 2 | **Activate licence** (Machine ID is Mac-specific). | ☐ |
| 3 | **Add folder** + **Update Search Index**. | ☐ |
| 4 | **Search by one photo**. | ☐ |
| 5 | **Multi-photo** search (2–3 photos). | ☐ |
| 6 | **Auto / Precise / Manual crop**. | ☐ |
| 7 | **Description search** (internet once). | ☐ |
| 8 | **Export Catalogue** PDF with several selected. | ☐ |
| 9 | **Help** — 12 steps; images match (light and dark). | ☐ |
| 10 | **Duplicates** page buttons readable. | ☐ |
| 11 | **Quit** the app. | ☐ |
| 12 | Confirm **no loose `tiles.db`** next to `tiles.db.enc` under `~/.tilevision_ai/database/`. | ☐ |
| 13 | Open `~/.tilevision_ai/logs/tilevision.log` — **“Catalogue database encrypted at rest”**. | ☐ |
| 14 | **Settings → Copy Diagnostics** → paste into your reply. | ☐ |

Build / version: _______________  Tester: _______________  Date: _______________

---

## Optional automated checks (not a substitute)

- Windows post-install bundle: `scripts/smoke_test_windows.ps1`
- Frozen verify: `scripts/verify_frozen_windows.ps1`, `scripts/verify_frozen_mac_app.sh`

See also `docs/CUSTOMER_RELEASE.md` for the release workflow around this checklist.
