# Manual QA — Tray background monitoring + mid-session license cutover

## Mechanism chosen (Task 1)

**Minimize-to-tray** (opt-in, default **off**), not a Windows Service / macOS LaunchAgent.

- Closing the window with the setting enabled hides the UI and keeps the existing
  process + `watchdog.Observer` alive via `QSystemTrayIcon`.
- Full exit is only via tray **Quit** (same clean shutdown path as today).
- A true OS service would need a separate install/update path, elevated
  permissions on Windows, and a process boundary from the GUI — out of scope.

Setting: `minimize_to_tray_on_close` in `config.json` / Settings → Auto Folder Monitoring.

---

## Automated evidence (dev / CI-friendly)

Headless cutover demo (no DINOv2 warm-up):

```bash
QT_QPA_PLATFORM=offscreen python scripts/demo_license_cutover.py
```

Unit tests:

```bash
QT_QPA_PLATFORM=offscreen python -m pytest \
  tests/test_monitor_folder_reconcile.py \
  tests/test_license_session_cutover.py \
  tests/test_tray_minimize.py -v
```

---

## Manual QA on a real desktop (Windows or macOS)

### A. Minimize-to-tray + monitoring

1. Start TileVision AI with a valid license and at least one watched folder.
2. Settings → Auto Folder Monitoring → enable  
   **Keep running in the system tray when I close the window**.
3. Close the main window (window chrome close button).
4. Confirm the tray icon appears with tooltip  
   `TileVision AI is running in the background`.
5. Drop a new image into a watched folder (or subfolder). Confirm it is
   auto-indexed (tray app still running; reopen via tray **Open** and check
   status bar / catalog).
6. Tray → **Quit**. Confirm the process exits and monitoring stops.

With the setting **off** (default): closing the window must quit the app
exactly as before.

### B. Reconcile-on-start

1. Quit the app completely.
2. Copy new images into a watched folder while the app is closed.
3. Relaunch. Without waiting for a new filesystem event, those images should
   appear as auto-indexed shortly after start (reconcile pass).

### C. Mid-session license expiry cutover

Use a short re-check interval for QA:

```bash
# Windows PowerShell
$env:TILEVISION_LICENSE_RECHECK_MS = "30000"   # 30 seconds
python main.py

# macOS / Linux
TILEVISION_LICENSE_RECHECK_MS=30000 python main.py
```

1. Activate with a trial/license that expires imminently, **or** replace the
   stored license with an already-expired key via the admin/dev license tools
   while the app stays open (same outcome as natural expiry).
2. Optionally enable minimize-to-tray and hide the main window.
3. Wait for the periodic re-check (default 15 minutes, or the env override).
4. Confirm **all** of the following:
   - Folder monitoring stops (no further auto-index of new drops).
   - No further auto-index status-bar / tray notifications after cutover.
   - Tray icon is hidden/removed if it was shown.
   - Main window is gone; only the **License Activation** screen is visible
     (same `LicenseView` path as startup).
5. Enter a new valid key → app restores the main window and restarts folder
   monitoring if `watch_folders` is configured (no full restart required).
6. Decline/skip activation → app exits with the same “License Required” message
   as startup.

### Log lines to look for

```
Mid-session license check failed — license missing or invalid.
License expired or invalidated mid-session — starting cutover.
Folder monitoring stopped due to license cutover.
Showing license activation dialog.
```

On successful renew:

```
License renewed mid-session — main window and monitoring restored.
```
