"""
Apply a downloaded TileVision AI update without a browser.

Windows: run a detached helper that waits for this process to exit, runs the
Inno Setup installer silently, then relaunches TileVisionAI.exe.

macOS: detach a helper shell that waits for this process to exit, mounts the
DMG, replaces ``/Applications/TileVisionAI.app``, then relaunches.
"""

from __future__ import annotations

import logging
import os
import shlex
import subprocess
import sys
import tempfile
from pathlib import Path

logger = logging.getLogger("tilevision.update_installer")

MAC_APP_BUNDLE_NAME = "TileVisionAI.app"
MAC_APPLICATIONS_DIR = Path("/Applications")
WINDOWS_EXE_NAME = "TileVisionAI.exe"

# Set while quitting so MainWindow skips the indexing confirm prompt and
# app.py can hard-exit after the Qt loop (non-daemon AI threads otherwise keep
# the PID alive and the helper never installs / relaunches).
_FORCE_QUIT_FOR_UPDATE = False


class UpdateInstallError(RuntimeError):
    """Raised when the downloaded update cannot be applied."""


def begin_force_quit_for_update() -> None:
    """Mark the process as exiting for an in-app update install."""
    global _FORCE_QUIT_FOR_UPDATE
    _FORCE_QUIT_FOR_UPDATE = True


def is_force_quit_for_update() -> bool:
    """True when TileVision is quitting so a silent installer can replace files."""
    return bool(_FORCE_QUIT_FOR_UPDATE)


def reset_force_quit_for_update_for_tests() -> None:
    """Test-only: clear the force-quit flag between cases."""
    global _FORCE_QUIT_FOR_UPDATE
    _FORCE_QUIT_FOR_UPDATE = False


def installed_mac_app_path() -> Path:
    """Preferred install location for the packaged Mac app."""
    return MAC_APPLICATIONS_DIR / MAC_APP_BUNDLE_NAME


def windows_silent_install_args(setup_exe: Path) -> list[str]:
    """Inno Setup flags for unattended upgrade (no postinstall GUI launch)."""
    return [
        str(setup_exe),
        "/VERYSILENT",
        "/SUPPRESSMSGBOXES",
        "/CLOSEAPPLICATIONS",
        "/FORCECLOSEAPPLICATIONS",
        "/NORESTART",
        "/SP-",
    ]


def build_windows_apply_script(
    setup_exe: Path,
    *,
    wait_pid: int | None = None,
    exe_name: str = WINDOWS_EXE_NAME,
) -> str:
    """
    Return a cmd.exe script that installs silently after ``wait_pid`` exits,
    then relaunches TileVision from Program Files.
    """
    setup = Path(setup_exe).resolve()
    pid = int(wait_pid if wait_pid is not None else os.getpid())
    # Escape for cmd: use short quoted paths.
    setup_q = str(setup).replace('"', "")
    exe_q = exe_name.replace('"', "")
    return f"""@echo off
setlocal
set "SETUP={setup_q}"
set "WAIT_PID={pid}"
set "EXE_NAME={exe_q}"

REM Wait for TileVision to exit (max ~2 minutes).
:waitloop
tasklist /FI "PID eq %WAIT_PID%" 2>NUL | find "%WAIT_PID%" >NUL
if errorlevel 1 goto runsetup
timeout /T 1 /NOBREAK >NUL
set /A _n+=1
if %_n% GEQ 120 goto runsetup
goto waitloop

:runsetup
timeout /T 1 /NOBREAK >NUL
"%SETUP%" /VERYSILENT /SUPPRESSMSGBOXES /CLOSEAPPLICATIONS /FORCECLOSEAPPLICATIONS /NORESTART /SP-
if errorlevel 1 exit /B %ERRORLEVEL%

REM Prefer 64-bit Program Files, then x86.
set "APP="
if exist "%ProgramFiles%\\TileVision AI\\%EXE_NAME%" set "APP=%ProgramFiles%\\TileVision AI\\%EXE_NAME%"
if not defined APP if exist "%ProgramFiles(x86)%\\TileVision AI\\%EXE_NAME%" set "APP=%ProgramFiles(x86)%\\TileVision AI\\%EXE_NAME%"
if defined APP (
  start "" "%APP%"
)
exit /B 0
"""


def write_windows_apply_script(
    setup_exe: Path,
    *,
    wait_pid: int | None = None,
    script_path: Path | None = None,
) -> Path:
    """Write the Windows apply helper to a temp .cmd file."""
    body = build_windows_apply_script(setup_exe, wait_pid=wait_pid)
    if script_path is None:
        fd, name = tempfile.mkstemp(prefix="tilevision_update_", suffix=".cmd")
        os.close(fd)
        script_path = Path(name)
    script_path.write_text(body, encoding="utf-8", newline="\r\n")
    return script_path


def launch_windows_silent_installer(setup_exe: Path) -> subprocess.Popen:
    """
    Schedule a silent Windows upgrade + relaunch after this process exits.
    """
    setup_exe = Path(setup_exe)
    if not setup_exe.is_file():
        raise UpdateInstallError(f"Installer not found: {setup_exe}")
    if setup_exe.suffix.lower() != ".exe":
        raise UpdateInstallError(f"Expected a Windows .exe installer, got: {setup_exe.name}")

    script = write_windows_apply_script(setup_exe, wait_pid=os.getpid())
    logger.info("Scheduling silent Windows installer via %s", script)

    creationflags = 0
    if sys.platform == "win32":
        creationflags = getattr(subprocess, "DETACHED_PROCESS", 0x00000008)
        creationflags |= getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0x00000200)
        creationflags |= getattr(subprocess, "CREATE_NO_WINDOW", 0x08000000)

    return subprocess.Popen(
        ["cmd.exe", "/c", str(script)],
        cwd=str(setup_exe.parent),
        close_fds=True,
        start_new_session=True,
        creationflags=creationflags,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )


def _find_app_bundle(mount_root: Path) -> Path:
    direct = mount_root / MAC_APP_BUNDLE_NAME
    if direct.is_dir():
        return direct
    matches = sorted(mount_root.rglob(MAC_APP_BUNDLE_NAME))
    for candidate in matches:
        if candidate.is_dir():
            return candidate
    raise UpdateInstallError(
        f"No {MAC_APP_BUNDLE_NAME} found inside the mounted disk image."
    )


def build_macos_apply_script(
    dmg_path: Path,
    *,
    dest_app: Path | None = None,
    wait_pid: int | None = None,
) -> str:
    """
    Return a bash script that applies a DMG update after ``wait_pid`` exits.
    """
    dmg = Path(dmg_path).resolve()
    dest = Path(dest_app) if dest_app is not None else installed_mac_app_path()
    pid = int(wait_pid if wait_pid is not None else os.getpid())
    dmg_q = shlex.quote(str(dmg))
    dest_q = shlex.quote(str(dest))
    dest_parent_q = shlex.quote(str(dest.parent))
    app_name_q = shlex.quote(MAC_APP_BUNDLE_NAME)

    return f"""#!/bin/bash
set -euo pipefail
DMG={dmg_q}
DEST={dest_q}
DEST_PARENT={dest_parent_q}
APP_NAME={app_name_q}
WAIT_PID={pid}

# Wait for the running TileVision process to exit (max ~2 minutes).
for _ in $(seq 1 120); do
  if ! kill -0 "$WAIT_PID" 2>/dev/null; then
    break
  fi
  sleep 1
done
sleep 1

ATTACH_OUT="$(hdiutil attach -nobrowse -readonly -mountrandom /tmp "$DMG")"
# Last field of the last non-empty line is typically the mount point.
MOUNT="$(printf '%s\\n' "$ATTACH_OUT" | awk 'NF{{mp=$NF}} END{{print mp}}')"
if [ -z "${{MOUNT}}" ] || [ ! -d "${{MOUNT}}" ]; then
  echo "Failed to mount DMG" >&2
  echo "$ATTACH_OUT" >&2
  exit 1
fi

cleanup() {{
  hdiutil detach "$MOUNT" -quiet -force 2>/dev/null || true
}}
trap cleanup EXIT

SRC=""
if [ -d "$MOUNT/$APP_NAME" ]; then
  SRC="$MOUNT/$APP_NAME"
else
  SRC="$(find "$MOUNT" -maxdepth 3 -type d -name "$APP_NAME" | head -n 1 || true)"
fi
if [ -z "$SRC" ] || [ ! -d "$SRC" ]; then
  echo "App bundle not found in DMG" >&2
  exit 1
fi

mkdir -p "$DEST_PARENT"
# Replace atomically via staged copy then swap.
STAGE="${{DEST_PARENT}}/.TileVisionAI.update.$$"
rm -rf "$STAGE"
ditto "$SRC" "$STAGE"
# Clear download quarantine so Gatekeeper does not block first launch.
xattr -dr com.apple.quarantine "$STAGE" 2>/dev/null || true
if [ -d "$DEST" ]; then
  rm -rf "$DEST"
fi
mv "$STAGE" "$DEST"

cleanup
trap - EXIT

open -n "$DEST"
"""


def write_macos_apply_script(
    dmg_path: Path,
    *,
    dest_app: Path | None = None,
    wait_pid: int | None = None,
    script_path: Path | None = None,
) -> Path:
    """Write the apply script to a temp file and return its path."""
    body = build_macos_apply_script(
        dmg_path,
        dest_app=dest_app,
        wait_pid=wait_pid,
    )
    if script_path is None:
        fd, name = tempfile.mkstemp(prefix="tilevision_update_", suffix=".sh")
        os.close(fd)
        script_path = Path(name)
    script_path.write_text(body, encoding="utf-8")
    script_path.chmod(0o755)
    return script_path


def launch_macos_dmg_installer(
    dmg_path: Path,
    *,
    dest_app: Path | None = None,
) -> subprocess.Popen:
    """
    Schedule Mac DMG apply after this process exits, then caller should quit.
    """
    dmg_path = Path(dmg_path)
    if not dmg_path.is_file():
        raise UpdateInstallError(f"Disk image not found: {dmg_path}")
    if dmg_path.suffix.lower() != ".dmg":
        raise UpdateInstallError(f"Expected a macOS .dmg installer, got: {dmg_path.name}")

    script = write_macos_apply_script(dmg_path, dest_app=dest_app, wait_pid=os.getpid())
    logger.info("Scheduling macOS DMG apply via %s", script)
    return subprocess.Popen(
        ["/bin/bash", str(script)],
        start_new_session=True,
        close_fds=True,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )


def launch_update_installer(installer_path: Path) -> subprocess.Popen:
    """Platform dispatch: start install then expect the UI to quit."""
    path = Path(installer_path)
    if sys.platform == "win32":
        return launch_windows_silent_installer(path)
    if sys.platform == "darwin":
        return launch_macos_dmg_installer(path)
    raise UpdateInstallError(f"In-app install is not supported on {sys.platform}.")
