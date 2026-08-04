"""
Apply a downloaded TileVision AI update without a browser.

Windows: elevate the Inno Setup installer with a visible UAC prompt
(``ShellExecuteEx`` + ``runas``). Inno force-closes the running app and
relaunches via ``[Run]``. A hidden CREATE_NO_WINDOW helper cannot show UAC
and silently fails elevation — that was the customer restart hang.

macOS: detach a helper that waits briefly for this process to exit (or
force-kills it), mounts the DMG, replaces ``/Applications/TileVisionAI.app``,
then relaunches. The UI must hard-exit so the helper can replace the bundle.
"""

from __future__ import annotations

import logging
import os
import shlex
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any

logger = logging.getLogger("tilevision.update_installer")

MAC_APP_BUNDLE_NAME = "TileVisionAI.app"
MAC_APPLICATIONS_DIR = Path("/Applications")
WINDOWS_EXE_NAME = "TileVisionAI.exe"

# ERROR_CANCELLED — user clicked No on UAC.
_ERROR_CANCELLED = 1223

# Set while quitting so MainWindow skips the indexing confirm prompt and
# app.py can hard-exit after the Qt loop (non-daemon AI threads otherwise keep
# the PID alive and the Mac helper cannot replace the .app).
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
    """Inno Setup flags for unattended upgrade + force-close running app."""
    return [
        str(setup_exe),
        "/VERYSILENT",
        "/SUPPRESSMSGBOXES",
        "/CLOSEAPPLICATIONS",
        "/FORCECLOSEAPPLICATIONS",
        "/NORESTART",
        "/SP-",
    ]


def windows_silent_install_parameters() -> str:
    """Parameter string for ShellExecute (exe path is lpFile, not here)."""
    return (
        "/VERYSILENT /SUPPRESSMSGBOXES /CLOSEAPPLICATIONS "
        "/FORCECLOSEAPPLICATIONS /NORESTART /SP-"
    )


def build_windows_apply_script(
    setup_exe: Path,
    *,
    wait_pid: int | None = None,
    exe_name: str = WINDOWS_EXE_NAME,
) -> str:
    """
    Fallback cmd helper (tests / non-elevated labs).

    Production Windows updates use ``launch_windows_elevated_setup`` so UAC is
    visible. This script still force-closes via Inno and relaunches.
    """
    setup = Path(setup_exe).resolve()
    pid = int(wait_pid if wait_pid is not None else os.getpid())
    setup_q = str(setup).replace('"', "")
    exe_q = exe_name.replace('"', "")
    params = windows_silent_install_parameters()
    return f"""@echo off
setlocal
set "SETUP={setup_q}"
set "WAIT_PID={pid}"
set "EXE_NAME={exe_q}"

REM Brief wait only — Inno FORCECLOSEAPPLICATIONS kills TileVision if needed.
set /A _n=0
:waitloop
tasklist /FI "PID eq %WAIT_PID%" 2>NUL | find "%WAIT_PID%" >NUL
if errorlevel 1 goto runsetup
timeout /T 1 /NOBREAK >NUL
set /A _n+=1
if %_n% GEQ 15 goto runsetup
goto waitloop

:runsetup
timeout /T 1 /NOBREAK >NUL
"%SETUP%" {params}
if errorlevel 1 exit /B %ERRORLEVEL%

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


def build_windows_relaunch_script(
    *,
    wait_pid: int | None = None,
    setup_exe_name: str,
    exe_name: str = WINDOWS_EXE_NAME,
) -> str:
    """
    Wait for the old app + elevated setup to finish, then relaunch TileVision.

    Used with ShellExecuteEx(runas) because silent Inno builds use
    ``skipifsilent`` and will not auto-start the app.
    """
    pid = int(wait_pid if wait_pid is not None else os.getpid())
    setup_name = Path(setup_exe_name).name.replace('"', "")
    exe_q = exe_name.replace('"', "")
    return f"""@echo off
setlocal
set "WAIT_PID={pid}"
set "SETUP_NAME={setup_name}"
set "EXE_NAME={exe_q}"

REM Wait for the running TileVision process to exit (Inno force-close).
set /A _n=0
:waitapp
tasklist /FI "PID eq %WAIT_PID%" 2>NUL | find "%WAIT_PID%" >NUL
if errorlevel 1 goto waitsetup
timeout /T 1 /NOBREAK >NUL
set /A _n+=1
if %_n% GEQ 180 goto waitsetup
goto waitapp

:waitsetup
REM Wait for the elevated setup.exe to finish installing.
set /A _s=0
:setuploop
tasklist /FI "IMAGENAME eq %SETUP_NAME%" 2>NUL | find /I "%SETUP_NAME%" >NUL
if errorlevel 1 goto relaunch
timeout /T 2 /NOBREAK >NUL
set /A _s+=1
if %_s% GEQ 900 goto relaunch
goto setuploop

:relaunch
timeout /T 2 /NOBREAK >NUL
set "APP="
if exist "%ProgramFiles%\\TileVision AI\\%EXE_NAME%" set "APP=%ProgramFiles%\\TileVision AI\\%EXE_NAME%"
if not defined APP if exist "%ProgramFiles(x86)%\\TileVision AI\\%EXE_NAME%" set "APP=%ProgramFiles(x86)%\\TileVision AI\\%EXE_NAME%"
if defined APP (
  start "" "%APP%"
)
exit /B 0
"""


def write_windows_relaunch_script(
    setup_exe: Path,
    *,
    wait_pid: int | None = None,
    script_path: Path | None = None,
) -> Path:
    """Write the post-setup relaunch helper to a temp .cmd file."""
    body = build_windows_relaunch_script(
        wait_pid=wait_pid,
        setup_exe_name=Path(setup_exe).name,
    )
    if script_path is None:
        fd, name = tempfile.mkstemp(prefix="tilevision_relaunch_", suffix=".cmd")
        os.close(fd)
        script_path = Path(name)
    script_path.write_text(body, encoding="utf-8", newline="\r\n")
    return script_path


def _spawn_detached_cmd(script: Path, *, cwd: Path) -> subprocess.Popen:
    creationflags = 0
    if sys.platform == "win32":
        creationflags = getattr(subprocess, "DETACHED_PROCESS", 0x00000008)
        creationflags |= getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0x00000200)
        creationflags |= getattr(subprocess, "CREATE_NO_WINDOW", 0x08000000)
    return subprocess.Popen(
        ["cmd.exe", "/c", str(script)],
        cwd=str(cwd),
        close_fds=True,
        start_new_session=True,
        creationflags=creationflags,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )


def launch_windows_elevated_setup(setup_exe: Path) -> None:
    """
    Start the Inno installer elevated with a **visible** UAC prompt.

    Must be called from the interactive GUI process. A detached CREATE_NO_WINDOW
    helper cannot display UAC and Windows then auto-denies elevation — install
    never runs and the UI stays on "Installing… will restart".

    After UAC accepts, a detached relaunch helper waits for setup to finish and
    starts TileVisionAI.exe (silent Inno uses skipifsilent).
    """
    setup_exe = Path(setup_exe)
    if not setup_exe.is_file():
        raise UpdateInstallError(f"Installer not found: {setup_exe}")
    if setup_exe.suffix.lower() != ".exe":
        raise UpdateInstallError(f"Expected a Windows .exe installer, got: {setup_exe.name}")
    if sys.platform != "win32":
        raise UpdateInstallError("Elevated Windows install is only available on Windows.")

    import ctypes
    from ctypes import wintypes

    class SHELLEXECUTEINFOW(ctypes.Structure):
        _fields_ = [
            ("cbSize", wintypes.DWORD),
            ("fMask", wintypes.ULONG),
            ("hwnd", wintypes.HWND),
            ("lpVerb", wintypes.LPCWSTR),
            ("lpFile", wintypes.LPCWSTR),
            ("lpParameters", wintypes.LPCWSTR),
            ("lpDirectory", wintypes.LPCWSTR),
            ("nShow", ctypes.c_int),
            ("hInstApp", wintypes.HINSTANCE),
            ("lpIDList", ctypes.c_void_p),
            ("lpClass", wintypes.LPCWSTR),
            ("hKeyClass", wintypes.HKEY),
            ("dwHotKey", wintypes.DWORD),
            ("hIcon", wintypes.HANDLE),
            ("hProcess", wintypes.HANDLE),
        ]

    SEE_MASK_NOCLOSEPROCESS = 0x00000040
    SW_SHOWNORMAL = 1

    # Schedule relaunch BEFORE elevation so it survives FORCECLOSEAPPLICATIONS.
    relaunch = write_windows_relaunch_script(setup_exe, wait_pid=os.getpid())
    logger.info("Scheduling Windows post-setup relaunch via %s", relaunch)
    _spawn_detached_cmd(relaunch, cwd=setup_exe.resolve().parent)

    sei = SHELLEXECUTEINFOW()
    sei.cbSize = ctypes.sizeof(SHELLEXECUTEINFOW)
    sei.fMask = SEE_MASK_NOCLOSEPROCESS
    sei.hwnd = None
    sei.lpVerb = "runas"
    sei.lpFile = str(setup_exe.resolve())
    sei.lpParameters = windows_silent_install_parameters()
    sei.lpDirectory = str(setup_exe.resolve().parent)
    sei.nShow = SW_SHOWNORMAL

    logger.info("Launching elevated Windows installer via ShellExecuteEx runas: %s", setup_exe)
    ok = ctypes.windll.shell32.ShellExecuteExW(ctypes.byref(sei))
    if not ok:
        err = ctypes.GetLastError()
        if err == _ERROR_CANCELLED:
            raise UpdateInstallError(
                "Windows permission prompt was cancelled.\n"
                "Click Yes on the UAC dialog to install the update."
            )
        raise UpdateInstallError(
            f"Could not start the Windows installer (error {err}).\n"
            "Use Open File… to run the installer manually."
        )


def launch_windows_silent_installer(setup_exe: Path) -> Any:
    """
    Production: elevate setup with visible UAC (returns None).

    Non-Windows callers / unit tests may still exercise the cmd helper path
    when ``sys.platform != "win32"`` is monkeypatched carefully; prefer
    ``launch_windows_elevated_setup`` in production.
    """
    setup_exe = Path(setup_exe)
    if not setup_exe.is_file():
        raise UpdateInstallError(f"Installer not found: {setup_exe}")
    if setup_exe.suffix.lower() != ".exe":
        raise UpdateInstallError(f"Expected a Windows .exe installer, got: {setup_exe.name}")

    if sys.platform == "win32":
        launch_windows_elevated_setup(setup_exe)
        return None

    # Fallback for mocked platform tests: detached helper script.
    script = write_windows_apply_script(setup_exe, wait_pid=os.getpid())
    logger.info("Scheduling silent Windows installer via %s", script)
    creationflags = 0
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

    If the UI fails to quit, force-kill the PID after ~20s so replace can proceed.
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

# Wait for TileVision to exit; force-kill if the UI hung on "Installing…".
for _ in $(seq 1 20); do
  if ! kill -0 "$WAIT_PID" 2>/dev/null; then
    break
  fi
  sleep 1
done
if kill -0 "$WAIT_PID" 2>/dev/null; then
  echo "TileVision PID $WAIT_PID still alive; force killing for update" >&2
  kill -TERM "$WAIT_PID" 2>/dev/null || true
  sleep 2
  kill -KILL "$WAIT_PID" 2>/dev/null || true
fi
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
    # Log helper output for support; do not use DEVNULL so failures are diagnosable.
    log_path = Path(tempfile.gettempdir()) / f"tilevision_update_{os.getpid()}.log"
    log_handle = open(log_path, "w", encoding="utf-8")  # noqa: SIM115
    logger.info("macOS update helper log: %s", log_path)
    return subprocess.Popen(
        ["/bin/bash", str(script)],
        start_new_session=True,
        close_fds=True,
        stdout=log_handle,
        stderr=subprocess.STDOUT,
    )


def launch_update_installer(installer_path: Path) -> Any:
    """Platform dispatch: start install then expect the UI to hard-exit."""
    path = Path(installer_path)
    if sys.platform == "win32":
        return launch_windows_silent_installer(path)
    if sys.platform == "darwin":
        return launch_macos_dmg_installer(path)
    raise UpdateInstallError(f"In-app install is not supported on {sys.platform}.")
