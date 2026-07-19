"""
Hardware fingerprinting module for TileVision AI licensing system.

Generates a stable, unique machine identifier per operating system:
  - Windows: registry MachineGuid + BIOS UUID + CPU ID
  - macOS: IOPlatformUUID + hardware serial
  - Linux: /etc/machine-id + DMI product UUID

The Windows fingerprint format is unchanged so existing license keys keep working.
"""

from __future__ import annotations

import hashlib
import logging
import platform
import subprocess
import sys
from pathlib import Path
from typing import Optional, Sequence

try:
    import winreg
except ImportError:
    winreg = None

logger = logging.getLogger("tilevision.licensing.hardware")


def _run_command(args: Sequence[str], *, timeout: float = 15.0) -> Optional[str]:
    """Run a subprocess quietly and return stripped stdout."""
    try:
        kwargs: dict = {
            "capture_output": True,
            "text": True,
            "timeout": timeout,
            "check": False,
        }
        if sys.platform == "win32":
            startupinfo = subprocess.STARTUPINFO()
            startupinfo.dwFlags |= subprocess.STARTF_USESHOWWINDOW
            kwargs["startupinfo"] = startupinfo

        result = subprocess.run(list(args), **kwargs)
        if result.returncode != 0:
            return None
        output = (result.stdout or "").strip()
        return output or None
    except Exception as exc:
        logger.debug("Command %s failed: %s", args, exc)
        return None


def get_registry_machine_guid() -> Optional[str]:
    """Windows: MachineGuid from registry."""
    if winreg is None:
        return None
    try:
        key = winreg.OpenKey(
            winreg.HKEY_LOCAL_MACHINE,
            r"SOFTWARE\Microsoft\Cryptography",
            0,
            winreg.KEY_READ | winreg.KEY_WOW64_64KEY,
        )
        value, _ = winreg.QueryValueEx(key, "MachineGuid")
        winreg.CloseKey(key)
        return str(value).strip()
    except OSError as exc:
        logger.error("Failed to read MachineGuid from registry: %s", exc)
        return None


def get_bios_uuid() -> Optional[str]:
    """Windows: BIOS / motherboard UUID via WMIC or PowerShell."""
    if sys.platform != "win32":
        return None

    output = _run_command(["wmic", "bios", "get", "serialnumber"])
    if output:
        lines = [line.strip() for line in output.split("\n") if line.strip()]
        if len(lines) > 1 and lines[0].lower() == "serialnumber":
            serial = lines[1]
            if "default" not in serial.lower() and "to be filled" not in serial.lower():
                return serial

    output = _run_command(
        [
            "powershell",
            "-NoProfile",
            "-Command",
            "Get-CimInstance Win32_ComputerSystemProduct | Select-Object -ExpandProperty UUID",
        ]
    )
    return output


def get_cpu_id() -> Optional[str]:
    """Windows: CPU processor ID via WMIC."""
    if sys.platform != "win32":
        return None

    output = _run_command(["wmic", "cpu", "get", "processorid"])
    if not output:
        return None
    lines = [line.strip() for line in output.split("\n") if line.strip()]
    if len(lines) > 1 and lines[0].lower() == "processorid":
        return lines[1]
    return None


def get_linux_machine_id() -> Optional[str]:
    """Linux: persistent machine-id from systemd/dbus."""
    for path in (Path("/etc/machine-id"), Path("/var/lib/dbus/machine-id")):
        try:
            if not path.is_file():
                continue
            value = path.read_text(encoding="utf-8").strip()
            if value and value.lower() not in {"uninitialized", "unknown"}:
                return value
        except OSError as exc:
            logger.debug("Could not read %s: %s", path, exc)
    return None


def get_linux_product_uuid() -> Optional[str]:
    """Linux: DMI product UUID when exposed by the kernel."""
    path = Path("/sys/class/dmi/id/product_uuid")
    try:
        if not path.is_file():
            return None
        value = path.read_text(encoding="utf-8").strip()
        if value and "not set" not in value.lower() and "not available" not in value.lower():
            return value
    except OSError as exc:
        logger.debug("Could not read Linux product UUID: %s", exc)
    return None


def get_macos_platform_uuid() -> Optional[str]:
    """macOS: IOPlatformUUID from I/O Registry."""
    output = _run_command(["ioreg", "-rd1", "-c", "IOPlatformExpertDevice"])
    if not output:
        return None
    for line in output.splitlines():
        if "IOPlatformUUID" not in line:
            continue
        parts = line.split('"')
        if len(parts) >= 4:
            return parts[3].strip()
    return None


def get_macos_serial_number() -> Optional[str]:
    """macOS: hardware serial number."""
    output = _run_command(["system_profiler", "SPHardwareDataType"])
    if not output:
        return None
    for line in output.splitlines():
        if "Serial Number" in line:
            return line.split(":", 1)[-1].strip()
    return None


def _windows_raw_id() -> Optional[str]:
    guid = get_registry_machine_guid() or ""
    bios = get_bios_uuid() or ""
    cpu = get_cpu_id() or ""
    if not (guid or bios or cpu):
        return None
    return f"GUID:{guid}|BIOS:{bios}|CPU:{cpu}"


def _macos_raw_id() -> Optional[str]:
    platform_uuid = get_macos_platform_uuid() or ""
    serial = get_macos_serial_number() or ""
    host = platform.node() or ""
    if not (platform_uuid or serial):
        return None
    return f"MAC:UUID:{platform_uuid}|SERIAL:{serial}|HOST:{host}"


def _linux_raw_id() -> Optional[str]:
    machine_id = get_linux_machine_id() or ""
    product_uuid = get_linux_product_uuid() or ""
    host = platform.node() or ""
    if not (machine_id or product_uuid):
        return None
    return f"LINUX:MID:{machine_id}|DMI:{product_uuid}|HOST:{host}"


def _network_fallback_raw_id() -> str:
    import socket
    import uuid

    logger.warning(
        "Standard hardware identifiers could not be read on %s — using fallback.",
        sys.platform,
    )
    return f"FALLBACK:{sys.platform}|{socket.gethostname()}|{uuid.getnode()}"


def get_machine_fingerprint() -> str:
    """
    Generate a stable SHA-256 fingerprint hash for the current machine.

    Returns:
        A 64-character hexadecimal SHA-256 string representing the hardware fingerprint.
    """
    if sys.platform == "win32":
        raw_id = _windows_raw_id()
    elif sys.platform == "darwin":
        raw_id = _macos_raw_id()
    else:
        raw_id = _linux_raw_id()

    if not raw_id:
        raw_id = _network_fallback_raw_id()

    fingerprint = hashlib.sha256(raw_id.encode("utf-8")).hexdigest()
    logger.debug("Generated hardware fingerprint on %s", sys.platform)
    return fingerprint
