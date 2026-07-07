"""
Hardware fingerprinting module for TileVision AI licensing system.

Generates a stable, unique machine identifier on Windows by querying registry keys
and system hardware parameters.
"""

import hashlib
import logging
import subprocess
import winreg
from typing import Optional

logger = logging.getLogger("tilevision.licensing.hardware")


def get_registry_machine_guid() -> Optional[str]:
    """
    Retrieve the OS installation unique identifier (MachineGuid) from the Windows registry.

    Returns:
        The MachineGuid string if found, otherwise None.
    """
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
    except OSError as e:
        logger.error(f"Failed to read MachineGuid from registry: {e}")
        return None


def get_bios_uuid() -> Optional[str]:
    """
    Retrieve the System BIOS / Motherboard UUID via Windows Command Line.

    Returns:
        The BIOS UUID string if found, otherwise None.
    """
    try:
        # Query BIOS Serial Number / UUID using WMIC
        # Although wmic is deprecated, it is highly compatible on target showroom Windows versions.
        startupinfo = subprocess.STARTUPINFO()
        startupinfo.dwFlags |= subprocess.STARTF_USESHOWWINDOW
        
        output = subprocess.check_output(
            ["wmic", "bios", "get", "serialnumber"],
            startupinfo=startupinfo,
            stderr=subprocess.DEVNULL,
            text=True
        )
        lines = [line.strip() for line in output.split("\n") if line.strip()]
        if len(lines) > 1 and lines[0].lower() == "serialnumber":
            serial = lines[1]
            if "default" not in serial.lower() and "to be filled" not in serial.lower():
                return serial
    except Exception as e:
        logger.debug(f"WMIC BIOS lookup failed: {e}. Trying PowerShell...")

    try:
        # PowerShell Fallback for CIM instance
        startupinfo = subprocess.STARTUPINFO()
        startupinfo.dwFlags |= subprocess.STARTF_USESHOWWINDOW
        
        output = subprocess.check_output(
            ["powershell", "-NoProfile", "-Command", "Get-CimInstance Win32_ComputerSystemProduct | Select-Object -ExpandProperty UUID"],
            startupinfo=startupinfo,
            stderr=subprocess.DEVNULL,
            text=True
        )
        uuid = output.strip()
        if uuid:
            return uuid
    except Exception as e:
        logger.error(f"PowerShell BIOS UUID lookup failed: {e}")

    return None


def get_cpu_id() -> Optional[str]:
    """
    Retrieve the CPU Processor ID.

    Returns:
        The Processor ID string if found, otherwise None.
    """
    try:
        startupinfo = subprocess.STARTUPINFO()
        startupinfo.dwFlags |= subprocess.STARTF_USESHOWWINDOW
        
        output = subprocess.check_output(
            ["wmic", "cpu", "get", "processorid"],
            startupinfo=startupinfo,
            stderr=subprocess.DEVNULL,
            text=True
        )
        lines = [line.strip() for line in output.split("\n") if line.strip()]
        if len(lines) > 1 and lines[0].lower() == "processorid":
            return lines[1]
    except Exception as e:
        logger.debug(f"WMIC CPU ID lookup failed: {e}")

    return None


def get_machine_fingerprint() -> str:
    """
    Generate a stable SHA-256 fingerprint hash of the physical Windows machine.
    
    Combines registry MachineGuid, system BIOS UUID, and CPU Processor ID to build
    a hardware-locked identifier. Falls back gracefully if components are missing.

    Returns:
        A 64-character hexadecimal SHA-256 string representing the hardware fingerprint.
    """
    guid = get_registry_machine_guid() or ""
    bios = get_bios_uuid() or ""
    cpu = get_cpu_id() or ""

    # Combine all pieces. If all fail, fall back to hostname and system variables
    if not (guid or bios or cpu):
        import socket
        import uuid
        logger.critical("Standard hardware identifiers could not be read! Falling back to network adapters.")
        fallback = f"{socket.gethostname()}-{uuid.getnode()}"
        raw_id = fallback
    else:
        raw_id = f"GUID:{guid}|BIOS:{bios}|CPU:{cpu}"

    fingerprint = hashlib.sha256(raw_id.encode("utf-8")).hexdigest()
    logger.debug(f"Generated hardware fingerprint: {fingerprint}")
    return fingerprint
