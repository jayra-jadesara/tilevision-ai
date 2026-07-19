"""Tests for cross-platform hardware fingerprinting."""

import hashlib
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import src.licensing.hardware as hardware


def test_windows_fingerprint_format_unchanged(monkeypatch):
    monkeypatch.setattr(hardware.sys, "platform", "win32")
    monkeypatch.setattr(hardware, "get_registry_machine_guid", lambda: "guid-1")
    monkeypatch.setattr(hardware, "get_bios_uuid", lambda: "bios-1")
    monkeypatch.setattr(hardware, "get_cpu_id", lambda: "cpu-1")

    expected_raw = "GUID:guid-1|BIOS:bios-1|CPU:cpu-1"
    expected = hashlib.sha256(expected_raw.encode("utf-8")).hexdigest()
    assert hardware.get_machine_fingerprint() == expected


def test_macos_fingerprint_uses_platform_uuid_and_serial(monkeypatch):
    monkeypatch.setattr(hardware.sys, "platform", "darwin")
    monkeypatch.setattr(hardware, "get_macos_platform_uuid", lambda: "MAC-UUID-123")
    monkeypatch.setattr(hardware, "get_macos_serial_number", lambda: "SN-456")
    monkeypatch.setattr(hardware.platform, "node", lambda: "MacBook-Pro")

    expected_raw = "MAC:UUID:MAC-UUID-123|SERIAL:SN-456|HOST:MacBook-Pro"
    expected = hashlib.sha256(expected_raw.encode("utf-8")).hexdigest()
    assert hardware.get_machine_fingerprint() == expected


def test_linux_fingerprint_uses_machine_id_and_dmi(monkeypatch):
    monkeypatch.setattr(hardware.sys, "platform", "linux")
    monkeypatch.setattr(hardware, "get_linux_machine_id", lambda: "linux-mid-abc")
    monkeypatch.setattr(hardware, "get_linux_product_uuid", lambda: "dmi-uuid-xyz")
    monkeypatch.setattr(hardware.platform, "node", lambda: "tile-pc")

    expected_raw = "LINUX:MID:linux-mid-abc|DMI:dmi-uuid-xyz|HOST:tile-pc"
    expected = hashlib.sha256(expected_raw.encode("utf-8")).hexdigest()
    assert hardware.get_machine_fingerprint() == expected


def test_fingerprint_falls_back_when_identifiers_missing(monkeypatch):
    monkeypatch.setattr(hardware.sys, "platform", "linux")
    monkeypatch.setattr(hardware, "get_linux_machine_id", lambda: None)
    monkeypatch.setattr(hardware, "get_linux_product_uuid", lambda: None)
    monkeypatch.setattr(hardware, "_network_fallback_raw_id", lambda: "FALLBACK:test")

    expected = hashlib.sha256(b"FALLBACK:test").hexdigest()
    assert hardware.get_machine_fingerprint() == expected


def test_get_linux_machine_id_reads_etc_machine_id(tmp_path, monkeypatch):
    machine_id_file = tmp_path / "machine-id"
    machine_id_file.write_text("abc123def456\n", encoding="utf-8")

    class _FakePath:
        def __init__(self, path_str: str):
            self._path_str = path_str

        def is_file(self) -> bool:
            return self._path_str == "/etc/machine-id"

        def read_text(self, encoding: str = "utf-8") -> str:
            return machine_id_file.read_text(encoding=encoding)

    def _path_factory(path_str="/"):
        if path_str in ("/etc/machine-id", "/var/lib/dbus/machine-id"):
            return _FakePath(path_str)
        return Path(path_str)

    monkeypatch.setattr(hardware, "Path", _path_factory)
    assert hardware.get_linux_machine_id() == "abc123def456"

