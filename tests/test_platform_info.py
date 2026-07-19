"""Tests for cross-platform platform_info helpers."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import src.utils.platform_info as platform_info


def test_detect_linux_graphics_from_lspci(monkeypatch):
    monkeypatch.setattr(platform_info, "is_linux", lambda: True)
    monkeypatch.setattr(platform_info, "is_windows", lambda: False)
    monkeypatch.setattr(platform_info, "is_macos", lambda: False)
    monkeypatch.setattr(
        platform_info.shutil,
        "which",
        lambda cmd: "/usr/bin/lspci" if cmd == "lspci" else None,
    )
    monkeypatch.setattr(
        platform_info,
        "_run_command",
        lambda args, **kwargs: (
            '00:02.0 "VGA compatible controller" "Display controller" '
            '"Advanced Micro Devices, Inc. [AMD/ATI] Radeon RX 580" "1002" "67df"'
            if args[0] == "lspci"
            else None
        ),
    )

    adapters = platform_info.detect_linux_graphics()
    assert any("Radeon" in name for name in adapters)


def test_has_nvidia_gpu_from_nvidia_smi(monkeypatch):
    monkeypatch.setattr(
        platform_info.shutil,
        "which",
        lambda cmd: "/usr/bin/nvidia-smi" if cmd == "nvidia-smi" else None,
    )
    assert platform_info.has_nvidia_gpu() is True


def test_app_icon_prefers_png_off_windows(monkeypatch, tmp_path):
    png = tmp_path / "app_icon.png"
    ico = tmp_path / "app_icon.ico"
    png.write_bytes(b"png")
    ico.write_bytes(b"ico")

    monkeypatch.setattr(platform_info, "APP_ICON_PNG_PATH", png)
    monkeypatch.setattr(platform_info, "APP_ICON_PATH", ico)
    monkeypatch.setattr(platform_info, "is_windows", lambda: False)

    assert platform_info.app_icon_path() == png
