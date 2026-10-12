"""Vendor admin sidebar shows app version and company name."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "admin_tool"))

from PySide6.QtWidgets import QApplication, QLabel

from src.version import APP_VERSION


@pytest.fixture(scope="module")
def qapp():
    return QApplication.instance() or QApplication(sys.argv)


def _load_admin_main():
    """Load admin_tool/main.py explicitly (repo root also has main.py)."""
    path = ROOT / "admin_tool" / "main.py"
    spec = importlib.util.spec_from_file_location("tilevision_admin_main", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules["tilevision_admin_main"] = module
    spec.loader.exec_module(module)
    return module


def test_admin_sidebar_shows_version_and_company(qapp, tmp_path, monkeypatch):
    monkeypatch.setenv("HOME", str(tmp_path))
    admin_main = _load_admin_main()

    monkeypatch.setattr(admin_main, "_VENDOR_DIR", tmp_path / ".tilevision_ai_vendor")
    monkeypatch.setattr(
        admin_main,
        "_SETTINGS_PATH",
        tmp_path / ".tilevision_ai_vendor" / "admin_settings.json",
    )
    monkeypatch.setattr(
        admin_main,
        "_VENDOR_KEY_PATH",
        tmp_path / ".tilevision_ai_vendor" / "vendor_private_key.pem",
    )

    win = admin_main.AdminLicenseWindow()
    labels = win.findChildren(QLabel)
    version_labels = [lb for lb in labels if lb.objectName() == "VersionLabel"]
    assert version_labels, "Expected a VersionLabel in the admin sidebar"
    text = version_labels[0].text()
    assert f"v{APP_VERSION}" in text
    assert "Adesara Tech" in text
    win.close()
    win.deleteLater()
    qapp.processEvents()
