"""Regression tests for application startup (Windows and Mac)."""

import ast
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

PySide6 = pytest.importorskip("PySide6")

from PySide6.QtGui import QFont, QIcon
from PySide6.QtWidgets import QApplication

from src.utils.platform_info import app_icon_path, default_ui_font_family


@pytest.fixture(scope="module")
def qapp():
    app = QApplication.instance() or QApplication(sys.argv)
    yield app


def test_app_py_imports_platform_startup_helpers():
    """app.py must import helpers used in build_application early setup."""
    tree = ast.parse(Path("src/app.py").read_text(encoding="utf-8"))
    imported: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and node.module == "src.utils.platform_info":
            imported.update(alias.name for alias in node.names)
    assert "app_icon_path" in imported
    assert "default_ui_font_family" in imported


def test_startup_icon_and_font_setup(qapp):
    """Same code path as build_application right after QApplication is created."""
    icon_path = app_icon_path()
    if icon_path is not None:
        qapp.setWindowIcon(QIcon(str(icon_path)))
    font = QFont(default_ui_font_family(), 10)
    assert font.family()


def test_app_module_imports_build_application():
    from src.app import build_application

    assert callable(build_application)
