"""Duplicates page action buttons must not clip their labels."""

from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import MagicMock

import pytest
from PySide6.QtWidgets import QAbstractButton, QApplication

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.core.models import TileImage
from src.core.use_cases.find_duplicates import FindDuplicatesUseCase
from src.presentation.views.duplicates_view import DuplicatesView


@pytest.fixture
def qapp():
    app = QApplication.instance()
    if app is None:
        app = QApplication([])
    return app


def _sample_group(tmp_path: Path) -> list[TileImage]:
    paths = []
    for name in ("a.png", "b.png"):
        p = tmp_path / name
        # Minimal valid 1×1 PNG
        p.write_bytes(
            bytes.fromhex(
                "89504e470d0a1a0a0000000d49484452000000010000000108060000001f15c489"
                "0000000a49444154789c63000100000500010d0a2db40000000049454e44ae426082"
            )
        )
        paths.append(p)
    return [
        TileImage(
            file_path=str(paths[0]),
            file_name=paths[0].name,
            file_size=paths[0].stat().st_size,
            dimensions="1x1",
            brand="A",
            category="Floor",
            product_code="T1",
            width=1,
            height=1,
            perceptual_hash="abcd" * 4,
            id=1,
        ),
        TileImage(
            file_path=str(paths[1]),
            file_name=paths[1].name,
            file_size=paths[1].stat().st_size,
            dimensions="1x1",
            brand="A",
            category="Floor",
            product_code="T1",
            width=1,
            height=1,
            perceptual_hash="abcd" * 4,
            id=2,
        ),
    ]


@pytest.mark.parametrize("theme", ["light", "dark"])
@pytest.mark.parametrize("width", [1100, 720, 520])
def test_duplicate_action_button_text_fits(qapp, tmp_path, theme, width):
    """Every labelled button's text fits: advance <= width minus padding."""
    view = DuplicatesView(
        use_case=FindDuplicatesUseCase(MagicMock()),
        theme=theme,
    )
    view.resize(width, 640)
    view.show()
    qapp.processEvents()
    view._on_scan_completed([_sample_group(tmp_path)], [])
    qapp.processEvents()
    view.adjustSize()
    qapp.processEvents()

    # Horizontal padding from #TileCard QPushButton / #ScanButton QSS.
    padding = 12
    checked = 0
    for btn in view.findChildren(QAbstractButton):
        text = (btn.text() or "").replace("&", "").strip()
        if not text:
            continue
        # Force layout so width() reflects sizeHint / minimumWidth.
        btn.ensurePolished()
        qapp.processEvents()
        advance = btn.fontMetrics().horizontalAdvance(text)
        usable = max(0, btn.width() - padding)
        assert advance <= usable, (
            f"Button {btn.objectName()!r} text {text!r} advance={advance} "
            f"exceeds usable width {usable} (btn.width={btn.width()}, "
            f"theme={theme}, window_width={width})"
        )
        checked += 1

    assert checked >= 5, "Expected Scan + tile action labels"
    # No emoji left on action buttons.
    for btn in view.findChildren(QAbstractButton):
        text = btn.text() or ""
        assert "📂" not in text and "📍" not in text

    view.close()
    view.deleteLater()
