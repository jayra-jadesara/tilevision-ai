"""Tests for in-app Pricing Quote PDF viewer and sidebar entry."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

PySide6 = pytest.importorskip("PySide6")

from PySide6.QtWidgets import QApplication

from src.presentation.views.pricing_quote_view import PricingQuoteView
from src.utils.brand_assets import nav_icon


@pytest.fixture(scope="module")
def qapp():
    app = QApplication.instance() or QApplication(sys.argv)
    yield app


def test_pricing_nav_icons_exist():
    icons_root = Path(__file__).resolve().parents[1] / "src" / "resources" / "icons"
    assert (icons_root / "dark" / "nav_pricing.svg").is_file()
    assert (icons_root / "light" / "nav_pricing.svg").is_file()
    assert not nav_icon("pricing", "dark").isNull()
    assert not nav_icon("pricing", "light").isNull()


def test_main_window_wires_pricing_nav_below_help():
    """Structural check: Pricing nav is declared immediately after Help."""
    source = (
        Path(__file__).resolve().parents[1]
        / "src"
        / "presentation"
        / "views"
        / "main_window.py"
    ).read_text(encoding="utf-8")
    help_pos = source.find('NavButton(\n            "Help", "help"')
    pricing_pos = source.find('NavButton(\n            "Pricing", "pricing"')
    assert help_pos != -1
    assert pricing_pos != -1
    assert pricing_pos > help_pos
    assert "_on_pricing_clicked" in source
    assert "PricingQuoteView" in source


def test_help_opens_in_app_viewer_not_browser():
    source = (
        Path(__file__).resolve().parents[1]
        / "src"
        / "presentation"
        / "views"
        / "help_view.py"
    ).read_text(encoding="utf-8")
    assert "PricingQuoteView" in source
    assert "QDesktopServices.openUrl" not in source


def test_pricing_quote_view_renders_pdf_in_app(qapp, tmp_path, monkeypatch):
    from src.services import pricing_quote_service as pqs
    from tests.test_pricing_quote_service import _sample_payload

    sample = _sample_payload()
    out = tmp_path / "quote.pdf"

    def _fake_create(**kwargs):
        path = pqs.render_pricing_pdf(sample, output_path=out)
        return pqs.PricingQuoteResult(pdf_path=path, source="bundled", data=sample)

    monkeypatch.setattr(
        "src.presentation.views.pricing_quote_view.create_pricing_quote_pdf",
        _fake_create,
    )
    # Avoid modal dialogs if anything unexpected fails in offscreen CI.
    monkeypatch.setattr(
        "src.presentation.views.pricing_quote_view.message_box.warning",
        lambda *a, **k: None,
    )

    opened_urls: list[str] = []

    def _fake_open(url):
        opened_urls.append(url.toString())
        return True

    monkeypatch.setattr(
        "PySide6.QtGui.QDesktopServices.openUrl",
        _fake_open,
        raising=False,
    )

    viewer = PricingQuoteView(theme="dark")
    qapp.processEvents()
    # Ensure pages are present even if the queued timer already ran.
    if viewer.pages_layout.count() == 0:
        viewer._load_pdf()
        qapp.processEvents()

    assert viewer.pages_layout.count() >= 1
    assert opened_urls == [], "Pricing PDF must stay inside the app (no OS/browser open)"
    assert "in-app PDF" in viewer.status_label.text()
    viewer.close()
    viewer.deleteLater()
