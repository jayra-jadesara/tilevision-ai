"""Tests for in-app Pricing Quote page and sidebar entry."""

from __future__ import annotations

import sys
import time
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


def _wait_until(predicate, *, timeout: float = 5.0, qapp=None) -> None:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if qapp is not None:
            qapp.processEvents()
        if predicate():
            return
        time.sleep(0.02)
    raise AssertionError(f"Condition not met within {timeout}s")


def test_pricing_nav_icons_exist():
    icons_root = Path(__file__).resolve().parents[1] / "src" / "resources" / "icons"
    assert (icons_root / "dark" / "nav_pricing.svg").is_file()
    assert (icons_root / "light" / "nav_pricing.svg").is_file()
    assert not nav_icon("pricing", "dark").isNull()
    assert not nav_icon("pricing", "light").isNull()


def test_main_window_wires_pricing_as_stack_page():
    """Help + Pricing are checkable stack pages (not modal dialogs)."""
    source = (
        Path(__file__).resolve().parents[1]
        / "src"
        / "presentation"
        / "views"
        / "main_window.py"
    ).read_text(encoding="utf-8")
    assert 'NavButton("Help", "help"' in source
    assert 'NavButton("Pricing", "pricing"' in source
    assert "_navigate(4)" in source
    assert "_navigate(5)" in source
    assert "PricingQuoteView(theme=" in source
    assert "HelpView(theme=" in source
    assert "_on_pricing_clicked" not in source
    assert "_on_help_clicked" not in source


def test_help_has_no_pricing_quote_button():
    source = (
        Path(__file__).resolve().parents[1]
        / "src"
        / "presentation"
        / "views"
        / "help_view.py"
    ).read_text(encoding="utf-8")
    assert "Pricing Quote (PDF)" not in source
    assert "PricingQuoteView" not in source
    assert "QDesktopServices.openUrl" not in source
    assert "class HelpView(QWidget)" in source


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
    viewer.show()
    qapp.processEvents()
    _wait_until(lambda: viewer.pages_layout.count() >= 1, qapp=qapp)

    assert viewer.pages_layout.count() >= 1
    assert opened_urls == [], "Pricing PDF must stay inside the app (no OS/browser open)"
    assert not viewer.status_label.isVisible()
    assert "prices.json" not in viewer.status_label.text()
    assert viewer.download_btn.isEnabled()
    assert viewer._progress_bar.isVisible() is False
    # A4 preview: page is capped and not full-bleed.
    page = viewer.pages_layout.itemAt(0).widget()
    assert page is not None
    assert page.width() <= 680
    viewer.close()
    viewer.deleteLater()


def test_pricing_download_copies_pdf(qapp, tmp_path, monkeypatch):
    from src.services import pricing_quote_service as pqs
    from tests.test_pricing_quote_service import _sample_payload

    sample = _sample_payload()
    out = tmp_path / "quote.pdf"
    dest = tmp_path / "saved" / "quote.pdf"
    dest.parent.mkdir()

    def _fake_create(**kwargs):
        path = pqs.render_pricing_pdf(sample, output_path=out)
        return pqs.PricingQuoteResult(pdf_path=path, source="bundled", data=sample)

    monkeypatch.setattr(
        "src.presentation.views.pricing_quote_view.create_pricing_quote_pdf",
        _fake_create,
    )
    monkeypatch.setattr(
        "src.presentation.views.pricing_quote_view.message_box.warning",
        lambda *a, **k: None,
    )
    monkeypatch.setattr(
        "src.presentation.views.pricing_quote_view.QFileDialog.getSaveFileName",
        lambda *a, **k: (str(dest), "PDF Files (*.pdf)"),
    )

    viewer = PricingQuoteView(theme="dark")
    viewer.show()
    qapp.processEvents()
    _wait_until(lambda: viewer.download_btn.isEnabled(), qapp=qapp)
    viewer._download_pdf()
    assert dest.is_file()
    assert dest.stat().st_size > 0
    viewer.close()
    viewer.deleteLater()
