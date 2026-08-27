"""Tests for live pricing quote JSON + PDF generation."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from src.services.pricing_quote_service import (
    PricingQuoteError,
    bundled_prices_path,
    create_pricing_quote_pdf,
    load_prices_file,
    load_prices_json_text,
    load_quote_data,
    render_pricing_pdf,
    validate_prices_data,
)


def _sample_payload() -> dict:
    return {
        "version": 1,
        "currency": "INR",
        "currency_symbol": "Rs",
        "location": "Rajkot, Gujarat",
        "audience": "FOR TILE SHOWROOMS & MANUFACTURERS",
        "product_name": "TileVision AI",
        "tagline": "AI Visual Tile Search · Pricing Quote",
        "hero_title": "Find the right tile from a photo — in seconds.",
        "hero_body": "Drop a photo. TileVision AI ranks the closest matches.",
        "features_heading": "What TileVision AI does",
        "features": [
            {"title": "Visual search", "body": "Match from photos."},
            {"title": "Works offline", "body": "Catalogue stays on your PC."},
        ],
        "pricing_heading": "License pricing (INR)",
        "plans": [
            {
                "id": "1y",
                "label": "1 Year",
                "price": 38000,
                "effective_per_year": 38000,
                "discount_note": "-",
            },
            {
                "id": "3y",
                "label": "3 Year",
                "price": 96900,
                "effective_per_year": 32300,
                "discount_note": "15% off",
                "badge": "Best value",
            },
            {
                "id": "lifetime",
                "label": "Lifetime",
                "price": 200000,
                "effective_label": "One-time",
                "discount_note": "Best for long-term",
            },
        ],
        "included_heading": "Included with every license",
        "included": ["Single PC activation", "Setup guidance"],
        "why_heading": "Why showrooms choose TileVision AI",
        "why_points": ["Faster matching.", "Fewer wrong tiles."],
        "vendor": {
            "name": "JD Software",
            "email": "jayrajadesara@gmail.com",
            "phone": "(+91) 88662 77767",
            "phone_display": "Mobile / WhatsApp: (+91) 88662 77767",
        },
        "footer": {
            "validity": "Valid for discussion · Prices in INR",
            "taxes": "Taxes included",
            "confidential": "TileVision AI · Confidential quote",
            "made_by_prefix": "Software by",
        },
        "updated_at": "2026-08-27",
    }


def test_bundled_prices_json_exists_and_validates():
    path = bundled_prices_path()
    assert path.is_file(), f"missing bundled prices at {path}"
    data = load_prices_file(path)
    assert data["vendor"]["name"] == "JD Software"
    assert data["location"] == "Rajkot, Gujarat"
    assert data["footer"]["taxes"] == "Taxes included"
    assert any(p.get("id") == "lifetime" for p in data["plans"])


def test_repo_pricing_json_matches_bundled_keys():
    repo = Path("pricing/prices.json")
    if not repo.is_file():
        pytest.skip("repo pricing/prices.json not present")
    bundled = load_prices_file(bundled_prices_path())
    remote = load_prices_file(repo)
    assert remote["product_name"] == bundled["product_name"]
    assert remote["vendor"]["phone"] == bundled["vendor"]["phone"]
    assert len(remote["plans"]) == len(bundled["plans"])


def test_validate_rejects_missing_plans():
    payload = _sample_payload()
    del payload["plans"]
    with pytest.raises(PricingQuoteError, match="plans"):
        validate_prices_data(payload)


def test_validate_rejects_bad_price():
    payload = _sample_payload()
    payload["plans"][0]["price"] = "not-a-number"
    with pytest.raises(PricingQuoteError, match="invalid price"):
        validate_prices_data(payload)


def test_load_quote_data_uses_bundled_when_remote_fails(monkeypatch, tmp_path):
    from src.services import pricing_quote_service as svc

    monkeypatch.setattr(svc, "_CACHE_JSON", tmp_path / "missing.json")

    def boom(*_a, **_k):
        raise PricingQuoteError("offline")

    monkeypatch.setattr(svc, "fetch_remote_prices", boom)
    data, source = load_quote_data(prefer_remote=True)
    assert source == "bundled"
    assert data["vendor"]["name"] == "JD Software"


def test_load_quote_data_prefers_cache(monkeypatch, tmp_path):
    from src.services import pricing_quote_service as svc

    cache = tmp_path / "prices.json"
    payload = _sample_payload()
    payload["location"] = "Cached City"
    cache.write_text(json.dumps(payload), encoding="utf-8")
    monkeypatch.setattr(svc, "_CACHE_JSON", cache)

    def boom(*_a, **_k):
        raise PricingQuoteError("offline")

    monkeypatch.setattr(svc, "fetch_remote_prices", boom)
    data, source = load_quote_data(prefer_remote=True)
    assert source == "cache"
    assert data["location"] == "Cached City"


def test_render_pricing_pdf_writes_file(tmp_path):
    out = tmp_path / "quote.pdf"
    path = render_pricing_pdf(_sample_payload(), output_path=out)
    assert path == out
    assert out.is_file()
    assert out.stat().st_size > 1000


def test_create_pricing_quote_pdf_offline(monkeypatch, tmp_path):
    from src.services import pricing_quote_service as svc

    monkeypatch.setattr(svc, "_CACHE_DIR", tmp_path)
    monkeypatch.setattr(svc, "_CACHE_JSON", tmp_path / "prices.json")
    monkeypatch.setattr(svc, "_CACHE_PDF", tmp_path / "quote.pdf")

    def boom(*_a, **_k):
        raise PricingQuoteError("offline")

    monkeypatch.setattr(svc, "fetch_remote_prices", boom)
    result = create_pricing_quote_pdf()
    assert result.source == "bundled"
    assert result.pdf_path.is_file()
    assert "JD Software" in json.dumps(result.data)


def test_help_view_has_no_pricing_button():
    source = Path("src/presentation/views/help_view.py").read_text(encoding="utf-8")
    assert "Pricing Quote (PDF)" not in source
    assert "class HelpView(QWidget)" in source
    assert "QDesktopServices.openUrl" not in source
