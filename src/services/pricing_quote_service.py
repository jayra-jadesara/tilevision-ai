"""
Live pricing quote for TileVision AI.

Fetches ``pricing/prices.json`` from GitHub (or a vendor URL), caches it under
``~/.tilevision_ai/cache/``, and renders a one-page PDF with reportlab.

Edit ``pricing/prices.json`` on ``main`` to change prices without shipping a
new app installer. Offline users fall back to cache, then the bundled copy.
"""

from __future__ import annotations

import json
import logging
import ssl
import tempfile
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Optional

logger = logging.getLogger("tilevision.services.pricing_quote")

DEFAULT_PRICES_URL = (
    "https://raw.githubusercontent.com/jayra-jadesara/tilevision-ai/"
    "main/pricing/prices.json"
)

_CACHE_DIR = Path.home() / ".tilevision_ai" / "cache"
_CACHE_JSON = _CACHE_DIR / "prices.json"
_CACHE_PDF = _CACHE_DIR / "TileVision_AI_Pricing_Quote.pdf"


@dataclass(frozen=True, slots=True)
class PricingQuoteResult:
    """Outcome of loading quote data and rendering a PDF."""

    pdf_path: Path
    source: str  # "remote" | "cache" | "bundled"
    data: dict[str, Any]


class PricingQuoteError(RuntimeError):
    """Raised when quote data cannot be loaded or the PDF cannot be built."""


def bundled_prices_path() -> Path:
    """Packaged fallback shipped with the app."""
    resource = (
        Path(__file__).resolve().parents[1] / "resources" / "pricing" / "prices.json"
    )
    if resource.is_file():
        return resource
    # Dev checkout (repo root /pricing/prices.json)
    return Path(__file__).resolve().parents[2] / "pricing" / "prices.json"


def cache_prices_path() -> Path:
    return _CACHE_JSON


def _ssl_context() -> ssl.SSLContext:
    try:
        import certifi

        return ssl.create_default_context(cafile=certifi.where())
    except Exception:
        return ssl.create_default_context()


def _http_get_text(url: str, *, timeout: float) -> str:
    from src.version import APP_VERSION

    request = urllib.request.Request(
        url,
        headers={
            "User-Agent": f"TileVisionAI/{APP_VERSION}",
            "Accept": "application/json",
            "Cache-Control": "no-cache",
        },
    )
    with urllib.request.urlopen(
        request, timeout=timeout, context=_ssl_context()
    ) as response:
        return response.read().decode("utf-8")


def validate_prices_data(data: Mapping[str, Any]) -> dict[str, Any]:
    """Validate and normalize quote JSON. Raises PricingQuoteError on bad data."""
    if not isinstance(data, Mapping):
        raise PricingQuoteError("Pricing data must be a JSON object.")

    required = (
        "product_name",
        "plans",
        "features",
        "vendor",
        "hero_title",
        "hero_body",
    )
    missing = [key for key in required if key not in data]
    if missing:
        raise PricingQuoteError(
            f"Pricing JSON missing required fields: {', '.join(missing)}"
        )

    plans = data.get("plans")
    if not isinstance(plans, list) or not plans:
        raise PricingQuoteError("Pricing JSON must include a non-empty plans list.")

    for index, plan in enumerate(plans):
        if not isinstance(plan, Mapping):
            raise PricingQuoteError(f"Plan #{index + 1} must be an object.")
        if "label" not in plan or "price" not in plan:
            raise PricingQuoteError(
                f"Plan #{index + 1} needs at least label and price."
            )
        try:
            float(plan["price"])
        except (TypeError, ValueError) as exc:
            raise PricingQuoteError(
                f"Plan #{index + 1} has an invalid price."
            ) from exc

    vendor = data.get("vendor")
    if not isinstance(vendor, Mapping) or not vendor.get("name"):
        raise PricingQuoteError("Pricing JSON needs vendor.name.")

    return dict(data)


def load_prices_json_text(text: str) -> dict[str, Any]:
    try:
        payload = json.loads(text)
    except json.JSONDecodeError as exc:
        raise PricingQuoteError(f"Pricing JSON is invalid: {exc}") from exc
    return validate_prices_data(payload)


def load_prices_file(path: Path) -> dict[str, Any]:
    try:
        text = path.read_text(encoding="utf-8")
    except OSError as exc:
        raise PricingQuoteError(f"Could not read pricing file: {path}") from exc
    return load_prices_json_text(text)


def fetch_remote_prices(
    url: str = DEFAULT_PRICES_URL,
    *,
    timeout: float = 10.0,
) -> dict[str, Any]:
    try:
        text = _http_get_text(url, timeout=timeout)
    except (urllib.error.URLError, urllib.error.HTTPError, TimeoutError, OSError) as exc:
        raise PricingQuoteError(f"Could not download live pricing: {exc}") from exc
    data = load_prices_json_text(text)
    try:
        _CACHE_DIR.mkdir(parents=True, exist_ok=True)
        _CACHE_JSON.write_text(
            json.dumps(data, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
    except OSError as exc:
        logger.warning("Could not cache pricing JSON: %s", exc)
    return data


def load_quote_data(
    *,
    url: str = DEFAULT_PRICES_URL,
    timeout: float = 10.0,
    prefer_remote: bool = True,
) -> tuple[dict[str, Any], str]:
    """
    Return ``(data, source)`` with source in remote|cache|bundled.
    """
    if prefer_remote:
        try:
            return fetch_remote_prices(url, timeout=timeout), "remote"
        except PricingQuoteError as exc:
            logger.info("Live pricing unavailable (%s) — trying cache/bundled.", exc)

    if _CACHE_JSON.is_file():
        try:
            return load_prices_file(_CACHE_JSON), "cache"
        except PricingQuoteError as exc:
            logger.warning("Cached pricing unusable: %s", exc)

    bundled = bundled_prices_path()
    if bundled.is_file():
        return load_prices_file(bundled), "bundled"

    raise PricingQuoteError(
        "No pricing data available (offline and no cached/bundled quote)."
    )


def _format_inr(amount: Any, symbol: str = "Rs") -> str:
    try:
        value = int(round(float(amount)))
    except (TypeError, ValueError):
        return str(amount)
    # Indian grouping: 12,34,567
    text = f"{value:d}"
    if len(text) <= 3:
        return f"{symbol} {text}"
    last3 = text[-3:]
    rest = text[:-3]
    parts: list[str] = []
    while rest:
        parts.append(rest[-2:])
        rest = rest[:-2]
    grouped = ",".join(reversed(parts)) + "," + last3
    return f"{symbol} {grouped}"


def render_pricing_pdf(
    data: Mapping[str, Any],
    output_path: Path | None = None,
    *,
    logo_path: Path | None = None,
) -> Path:
    """Render a one-page A4 pricing PDF from quote JSON."""
    from reportlab.lib.colors import HexColor, white
    from reportlab.lib.pagesizes import A4
    from reportlab.lib.units import mm
    from reportlab.lib.utils import ImageReader
    from reportlab.pdfgen import canvas

    validated = validate_prices_data(data)
    if output_path is None:
        _CACHE_DIR.mkdir(parents=True, exist_ok=True)
        output_path = _CACHE_PDF

    if logo_path is None:
        logo_path = (
            Path(__file__).resolve().parents[1] / "resources" / "app_icon.png"
        )

    vendor = validated.get("vendor") or {}
    footer = validated.get("footer") or {}
    symbol = str(validated.get("currency_symbol") or "Rs")

    navy = HexColor("#0F172A")
    navy_soft = HexColor("#1E293B")
    accent = HexColor("#0284C7")
    accent_bright = HexColor("#0EA5E9")
    accent_light = HexColor("#E0F2FE")
    panel = HexColor("#F8FAFC")
    border = HexColor("#CBD5E1")
    muted = HexColor("#64748B")
    text = HexColor("#0F172A")

    width, height = A4
    pdf = canvas.Canvas(str(output_path), pagesize=A4)

    pdf.setFillColor(HexColor("#F1F5F9"))
    pdf.rect(0, 0, width, height, fill=1, stroke=0)

    pdf.setFillColor(navy)
    pdf.rect(0, height - 42 * mm, width, 42 * mm, fill=1, stroke=0)
    pdf.setFillColor(accent_bright)
    pdf.rect(0, height - 43.2 * mm, width, 1.2 * mm, fill=1, stroke=0)

    logo_size = 30 * mm
    pdf.setFillColor(white)
    pdf.roundRect(
        12 * mm,
        height - 38 * mm,
        logo_size + 2 * mm,
        logo_size + 2 * mm,
        3 * mm,
        fill=1,
        stroke=0,
    )
    if logo_path.is_file():
        try:
            from PIL import Image

            tmp = Path(tempfile.gettempdir()) / "tv_pricing_logo_plate.png"
            img = Image.open(logo_path).convert("RGBA")
            plate = Image.new(
                "RGBA", (img.width + 80, img.height + 80), (255, 255, 255, 255)
            )
            plate.paste(img, (40, 40), img)
            plate.save(tmp)
            pdf.drawImage(
                ImageReader(str(tmp)),
                13 * mm,
                height - 37 * mm,
                width=logo_size,
                height=logo_size,
                mask="auto",
                preserveAspectRatio=True,
            )
        except Exception as exc:
            logger.debug("Pricing PDF logo skipped: %s", exc)

    pdf.setFillColor(white)
    pdf.setFont("Helvetica-Bold", 22)
    pdf.drawString(
        48 * mm,
        height - 20 * mm,
        str(validated.get("product_name") or "TileVision AI"),
    )
    pdf.setFont("Helvetica", 10)
    pdf.setFillColor(HexColor("#94A3B8"))
    pdf.drawString(
        48 * mm,
        height - 26 * mm,
        str(validated.get("tagline") or "AI Visual Tile Search"),
    )
    pdf.setFillColor(accent_bright)
    pdf.setFont("Helvetica-Bold", 9)
    audience = str(validated.get("audience") or "")
    if audience:
        # Split long audience across two lines when needed
        if " & " in audience:
            left, right = audience.split(" & ", 1)
            pdf.drawRightString(width - 14 * mm, height - 18 * mm, left + " &")
            pdf.drawRightString(width - 14 * mm, height - 22.5 * mm, right)
        else:
            pdf.drawRightString(width - 14 * mm, height - 20 * mm, audience)
    pdf.setFillColor(HexColor("#94A3B8"))
    pdf.setFont("Helvetica", 8)
    pdf.drawRightString(
        width - 14 * mm,
        height - 29 * mm,
        str(validated.get("location") or ""),
    )

    y = height - 58 * mm
    pdf.setFillColor(white)
    pdf.roundRect(12 * mm, y - 2 * mm, width - 24 * mm, 14 * mm, 3 * mm, fill=1, stroke=0)
    pdf.setStrokeColor(border)
    pdf.setLineWidth(0.6)
    pdf.roundRect(12 * mm, y - 2 * mm, width - 24 * mm, 14 * mm, 3 * mm, fill=0, stroke=1)
    pdf.setFillColor(text)
    pdf.setFont("Helvetica-Bold", 10)
    pdf.drawString(16 * mm, y + 7 * mm, str(validated.get("hero_title") or ""))
    pdf.setFont("Helvetica", 8.5)
    pdf.setFillColor(muted)
    pdf.drawString(16 * mm, y + 2 * mm, str(validated.get("hero_body") or ""))

    y = height - 72 * mm
    pdf.setFillColor(navy)
    pdf.setFont("Helvetica-Bold", 11)
    features_heading = str(validated.get("features_heading") or "What it does")
    pdf.drawString(14 * mm, y, features_heading)
    pdf.setStrokeColor(accent_bright)
    pdf.setLineWidth(2)
    pdf.line(14 * mm, y - 1.5 * mm, 14 * mm + 48 * mm, y - 1.5 * mm)

    y -= 8 * mm
    for feature in validated.get("features") or []:
        if not isinstance(feature, Mapping):
            continue
        title = str(feature.get("title") or "")
        body = str(feature.get("body") or "")
        pdf.setFillColor(accent)
        pdf.circle(17 * mm, y + 1.2 * mm, 1.4 * mm, fill=1, stroke=0)
        pdf.setFillColor(text)
        pdf.setFont("Helvetica-Bold", 9)
        pdf.drawString(21 * mm, y, title)
        tw = pdf.stringWidth(title, "Helvetica-Bold", 9)
        pdf.setFont("Helvetica", 8.5)
        pdf.setFillColor(muted)
        pdf.drawString(21 * mm + tw + 2 * mm, y, "—  " + body)
        y -= 6.2 * mm

    y -= 3 * mm
    pdf.setFillColor(navy)
    pdf.setFont("Helvetica-Bold", 11)
    pricing_heading = str(validated.get("pricing_heading") or "License pricing")
    pdf.drawString(14 * mm, y, pricing_heading)
    pdf.setStrokeColor(accent_bright)
    pdf.setLineWidth(2)
    pdf.line(14 * mm, y - 1.5 * mm, 14 * mm + 44 * mm, y - 1.5 * mm)

    y -= 4 * mm
    table_top = y
    row_h = 8.2 * mm
    cols = [14 * mm, 52 * mm, 95 * mm, 138 * mm]
    headers = ["Plan", f"Price ({validated.get('currency') or 'INR'})", "Effective / year", "Discount"]

    pdf.setFillColor(navy)
    pdf.roundRect(14 * mm, table_top - row_h, width - 28 * mm, row_h, 2 * mm, fill=1, stroke=0)
    pdf.rect(14 * mm, table_top - row_h, width - 28 * mm, 2 * mm, fill=1, stroke=0)
    pdf.setFillColor(white)
    pdf.setFont("Helvetica-Bold", 8.5)
    for i, header in enumerate(headers):
        pdf.drawString(cols[i] + 2 * mm, table_top - 5.4 * mm, header)

    y = table_top - row_h
    for index, plan in enumerate(validated.get("plans") or []):
        if not isinstance(plan, Mapping):
            continue
        y -= row_h
        badge = plan.get("badge")
        highlight = bool(badge)
        label = str(plan.get("label") or "")
        if badge:
            label = f"{label}  * {badge}"
        price = _format_inr(plan.get("price"), symbol)
        if plan.get("effective_label"):
            per_year = str(plan.get("effective_label"))
        elif plan.get("effective_per_year") is not None:
            per_year = _format_inr(plan.get("effective_per_year"), symbol)
        else:
            per_year = "—"
        discount = str(plan.get("discount_note") or "—")

        bg = accent_light if highlight else (white if index % 2 == 0 else panel)
        pdf.setFillColor(bg)
        pdf.rect(14 * mm, y, width - 28 * mm, row_h, fill=1, stroke=0)
        if highlight:
            pdf.setStrokeColor(accent)
            pdf.setLineWidth(1.2)
            pdf.rect(14 * mm, y, width - 28 * mm, row_h, fill=0, stroke=1)

        pdf.setFillColor(text)
        pdf.setFont(
            "Helvetica-Bold" if highlight or str(plan.get("id")) == "lifetime" else "Helvetica",
            9,
        )
        pdf.drawString(cols[0] + 2 * mm, y + 3 * mm, label)
        pdf.setFont("Helvetica-Bold", 9)
        pdf.setFillColor(accent if highlight else text)
        pdf.drawString(cols[1] + 2 * mm, y + 3 * mm, price)
        pdf.setFont("Helvetica", 8.5)
        pdf.setFillColor(muted)
        pdf.drawString(cols[2] + 2 * mm, y + 3 * mm, per_year)
        pdf.drawString(cols[3] + 2 * mm, y + 3 * mm, discount)

    pdf.setStrokeColor(border)
    pdf.setLineWidth(0.7)
    pdf.rect(14 * mm, y, width - 28 * mm, table_top - y, fill=0, stroke=1)

    y -= 10 * mm
    pdf.setFillColor(navy)
    pdf.setFont("Helvetica-Bold", 10)
    pdf.drawString(
        14 * mm,
        y,
        str(validated.get("included_heading") or "Included with every license"),
    )
    y -= 5.5 * mm
    pdf.setFont("Helvetica", 8.5)
    for item in validated.get("included") or []:
        pdf.setFillColor(accent)
        pdf.drawString(16 * mm, y, ">")
        pdf.setFillColor(text)
        pdf.drawString(21 * mm, y, str(item))
        y -= 5 * mm

    y -= 2 * mm
    box_h = 22 * mm
    pdf.setFillColor(navy)
    pdf.roundRect(14 * mm, y - box_h, width - 28 * mm, box_h, 3 * mm, fill=1, stroke=0)
    pdf.setFillColor(accent_bright)
    pdf.setFont("Helvetica-Bold", 9)
    pdf.drawString(
        18 * mm,
        y - 6 * mm,
        str(validated.get("why_heading") or "Why choose TileVision AI"),
    )
    pdf.setFillColor(HexColor("#CBD5E1"))
    pdf.setFont("Helvetica", 8)
    yy = y - 11.5 * mm
    for point in validated.get("why_points") or []:
        pdf.drawString(18 * mm, yy, f"- {point}")
        yy -= 4.2 * mm

    pdf.setFillColor(navy_soft)
    pdf.rect(0, 0, width, 28 * mm, fill=1, stroke=0)
    pdf.setFillColor(accent_bright)
    pdf.rect(0, 28 * mm, width, 1 * mm, fill=1, stroke=0)

    made_by = str(footer.get("made_by_prefix") or "Software by")
    vendor_name = str(vendor.get("name") or "JD Software")
    pdf.setFillColor(white)
    pdf.setFont("Helvetica-Bold", 10)
    pdf.drawString(14 * mm, 18 * mm, f"{made_by} {vendor_name}")
    pdf.setFont("Helvetica", 8.5)
    pdf.setFillColor(HexColor("#94A3B8"))
    email = str(vendor.get("email") or "")
    if email:
        pdf.drawString(14 * mm, 12 * mm, f"Email: {email}")
    pdf.setFillColor(accent_bright)
    pdf.setFont("Helvetica-Bold", 9)
    phone_line = str(
        vendor.get("phone_display")
        or (f"Mobile / WhatsApp: {vendor.get('phone')}" if vendor.get("phone") else "")
    )
    if phone_line:
        pdf.drawString(14 * mm, 6.5 * mm, phone_line)

    pdf.setFillColor(HexColor("#94A3B8"))
    pdf.setFont("Helvetica", 8)
    pdf.drawRightString(
        width - 14 * mm,
        18 * mm,
        str(footer.get("validity") or "Prices in INR"),
    )
    pdf.drawRightString(
        width - 14 * mm,
        12 * mm,
        str(footer.get("taxes") or "Taxes included"),
    )
    pdf.setFillColor(accent_bright)
    pdf.setFont("Helvetica-Bold", 8)
    pdf.drawRightString(
        width - 14 * mm,
        7 * mm,
        str(footer.get("confidential") or "TileVision AI · Confidential quote"),
    )

    pdf.showPage()
    pdf.save()
    return output_path


def create_pricing_quote_pdf(
    *,
    url: str = DEFAULT_PRICES_URL,
    timeout: float = 10.0,
    output_path: Path | None = None,
) -> PricingQuoteResult:
    """Fetch (or fall back) quote data and write the PDF."""
    data, source = load_quote_data(url=url, timeout=timeout, prefer_remote=True)
    path = render_pricing_pdf(data, output_path=output_path)
    logger.info("Pricing quote PDF ready (%s) → %s", source, path)
    return PricingQuoteResult(pdf_path=path, source=source, data=data)
