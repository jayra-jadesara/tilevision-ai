# src/services/pdf_export_service.py

"""
PDF export for TileVision AI search results.

Generates a professional B2B tile catalogue PDF suitable for
manufacturers to share with dealers — reference image, recommended
match, and ranked product selection sheets.

Requires:
    pip install reportlab pillow
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from html import escape as html_escape
from pathlib import Path
from typing import Iterable, Optional, Sequence

from PIL import Image
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_LEFT
from reportlab.lib.pagesizes import A4, landscape
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import mm
from reportlab.platypus import (
    HRFlowable,
    Image as RLImage,
    KeepTogether,
    PageBreak,
    Paragraph,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
)

try:
    from src.core.models import SearchResult
except Exception:  # pragma: no cover
    SearchResult = object  # type: ignore

# Brand palette aligned with TileVision UI (slate + sky blue)
_PRIMARY = colors.HexColor("#0F172A")
_ACCENT = colors.HexColor("#0284C7")
_ACCENT_DEEP = colors.HexColor("#0369A1")
_ACCENT_LIGHT = colors.HexColor("#E0F2FE")
_BORDER = colors.HexColor("#CBD5E1")
_MUTED = colors.HexColor("#64748B")
_PANEL = colors.HexColor("#F8FAFC")
_SUCCESS = colors.HexColor("#15803D")
_WHITE = colors.white


@dataclass
class PdfExportOptions:
    title: str = "Tile Selection Catalogue"
    company_name: str = "TileVision AI"
    company_email: Optional[str] = None
    company_phone: Optional[str] = None
    company_website: Optional[str] = None
    company_address: Optional[str] = None
    logo_path: Optional[str] = None
    include_search_image: bool = True
    include_image_path: bool = False
    include_similarity: bool = True
    include_selected_only: bool = False
    max_results: int = 12
    landscape_mode: bool = False
    watermark_text: Optional[str] = None
    generated_by: str = "TileVision AI"


class PdfExportError(RuntimeError):
    pass


class PDFExportService:
    def __init__(self) -> None:
        base = getSampleStyleSheet()
        self.styles = base
        self.styles.add(
            ParagraphStyle(
                name="CoverTitle",
                parent=base["Title"],
                fontSize=26,
                leading=30,
                alignment=TA_CENTER,
                textColor=_PRIMARY,
                spaceAfter=6,
            )
        )
        self.styles.add(
            ParagraphStyle(
                name="CoverSubtitle",
                parent=base["BodyText"],
                fontSize=12,
                leading=16,
                alignment=TA_CENTER,
                textColor=_MUTED,
                spaceAfter=4,
            )
        )
        self.styles.add(
            ParagraphStyle(
                name="SectionHeading",
                parent=base["Heading2"],
                fontSize=14,
                leading=18,
                textColor=_PRIMARY,
                spaceBefore=4,
                spaceAfter=8,
            )
        )
        self.styles.add(
            ParagraphStyle(
                name="MetaLabel",
                parent=base["BodyText"],
                fontSize=8,
                leading=10,
                textColor=_MUTED,
                alignment=TA_LEFT,
            )
        )
        self.styles.add(
            ParagraphStyle(
                name="MetaValue",
                parent=base["BodyText"],
                fontSize=9,
                leading=12,
                textColor=_PRIMARY,
                alignment=TA_LEFT,
            )
        )
        self.styles.add(
            ParagraphStyle(
                name="BodySmall",
                parent=base["BodyText"],
                fontSize=9,
                leading=12,
                textColor=_MUTED,
                alignment=TA_LEFT,
            )
        )
        self.styles.add(
            ParagraphStyle(
                name="CenterSmall",
                parent=base["BodyText"],
                fontSize=8,
                leading=10,
                alignment=TA_CENTER,
                textColor=_MUTED,
            )
        )
        self.styles.add(
            ParagraphStyle(
                name="ProductCode",
                parent=base["BodyText"],
                fontSize=11,
                leading=14,
                textColor=_WHITE,
                alignment=TA_CENTER,
            )
        )
        self.styles.add(
            ParagraphStyle(
                name="ClosingTitle",
                parent=base["Title"],
                fontSize=20,
                leading=24,
                alignment=TA_CENTER,
                textColor=_PRIMARY,
                spaceAfter=10,
            )
        )

    def export_catalogue(
        self,
        output_file: str | Path,
        query_image_path: Optional[str],
        results: Sequence[SearchResult],
        options: Optional[PdfExportOptions] = None,
        selected_indices: Optional[Iterable[int]] = None,
    ) -> str:
        options = options or PdfExportOptions()
        output_path = Path(output_file).expanduser().resolve()

        try:
            output_path.parent.mkdir(parents=True, exist_ok=True)
        except Exception as exc:
            raise PdfExportError(f"Unable to create output folder: {output_path.parent}") from exc

        filtered_results = self._filter_results(results, selected_indices, options.max_results)
        if not filtered_results:
            raise PdfExportError("No search results available to export.")

        reference_id = datetime.now().strftime("CAT-%Y%m%d-%H%M%S")
        generated_at = datetime.now().strftime("%d %B %Y, %I:%M %p")

        page_size = landscape(A4) if options.landscape_mode else A4
        doc = SimpleDocTemplate(
            str(output_path),
            pagesize=page_size,
            rightMargin=14 * mm,
            leftMargin=14 * mm,
            topMargin=16 * mm,
            bottomMargin=18 * mm,
            title=options.title,
            author=options.generated_by,
        )
        content_width = doc.width

        story: list = []
        story.extend(
            self._build_cover_page(options, reference_id, generated_at, len(filtered_results), content_width)
        )
        story.append(PageBreak())

        if options.include_search_image and query_image_path and Path(query_image_path).exists():
            story.extend(
                self._build_reference_page(
                    query_image_path,
                    filtered_results[0],
                    options,
                    content_width,
                )
            )
            story.append(PageBreak())
        elif filtered_results:
            story.extend(self._build_best_match_only(filtered_results[0], options, content_width))
            story.append(Spacer(1, 6 * mm))

        story.extend(
            self._build_catalogue_section(filtered_results, options, content_width)
        )
        story.append(PageBreak())
        story.extend(self._build_closing_page(options, reference_id, content_width))

        doc.build(
            story,
            onFirstPage=lambda c, d: self._decorate_page(c, d, options, reference_id),
            onLaterPages=lambda c, d: self._decorate_page(c, d, options, reference_id),
        )
        return str(output_path)

    def _filter_results(
        self,
        results: Sequence[SearchResult],
        selected_indices: Optional[Iterable[int]],
        max_results: int,
    ) -> list[SearchResult]:
        if selected_indices is not None:
            index_set = {int(i) for i in selected_indices}
            filtered = [r for idx, r in enumerate(results) if idx in index_set]
        else:
            filtered = list(results)
        return filtered[:max_results]

    # ── Cover ─────────────────────────────────────────────────────────────

    def _build_cover_page(
        self,
        options: PdfExportOptions,
        reference_id: str,
        generated_at: str,
        result_count: int,
        width: float,
    ) -> list:
        elements: list = []
        elements.append(Spacer(1, 18 * mm))

        logo_row = []
        if options.logo_path and Path(options.logo_path).exists():
            try:
                logo_row.append(
                    self._make_rl_image(options.logo_path, width=28 * mm, height=28 * mm)
                )
            except Exception:
                logo_row = []

        company_block = Paragraph(
            f'<font size="22" color="#0F172A"><b>{self._escape(options.company_name)}</b></font>',
            self.styles["BodyText"],
        )

        if logo_row:
            header = Table(
                [[logo_row[0], company_block]],
                colWidths=[34 * mm, width - 34 * mm],
            )
            header.setStyle(
                TableStyle(
                    [
                        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
                        ("LEFTPADDING", (0, 0), (-1, -1), 0),
                        ("RIGHTPADDING", (0, 0), (-1, -1), 0),
                    ]
                )
            )
            elements.append(header)
        else:
            elements.append(company_block)

        elements.append(Spacer(1, 22 * mm))
        elements.append(Paragraph(self._escape(options.title), self.styles["CoverTitle"]))
        elements.append(
            Paragraph(
                "Dealer Reference Catalogue &mdash; Visual Search Results",
                self.styles["CoverSubtitle"],
            )
        )
        elements.append(Spacer(1, 6 * mm))
        elements.append(
            HRFlowable(width="100%", thickness=1, color=_ACCENT, spaceBefore=4, spaceAfter=14)
        )

        meta_rows = [
            ["Document Ref.", reference_id],
            ["Generated", generated_at],
            ["Products Listed", str(result_count)],
        ]
        meta_table = Table(meta_rows, colWidths=[38 * mm, width - 38 * mm])
        meta_table.setStyle(
            TableStyle(
                [
                    ("FONTNAME", (0, 0), (0, -1), "Helvetica-Bold"),
                    ("FONTSIZE", (0, 0), (-1, -1), 9),
                    ("TEXTCOLOR", (0, 0), (0, -1), _MUTED),
                    ("TEXTCOLOR", (1, 0), (1, -1), _PRIMARY),
                    ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
                    ("TOPPADDING", (0, 0), (-1, -1), 6),
                    ("LINEBELOW", (0, -1), (-1, -1), 0.5, _BORDER),
                ]
            )
        )
        elements.append(meta_table)
        elements.append(Spacer(1, 10 * mm))

        contact_lines = self._contact_lines(options)
        if contact_lines:
            contact = Paragraph("<br/>".join(contact_lines), self.styles["BodySmall"])
            contact_box = Table([[contact]], colWidths=[width])
            contact_box.setStyle(
                TableStyle(
                    [
                        ("BOX", (0, 0), (-1, -1), 0.5, _BORDER),
                        ("BACKGROUND", (0, 0), (-1, -1), _PANEL),
                        ("TOPPADDING", (0, 0), (-1, -1), 12),
                        ("BOTTOMPADDING", (0, 0), (-1, -1), 12),
                        ("LEFTPADDING", (0, 0), (-1, -1), 14),
                        ("RIGHTPADDING", (0, 0), (-1, -1), 14),
                    ]
                )
            )
            elements.append(contact_box)

        elements.append(Spacer(1, 16 * mm))
        elements.append(
            Paragraph(
                "Prepared for dealer review. Please verify product codes, "
                "availability, and pricing before confirming orders.",
                self.styles["CenterSmall"],
            )
        )
        return elements

    # ── Reference + recommended match ─────────────────────────────────────

    def _build_reference_page(
        self,
        query_image_path: str,
        best: SearchResult,
        options: PdfExportOptions,
        width: float,
    ) -> list:
        elements: list = []
        elements.append(Paragraph("Reference &amp; Recommended Match", self.styles["SectionHeading"]))
        elements.append(
            Paragraph(
                "The reference image below was used for visual search. "
                "Our top recommended match is highlighted for your review.",
                self.styles["BodySmall"],
            )
        )
        elements.append(Spacer(1, 6 * mm))

        ref_img = self._make_rl_image(query_image_path, width=78 * mm, height=78 * mm)
        best_path = self._result_image_path(best)
        if best_path:
            best_img = self._make_rl_image(best_path, width=78 * mm, height=78 * mm)
        else:
            best_img = Paragraph("No image", self.styles["BodySmall"])

        compare = Table(
            [
                [
                    Paragraph("<b>Reference Image</b>", self.styles["MetaValue"]),
                    Paragraph("<b>Recommended Match</b>", self.styles["MetaValue"]),
                ],
                [ref_img, best_img],
            ],
            colWidths=[width / 2 - 2 * mm, width / 2 - 2 * mm],
        )
        compare.setStyle(
            TableStyle(
                [
                    ("ALIGN", (0, 0), (-1, -1), "CENTER"),
                    ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
                    ("BACKGROUND", (0, 0), (-1, 0), _ACCENT_LIGHT),
                    ("BOX", (0, 0), (-1, -1), 0.5, _BORDER),
                    ("INNERGRID", (0, 0), (-1, -1), 0.5, _BORDER),
                    ("TOPPADDING", (0, 0), (-1, -1), 8),
                    ("BOTTOMPADDING", (0, 0), (-1, -1), 8),
                ]
            )
        )
        elements.append(compare)
        elements.append(Spacer(1, 8 * mm))
        elements.extend(self._build_featured_product_table(best, options, width, rank=1))
        return elements

    def _build_best_match_only(
        self,
        best: SearchResult,
        options: PdfExportOptions,
        width: float,
    ) -> list:
        elements: list = []
        elements.append(Paragraph("Recommended Match", self.styles["SectionHeading"]))
        elements.extend(self._build_featured_product_table(best, options, width, rank=1))
        return elements

    def _build_featured_product_table(
        self,
        result: SearchResult,
        options: PdfExportOptions,
        width: float,
        rank: int,
    ) -> list:
        tile = result.tile
        rows = [
            ["Rank", f"#{rank}"],
            ["Product Code", self._escape(getattr(tile, "product_code", None) or "—")],
            ["Brand", self._escape(getattr(tile, "brand", None) or "—")],
            ["Category", self._escape(getattr(tile, "category", None) or "—")],
            ["Size", self._escape(getattr(tile, "size", None) or "—")],
            ["Color", self._escape(getattr(tile, "color", None) or "—")],
        ]
        if options.include_similarity:
            rows.append(["Match Score", f"{result.similarity_score:.1f}%"])
        if options.include_image_path and tile.file_path:
            rows.append(["Image Path", self._escape(tile.file_path)])

        table = Table(rows, colWidths=[42 * mm, width - 42 * mm])
        table.setStyle(
            TableStyle(
                [
                    ("FONTNAME", (0, 0), (0, -1), "Helvetica-Bold"),
                    ("FONTSIZE", (0, 0), (-1, -1), 9),
                    ("TEXTCOLOR", (0, 0), (0, -1), _MUTED),
                    ("TEXTCOLOR", (1, 0), (1, -1), _PRIMARY),
                    ("BACKGROUND", (0, 0), (-1, 0), _ACCENT_LIGHT),
                    ("BOX", (0, 0), (-1, -1), 0.5, _BORDER),
                    ("INNERGRID", (0, 0), (-1, -1), 0.25, _BORDER),
                    ("TOPPADDING", (0, 0), (-1, -1), 5),
                    ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
                    ("LEFTPADDING", (0, 0), (-1, -1), 8),
                ]
            )
        )
        return [table]

    # ── Product catalogue grid ────────────────────────────────────────────

    def _build_catalogue_section(
        self,
        results: Sequence[SearchResult],
        options: PdfExportOptions,
        width: float,
    ) -> list:
        story: list = []
        story.append(Paragraph("Product Selection Sheet", self.styles["SectionHeading"]))
        story.append(
            Paragraph(
                f"{len(results)} matching product(s) ranked by visual similarity. "
                "Share this sheet with your dealer network for selection and confirmation.",
                self.styles["BodySmall"],
            )
        )
        story.append(Spacer(1, 6 * mm))

        col_width = (width - 6 * mm) / 2
        cards: list = []
        for index, result in enumerate(results, start=1):
            cards.append(self._build_product_card(result, options, col_width, index))

        rows: list = []
        for i in range(0, len(cards), 2):
            left = cards[i]
            right = cards[i + 1] if i + 1 < len(cards) else ""
            rows.append([left, right])

        grid = Table(rows, colWidths=[col_width + 3 * mm, col_width + 3 * mm])
        grid.setStyle(
            TableStyle(
                [
                    ("VALIGN", (0, 0), (-1, -1), "TOP"),
                    ("LEFTPADDING", (0, 0), (-1, -1), 0),
                    ("RIGHTPADDING", (0, 0), (-1, -1), 0),
                    ("BOTTOMPADDING", (0, 0), (-1, -1), 8),
                ]
            )
        )
        story.append(grid)
        return story

    def _build_product_card(
        self,
        result: SearchResult,
        options: PdfExportOptions,
        card_width: float,
        rank: int,
    ) -> Table:
        tile = result.tile
        image_path = self._result_image_path(result)
        img_w = card_width - 12 * mm

        if image_path:
            image = self._make_rl_image(image_path, width=img_w, height=img_w)
        else:
            image = Paragraph("No image available", self.styles["BodySmall"])

        code = self._escape(getattr(tile, "product_code", None) or "N/A")
        header = Paragraph(f"<b>{code}</b>", self.styles["ProductCode"])

        spec_lines = [
            f"<b>Brand</b>&nbsp;&nbsp; {self._escape(getattr(tile, 'brand', None) or '—')}",
            f"<b>Category</b>&nbsp;&nbsp; {self._escape(getattr(tile, 'category', None) or '—')}",
            f"<b>Size</b>&nbsp;&nbsp; {self._escape(getattr(tile, 'size', None) or '—')}",
            f"<b>Color</b>&nbsp;&nbsp; {self._escape(getattr(tile, 'color', None) or '—')}",
        ]
        if options.include_similarity:
            spec_lines.append(
                f'<font color="#15803D"><b>Match: {result.similarity_score:.1f}%</b></font>'
            )
        if options.include_image_path and tile.file_path:
            spec_lines.append(
                f'<font size="7" color="#64748B">{self._escape(tile.file_path)}</font>'
            )

        rank_badge = Paragraph(
            f'<font color="#64748B">Rank #{rank}</font>',
            self.styles["MetaLabel"],
        )

        card = Table(
            [
                [header],
                [image],
                [Paragraph("<br/>".join(spec_lines), self.styles["MetaValue"])],
                [rank_badge],
            ],
            colWidths=[card_width],
        )
        card.setStyle(
            TableStyle(
                [
                    ("BACKGROUND", (0, 0), (-1, 0), _ACCENT),
                    ("BOX", (0, 0), (-1, -1), 0.5, _BORDER),
                    ("ALIGN", (0, 0), (-1, -1), "CENTER"),
                    ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
                    ("TOPPADDING", (0, 0), (-1, 0), 6),
                    ("BOTTOMPADDING", (0, 0), (-1, 0), 6),
                    ("TOPPADDING", (0, 1), (-1, 1), 8),
                    ("BOTTOMPADDING", (0, 1), (-1, 1), 4),
                    ("TOPPADDING", (0, 2), (-1, 2), 4),
                    ("BOTTOMPADDING", (0, 2), (-1, 2), 6),
                    ("LEFTPADDING", (0, 0), (-1, -1), 6),
                    ("RIGHTPADDING", (0, 0), (-1, -1), 6),
                    ("BACKGROUND", (0, 1), (-1, -1), _WHITE),
                ]
            )
        )
        return card

    # ── Closing page ──────────────────────────────────────────────────────

    def _build_closing_page(
        self,
        options: PdfExportOptions,
        reference_id: str,
        width: float,
    ) -> list:
        story: list = []
        story.append(Spacer(1, 20 * mm))
        story.append(Paragraph("Next Steps", self.styles["ClosingTitle"]))
        story.append(
            Paragraph(
                "Review the products listed in this catalogue and confirm your selection "
                f"with <b>{self._escape(options.company_name)}</b>. "
                "Our team will assist with availability, pricing, and order placement.",
                self.styles["BodySmall"],
            )
        )
        story.append(Spacer(1, 10 * mm))

        if options.logo_path and Path(options.logo_path).exists():
            try:
                story.append(
                    KeepTogether(
                        [
                            self._make_rl_image(options.logo_path, width=22 * mm, height=22 * mm),
                            Spacer(1, 6 * mm),
                        ]
                    )
                )
            except Exception:
                pass

        contact_lines = self._contact_lines(options, bold_name=True)
        if contact_lines:
            story.append(Paragraph("<br/>".join(contact_lines), self.styles["CoverSubtitle"]))
            story.append(Spacer(1, 10 * mm))

        story.append(
            HRFlowable(width="60%", thickness=0.5, color=_BORDER, spaceBefore=6, spaceAfter=10)
        )
        story.append(
            Paragraph(
                f"Document reference: <b>{reference_id}</b><br/>"
                "This catalogue was generated by TileVision AI visual search.",
                self.styles["CenterSmall"],
            )
        )
        return story

    # ── Page chrome ───────────────────────────────────────────────────────

    def _decorate_page(
        self,
        canvas,
        doc,
        options: PdfExportOptions,
        reference_id: str,
    ) -> None:
        canvas.saveState()
        width, height = doc.pagesize
        page_num = canvas.getPageNumber()

        # Header bar
        canvas.setFillColor(_PRIMARY)
        canvas.rect(0, height - 14 * mm, width, 14 * mm, stroke=0, fill=1)

        canvas.setFillColor(_WHITE)
        canvas.setFont("Helvetica-Bold", 10)
        canvas.drawString(doc.leftMargin, height - 9.5 * mm, options.company_name[:60])

        canvas.setFont("Helvetica", 8)
        canvas.drawRightString(
            width - doc.rightMargin,
            height - 9.5 * mm,
            options.title[:50],
        )

        # Accent line under header
        canvas.setStrokeColor(_ACCENT)
        canvas.setLineWidth(1.5)
        canvas.line(0, height - 14 * mm, width, height - 14 * mm)

        # Footer
        canvas.setFillColor(_PANEL)
        canvas.rect(0, 0, width, 12 * mm, stroke=0, fill=1)
        canvas.setStrokeColor(_BORDER)
        canvas.setLineWidth(0.5)
        canvas.line(0, 12 * mm, width, 12 * mm)

        canvas.setFillColor(_MUTED)
        canvas.setFont("Helvetica", 7)
        footer_parts = [p for p in (options.company_phone, options.company_email) if p]
        if footer_parts:
            canvas.drawString(doc.leftMargin, 4.5 * mm, "  |  ".join(footer_parts)[:90])

        canvas.drawCentredString(width / 2, 4.5 * mm, reference_id)
        canvas.drawRightString(width - doc.rightMargin, 4.5 * mm, f"Page {page_num}")

        if options.watermark_text:
            canvas.saveState()
            canvas.setFillColor(colors.HexColor("#E2E8F0"))
            canvas.setFont("Helvetica-Bold", 48)
            canvas.translate(width / 2, height / 2)
            canvas.rotate(42)
            canvas.drawCentredString(0, 0, options.watermark_text[:30])
            canvas.restoreState()

        canvas.restoreState()

    # ── Helpers ───────────────────────────────────────────────────────────

    def _contact_lines(self, options: PdfExportOptions, *, bold_name: bool = False) -> list[str]:
        lines: list[str] = []
        if bold_name:
            lines.append(f"<b>{self._escape(options.company_name)}</b>")
        if options.company_phone:
            lines.append(f"<b>Phone</b>&nbsp; {self._escape(options.company_phone)}")
        if options.company_email:
            lines.append(f"<b>Email</b>&nbsp; {self._escape(options.company_email)}")
        if options.company_website:
            lines.append(f"<b>Website</b>&nbsp; {self._escape(options.company_website)}")
        if options.company_address:
            lines.append(f"<b>Address</b>&nbsp; {self._escape(options.company_address)}")
        return lines

    def _result_image_path(self, result: SearchResult) -> Optional[str]:
        tile = getattr(result, "tile", None)
        if tile and getattr(tile, "file_path", None):
            path = Path(tile.file_path)
            if path.is_file():
                return str(path)
        thumb = getattr(result, "thumbnail_path", None)
        if thumb and Path(thumb).is_file():
            return str(thumb)
        return None

    def _make_rl_image(self, image_path: str, width: float, height: float) -> RLImage:
        with Image.open(image_path) as img:
            img_w, img_h = img.size
        ratio = min(width / img_w, height / img_h)
        image = RLImage(image_path)
        image.drawWidth = img_w * ratio
        image.drawHeight = img_h * ratio
        return image

    @staticmethod
    def _escape(value: Optional[str]) -> str:
        if value is None:
            return "—"
        return html_escape(str(value), quote=False)
