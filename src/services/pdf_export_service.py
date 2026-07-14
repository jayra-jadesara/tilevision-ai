# src/services/pdf_export_service.py

"""
PDF export for TileVision AI search results.

Generates a catalogue-style PDF for a visual tile search:
- search image
- best match highlight
- top matches grid/list
- product metadata
- optional selected-only export

Requires:
    pip install reportlab pillow
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Iterable, Optional, Sequence

from PIL import Image

import tempfile

from reportlab.lib.utils import ImageReader
from PIL import Image

from reportlab.lib import colors
from reportlab.lib.enums import TA_LEFT, TA_CENTER
from reportlab.lib.pagesizes import A4, landscape
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import mm
from reportlab.platypus import (
    PageBreak,
    SimpleDocTemplate,
    Paragraph,
    Spacer,
    Image as RLImage,
    Table,
    TableStyle,
    KeepTogether,
)

# Adjust these imports to match your repo if needed.
try:
    from src.core.models import SearchResult
except Exception:  # pragma: no cover
    SearchResult = object  # type: ignore

@dataclass
class PdfExportOptions:
    title: str = "Tile Catalogue"
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
        self.styles = getSampleStyleSheet()
        self.styles.add(
            ParagraphStyle(
                name="TitleCenter",
                parent=self.styles["Title"],
                alignment=TA_CENTER,
                fontSize=18,
                leading=22,
                spaceAfter=8,
            )
        )
        self.styles.add(
            ParagraphStyle(
                name="Meta",
                parent=self.styles["BodyText"],
                fontSize=9,
                leading=11,
                alignment=TA_LEFT,
            )
        )
        self.styles.add(
            ParagraphStyle(
                name="SmallCenter",
                parent=self.styles["BodyText"],
                fontSize=8,
                leading=10,
                alignment=TA_CENTER,
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
        """
        Export a tile match catalogue PDF.

        Args:
            output_file: destination PDF file path
            query_image_path: searched image path
            results: ranked search results
            options: PDF options
            selected_indices: zero-based result indices to include; when given,
                              only those results are exported.

        Returns:
            Absolute path to the created PDF.

        Raises:
            PdfExportError on validation or generation failure.
        """
        options = options or PdfExportOptions()
        output_path = Path(output_file).expanduser().resolve()

        try:
            output_path.parent.mkdir(parents=True, exist_ok=True)
        except Exception as exc:
            raise PdfExportError(f"Unable to create output folder: {output_path.parent}") from exc

        filtered_results = self._filter_results(results, selected_indices, options.max_results)
        if not filtered_results:
            raise PdfExportError("No search results available to export.")

        page_size = landscape(A4) if options.landscape_mode else A4
        doc = SimpleDocTemplate(
            str(output_path),
            pagesize=page_size,
            rightMargin=8 * mm,
            leftMargin=8 * mm,
            topMargin=22 * mm,
            bottomMargin=20 * mm,
            title=options.title,
            author=options.generated_by,
        )

        story = []
        story.append(self._build_header(options))

        story.append(Spacer(1, 12 * mm))
        if options.include_search_image and query_image_path:
            story.extend(
                self._build_search_image_block(query_image_path)
            )
        story.append(PageBreak())

        story.append(Spacer(1, 4 * mm))

        if options.include_search_image and query_image_path:
            story.extend(self._build_search_image_block(query_image_path))
            story.append(Spacer(1, 6 * mm))

        story.extend(self._build_summary_block(filtered_results))
        story.append(Spacer(1, 6 * mm))

        story.extend(self._build_results_block(filtered_results, options))
        story.append(PageBreak())

        story.extend(
            self._build_thank_you_page(
                options
            )
        )

        doc.build(
            story,
            onFirstPage=lambda canvas, doc_obj: self._decorate_page(canvas, doc_obj, options, 1),
            onLaterPages=lambda canvas, doc_obj: self._decorate_page(canvas, doc_obj, options, 2),
        )

        return str(output_path)

    def _filter_results(
        self,
        results: Sequence[SearchResult],
        selected_indices: Optional[Iterable[int]],
        max_results: int,
    ) -> list[SearchResult]:
        if selected_indices is not None:
            index_set = set(int(i) for i in selected_indices)
            filtered = [r for idx, r in enumerate(results) if idx in index_set]
        else:
            filtered = list(results)
        return filtered[:max_results]

    def _build_header(self, options: PdfExportOptions):
        elements = []
        header_data = []

        # ---------------- LOGO ----------------

        if (
            options.logo_path
            and Path(options.logo_path).exists()
        ):
            try:
                logo = self._make_rl_image(
                    options.logo_path,
                    width=18 * mm,
                    height=18 * mm,
                )

                company = Paragraph(
                    f"""
                    <font size="18">
                    <b>{self._escape(options.company_name)}</b>
                    </font>
                    """,
                    self.styles["BodyText"],
                )

                header = Table(
                    [
                        [logo, company]
                    ],
                    colWidths=[22 * mm, 150 * mm],
                )

                header.setStyle(
                    TableStyle(
                        [
                            ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
                            ("LEFTPADDING", (0, 0), (-1, -1), 0),
                            ("RIGHTPADDING", (0, 0), (-1, -1), 0),
                            ("BOTTOMPADDING", (0, 0), (-1, -1), 8),
                        ]
                    )
                )

                elements.append(header)

            except Exception:
                pass

        if not header_data:

            header_data.append(
                [
                    Paragraph(
                        f"""
                        <font size="22">
                        <b>{self._escape(options.company_name)}</b>
                        </font>
                        """,
                        self.styles["TitleCenter"],
                    )
                ]
            )

        table = Table(
            header_data,
            colWidths=[45 * mm, 135 * mm],
        )

        table.setStyle(
            TableStyle(
                [
                    ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
                    ("BOTTOMPADDING", (0, 0), (-1, -1), 10),
                ]
            )
        )

        elements.append(table)

        # ---------------- TITLE ----------------

        elements.append(
            Spacer(1, 20 * mm)
        )

        elements.append(
            Paragraph(
                """
                <font size="28">
                <b>TILE COLLECTION</b>
                </font>
                """,
                self.styles["TitleCenter"],
            )
        )

        elements.append(
            Paragraph(
                """
                <font size="14" color="#777777">
                Visual Search Catalogue
                </font>
                """,
                self.styles["TitleCenter"],
            )
        )

        elements.append(Spacer(1, 8 * mm))

        # ---------------- COMPANY INFO ----------------

        contact = Table(
            [[
                Paragraph(
                    f"""
                    <b>Phone</b> : {options.company_phone}<br/>
                    <b>Email</b> : {options.company_email}<br/>
                    <b>Website</b> : {options.company_website}<br/>
                    <b>Address</b> : {options.company_address}
                    """,
                    self.styles["BodyText"],
                )
            ]],
            colWidths=[165 * mm],
        )

        contact.setStyle(
            TableStyle(
                [
                    ("BOX", (0, 0), (-1, -1), 0.5, colors.HexColor("#DDDDDD")),
                    ("BACKGROUND", (0, 0), (-1, -1), colors.HexColor("#F8F8F8")),
                    ("TOPPADDING", (0, 0), (-1, -1), 10),
                    ("BOTTOMPADDING", (0, 0), (-1, -1), 10),
                    ("LEFTPADDING", (0, 0), (-1, -1), 12),
                ]
            )
        )

        elements.append(Spacer(1, 20 * mm))
        elements.append(contact)

        elements.append(Spacer(1, 8 * mm))

        elements.append(
            Paragraph(
                f"""
                Generated on
                <b>{datetime.now().strftime('%d %B %Y %I:%M %p')}</b>
                """,
                self.styles["SmallCenter"],
            )
        )

        return KeepTogether(elements)



    def _build_search_image_block(self, query_image_path: str):
        img = self._make_rl_image(
            query_image_path,
            width=150 * mm,
            height=150 * mm,
        )

        table = Table(
            [
                [
                    Paragraph(
                        "<b>Customer Search Image</b>",
                        self.styles["Meta"],
                    )
                ],
                [img],
            ],
            colWidths=[170 * mm],
        )

        table.setStyle(
            TableStyle(
                [
                    ("BOX", (0, 0), (-1, -1), 1, colors.grey),
                    ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#F4F4F4")),
                    ("ALIGN", (0, 0), (-1, -1), "CENTER"),
                    ("TOPPADDING", (0, 0), (-1, -1), 10),
                    ("BOTTOMPADDING", (0, -1), (-1, -1), 10),
                ]
            )
        )

        return [table]

    def _build_summary_block(self, results: Sequence[SearchResult]):
        best = results[0]
        tile = getattr(best, "tile", None)

        rows = [
            [Paragraph("<b>Best Match</b>", self.styles["Meta"])],
            [Paragraph(f"<b>Similarity:</b> {best.similarity_score:.1f}%", self.styles["BodyText"])],
        ]

        if tile is not None:
            rows.extend(
                [
                    [Paragraph(f"<b>Product Code:</b> {self._escape(getattr(tile, 'product_code', None) or '—')}", self.styles["BodyText"])],
                    [Paragraph(f"<b>Brand:</b> {self._escape(getattr(tile, 'brand', None) or '—')}", self.styles["BodyText"])],
                    [Paragraph(f"<b>Category:</b> {self._escape(getattr(tile, 'category', None) or '—')}", self.styles["BodyText"])],
                ]
            )

        table = Table(rows, colWidths=[170 * mm])
        table.setStyle(
            TableStyle(
                [
                    ("BOX", (0, 0), (-1, -1), 0.8, colors.HexColor("#888888")),
                    ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#e9eef7")),
                    ("TOPPADDING", (0, 0), (-1, -1), 6),
                    ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
                    ("LEFTPADDING", (0, 0), (-1, -1), 8),
                    ("RIGHTPADDING", (0, 0), (-1, -1), 8),
                ]
            )
        )
        return [table]


    def _build_results_block(
        self,
        results: Sequence[SearchResult],
        options: PdfExportOptions,
    ):
        story = []

        story.append(
            Paragraph(
                "<font size='18'><b>Matching Tiles</b></font>",
                self.styles["Heading2"],
            )
        )

        story.append(Spacer(1, 6 * mm))

        cards = []

        for index, result in enumerate(results, start=1):

            tile = result.tile

            image_path = self._result_image_path(result)

            if image_path:
                image = self._make_rl_image(
                    image_path,
                    width=82 * mm,
                    height=82 * mm,
                )
            else:
                image = Paragraph(
                    "No Image",
                    self.styles["BodyText"],
                )

            info = [
                f"<b>{tile.product_code or 'N/A'}</b>",
                "<br/>",
                f"Brand : {tile.brand or '-'}",
                "<br/>",
                f"Category : {tile.category or '-'}",
            ]

            if getattr(tile, "size", None):
                info.append(f"<br/>Size : {tile.size}")

            if getattr(tile, "finish", None):
                info.append(f"<br/>Finish : {tile.finish}")

            if getattr(tile, "color", None):
                info.append(f"<br/>Color : {tile.color}")

            info.append(
                f"<br/><br/><font color='green'><b>{result.similarity_score:.2f}% Match</b></font>"
            )

            card = Table(
                [
                    [image],
                    [Paragraph("".join(info), self.styles["BodyText"])],
                ],
                colWidths=[85 * mm],
            )

            card.setStyle(
                TableStyle(
                    [
                        ("BOX", (0, 0), (-1, -1), 0.7, colors.HexColor("#D5D5D5")),
                        ("BACKGROUND", (0, 0), (-1, 0), colors.white),
                        ("BACKGROUND", (0, 1), (-1, 1), colors.HexColor("#FAFAFA")),
                        ("ALIGN", (0, 0), (-1, -1), "CENTER"),
                        ("VALIGN", (0, 0), (-1, -1), "TOP"),
                        ("BOTTOMPADDING", (0, 0), (-1, -1), 8),
                        ("TOPPADDING", (0, 0), (-1, -1), 8),
                    ]
                )
            )

            cards.append(card)

        rows = []

        for i in range(0, len(cards), 2):

            left = cards[i]

            right = ""

            if i + 1 < len(cards):
                right = cards[i + 1]

            rows.append([left, right])

        table = Table(
            rows,
            colWidths=[90 * mm, 90 * mm],
            rowHeights=None,
        )

        table.setStyle(
            TableStyle(
                [
                    ("VALIGN", (0, 0), (-1, -1), "TOP"),
                    ("BOTTOMPADDING", (0, 0), (-1, -1), 10),
                ]
            )
        )

        story.append(table)

        return story


    def _build_thank_you_page(
        self,
        options: PdfExportOptions,
    ):

        story = []

        story.append(
            Spacer(1, 30 * mm)
        )

        story.append(
            Paragraph(
                """
                <font size="30">
                <b>Thank You</b>
                </font>
                """,
                self.styles["TitleCenter"],
            )
        )

        story.append(
            Spacer(1, 12 * mm)
        )

        story.append(
            Paragraph(
                """
                Thank you for choosing our products.
                We look forward to serving you again.
                """,
                self.styles["BodyText"],
            )
        )

        story.append(
            Spacer(1, 12 * mm)
        )

        if options.logo_path and Path(options.logo_path).exists():

            story.append(
                self._make_rl_image(
                    options.logo_path,
                    width=45 * mm,
                    height=45 * mm,
                )
            )

            story.append(
                Spacer(1, 8 * mm)
            )

        info = []

        info.append(
            f"<b>{self._escape(options.company_name)}</b>"
        )

        if options.company_phone:
            info.append(
                f"Phone : {self._escape(options.company_phone)}"
            )

        if options.company_email:
            info.append(
                f"Email : {self._escape(options.company_email)}"
            )

        if options.company_website:
            info.append(
                f"Website : {self._escape(options.company_website)}"
            )

        if options.company_address:
            info.append(
                f"Address : {self._escape(options.company_address)}"
            )

        story.append(
            Paragraph(
                "<br/><br/>".join(info),
                self.styles["TitleCenter"],
            )
        )

        return story

    def _decorate_page(self, canvas, doc, options: PdfExportOptions, page_no: int):
        canvas.saveState()

        width, height = doc.pagesize

        primary = colors.HexColor("#1E3A5F")
        light = colors.HexColor("#F4F6F9")
        text = colors.HexColor("#555555")

        # ---------------- HEADER ----------------

        canvas.setFillColor(primary)
        canvas.rect(
            0,
            height - 18 * mm,
            width,
            18 * mm,
            stroke=0,
            fill=1,
        )

        canvas.setFillColor(colors.white)
        canvas.setFont("Helvetica-Bold", 16)

        canvas.drawString(
            doc.leftMargin,
            height - 12 * mm,
            options.company_name,
        )

        # ---------------- FOOTER ----------------

        canvas.setFillColor(light)

        canvas.rect(
            0,
            0,
            width,
            15 * mm,
            stroke=0,
            fill=1,
        )

        canvas.setFont("Helvetica", 8)
        canvas.setFillColor(text)

        x = doc.leftMargin

        if options.company_phone:
            canvas.drawString(x, 9 * mm, options.company_phone)
            x += 55 * mm

        if options.company_email:
            canvas.drawString(x, 9 * mm, options.company_email)
            x += 70 * mm

        if options.company_website:
            canvas.drawString(x, 9 * mm, options.company_website)

        canvas.drawRightString(
            width - doc.rightMargin,
            4 * mm,
            f"Page {page_no}",
        )

        # ---------------- WATERMARK ----------------

        if options.watermark_text:

            canvas.saveState()

            canvas.setFillColor(colors.HexColor("#EEEEEE"))

            canvas.setFont(
                "Helvetica-Bold",
                60,
            )

            canvas.translate(
                width / 2,
                height / 2,
            )

            canvas.rotate(45)

            canvas.drawCentredString(
                0,
                0,
                options.watermark_text,
            )

            canvas.restoreState()

        canvas.restoreState()

    def _result_image_path(self, result):
        tile = getattr(result, "tile", None)

        if tile and tile.file_path:
            return tile.file_path

        return None


    def _make_rl_image(
        self,
        image_path: str,
        width: float,
        height: float,
    ):
        """
        Use original image.
        Keep aspect ratio.
        No resizing.
        """

        img = Image.open(image_path)

        img_w, img_h = img.size

        ratio = min(width / img_w, height / img_h)

        draw_w = img_w * ratio
        draw_h = img_h * ratio

        image = RLImage(image_path)

        image.drawWidth = draw_w
        image.drawHeight = draw_h

        return image


    @staticmethod
    def _escape(value: Optional[str]) -> str:
        if value is None:
            return "—"
        return (
            str(value)
            .replace("&", "&amp;")
            .replace("<", "&lt;")
            .replace(">", "&gt;")
        )