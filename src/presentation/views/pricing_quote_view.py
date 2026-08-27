"""In-app Pricing Quote PDF viewer (never opens a browser)."""

from __future__ import annotations

from pathlib import Path

from PySide6.QtCore import Qt, QSize, QTimer
from PySide6.QtGui import QPixmap, QImage
from PySide6.QtWidgets import (
    QDialog,
    QVBoxLayout,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QScrollArea,
    QWidget,
    QFrame,
)

import logging

from src.presentation.dialogs import message_box
from src.services.pricing_quote_service import (
    PricingQuoteError,
    create_pricing_quote_pdf,
)

logger = logging.getLogger("tilevision.presentation.views.pricing_quote_view")

try:
    from PySide6.QtPdf import QPdfDocument

    _HAS_QTPDF = True
except ImportError:  # pragma: no cover
    QPdfDocument = None  # type: ignore[misc, assignment]
    _HAS_QTPDF = False


class PricingQuoteView(QDialog):
    """Modal dialog that generates and displays the Pricing Quote PDF inside the app."""

    def __init__(self, parent=None, theme: str = "dark"):
        super().__init__(parent)
        self._theme = theme if theme in ("dark", "light") else "dark"
        self._pdf_path: Path | None = None
        self.setWindowTitle("Pricing Quote")
        self.setMinimumSize(720, 860)
        self.resize(780, 920)
        self.setModal(True)
        self._build_ui()
        self._apply_theme()
        QTimer.singleShot(0, self._load_pdf)

    def _build_ui(self) -> None:
        root = QVBoxLayout(self)
        root.setContentsMargins(16, 16, 16, 16)
        root.setSpacing(12)

        header = QHBoxLayout()
        title = QLabel("Pricing Quote")
        title.setObjectName("pricingTitle")
        header.addWidget(title)
        header.addStretch()

        self.refresh_btn = QPushButton("Refresh prices")
        self.refresh_btn.setCursor(Qt.PointingHandCursor)
        self.refresh_btn.clicked.connect(self._load_pdf)
        header.addWidget(self.refresh_btn)
        root.addLayout(header)

        self.status_label = QLabel("Loading latest prices…")
        self.status_label.setObjectName("pricingStatus")
        self.status_label.setWordWrap(True)
        root.addWidget(self.status_label)

        self.scroll = QScrollArea()
        self.scroll.setWidgetResizable(True)
        self.scroll.setFrameShape(QFrame.NoFrame)
        self.scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        self.scroll.setVerticalScrollBarPolicy(Qt.ScrollBarAsNeeded)

        self.pages_host = QWidget()
        self.pages_layout = QVBoxLayout(self.pages_host)
        self.pages_layout.setContentsMargins(8, 8, 8, 8)
        self.pages_layout.setSpacing(16)
        self.pages_layout.setAlignment(Qt.AlignHCenter | Qt.AlignTop)
        self.scroll.setWidget(self.pages_host)
        root.addWidget(self.scroll, 1)

        footer = QHBoxLayout()
        footer.addStretch()
        close_btn = QPushButton("Close")
        close_btn.setCursor(Qt.PointingHandCursor)
        close_btn.clicked.connect(self.accept)
        footer.addWidget(close_btn)
        root.addLayout(footer)

    def _apply_theme(self) -> None:
        if self._theme == "light":
            self.setStyleSheet(
                """
                QDialog { background: #f4f7fb; color: #0f172a; }
                QLabel#pricingTitle { font-size: 20px; font-weight: 700; color: #0b1f3a; }
                QLabel#pricingStatus { color: #475569; font-size: 12px; }
                QScrollArea { background: #e8eef6; border: 1px solid #cbd5e1; border-radius: 10px; }
                QPushButton {
                    background: #0b1f3a; color: white; border: none;
                    border-radius: 8px; padding: 8px 14px; font-weight: 600;
                }
                QPushButton:hover { background: #16325a; }
                QPushButton:disabled { background: #94a3b8; }
                """
            )
        else:
            self.setStyleSheet(
                """
                QDialog { background: #0b1220; color: #e2e8f0; }
                QLabel#pricingTitle { font-size: 20px; font-weight: 700; color: #f8fafc; }
                QLabel#pricingStatus { color: #94a3b8; font-size: 12px; }
                QScrollArea { background: #111827; border: 1px solid #334155; border-radius: 10px; }
                QPushButton {
                    background: #38bdf8; color: #0b1f3a; border: none;
                    border-radius: 8px; padding: 8px 14px; font-weight: 600;
                }
                QPushButton:hover { background: #7dd3fc; }
                QPushButton:disabled { background: #475569; color: #cbd5e1; }
                """
            )

    def _clear_pages(self) -> None:
        while self.pages_layout.count():
            item = self.pages_layout.takeAt(0)
            widget = item.widget()
            if widget is not None:
                widget.deleteLater()

    def _load_pdf(self) -> None:
        self.refresh_btn.setEnabled(False)
        self.status_label.setText("Fetching latest prices and building PDF…")
        self._clear_pages()
        try:
            result = create_pricing_quote_pdf()
            self._pdf_path = result.pdf_path
            source_label = {
                "remote": "online prices.json",
                "cache": "cached prices.json",
                "bundled": "bundled offline prices.json",
            }.get(result.source, result.source)
            self.status_label.setText(
                f"Showing in-app PDF · prices from {source_label}"
            )
            if not self._render_pdf(result.pdf_path):
                raise RuntimeError(
                    "Could not render PDF pages inside the app "
                    "(QtPdf / PyMuPDF unavailable)."
                )
        except PricingQuoteError as exc:
            logger.warning("Pricing quote unavailable: %s", exc)
            self.status_label.setText(f"Could not load pricing quote: {exc}")
            message_box.warning(
                self,
                "Pricing Quote",
                "Could not load the pricing quote right now.\n\n"
                f"{exc}\n\n"
                "Check your internet connection and try again.",
            )
        except Exception as exc:
            logger.exception("Failed to load pricing quote PDF in-app")
            self.status_label.setText(f"Could not load pricing quote: {exc}")
            message_box.warning(
                self,
                "Pricing Quote",
                f"Could not load the pricing quote PDF inside the app.\n\n{exc}",
            )
        finally:
            self.refresh_btn.setEnabled(True)

    def _render_pdf(self, path: Path) -> bool:
        """Render PDF pages as images inside the dialog. Never opens a browser."""
        if _HAS_QTPDF and QPdfDocument is not None:
            return self._render_with_qtpdf(path)
        return self._render_with_pymupdf(path)

    def _render_with_qtpdf(self, path: Path) -> bool:
        assert QPdfDocument is not None
        doc = QPdfDocument(self)
        err = doc.load(str(path))
        if err != QPdfDocument.Error.None_:
            logger.warning("QPdfDocument.load failed: %s", err)
            return self._render_with_pymupdf(path)

        page_count = doc.pageCount()
        if page_count <= 0:
            return False

        target_width = max(640, self.scroll.viewport().width() - 40)
        for i in range(page_count):
            point_size = doc.pagePointSize(i)
            if point_size.width() <= 0:
                continue
            scale = target_width / float(point_size.width())
            img_size = QSize(
                int(point_size.width() * scale),
                int(point_size.height() * scale),
            )
            image: QImage = doc.render(i, img_size)
            if image.isNull():
                continue
            self._add_page_pixmap(QPixmap.fromImage(image))
        doc.deleteLater()
        return self.pages_layout.count() > 0

    def _render_with_pymupdf(self, path: Path) -> bool:
        try:
            import fitz  # PyMuPDF
        except ImportError:
            return False
        try:
            doc = fitz.open(str(path))
            target_width = max(640, self.scroll.viewport().width() - 40)
            for page in doc:
                rect = page.rect
                scale = target_width / float(rect.width) if rect.width else 2.0
                pix = page.get_pixmap(matrix=fitz.Matrix(scale, scale), alpha=False)
                image = QImage(
                    pix.samples,
                    pix.width,
                    pix.height,
                    pix.stride,
                    QImage.Format_RGB888,
                ).copy()
                self._add_page_pixmap(QPixmap.fromImage(image))
            doc.close()
            return self.pages_layout.count() > 0
        except Exception:
            logger.exception("PyMuPDF PDF render failed")
            return False

    def _add_page_pixmap(self, pixmap: QPixmap) -> None:
        page = QLabel()
        page.setAlignment(Qt.AlignCenter)
        page.setPixmap(pixmap)
        page.setStyleSheet(
            "QLabel { background: white; border: 1px solid #94a3b8; "
            "border-radius: 4px; }"
        )
        self.pages_layout.addWidget(page)
