"""In-app Pricing Quote page (stack page — never opens a browser)."""

from __future__ import annotations

import logging
import shutil
from pathlib import Path

from PySide6.QtCore import Qt, QSize, QThread, Signal
from PySide6.QtGui import QPixmap, QImage
from PySide6.QtWidgets import (
    QVBoxLayout,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QScrollArea,
    QWidget,
    QFrame,
    QProgressBar,
    QFileDialog,
)

from src.presentation.dialogs import message_box
from src.services.pricing_quote_service import (
    PricingQuoteError,
    PricingQuoteResult,
    create_pricing_quote_pdf,
)

logger = logging.getLogger("tilevision.presentation.views.pricing_quote_view")

# On-screen A4 preview width (points ≈ px at 96dpi would be ~794; keep a
# readable centered sheet with left/right gutters instead of full stretch).
_A4_PREVIEW_MAX_WIDTH = 680
_A4_SIDE_GUTTER = 48

try:
    from PySide6.QtPdf import QPdfDocument

    _HAS_QTPDF = True
except ImportError:  # pragma: no cover
    QPdfDocument = None  # type: ignore[misc, assignment]
    _HAS_QTPDF = False


class _PricingLoadWorker(QThread):
    """Fetch prices.json + build PDF off the UI thread so the loader can animate."""

    finished_ok = Signal(object)
    finished_err = Signal(str, bool)  # message, is_pricing_quote_error

    def run(self) -> None:
        try:
            result = create_pricing_quote_pdf()
            self.finished_ok.emit(result)
        except PricingQuoteError as exc:
            self.finished_err.emit(str(exc), True)
        except Exception as exc:  # pragma: no cover - surfaced to UI
            logger.exception("Pricing quote worker failed")
            self.finished_err.emit(str(exc), False)


class PricingQuoteView(QWidget):
    """Content-stack page that generates and displays the Pricing Quote PDF."""

    def __init__(self, parent=None, theme: str = "dark"):
        super().__init__(parent)
        self._theme = theme if theme in ("dark", "light") else "dark"
        self._pdf_path: Path | None = None
        self._worker: _PricingLoadWorker | None = None
        self._loaded_once = False
        self.setObjectName("PricingQuoteView")
        self._build_ui()
        self._apply_theme()

    def set_theme(self, theme: str) -> None:
        self._theme = theme if theme in ("dark", "light") else "dark"
        self._apply_theme()

    def showEvent(self, event) -> None:  # noqa: N802
        super().showEvent(event)
        if not self._loaded_once:
            self._loaded_once = True
            self.refresh_prices()

    def refresh_prices(self) -> None:
        """Public entry used by Refresh button and first show."""
        self._load_pdf()

    def _build_ui(self) -> None:
        root = QVBoxLayout(self)
        root.setContentsMargins(24, 20, 24, 16)
        root.setSpacing(12)

        header = QHBoxLayout()
        title = QLabel("Pricing Quote")
        title.setObjectName("pricingTitle")
        header.addWidget(title)
        header.addStretch()

        self.download_btn = QPushButton("Download PDF")
        self.download_btn.setCursor(Qt.PointingHandCursor)
        self.download_btn.setEnabled(False)
        self.download_btn.setToolTip("Save the pricing quote PDF to your computer")
        self.download_btn.clicked.connect(self._download_pdf)
        header.addWidget(self.download_btn)

        self.refresh_btn = QPushButton("Refresh prices")
        self.refresh_btn.setCursor(Qt.PointingHandCursor)
        self.refresh_btn.clicked.connect(self._load_pdf)
        header.addWidget(self.refresh_btn)
        root.addLayout(header)

        self.status_label = QLabel("")
        self.status_label.setObjectName("pricingStatus")
        self.status_label.setWordWrap(True)
        self.status_label.setVisible(False)
        root.addWidget(self.status_label)

        self._progress_bar = QProgressBar()
        self._progress_bar.setObjectName("PricingProgressBar")
        self._progress_bar.setRange(0, 0)  # indeterminate — matches Search/Duplicates
        self._progress_bar.setTextVisible(False)
        self._progress_bar.setFixedHeight(4)
        self._progress_bar.setVisible(False)
        root.addWidget(self._progress_bar)

        self.scroll = QScrollArea()
        self.scroll.setWidgetResizable(True)
        self.scroll.setFrameShape(QFrame.NoFrame)
        self.scroll.setAlignment(Qt.AlignHCenter | Qt.AlignTop)
        self.scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        self.scroll.setVerticalScrollBarPolicy(Qt.ScrollBarAsNeeded)

        self.pages_host = QWidget()
        self.pages_layout = QVBoxLayout(self.pages_host)
        self.pages_layout.setContentsMargins(
            _A4_SIDE_GUTTER, 24, _A4_SIDE_GUTTER, 24
        )
        self.pages_layout.setSpacing(16)
        self.pages_layout.setAlignment(Qt.AlignHCenter | Qt.AlignTop)
        self.scroll.setWidget(self.pages_host)
        root.addWidget(self.scroll, 1)

    def _apply_theme(self) -> None:
        if self._theme == "light":
            self.setStyleSheet(
                """
                #PricingQuoteView { background: #f4f7fb; color: #0f172a; }
                QLabel#pricingTitle { font-size: 20px; font-weight: 700; color: #0b1f3a; }
                QLabel#pricingStatus { color: #475569; font-size: 12px; }
                QScrollArea { background: #e8eef6; border: 1px solid #cbd5e1; border-radius: 10px; }
                QProgressBar#PricingProgressBar {
                    background: #e2e8f0; border: none; border-radius: 2px;
                }
                QProgressBar#PricingProgressBar::chunk { background: #0284c7; }
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
                #PricingQuoteView { background: #0b1220; color: #e2e8f0; }
                QLabel#pricingTitle { font-size: 20px; font-weight: 700; color: #f8fafc; }
                QLabel#pricingStatus { color: #94a3b8; font-size: 12px; }
                QScrollArea { background: #111827; border: 1px solid #334155; border-radius: 10px; }
                QProgressBar#PricingProgressBar {
                    background: #1e293b; border: none; border-radius: 2px;
                }
                QProgressBar#PricingProgressBar::chunk { background: #38bdf8; }
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

    def _set_loading(self, loading: bool) -> None:
        self.refresh_btn.setEnabled(not loading)
        self.download_btn.setEnabled(not loading and self._pdf_path is not None)
        self._progress_bar.setVisible(loading)

    def _show_status(self, text: str) -> None:
        self.status_label.setText(text)
        self.status_label.setVisible(bool(text))

    def _clear_status(self) -> None:
        self.status_label.clear()
        self.status_label.setVisible(False)

    def _load_pdf(self) -> None:
        if self._worker is not None and self._worker.isRunning():
            return

        self._set_loading(True)
        self._show_status("Fetching latest prices…")
        self._clear_pages()
        self._pdf_path = None
        self.download_btn.setEnabled(False)

        worker = _PricingLoadWorker(self)
        worker.finished_ok.connect(self._on_load_ok)
        worker.finished_err.connect(self._on_load_err)
        worker.finished.connect(worker.deleteLater)
        self._worker = worker
        worker.start()

    def _on_load_ok(self, result: object) -> None:
        assert isinstance(result, PricingQuoteResult)
        self._pdf_path = result.pdf_path
        try:
            if not self._render_pdf(result.pdf_path):
                raise RuntimeError(
                    "Could not render PDF pages inside the app "
                    "(QtPdf / PyMuPDF unavailable)."
                )
            # No source/prices.json status line — keep the page clean.
            self._clear_status()
            logger.info(
                "Pricing quote PDF ready in-app (source=%s path=%s)",
                result.source,
                result.pdf_path,
            )
        except Exception as exc:
            logger.exception("Failed to render pricing quote PDF in-app")
            self._show_status(f"Could not load pricing quote: {exc}")
            message_box.warning(
                self,
                "Pricing Quote",
                f"Could not load the pricing quote PDF inside the app.\n\n{exc}",
            )
        finally:
            self._set_loading(False)

    def _on_load_err(self, message: str, is_pricing_error: bool) -> None:
        self._show_status(f"Could not load pricing quote: {message}")
        if is_pricing_error:
            message_box.warning(
                self,
                "Pricing Quote",
                "Could not load the pricing quote right now.\n\n"
                f"{message}\n\n"
                "Check your internet connection and try again.",
            )
        else:
            message_box.warning(
                self,
                "Pricing Quote",
                f"Could not load the pricing quote PDF inside the app.\n\n{message}",
            )
        self._set_loading(False)

    def _download_pdf(self) -> None:
        if self._pdf_path is None or not self._pdf_path.is_file():
            message_box.information(
                self,
                "Download PDF",
                "No pricing quote PDF is ready yet. Click Refresh prices first.",
            )
            return

        default_name = "TileVision_AI_Pricing_Quote.pdf"
        path, _ = QFileDialog.getSaveFileName(
            self,
            "Download Pricing Quote PDF",
            str(Path.home() / default_name),
            "PDF Files (*.pdf)",
        )
        if not path:
            return
        if not path.lower().endswith(".pdf"):
            path += ".pdf"
        try:
            shutil.copy2(self._pdf_path, path)
            self._show_status(f"PDF saved to {path}")
            logger.info("Pricing quote PDF downloaded to %s", path)
        except OSError as exc:
            logger.exception("Failed to save pricing quote PDF")
            message_box.warning(
                self,
                "Download PDF",
                f"Could not save the PDF:\n{exc}",
            )

    def _logical_target_width(self) -> int:
        """A4 sheet preview width — never stretch to full content width."""
        viewport_w = self.scroll.viewport().width()
        available = viewport_w - (2 * _A4_SIDE_GUTTER)
        if available <= 0:
            available = _A4_PREVIEW_MAX_WIDTH
        return max(420, min(_A4_PREVIEW_MAX_WIDTH, available))

    def _device_pixel_ratio(self) -> float:
        return max(1.0, float(self.devicePixelRatioF() or 1.0))

    def _render_pdf(self, path: Path) -> bool:
        """Render PDF pages as crisp HiDPI A4 previews. Never opens a browser."""
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

        dpr = self._device_pixel_ratio()
        logical_width = self._logical_target_width()
        # Extra sharpness: render at least 2x even on 1.0 DPR displays.
        render_scale = max(dpr, 2.0)
        pixel_width = int(logical_width * render_scale)

        for i in range(page_count):
            point_size = doc.pagePointSize(i)
            if point_size.width() <= 0:
                continue
            scale = pixel_width / float(point_size.width())
            img_size = QSize(
                int(point_size.width() * scale),
                int(point_size.height() * scale),
            )
            image: QImage = doc.render(i, img_size)
            if image.isNull():
                continue
            pixmap = QPixmap.fromImage(image)
            pixmap.setDevicePixelRatio(render_scale)
            self._add_page_pixmap(pixmap, logical_width)
        doc.deleteLater()
        return self.pages_layout.count() > 0

    def _render_with_pymupdf(self, path: Path) -> bool:
        try:
            import fitz  # PyMuPDF
        except ImportError:
            return False
        try:
            doc = fitz.open(str(path))
            dpr = self._device_pixel_ratio()
            logical_width = self._logical_target_width()
            render_scale = max(dpr, 2.0)
            pixel_width = int(logical_width * render_scale)
            for page in doc:
                rect = page.rect
                scale = pixel_width / float(rect.width) if rect.width else 2.0 * render_scale
                pix = page.get_pixmap(matrix=fitz.Matrix(scale, scale), alpha=False)
                image = QImage(
                    pix.samples,
                    pix.width,
                    pix.height,
                    pix.stride,
                    QImage.Format_RGB888,
                ).copy()
                pixmap = QPixmap.fromImage(image)
                pixmap.setDevicePixelRatio(render_scale)
                self._add_page_pixmap(pixmap, logical_width)
            doc.close()
            return self.pages_layout.count() > 0
        except Exception:
            logger.exception("PyMuPDF PDF render failed")
            return False

    def _add_page_pixmap(self, pixmap: QPixmap, logical_width: int) -> None:
        page = QLabel()
        page.setAlignment(Qt.AlignCenter)
        page.setPixmap(pixmap)
        # Lock to A4 preview width so the sheet does not stretch edge-to-edge.
        dpr = max(1.0, float(pixmap.devicePixelRatio() or 1.0))
        logical_height = int(round(pixmap.height() / dpr))
        page.setFixedSize(logical_width, logical_height)
        page.setStyleSheet(
            "QLabel { background: white; border: 1px solid #94a3b8; "
            "border-radius: 2px; }"
        )
        self.pages_layout.addWidget(page, 0, Qt.AlignHCenter)
