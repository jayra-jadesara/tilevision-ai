"""Background worker for Export Catalogue PDF generation."""

from __future__ import annotations

import logging
import threading
from pathlib import Path
from typing import Optional, Sequence

from PySide6.QtCore import QThread, Signal

from src.core.models import SearchResult
from src.services.pdf_export_service import PdfExportOptions, PDFExportService, PdfExportError

logger = logging.getLogger("tilevision.presentation.workers.export_catalogue")


class ExportCatalogueWorker(QThread):
    """
    Run ``PDFExportService.export_catalogue`` off the UI thread.

    Signals:
        progress(done, total, label)
        export_finished(output_path)
        export_failed(message)
    """

    progress = Signal(int, int, str)
    export_finished = Signal(str)
    export_failed = Signal(str)

    def __init__(
        self,
        *,
        output_file: str,
        query_image_path: Optional[str],
        results: Sequence[SearchResult],
        options: PdfExportOptions,
        parent=None,
    ) -> None:
        super().__init__(parent)
        self._output_file = output_file
        self._query_image_path = query_image_path
        self._results = list(results)
        self._options = options
        self._cancel_event = threading.Event()

    def cancel(self) -> None:
        logger.info("Export catalogue cancel requested")
        self._cancel_event.set()
        try:
            self.requestInterruption()
        except Exception:
            pass

    def run(self) -> None:
        service = PDFExportService()

        def _progress(done: int, total: int, label: str) -> None:
            try:
                self.progress.emit(int(done), int(total), str(label))
            except RuntimeError:
                pass

        try:
            created = service.export_catalogue(
                output_file=self._output_file,
                query_image_path=self._query_image_path,
                results=self._results,
                options=self._options,
                selected_indices=None,
                on_progress=_progress,
                cancel_event=self._cancel_event,
            )
            if self._cancel_event.is_set():
                self.export_failed.emit("Export cancelled.")
                return
            self.export_finished.emit(created)
        except PdfExportError as exc:
            logger.warning("Export catalogue failed: %s", exc)
            self.export_failed.emit(str(exc))
        except Exception as exc:
            logger.exception("Export catalogue crashed")
            self.export_failed.emit(str(exc))
