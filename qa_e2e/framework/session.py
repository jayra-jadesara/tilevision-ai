"""Live application session handles for E2E scenarios."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

from PySide6.QtWidgets import QApplication

from qa_e2e.framework.collectors import ArtifactCollector
from qa_e2e.framework.human import HumanSimulator
from qa_e2e.framework.log_capture import LogCapture


@dataclass
class AppSession:
    """Everything a customer journey needs after launch."""

    app: QApplication
    main_window: Any
    settings: Any
    home_dir: Path
    data_dir: Path
    catalog_dir: Path
    query_dir: Path
    artifacts: ArtifactCollector
    logs: LogCapture
    human: HumanSimulator
    # production objects (real stack)
    search_viewmodel: Any
    indexing_viewmodel: Any
    search_use_case: Any
    index_use_case: Any
    vector_index: Any
    image_repository: Any
    feature_extractor: Any
    db_context: Any
    embedder: Any
    folder_monitor: Any = None
    expected_manifest: dict = field(default_factory=dict)
    _closed: bool = False

    @property
    def search_view(self):
        from src.presentation.views.search_view import SearchView

        view = self.main_window.findChild(SearchView, "SearchView")
        if view is None:
            view = getattr(self.main_window, "_search_view", None)
        if view is None:
            raise AssertionError("SearchView not found on MainWindow")
        return view

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        try:
            if self.folder_monitor is not None:
                self.folder_monitor.stop_monitoring()
        except Exception:
            pass
        try:
            self.main_window.close()
        except Exception:
            pass
        try:
            self.logs.detach()
        except Exception:
            pass
        try:
            self.db_context.close_all()
        except Exception:
            pass
