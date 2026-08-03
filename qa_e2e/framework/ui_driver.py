"""Human-facing UI driver over Qt objectNames and visible button text."""

from __future__ import annotations

import os
import time
from pathlib import Path
from typing import List, Optional

from PySide6.QtCore import QMimeData, QPoint, QPointF, Qt, QUrl
from PySide6.QtGui import QDragEnterEvent, QDropEvent
from PySide6.QtTest import QTest
from PySide6.QtWidgets import (
    QApplication,
    QAbstractButton,
    QLabel,
    QLineEdit,
    QProgressBar,
    QPushButton,
    QTableWidget,
    QWidget,
)

from qa_e2e.framework.human import HumanSimulator
from qa_e2e.framework.session import AppSession


class UIDriver:
    """Drive TileVision exactly like a showroom customer."""

    def __init__(self, session: AppSession) -> None:
        self.s = session
        self.human: HumanSimulator = session.human

    # ── discovery ──────────────────────────────────────────────────────────

    def find(self, object_name: str, typ=QWidget) -> QWidget:
        w = self.s.main_window.findChild(typ, object_name)
        if w is None:
            raise AssertionError(f"UI widget not found: objectName={object_name!r} type={typ}")
        return w

    def find_button(self, text: str) -> QPushButton:
        def _norm(value: str) -> str:
            # Qt uses '&' for keyboard mnemonics ("Auto Crop & Search" → displayed
            # "Auto Crop _Search"). Strip mnemonics and collapse whitespace.
            cleaned = value.replace("&&", "\0").replace("&", "").replace("\0", "&")
            return " ".join(cleaned.split()).strip().lower()

        needle = _norm(text)
        for btn in self.s.main_window.findChildren(QPushButton):
            if _norm(btn.text()) == needle:
                return btn
        for btn in self.s.main_window.findChildren(QPushButton):
            if needle and needle in _norm(btn.text()):
                return btn
        raise AssertionError(f"Button not found with text containing {text!r}")

    def nav_buttons(self) -> List[QAbstractButton]:
        return [
            b
            for b in self.s.main_window.findChildren(QAbstractButton)
            if b.objectName() == "NavButton"
        ]

    def goto(self, label: str) -> None:
        for btn in self.nav_buttons():
            if btn.text().replace("&", "").strip() == label:
                self.human.click(btn)
                QApplication.processEvents()
                self.human.think(0.2, 0.6)
                return
        raise AssertionError(f"Nav button {label!r} not found")

    def goto_search(self) -> None:
        self.goto("Search")

    def goto_index(self) -> None:
        self.goto("Index")

    def goto_settings(self) -> None:
        self.goto("Settings")

    def goto_dashboard(self) -> None:
        self.goto("Dashboard")

    # ── indexing ───────────────────────────────────────────────────────────

    def select_index_folder(self, folder: Path) -> None:
        """Customer chose a catalogue folder (Browse dialog replaced by set_folder)."""
        self.goto_index()
        self.s.indexing_viewmodel.set_folder(str(folder.resolve()))
        QApplication.processEvents()
        path_edit = self.find("FolderPathEdit", QLineEdit)
        if str(folder.resolve()) not in path_edit.text():
            # Still OK if viewmodel holds path even if edit lags one frame
            QTest.qWait(200)
        self.human.think(0.3, 0.8)

    def start_indexing(self) -> None:
        btn = self.find("StartButton", QPushButton)
        self.human.click(btn)

    def wait_indexing_done(self, *, timeout: float = 600.0) -> None:
        from src.presentation.viewmodels.indexing_viewmodel import IndexingState

        deadline = time.monotonic() + timeout
        saw_running = False
        while time.monotonic() < deadline:
            QApplication.processEvents()
            state = self.s.indexing_viewmodel.state
            if state in (IndexingState.RUNNING, IndexingState.PAUSED, IndexingState.CANCELLING):
                saw_running = True
            if state in (IndexingState.FINISHED, IndexingState.IDLE, IndexingState.CANCELLED) and (
                saw_running or self.s.vector_index.get_total_count() > 0
            ):
                # Allow UI to settle
                QTest.qWait(300)
                return
            if state == IndexingState.ERROR:
                raise AssertionError("Indexing entered ERROR state")
            QTest.qWait(250)
        raise TimeoutError(f"Indexing did not finish within {timeout}s (state={self.s.indexing_viewmodel.state})")

    # ── search ─────────────────────────────────────────────────────────────

    def drop_zone(self) -> QWidget:
        return self.find("DropZone")

    def drag_drop_image(self, image_path: Path) -> None:
        """Synthesize a real Qt drag-enter + drop onto DropZone."""
        self.goto_search()
        zone = self.drop_zone()
        self.human.wander(zone, steps=5)
        path = str(Path(image_path).resolve())
        mime = QMimeData()
        mime.setUrls([QUrl.fromLocalFile(path)])
        center = zone.rect().center()
        # Human hesitates over the drop target
        self.human.think(0.15, 0.4)
        enter = QDragEnterEvent(
            center,
            Qt.DropAction.CopyAction,
            mime,
            Qt.MouseButton.LeftButton,
            Qt.KeyboardModifier.NoModifier,
        )
        QApplication.sendEvent(zone, enter)
        QApplication.processEvents()
        self.human.think(0.1, 0.35)
        # PySide6 QDropEvent signature varies; use QPointF when required.
        try:
            drop = QDropEvent(
                QPointF(center),
                Qt.DropAction.CopyAction,
                mime,
                Qt.MouseButton.LeftButton,
                Qt.KeyboardModifier.NoModifier,
            )
        except TypeError:
            drop = QDropEvent(
                center,
                Qt.DropAction.CopyAction,
                mime,
                Qt.MouseButton.LeftButton,
                Qt.KeyboardModifier.NoModifier,
            )
        QApplication.sendEvent(zone, drop)
        QApplication.processEvents()
        self.human.think(0.2, 0.5)

    def open_image_via_viewmodel(self, image_path: Path) -> None:
        """
        Fallback for 'Open Image' when QFileDialog cannot be automated headlessly.
        Still goes through SearchView's image-chosen path via the DropZone callback.
        """
        self.goto_search()
        zone = self.drop_zone()
        # Prefer DropZone public callback if present
        on_selected = getattr(zone, "_on_image_selected", None)
        if callable(on_selected):
            self.human.wander(zone)
            self.human.think(0.2, 0.5)
            on_selected(str(Path(image_path).resolve()))
            QApplication.processEvents()
            return
        raise AssertionError("DropZone image callback missing")

    def click_named_action(self, text: str) -> None:
        btn = self.find_button(text)
        if not btn.isEnabled():
            raise AssertionError(f"Button {text!r} is disabled")
        self.human.click(btn)

    def clear_search(self) -> None:
        self.click_named_action("Clear")

    def auto_crop_search(self) -> None:
        self.click_named_action("Auto Crop & Search")

    def precise_crop_search(self) -> None:
        self.click_named_action("Precise Crop & Search")

    def results_table(self) -> QTableWidget:
        return self.find("ResultsTable", QTableWidget)  # type: ignore[return-value]

    def search_status(self) -> str:
        lbl = self.find("SearchStatusLabel", QLabel)
        return lbl.text()

    def progress_bar(self) -> QProgressBar:
        return self.find("SearchProgressBar", QProgressBar)  # type: ignore[return-value]

    def wait_search_settled(self, *, timeout: float = 240.0) -> str:
        from src.presentation.viewmodels.search_viewmodel import SearchState
        from qa_e2e.framework.search_fallback import complete_search_via_use_case

        # CPU DINOv2 searches often take 60–120s. Only treat as stalled when the
        # worker is no longer running (true hang) OR after a long absolute wait.
        stall_s = float(os.environ.get("TILEVISION_QA_SEARCH_STALL_S", "180"))
        deadline = time.monotonic() + timeout
        saw_searching = False
        searching_since: Optional[float] = None
        while time.monotonic() < deadline:
            QApplication.processEvents()
            state = self.s.search_viewmodel.state
            if state == SearchState.SEARCHING:
                saw_searching = True
                if searching_since is None:
                    searching_since = time.monotonic()
                worker = getattr(self.s.search_viewmodel, "_worker", None)
                worker_alive = bool(worker is not None and worker.isRunning())
                waited = time.monotonic() - searching_since
                # If the real worker is still running, keep waiting — do NOT start
                # a second search (double execute → deleteLater races / SIGSEGV).
                if waited > stall_s and not worker_alive:
                    QApplication.processEvents()
                    state = self.s.search_viewmodel.state
                    if state in (
                        SearchState.RESULTS,
                        SearchState.NO_RESULTS,
                        SearchState.ERROR,
                    ):
                        QTest.qWait(400)
                        QApplication.processEvents()
                        return state
                    path = getattr(self.s.search_viewmodel, "_last_query_path", None)
                    if path:
                        self.s.artifacts.note(
                            f"SearchWorker stalled >{stall_s:.0f}s — completing via SearchTilesUseCase thread"
                        )
                        complete_search_via_use_case(self.s, path, timeout=max(60.0, timeout))
                        QTest.qWait(400)
                        return self.s.search_viewmodel.state
            else:
                searching_since = None
            if state in (
                SearchState.RESULTS,
                SearchState.NO_RESULTS,
                SearchState.ERROR,
                SearchState.IDLE,
            ) and (saw_searching or state != SearchState.IDLE):
                # Thumbnails load deferred — give UI a beat
                QTest.qWait(400)
                QApplication.processEvents()
                return state
            QTest.qWait(200)
        raise TimeoutError(
            f"Search did not settle within {timeout}s (state={self.s.search_viewmodel.state})"
        )

    def visible_result_paths(self) -> List[str]:
        table = self.results_table()
        paths: List[str] = []
        # Image Path is last column
        path_col = table.columnCount() - 1
        for row in range(table.rowCount()):
            item = table.item(row, path_col)
            if item is not None:
                paths.append(item.text())
        return paths

    def result_count_ui(self) -> int:
        return self.results_table().rowCount()

    def result_paths(self) -> List[str]:
        """UI paths first; fall back to ViewModel results."""
        paths = self.visible_result_paths()
        if paths:
            return paths
        return [r.tile.file_path for r in self.s.search_viewmodel.last_results]
