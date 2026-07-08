"""
Indexing View for TileVision AI.

Provides a fully implemented PySide6 QWidget panel for the Folder Indexing feature.
This view is purely presentation: it delegates all logic to IndexingViewModel.

UI Sections:
    1. Folder Selector — browse and display the chosen folder path.
    2. Progress Panel — animated progress bar, file counter, ETA display.
    3. Action Buttons — Start, Pause/Resume, Cancel.
    4. Log/Status Panel — scrollable live status messages.
    5. Summary Panel — final indexed/skipped count after completion.

Design Decision:
    Uses QSS (Qt Style Sheets) for dark-themed styling consistent with the app theme.
    The view is completely decoupled from business logic via the ViewModel pattern.
    Every widget is connected to ViewModel signals/slots, never to use-case classes directly.
"""

import logging
from typing import Optional

from PySide6.QtCore import Qt, Slot, QDateTime
from PySide6.QtGui import QFont, QIcon, QColor
from PySide6.QtWidgets import (
    QWidget,
    QVBoxLayout,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QPushButton,
    QProgressBar,
    QTextEdit,
    QGroupBox,
    QFileDialog,
    QFrame,
    QSizePolicy,
    QGridLayout,
    QScrollArea,
)

from src.presentation.viewmodels.indexing_viewmodel import IndexingViewModel, IndexingState

logger = logging.getLogger("tilevision.presentation.views.indexing_view")


class IndexingView(QWidget):
    """
    Full-featured Folder Indexing panel widget.

    Connects to an IndexingViewModel instance to drive all state and logic.
    """

    def __init__(self, viewmodel: IndexingViewModel, parent: Optional[QWidget] = None) -> None:
        """
        Initialize the IndexingView.

        Args:
            viewmodel: The bound IndexingViewModel instance.
            parent: Optional Qt parent widget.
        """
        super().__init__(parent)
        self._viewmodel = viewmodel
        self._setup_ui()
        self._connect_signals()
        self._apply_styles()
        logger.debug("IndexingView initialized.")

    # ── UI Construction ───────────────────────────────────────────────────────

    def _setup_ui(self) -> None:
        """Build and arrange all UI widgets in this view."""
        self.setObjectName("IndexingView")

        # ── Root layout
        root_layout = QVBoxLayout(self)
        root_layout.setContentsMargins(24, 20, 24, 20)
        root_layout.setSpacing(16)

        # ── Page Header
        root_layout.addWidget(self._build_header())

        # ── Folder Selector Section
        root_layout.addWidget(self._build_folder_selector())

        # ── Progress Section
        root_layout.addWidget(self._build_progress_section())

        # ── Action Buttons
        root_layout.addWidget(self._build_action_buttons())

        # ── Stats Summary Panel
        root_layout.addWidget(self._build_stats_panel())

        # ── Status Log (scrollable)
        root_layout.addWidget(self._build_status_log(), stretch=1)

        self.setLayout(root_layout)

    def _build_header(self) -> QWidget:
        """Build the page title / subtitle header block."""
        container = QFrame()
        container.setObjectName("HeaderFrame")
        layout = QVBoxLayout(container)
        layout.setContentsMargins(0, 0, 0, 8)
        layout.setSpacing(4)

        title = QLabel("📁  Folder Indexing")
        title.setObjectName("PageTitle")
        font = QFont()
        font.setPointSize(20)
        font.setBold(True)
        title.setFont(font)

        subtitle = QLabel(
            "Select a tile image folder to scan and generate visual embeddings for AI-powered similarity search."
        )
        subtitle.setObjectName("PageSubtitle")
        subtitle.setWordWrap(True)

        layout.addWidget(title)
        layout.addWidget(subtitle)

        separator = QFrame()
        separator.setFrameShape(QFrame.Shape.HLine)
        separator.setObjectName("Separator")
        layout.addWidget(separator)

        return container

    def _build_folder_selector(self) -> QGroupBox:
        """Build the folder selector group box with path display and browse button."""
        group = QGroupBox("📂  Tile Image Folder")
        group.setObjectName("SectionGroup")
        layout = QHBoxLayout(group)
        layout.setContentsMargins(16, 12, 16, 12)
        layout.setSpacing(12)

        self._folder_path_edit = QLineEdit()
        self._folder_path_edit.setObjectName("FolderPathEdit")
        self._folder_path_edit.setPlaceholderText("No folder selected — click Browse to choose...")
        self._folder_path_edit.setReadOnly(True)
        self._folder_path_edit.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)

        self._browse_button = QPushButton("  Browse")
        self._browse_button.setObjectName("BrowseButton")
        self._browse_button.setFixedWidth(120)
        self._browse_button.setFixedHeight(36)
        self._browse_button.setToolTip("Open folder picker dialog")

        layout.addWidget(self._folder_path_edit)
        layout.addWidget(self._browse_button)
        return group

    def _build_progress_section(self) -> QGroupBox:
        """Build the progress bar with file counter and ETA display."""
        group = QGroupBox("⚙️  Indexing Progress")
        group.setObjectName("SectionGroup")
        layout = QVBoxLayout(group)
        layout.setContentsMargins(16, 12, 16, 12)
        layout.setSpacing(10)

        # File name being processed
        self._current_file_label = QLabel("Ready to start.")
        self._current_file_label.setObjectName("CurrentFileLabel")
        self._current_file_label.setWordWrap(True)

        # Progress bar
        self._progress_bar = QProgressBar()
        self._progress_bar.setObjectName("IndexingProgressBar")
        self._progress_bar.setRange(0, 100)
        self._progress_bar.setValue(0)
        self._progress_bar.setTextVisible(True)
        self._progress_bar.setFormat("%p%")
        self._progress_bar.setFixedHeight(22)

        # Stats row: Processed / Total / ETA
        stats_row = QHBoxLayout()
        stats_row.setSpacing(24)

        self._processed_label = QLabel("Processed: 0")
        self._processed_label.setObjectName("StatLabel")
        self._total_label = QLabel("Total: 0")
        self._total_label.setObjectName("StatLabel")
        self._eta_label = QLabel("ETA: --")
        self._eta_label.setObjectName("StatLabel")

        stats_row.addWidget(self._processed_label)
        stats_row.addWidget(self._total_label)
        stats_row.addStretch()
        stats_row.addWidget(self._eta_label)

        layout.addWidget(self._current_file_label)
        layout.addWidget(self._progress_bar)
        layout.addLayout(stats_row)
        return group

    def _build_action_buttons(self) -> QWidget:
        """Build Start / Pause / Resume / Cancel action buttons."""
        container = QFrame()
        container.setObjectName("ButtonBar")
        layout = QHBoxLayout(container)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(12)

        self._start_button = QPushButton("▶  Start Indexing")
        self._start_button.setObjectName("StartButton")
        self._start_button.setFixedHeight(44)
        self._start_button.setToolTip("Start scanning the selected folder and indexing all tile images")

        self._pause_resume_button = QPushButton("⏸  Pause")
        self._pause_resume_button.setObjectName("PauseButton")
        self._pause_resume_button.setFixedHeight(44)
        self._pause_resume_button.setEnabled(False)
        self._pause_resume_button.setToolTip("Pause or resume the indexing process")

        self._cancel_button = QPushButton("✕  Cancel")
        self._cancel_button.setObjectName("CancelButton")
        self._cancel_button.setFixedHeight(44)
        self._cancel_button.setEnabled(False)
        self._cancel_button.setToolTip("Cancel the current indexing operation")

        layout.addWidget(self._start_button, stretch=2)
        layout.addWidget(self._pause_resume_button, stretch=1)
        layout.addWidget(self._cancel_button, stretch=1)
        return container

    def _build_stats_panel(self) -> QGroupBox:
        """Build summary statistics box showing indexed/skipped counts."""
        group = QGroupBox("📊  Results Summary")
        group.setObjectName("SectionGroup")
        layout = QGridLayout(group)
        layout.setContentsMargins(16, 12, 16, 12)
        layout.setSpacing(16)

        def _make_stat_block(icon: str, label_text: str, obj_name: str) -> tuple:
            """Helper to build a labelled stat value widget pair."""
            icon_label = QLabel(icon)
            icon_label.setObjectName("StatIcon")
            icon_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
            font = QFont()
            font.setPointSize(22)
            icon_label.setFont(font)

            value_label = QLabel("—")
            value_label.setObjectName(obj_name)
            value_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
            value_font = QFont()
            value_font.setPointSize(18)
            value_font.setBold(True)
            value_label.setFont(value_font)

            desc_label = QLabel(label_text)
            desc_label.setObjectName("StatDesc")
            desc_label.setAlignment(Qt.AlignmentFlag.AlignCenter)

            return icon_label, value_label, desc_label

        # Indexed stat
        ic1, self._indexed_value_label, dc1 = _make_stat_block("🆕", "Newly Indexed", "IndexedStatValue")
        layout.addWidget(ic1, 0, 0)
        layout.addWidget(self._indexed_value_label, 1, 0)
        layout.addWidget(dc1, 2, 0)

        # Skipped stat
        ic2, self._skipped_value_label, dc2 = _make_stat_block("⏭", "Skipped (Unchanged)", "SkippedStatValue")
        layout.addWidget(ic2, 0, 1)
        layout.addWidget(self._skipped_value_label, 1, 1)
        layout.addWidget(dc2, 2, 1)

        # Total stat
        ic3, self._total_stat_value_label, dc3 = _make_stat_block("🗂", "Total Scanned", "TotalStatValue")
        layout.addWidget(ic3, 0, 2)
        layout.addWidget(self._total_stat_value_label, 1, 2)
        layout.addWidget(dc3, 2, 2)

        layout.setColumnStretch(0, 1)
        layout.setColumnStretch(1, 1)
        layout.setColumnStretch(2, 1)
        return group

    def _build_status_log(self) -> QGroupBox:
        """Build the scrollable live status / log text panel."""
        group = QGroupBox("📋  Activity Log")
        group.setObjectName("SectionGroup")
        layout = QVBoxLayout(group)
        layout.setContentsMargins(12, 8, 12, 8)

        self._status_log = QTextEdit()
        self._status_log.setObjectName("StatusLog")
        self._status_log.setReadOnly(True)
        self._status_log.setLineWrapMode(QTextEdit.LineWrapMode.NoWrap)
        self._status_log.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)

        clear_button = QPushButton("Clear Log")
        clear_button.setObjectName("ClearLogButton")
        clear_button.setFixedHeight(28)
        clear_button.clicked.connect(self._status_log.clear)

        layout.addWidget(self._status_log)
        layout.addWidget(clear_button, alignment=Qt.AlignmentFlag.AlignRight)
        return group

    # ── Signal/Slot Connections ───────────────────────────────────────────────

    def _connect_signals(self) -> None:
        """Connect UI widget events to ViewModel slots and ViewModel signals to UI handlers."""

        # View → ViewModel
        self._browse_button.clicked.connect(self._on_browse_clicked)
        self._start_button.clicked.connect(self._viewmodel.start_indexing)
        self._pause_resume_button.clicked.connect(self._on_pause_resume_clicked)
        self._cancel_button.clicked.connect(self._viewmodel.cancel_indexing)

        # ViewModel → View
        self._viewmodel.progress_changed.connect(self._on_progress_changed)
        self._viewmodel.state_changed.connect(self._on_state_changed)
        self._viewmodel.indexing_completed.connect(self._on_indexing_completed)
        self._viewmodel.error_occurred.connect(self._on_error_occurred)
        self._viewmodel.folder_selected.connect(self._on_folder_selected)
        self._viewmodel.status_message.connect(self._log_message)

    # ── UI Event Handlers ─────────────────────────────────────────────────────

    @Slot()
    def _on_browse_clicked(self) -> None:
        """Open a native folder picker dialog and pass the selection to the ViewModel."""
        folder = QFileDialog.getExistingDirectory(
            self,
            "Select Tile Image Folder",
            "",
            QFileDialog.Option.ShowDirsOnly | QFileDialog.Option.DontResolveSymlinks,
        )
        if folder:
            self._viewmodel.set_folder(folder)

    @Slot()
    def _on_pause_resume_clicked(self) -> None:
        """Toggle between pause and resume based on current state."""
        if self._viewmodel.is_running:
            self._viewmodel.pause_indexing()
        elif self._viewmodel.is_paused:
            self._viewmodel.resume_indexing()

    # ── ViewModel Signal Handlers ─────────────────────────────────────────────

    @Slot(int, int, int, str, str)
    def _on_progress_changed(
        self, processed: int, total: int, percent: int, filename: str, eta: str
    ) -> None:
        """
        Update progress bar, counter labels, and current file display.

        Args:
            processed: Files processed so far.
            total: Total file count.
            percent: Integer percentage 0-100.
            filename: Current file name.
            eta: Formatted ETA string.
        """
        self._progress_bar.setValue(percent)
        self._processed_label.setText(f"Processed: {processed:,}")
        self._total_label.setText(f"Total: {total:,}")
        self._eta_label.setText(f"ETA: {eta}")
        self._current_file_label.setText(f"Processing: {filename}")

    @Slot(str)
    def _on_state_changed(self, state: str) -> None:
        """
        Update button states and visual indicators based on indexing lifecycle state.

        Args:
            state: One of the IndexingState constants.
        """
        logger.debug(f"UI state transition: {state}")

        if state == IndexingState.RUNNING:
            self._start_button.setEnabled(False)
            self._pause_resume_button.setEnabled(True)
            self._pause_resume_button.setText("⏸  Pause")
            self._cancel_button.setEnabled(True)
            self._browse_button.setEnabled(False)

        elif state == IndexingState.PAUSED:
            self._pause_resume_button.setText("▶  Resume")

        elif state == IndexingState.CANCELLING:
            # Background thread is still finishing its current file — keep
            # everything locked so a second job can't start underneath it.
            self._start_button.setEnabled(False)
            self._pause_resume_button.setEnabled(False)
            self._cancel_button.setEnabled(False)
            self._browse_button.setEnabled(False)
            self._current_file_label.setText("⏳ Cancelling — finishing current file...")

        elif state in (IndexingState.IDLE, IndexingState.FINISHED, IndexingState.CANCELLED):
            self._start_button.setEnabled(True)
            self._pause_resume_button.setEnabled(False)
            self._pause_resume_button.setText("⏸  Pause")
            self._cancel_button.setEnabled(False)
            self._browse_button.setEnabled(True)

            if state == IndexingState.FINISHED:
                self._progress_bar.setValue(100)
                self._current_file_label.setText("✅ Indexing completed successfully.")
            elif state == IndexingState.CANCELLED:
                self._current_file_label.setText("⛔ Indexing was cancelled.")

        elif state == IndexingState.ERROR:
            self._start_button.setEnabled(True)
            self._pause_resume_button.setEnabled(False)
            self._cancel_button.setEnabled(False)
            self._browse_button.setEnabled(True)
            self._current_file_label.setText("❌ An error occurred. Check the activity log.")

    @Slot(int, int, int)
    def _on_indexing_completed(self, indexed: int, skipped: int, total: int) -> None:
        """
        Populate the results summary stats panel after indexing finishes.

        Args:
            indexed: Count of tiles newly indexed.
            skipped: Count of tiles skipped as unchanged.
            total: Total tiles scanned.
        """
        self._indexed_value_label.setText(f"{indexed:,}")
        self._skipped_value_label.setText(f"{skipped:,}")
        self._total_stat_value_label.setText(f"{total:,}")

    @Slot(str)
    def _on_error_occurred(self, message: str) -> None:
        """
        Display an error message in the activity log with error styling.

        Args:
            message: Human-readable error description.
        """
        timestamp = QDateTime.currentDateTime().toString("HH:mm:ss")
        self._status_log.append(
            f'<span style="color:#FF6B6B;">[{timestamp}] ERROR: {message}</span>'
        )

    @Slot(str)
    def _on_folder_selected(self, folder_path: str) -> None:
        """
        Update the folder path display field.

        Args:
            folder_path: Absolute path string to the selected folder.
        """
        self._folder_path_edit.setText(folder_path)
        self._folder_path_edit.setToolTip(folder_path)

    @Slot(str)
    def _log_message(self, message: str) -> None:
        """
        Append a timestamped message to the activity log.

        Args:
            message: Status message string.
        """
        timestamp = QDateTime.currentDateTime().toString("HH:mm:ss")
        self._status_log.append(f"[{timestamp}] {message}")
        # Auto-scroll to bottom
        scrollbar = self._status_log.verticalScrollBar()
        scrollbar.setValue(scrollbar.maximum())

    # ── Styling ───────────────────────────────────────────────────────────────

    def _apply_styles(self) -> None:
        """Apply QSS dark theme styles to this view and all child widgets."""
        self.setStyleSheet("""
            /* ── Base ─────────────────────────────────────────────────────── */
            #IndexingView {
                background-color: #1A1D26;
            }

            /* ── Header ───────────────────────────────────────────────────── */
            #PageTitle {
                color: #E8EAF6;
                font-size: 20px;
                font-weight: bold;
            }
            #PageSubtitle {
                color: #9E9E9E;
                font-size: 12px;
            }
            #Separator {
                border: none;
                border-top: 1px solid #2D3250;
                margin: 4px 0;
            }

            /* ── Section Groups ───────────────────────────────────────────── */
            QGroupBox {
                background-color: #1E2130;
                border: 1px solid #2D3250;
                border-radius: 8px;
                margin-top: 12px;
                font-size: 13px;
                color: #B0BEC5;
                font-weight: 600;
                padding-top: 6px;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                subcontrol-position: top left;
                padding: 0 8px;
                left: 12px;
                top: -6px;
                color: #7C83D3;
            }

            /* ── Folder Path Input ────────────────────────────────────────── */
            #FolderPathEdit {
                background-color: #252837;
                border: 1px solid #3D4166;
                border-radius: 6px;
                color: #E8EAF6;
                font-size: 13px;
                padding: 6px 10px;
            }
            #FolderPathEdit:read-only {
                color: #B0BEC5;
            }

            /* ── Browse Button ────────────────────────────────────────────── */
            #BrowseButton {
                background-color: #3D4166;
                border: 1px solid #5C6BC0;
                border-radius: 6px;
                color: #E8EAF6;
                font-size: 13px;
                font-weight: 600;
                padding: 6px 14px;
            }
            #BrowseButton:hover {
                background-color: #5C6BC0;
                border-color: #7986CB;
            }
            #BrowseButton:pressed {
                background-color: #3949AB;
            }

            /* ── Progress Bar ─────────────────────────────────────────────── */
            #IndexingProgressBar {
                background-color: #252837;
                border: 1px solid #3D4166;
                border-radius: 4px;
                text-align: center;
                color: #E8EAF6;
                font-size: 12px;
            }
            #IndexingProgressBar::chunk {
                background: qlineargradient(
                    x1: 0, y1: 0, x2: 1, y2: 0,
                    stop: 0 #5C6BC0,
                    stop: 1 #7C4DFF
                );
                border-radius: 4px;
            }

            /* ── Stat Labels ──────────────────────────────────────────────── */
            #StatLabel, #CurrentFileLabel {
                color: #B0BEC5;
                font-size: 12px;
            }
            #StatIcon {
                font-size: 22px;
            }
            #IndexedStatValue {
                color: #69F0AE;
                font-size: 22px;
                font-weight: bold;
            }
            #SkippedStatValue {
                color: #FFD740;
                font-size: 22px;
                font-weight: bold;
            }
            #TotalStatValue {
                color: #82B1FF;
                font-size: 22px;
                font-weight: bold;
            }
            #StatDesc {
                color: #757575;
                font-size: 11px;
            }

            /* ── Start Button ─────────────────────────────────────────────── */
            #StartButton {
                background: qlineargradient(
                    x1: 0, y1: 0, x2: 1, y2: 0,
                    stop: 0 #3949AB,
                    stop: 1 #5C6BC0
                );
                border: none;
                border-radius: 8px;
                color: #FFFFFF;
                font-size: 14px;
                font-weight: bold;
            }
            #StartButton:hover {
                background: qlineargradient(
                    x1: 0, y1: 0, x2: 1, y2: 0,
                    stop: 0 #5C6BC0,
                    stop: 1 #7986CB
                );
            }
            #StartButton:pressed {
                background-color: #283593;
            }
            #StartButton:disabled {
                background-color: #2D3250;
                color: #546E7A;
            }

            /* ── Pause Button ─────────────────────────────────────────────── */
            #PauseButton {
                background-color: #2D3250;
                border: 1px solid #FF9800;
                border-radius: 8px;
                color: #FF9800;
                font-size: 13px;
                font-weight: 600;
            }
            #PauseButton:hover {
                background-color: #FF9800;
                color: #1A1D26;
            }
            #PauseButton:pressed {
                background-color: #E65100;
                color: #FFFFFF;
            }
            #PauseButton:disabled {
                border-color: #37474F;
                color: #546E7A;
                background-color: #1E2130;
            }

            /* ── Cancel Button ────────────────────────────────────────────── */
            #CancelButton {
                background-color: #2D3250;
                border: 1px solid #EF5350;
                border-radius: 8px;
                color: #EF5350;
                font-size: 13px;
                font-weight: 600;
            }
            #CancelButton:hover {
                background-color: #EF5350;
                color: #FFFFFF;
            }
            #CancelButton:pressed {
                background-color: #B71C1C;
            }
            #CancelButton:disabled {
                border-color: #37474F;
                color: #546E7A;
                background-color: #1E2130;
            }

            /* ── Status Log ───────────────────────────────────────────────── */
            #StatusLog {
                background-color: #12141E;
                border: 1px solid #2D3250;
                border-radius: 6px;
                color: #B0BEC5;
                font-family: "Consolas", "Courier New", monospace;
                font-size: 12px;
                selection-background-color: #3D4166;
            }

            /* ── Clear Log Button ─────────────────────────────────────────── */
            #ClearLogButton {
                background-color: transparent;
                border: 1px solid #3D4166;
                border-radius: 4px;
                color: #757575;
                font-size: 11px;
                padding: 2px 10px;
            }
            #ClearLogButton:hover {
                color: #B0BEC5;
                border-color: #546E7A;
            }
        """)
