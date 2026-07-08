"""
Application bootstrapper for TileVision AI.

Responsible for:
    1. Initialising structured logging (earliest possible).
    2. Loading application settings from config.json.
    3. Instantiating and wiring all dependency objects
       (DbContext → Repositories → UseCases → ViewModels → Views).
    4. Performing the offline license gate-check on startup.
    5. Launching the QApplication event loop.

Design Decision:
    All dependency construction is concentrated here (Composition Root / DI Root).
    No other module imports concrete implementations directly — they receive
    interfaces or pre-constructed instances via constructor injection.
    This makes every layer independently unit-testable with mock collaborators.
"""

import logging
import sys
from pathlib import Path

from PySide6.QtWidgets import QApplication, QMessageBox
from PySide6.QtGui import QFont

from src.utils.logger import setup_logger
from src.config.settings import AppSettings
from src.data.db_context import DatabaseContext
from src.data.sqlite_repository import SQLiteImageRepository, SQLiteLicenseRepository
from src.ai.embedder import OpenCLIPEmbedder
from src.ai.vector_index import FaissIndexManager
from src.core.use_cases.index_images import IndexImagesUseCase
from src.core.use_cases.validate_license import ValidateLicenseUseCase
from src.licensing.validator import LicenseValidator
from src.presentation.viewmodels.indexing_viewmodel import IndexingViewModel
from src.presentation.views.main_window import MainWindow
from src.presentation.views.license_view import LicenseView


def build_application() -> int:
    """
    Compose and launch the full TileVision AI application.

    This is the single entry point that wires everything together.

    Returns:
        The QApplication exit code (0 for clean exit).
    """
    # ── 1. Configure Logging first (before any other module runs) ────────────
    root_logger = setup_logger(
        name="tilevision",
        log_file_name="tilevision.log",
        log_level=logging.INFO,
    )
    logger = logging.getLogger("tilevision.app")
    logger.info("═" * 60)
    logger.info("TileVision AI — Starting application")
    logger.info("═" * 60)

    # ── 2. Load Settings ──────────────────────────────────────────────────────
    settings = AppSettings()
    logger.info(f"Configuration loaded. Data directory: {Path(settings.database_path).parent}")

    # ── 3. Create QApplication (must happen before any QWidget is created) ────
    app = QApplication(sys.argv)
    app.setApplicationName("TileVision AI")
    app.setOrganizationName("TileVision")
    app.setApplicationVersion("1.0.0")

    # Set modern default font
    default_font = QFont("Segoe UI", 10)
    app.setFont(default_font)

    # ── 4. Construct Data Layer ───────────────────────────────────────────────
    logger.info("Initializing database context...")
    db_context = DatabaseContext(db_path=settings.database_path)

    image_repository = SQLiteImageRepository(db_context=db_context)
    license_repository = SQLiteLicenseRepository(db_context=db_context)

    # ── 5. Construct Licensing Layer ──────────────────────────────────────────
    logger.info("Initializing license validator...")
    license_validator = LicenseValidator()
    validate_license_use_case = ValidateLicenseUseCase(
        license_repository=license_repository,
        validator=license_validator,
    )

    # ── 6. License Gate on Startup ────────────────────────────────────────────
    logger.info("Checking startup license status...")
    license_details = validate_license_use_case.verify_existing_license()

    if license_details is None:
        logger.warning("No valid license found — showing activation dialog.")
        license_dialog = LicenseView(validate_use_case=validate_license_use_case)
        result = license_dialog.exec()

        if not license_dialog.is_activated:
            logger.warning("License activation skipped or failed. Exiting.")
            QMessageBox.critical(
                None,
                "License Required",
                "TileVision AI requires a valid license to run.\n\n"
                "Please contact your supplier for a license key.\n\n"
                "The application will now close.",
            )
            return 1
    else:
        customer = license_details.get("customer_name", "Unknown")
        logger.info(f"Valid license found for: {customer}")

    # ── 7. Construct AI Layer ─────────────────────────────────────────────────
    logger.info("Initializing CLIP embedder and FAISS index manager...")
    embedder = OpenCLIPEmbedder(
        model_name=settings.model_name,
        pretrained=settings.pretrained,
    )
    vector_index = FaissIndexManager(
        index_path=settings.index_path,
        dimension=512,
    )

    # ── 8. Construct Use Cases ────────────────────────────────────────────────
    logger.info("Initializing use cases...")
    index_images_use_case = IndexImagesUseCase(
        image_repository=image_repository,
        embedder=embedder,
        vector_index=vector_index,
        thumbnail_dir=settings.thumbnail_dir,
    )

    # ── 9. Construct ViewModels ───────────────────────────────────────────────
    logger.info("Constructing view models...")
    indexing_viewmodel = IndexingViewModel(use_case=index_images_use_case)

    # ── 10. Launch Main Window ────────────────────────────────────────────────
    logger.info("Launching main application window...")
    main_window = MainWindow(indexing_viewmodel=indexing_viewmodel)
    main_window.show()

    logger.info("TileVision AI is running.")

    # ── 11. Run Qt Event Loop ─────────────────────────────────────────────────
    exit_code = app.exec()
    logger.info(f"TileVision AI exiting with code: {exit_code}")
    return exit_code
