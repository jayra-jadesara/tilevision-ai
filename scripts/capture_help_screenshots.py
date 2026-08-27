#!/usr/bin/env python3
"""
Capture authentic Help-dialog step screenshots from the live PySide6 UI.

Run with QT_QPA_PLATFORM=offscreen (or xvfb-run). Does not commit.
"""

from __future__ import annotations

import os
import sys
import types
from datetime import datetime
from pathlib import Path
from unittest.mock import MagicMock

# Must be set before QApplication
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
os.environ.setdefault("OMP_NUM_THREADS", "1")

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def _install_fake_ai_deps() -> None:
    if "torch" not in sys.modules:
        fake_torch = types.ModuleType("torch")

        class _FakeCuda:
            @staticmethod
            def is_available() -> bool:
                return False

        fake_torch.cuda = _FakeCuda()
        fake_torch.backends = types.SimpleNamespace(
            mps=types.SimpleNamespace(is_available=lambda: False)
        )
        sys.modules["torch"] = fake_torch
    if "open_clip" not in sys.modules:
        sys.modules["open_clip"] = types.ModuleType("open_clip")
    if "transformers" not in sys.modules:
        try:
            __import__("transformers")
        except ImportError:
            fake = types.ModuleType("transformers")
            fake.AutoImageProcessor = object
            fake.AutoModel = object
            sys.modules["transformers"] = fake
    # Lightweight stubs so IndexImagesUseCase import graph loads without GPU stack.
    try:
        __import__("skimage.feature")
    except ImportError:
        sk = types.ModuleType("skimage")
        feat = types.ModuleType("skimage.feature")
        feat.local_binary_pattern = lambda *a, **k: None
        sk.feature = feat
        sys.modules["skimage"] = sk
        sys.modules["skimage.feature"] = feat
    try:
        __import__("cv2")
    except ImportError:
        sys.modules["cv2"] = types.ModuleType("cv2")


_install_fake_ai_deps()

from PySide6.QtCore import QRect, Qt, QTimer
from PySide6.QtGui import QColor, QFont, QPainter, QPixmap
from PySide6.QtWidgets import QApplication

from src.config.settings import AppSettings
from src.core.models import SearchResult, TileImage
from src.core.use_cases.find_duplicates import FindDuplicatesUseCase
from src.data.db_context import DatabaseContext
from src.data.sqlite_repository import SQLiteCatalogueProfileRepository
from src.presentation.viewmodels.indexing_viewmodel import IndexingState, IndexingViewModel
from src.presentation.viewmodels.search_viewmodel import SearchState, SearchViewModel
from src.presentation.views.main_window import DashboardDataProviders, MainWindow
from src.services.catalogue_master_service import CatalogueMasterService


HELP_DIR = ROOT / "src" / "resources" / "help"
ASSETS = Path("/tmp/tilevision_help_assets")
TILES_DIR = ASSETS / "tiles"
QUERY_DIR = ASSETS / "query"
OUT_NAMES = [
    "step1_choose_folder.png",
    "step2_index_images.png",
    "step3_upload_customer_image.png",
    "step4_view_similar_tiles.png",
    "step5_double_click_to_open.png",
]

# Realistic showroom-style Windows path for the Index page.
FAKE_FOLDER_DISPLAY = r"E:\Showroom\Tile Catalog\Porcelain Collection"


class _FakeIndexUseCase:
    pass


def _make_tile_texture(path: Path, base: tuple[int, int, int], label: str, seed: int) -> None:
    """Paint a simple ceramic/marble-like square PNG (no external assets needed)."""
    size = 256
    img = QPixmap(size, size)
    img.fill(QColor(*base))
    painter = QPainter(img)
    painter.setRenderHint(QPainter.RenderHint.Antialiasing)

    # Subtle grain / veining
    rng = seed
    for i in range(80):
        rng = (1103515245 * rng + 12345) & 0x7FFFFFFF
        x = rng % size
        rng = (1103515245 * rng + 12345) & 0x7FFFFFFF
        y = rng % size
        rng = (1103515245 * rng + 12345) & 0x7FFFFFFF
        w = 8 + (rng % 40)
        shade = QColor(
            max(0, min(255, base[0] + ((rng >> 3) % 40) - 20)),
            max(0, min(255, base[1] + ((rng >> 5) % 40) - 20)),
            max(0, min(255, base[2] + ((rng >> 7) % 40) - 20)),
            90,
        )
        painter.setPen(Qt.PenStyle.NoPen)
        painter.setBrush(shade)
        painter.drawEllipse(x, y, w, w // 2 + 4)

    # Grid grout lines for a tile look
    painter.setPen(QColor(255, 255, 255, 35))
    for g in range(0, size, 64):
        painter.drawLine(g, 0, g, size)
        painter.drawLine(0, g, size, g)

    painter.setPen(QColor(40, 40, 40, 180))
    font = QFont("Segoe UI", 11, QFont.Weight.Bold)
    painter.setFont(font)
    painter.drawText(QRect(8, size - 36, size - 16, 28), Qt.AlignmentFlag.AlignLeft, label)
    painter.end()
    img.save(str(path), "PNG")


def _prepare_assets() -> list[Path]:
    TILES_DIR.mkdir(parents=True, exist_ok=True)
    QUERY_DIR.mkdir(parents=True, exist_ok=True)
    specs = [
        ((210, 195, 175), "KAR-2401 Beige"),
        ((120, 130, 140), "KAR-3108 Grey"),
        ((180, 95, 70), "RAK-8812 Terra"),
        ((90, 110, 95), "SOM-5520 Sage"),
        ((230, 225, 220), "ASI-1001 White"),
        ((70, 75, 85), "NIT-4400 Charcoal"),
    ]
    paths: list[Path] = []
    for i, (color, label) in enumerate(specs):
        p = TILES_DIR / f"{label.split()[0].lower()}.png"
        _make_tile_texture(p, color, label, seed=1000 + i * 17)
        paths.append(p)

    query = QUERY_DIR / "customer_whatsapp_photo.png"
    _make_tile_texture(query, (205, 190, 170), "Customer photo", seed=42)
    return paths


def _catalogue_service(tmp: Path) -> CatalogueMasterService:
    db = DatabaseContext(str(tmp / "catalogue_help.db"))
    repo = SQLiteCatalogueProfileRepository(db)
    return CatalogueMasterService(repository=repo, license_customer_name="Demo Showroom")


def _build_window(app: QApplication, tmp: Path, tile_paths: list[Path]) -> MainWindow:
    settings = AppSettings(config_dir=tmp)
    settings.theme = "light"
    settings.setup_wizard_completed = True

    search_use_case = MagicMock()
    search_use_case.execute.return_value = []
    search_use_case.get_filter_options.return_value = {
        "brand": ["Kajaria", "RAK", "Somany", "Asian Granito", "NITCO"],
        "category": ["Floor", "Wall", "Outdoor"],
        "color": ["Beige", "Grey", "White", "Terra"],
        "size": ["600x600", "800x800", "300x600"],
    }
    search_use_case.get_index_health.return_value = types.SimpleNamespace(
        is_compatible=True, stale_count=0, indexed_count=len(tile_paths)
    )
    search_use_case.get_searchable_count.return_value = len(tile_paths)

    search_vm = SearchViewModel(use_case=search_use_case, default_top_k=10)
    repo = MagicMock()
    repo.get_all.return_value = []
    duplicates_uc = FindDuplicatesUseCase(repo)

    dashboard = DashboardDataProviders(
        indexed_folder_count=lambda: 1,
        database_size=lambda: 12_000_000,
        faiss_size=lambda: 4_000_000,
        last_search=lambda: None,
        recent_activity=lambda: [],
        recent_searches=lambda: [],
    )

    window = MainWindow(
        indexing_viewmodel=IndexingViewModel(use_case=_FakeIndexUseCase()),
        search_viewmodel=search_vm,
        find_duplicates_use_case=duplicates_uc,
        settings=settings,
        catalog_count_provider=lambda: len(tile_paths),
        catalogue_master_service=_catalogue_service(tmp),
        dashboard_providers=dashboard,
        license_details={
            "license_type": "1-Year",
            "customer_name": "Demo Showroom",
            "is_trial": False,
        },
        on_check_updates=lambda: None,
    )
    # Avoid maximized offscreen extremes; Help cards scale to ~540px wide.
    window.showNormal()
    window.resize(1100, 720)
    window.setMinimumSize(1100, 720)
    app.processEvents()
    return window


def _process(app: QApplication, ms: int = 50) -> None:
    app.processEvents()
    if ms > 0:
        # Allow deferred thumbnail loads / polish
        loop_deadline = datetime.now().timestamp() + (ms / 1000.0)
        while datetime.now().timestamp() < loop_deadline:
            app.processEvents()


def _grab_main(window: MainWindow, out_path: Path, app: QApplication) -> None:
    """Grab the full main window (sidebar + content) — authentic product chrome."""
    window.raise_()
    window.activateWindow()
    _process(app, 80)
    pix = window.grab()
    # Soft crop to content if somehow oversized
    if pix.width() > 1200:
        pix = pix.copy(0, 0, 1100, min(720, pix.height()))
    # Scale down slightly if still huge file-wise
    if pix.width() > 1100:
        pix = pix.scaledToWidth(1100, Qt.TransformationMode.SmoothTransformation)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    ok = pix.save(str(out_path), "PNG")
    if not ok:
        raise RuntimeError(f"Failed to save {out_path}")
    print(
        f"Wrote {out_path} ({out_path.stat().st_size:,} bytes) {pix.width()}x{pix.height()}",
        flush=True,
    )


def _setup_index_idle_with_folder(window: MainWindow) -> None:
    iv = window._indexing_view
    # set_folder requires a real dir — use tiles dir, then overwrite display path
    window._indexing_viewmodel._selected_folder = TILES_DIR
    iv._on_folder_selected(FAKE_FOLDER_DISPLAY)
    iv._indexed_images_info_label.setText("Indexed Images: 1,248")
    iv._status_info_label.setText("Status: Ready")
    iv._last_indexed_info_label.setText("Last Indexed: Today 10:42 AM")
    iv._current_file_label.setText("Ready — click Start Indexing to check for changes.")
    iv._log_message(f"Folder selected: {FAKE_FOLDER_DISPLAY}")
    iv._log_message("Catalog ready. Click Start Indexing to scan for new or changed tiles.")


def _setup_index_progress(window: MainWindow) -> None:
    iv = window._indexing_view
    window._indexing_viewmodel._selected_folder = TILES_DIR
    window._indexing_viewmodel._state = IndexingState.RUNNING
    iv._on_folder_selected(FAKE_FOLDER_DISPLAY)
    iv._on_state_changed(IndexingState.RUNNING)
    iv._on_progress_changed(486, 1248, 39, "KAR-3108_grey_600x600.jpg", "2m 15s")
    iv._log_message(f"Scanning: Porcelain Collection...")
    iv._log_message("Found 1,248 images.")
    iv._log_message("Embedding KAR-2401_beige_600x600.jpg")
    iv._log_message("Embedding KAR-3108_grey_600x600.jpg")
    iv._new_value_label.setText("312")
    iv._modified_value_label.setText("18")
    iv._deleted_value_label.setText("0")
    iv._skipped_value_label.setText("156")


def _make_results(tile_paths: list[Path]) -> list[SearchResult]:
    meta = [
        ("KAR-2401", "Kajaria", "Floor", 96.4),
        ("KAR-3108", "Kajaria", "Floor", 91.2),
        ("RAK-8812", "RAK", "Wall", 87.5),
        ("SOM-5520", "Somany", "Floor", 84.1),
        ("ASI-1001", "Asian Granito", "Wall", 79.8),
        ("NIT-4400", "NITCO", "Outdoor", 74.3),
    ]
    results = []
    for path, (code, brand, cat, score) in zip(tile_paths, meta):
        # Real file for thumbnails; Windows-style path shown in the table.
        display_path = rf"E:\Showroom\Tile Catalog\Porcelain Collection\{code}.jpg"
        tile = TileImage(
            file_path=display_path,
            file_name=f"{code}.jpg",
            file_size=path.stat().st_size,
            dimensions="600x600",
            brand=brand,
            category=cat,
            color="Beige" if "beige" in path.name or "white" in path.name else "Grey",
            size="600x600",
            product_code=code,
            width=256,
            height=256,
        )
        results.append(
            SearchResult(tile=tile, similarity_score=score, thumbnail_path=str(path))
        )
    return results


def _setup_search_empty(window: MainWindow, app: QApplication) -> None:
    sv = window._search_view
    sv._on_clear_clicked()
    window._search_viewmodel.filters_available.emit(
        {
            "brand": ["Kajaria", "RAK", "Somany", "Asian Granito", "NITCO"],
            "category": ["Floor", "Wall", "Outdoor"],
            "color": ["Beige", "Grey", "White", "Terra"],
            "size": ["600x600", "800x800", "300x600"],
        }
    )
    sv._status_label.setText("Ready. Drag an image or click Browse to search.")
    _process(app, 40)


def _setup_search_with_query(window: MainWindow, app: QApplication) -> None:
    query = QUERY_DIR / "customer_whatsapp_photo.png"
    sv = window._search_view
    sv._current_query_image_path = str(query)
    sv._drop_zone.show_preview(str(query))
    sv._crop_button.setEnabled(True)
    sv._auto_crop_button.setEnabled(True)
    sv._precise_crop_button.setEnabled(True)
    sv._clear_button.setEnabled(True)
    sv._status_label.setText("Ready to search — or use Crop if the photo shows a room scene.")
    window._search_viewmodel.filters_available.emit(
        {
            "brand": ["Kajaria", "RAK", "Somany"],
            "category": ["Floor", "Wall"],
            "color": ["Beige", "Grey"],
            "size": ["600x600", "800x800"],
        }
    )
    _process(app, 40)


def _setup_search_results(window: MainWindow, app: QApplication, tile_paths: list[Path]) -> list[SearchResult]:
    query = QUERY_DIR / "customer_whatsapp_photo.png"
    sv = window._search_view
    results = _make_results(tile_paths)
    sv._current_query_image_path = str(query)
    sv._drop_zone.show_preview(str(query))
    sv._crop_button.setEnabled(True)
    sv._auto_crop_button.setEnabled(True)
    sv._precise_crop_button.setEnabled(True)
    sv._clear_button.setEnabled(True)
    window._search_viewmodel._state = SearchState.RESULTS
    window._search_viewmodel.state_changed.emit(SearchState.RESULTS)
    window._search_viewmodel.results_ready.emit(results)
    window._search_viewmodel.search_stats_ready.emit(len(results), 0.18)
    sv._status_label.setText(f"Found {len(results)} similar tiles.")
    window._search_viewmodel.filters_available.emit(
        {
            "brand": ["Kajaria", "RAK", "Somany", "Asian Granito", "NITCO"],
            "category": ["Floor", "Wall", "Outdoor"],
            "color": ["Beige", "Grey", "White", "Terra"],
            "size": ["600x600", "800x800", "300x600"],
        }
    )
    # Wait for deferred thumbnail load
    _process(app, 200)
    return results


def main() -> int:
    HELP_DIR.mkdir(parents=True, exist_ok=True)
    app = QApplication.instance() or QApplication(sys.argv)
    tile_paths = _prepare_assets()

    tmp = ASSETS / "config"
    tmp.mkdir(parents=True, exist_ok=True)
    window = _build_window(app, tmp, tile_paths)

    # ── Step 1: Choose folder (Index page, idle with folder / Browse) ──
    window._navigate(0)
    _setup_index_idle_with_folder(window)
    _grab_main(window, HELP_DIR / OUT_NAMES[0], app)

    # ── Step 2: Indexing in progress ──
    window._navigate(0)
    _setup_index_progress(window)
    _grab_main(window, HELP_DIR / OUT_NAMES[1], app)

    # ── Step 3: Upload customer image (empty drop zone ready for drag/browse) ──
    window._navigate(1)
    _setup_search_empty(window, app)
    _grab_main(window, HELP_DIR / OUT_NAMES[2], app)

    # ── Step 4: View similar tiles ──
    window._navigate(1)
    _setup_search_results(window, app, tile_paths)
    _grab_main(window, HELP_DIR / OUT_NAMES[3], app)

    # ── Step 5: Row selected — double-click / right-click affordance ──
    window._navigate(1)
    _setup_search_results(window, app, tile_paths)
    sv = window._search_view
    sv._results_table.selectRow(0)
    sv._results_table.setFocus()
    sv._status_label.setText(
        "Double-click a row to open the full photo. Right-click for more options."
    )
    _grab_main(window, HELP_DIR / OUT_NAMES[4], app)

    print("\nSummary:", flush=True)
    for name in OUT_NAMES:
        p = HELP_DIR / name
        print(f"  {p.relative_to(ROOT)}  {p.stat().st_size:,} bytes", flush=True)
    # Offscreen Qt teardown can hang; screenshots are already on disk.
    os._exit(0)


if __name__ == "__main__":
    raise SystemExit(main())
