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


# Real torch/transformers are available in CI / Cloud Agent images.
# Do not inject fake torch modules — that breaks transformers' find_spec.

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

# Synthetic codes only — no real customer brands or product names.
OUT_STEMS = [
    "step01_add_folder_index",
    "step02_update_vs_rebuild",
    "step03_search_by_photo",
    "step04_multi_photo",
    "step05_crop_modes",
    "step06_description_search",
    "step07_read_results",
    "step08_export_catalogue",
    "step09_duplicates",
    "step10_history_activity",
    "step11_settings",
    "step12_licence_support",
]

# Realistic showroom-style Windows path for the Index page.
FAKE_FOLDER_DISPLAY = r"E:\Showroom\Tile Catalog\Collection A"


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
        ((210, 195, 175), "TV-2401 Sand"),
        ((120, 130, 140), "TV-3108 Slate"),
        ((180, 95, 70), "TV-8812 Clay"),
        ((90, 110, 95), "TV-5520 Moss"),
        ((230, 225, 220), "TV-1001 Cloud"),
        ((70, 75, 85), "TV-4400 Ink"),
    ]
    paths: list[Path] = []
    for i, (color, label) in enumerate(specs):
        p = TILES_DIR / f"{label.split()[0].lower()}.png"
        _make_tile_texture(p, color, label, seed=1000 + i * 17)
        paths.append(p)

    query = QUERY_DIR / "sample_query_photo.png"
    _make_tile_texture(query, (205, 190, 170), "Sample query", seed=42)
    return paths


def _catalogue_service(tmp: Path) -> CatalogueMasterService:
    db = DatabaseContext(str(tmp / "catalogue_help.db"))
    repo = SQLiteCatalogueProfileRepository(db)
    return CatalogueMasterService(repository=repo, license_customer_name="Demo Showroom")


def _build_window(
    app: QApplication, tmp: Path, tile_paths: list[Path], *, theme: str = "light"
) -> MainWindow:
    settings = AppSettings(config_dir=tmp)
    settings.theme = theme
    settings.setup_wizard_completed = True

    search_use_case = MagicMock()
    search_use_case.execute.return_value = []
    search_use_case.get_filter_options.return_value = {
        "brand": ["Studio A", "Studio B", "Studio C"],
        "category": ["Floor", "Wall", "Outdoor"],
        "color": ["Sand", "Slate", "Cloud", "Clay"],
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
    # Large window so content-only grabs stay sharp in Help.
    window.showNormal()
    window.resize(1600, 1000)
    window.setMinimumSize(1600, 1000)
    app.processEvents()
    return window


def _process(app: QApplication, ms: int = 50) -> None:
    app.processEvents()
    if ms > 0:
        # Allow deferred thumbnail loads / polish
        loop_deadline = datetime.now().timestamp() + (ms / 1000.0)
        while datetime.now().timestamp() < loop_deadline:
            app.processEvents()


def _save_pix(pix: QPixmap, out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    ok = pix.save(str(out_path), "PNG")
    if not ok:
        raise RuntimeError(f"Failed to save {out_path}")
    print(
        f"Wrote {out_path} ({out_path.stat().st_size:,} bytes) {pix.width()}x{pix.height()}",
        flush=True,
    )


def _grab_content(
    window: MainWindow,
    out_path: Path,
    app: QApplication,
    *,
    max_height_ratio: float | None = None,
) -> None:
    """
    Grab the content stack only (no sidebar).

    Nested full-window shots inside Help look tiny and blurry — content-only
    panels stay readable when shown next to the real sidebar.
    """
    window.raise_()
    window.activateWindow()
    _process(app, 100)
    content = window._content_stack
    pix = content.grab()
    if pix.isNull() or pix.width() < 200:
        # Fallback if stack grab fails offscreen.
        pix = window.centralWidget().grab()

    if max_height_ratio is not None and 0.3 < max_height_ratio < 1.0:
        h = max(320, int(pix.height() * max_height_ratio))
        pix = pix.copy(0, 0, pix.width(), min(h, pix.height()))

    # Do not downscale — Help displays at ~980 logical × 2x pixels.
    _save_pix(pix, out_path)


def _grab_widget(widget, out_path: Path, app: QApplication, *, min_width: int = 1400) -> None:
    """Grab a dialog/widget; pad onto a wide canvas so Help keeps readable width."""
    from PySide6.QtWidgets import QWidget

    assert isinstance(widget, QWidget)
    widget.show()
    widget.raise_()
    _process(app, 120)
    pix = widget.grab()
    if pix.isNull() or pix.width() < 100:
        raise RuntimeError(f"Null grab for {out_path}")
    if pix.width() >= min_width:
        _save_pix(pix, out_path)
        return
    # Pad horizontally so Help step images stay ≥1400 px wide (test guard).
    canvas = QPixmap(min_width, max(pix.height(), 320))
    canvas.fill(QColor(245, 247, 250) if "light" in str(out_path) else QColor(28, 32, 38))
    painter = QPainter(canvas)
    x = (min_width - pix.width()) // 2
    painter.drawPixmap(x, 0, pix)
    painter.end()
    _save_pix(canvas, out_path)


def _setup_index_idle_with_folder(window: MainWindow) -> None:
    iv = window._indexing_view
    # set_folder requires a real dir — use tiles dir, then overwrite display path
    window._indexing_viewmodel._selected_folder = TILES_DIR
    iv._on_folder_selected(FAKE_FOLDER_DISPLAY)
    iv._indexed_images_info_label.setText("Indexed Images: 38")
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
    iv._on_progress_changed(12, 38, 32, "TV-3108_slate_600x600.jpg", "2m 15s")
    iv._log_message("Scanning: Collection A...")
    iv._log_message("Found 38 images.")
    iv._log_message("Embedding TV-2401_sand_600x600.jpg")
    iv._log_message("Embedding TV-3108_slate_600x600.jpg")
    iv._new_value_label.setText("12")
    iv._modified_value_label.setText("3")
    iv._deleted_value_label.setText("0")
    iv._skipped_value_label.setText("8")


def _make_results(tile_paths: list[Path]) -> list[SearchResult]:
    meta = [
        ("TV-2401", "Studio A", "Floor", 96.4),
        ("TV-3108", "Studio A", "Floor", 91.2),
        ("TV-8812", "Studio B", "Wall", 87.5),
        ("TV-5520", "Studio B", "Floor", 84.1),
        ("TV-1001", "Studio C", "Wall", 79.8),
        ("TV-4400", "Studio C", "Outdoor", 74.3),
    ]
    results = []
    for path, (code, brand, cat, score) in zip(tile_paths, meta):
        display_path = rf"E:\Showroom\Tile Catalog\Collection A\{code}.jpg"
        tile = TileImage(
            file_path=display_path,
            file_name=f"{code}.jpg",
            file_size=path.stat().st_size,
            dimensions="600x600",
            brand=brand,
            category=cat,
            color="Sand" if "sand" in path.name or "cloud" in path.name else "Slate",
            size="600x600",
            product_code=code,
            width=256,
            height=256,
        )
        results.append(
            SearchResult(
                tile=tile,
                similarity_score=score,
                thumbnail_path=str(path),
                copy_paths=[display_path, display_path.replace("Collection A", "Collection B")],
                extra_copy_count=1,
            )
        )
    return results


def _filter_opts() -> dict:
    return {
        "brand": ["Studio A", "Studio B", "Studio C"],
        "category": ["Floor", "Wall", "Outdoor"],
        "color": ["Sand", "Slate", "Cloud", "Clay"],
        "size": ["600x600", "800x800", "300x600"],
    }


def _setup_search_empty(window: MainWindow, app: QApplication) -> None:
    sv = window._search_view
    sv._on_clear_clicked()
    window._search_viewmodel.filters_available.emit(_filter_opts())
    sv._status_label.setText("Ready. Drag an image or click Browse to search.")
    _process(app, 40)


def _setup_search_results(window: MainWindow, app: QApplication, tile_paths: list[Path]) -> list[SearchResult]:
    query = QUERY_DIR / "sample_query_photo.png"
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
    window._search_viewmodel.filters_available.emit(_filter_opts())
    _process(app, 200)
    return results


def _capture_theme(app: QApplication, tile_paths: list[Path], theme: str) -> None:
    out_dir = HELP_DIR / theme
    out_dir.mkdir(parents=True, exist_ok=True)
    tmp = ASSETS / f"config_{theme}"
    tmp.mkdir(parents=True, exist_ok=True)
    window = _build_window(app, tmp, tile_paths, theme=theme)
    window._on_theme_changed_request(theme)

    def save(stem: str, *, ratio: float | None = 0.78) -> None:
        _grab_content(window, out_dir / f"{stem}.png", app, max_height_ratio=ratio)

    from PySide6.QtWidgets import QScrollArea
    from src.presentation.views.duplicates_view import DuplicatesView
    from src.presentation.views.license_view import LicenseView

    # 1 Index folder
    window._navigate(0)
    _setup_index_idle_with_folder(window)
    save(OUT_STEMS[0], ratio=0.72)

    # 2 Settings — scroll Maintenance into view so Update / Full rebuild show
    window._navigate(3)
    settings = window._settings_view
    scroll = settings.findChild(QScrollArea, "SettingsGeneralScroll")
    if scroll is not None and hasattr(settings, "_rebuild_button"):
        scroll.ensureWidgetVisible(settings._rebuild_button)
        # Prefer the Maintenance buttons near the top of the viewport.
        bar = scroll.verticalScrollBar()
        if bar is not None:
            target = max(0, settings._rebuild_button.mapTo(scroll.widget(), settings._rebuild_button.rect().topLeft()).y() - 40)
            bar.setValue(min(target, bar.maximum()))
        _process(app, 80)
    save(OUT_STEMS[1], ratio=None)  # full content — buttons must not be cropped

    # 3 Search by photo
    window._navigate(1)
    _setup_search_empty(window, app)
    save(OUT_STEMS[2], ratio=0.70)

    # 4 Multi-photo
    window._navigate(1)
    sv = window._search_view
    if hasattr(sv, "_drop_zone") and hasattr(sv._drop_zone, "set_multi_photo_enabled"):
        sv._drop_zone.set_multi_photo_enabled(True)
    _setup_search_empty(window, app)
    sv._status_label.setText("Multi-photo on — drop at least two tile photos.")
    save(OUT_STEMS[3], ratio=0.70)

    # 5 Crop modes with query loaded
    window._navigate(1)
    _setup_search_results(window, app, tile_paths)
    sv._status_label.setText("Use Auto Crop & Search, Precise, or Manual crop.")
    save(OUT_STEMS[4], ratio=0.75)

    # 6 Description search — typed query + description results (no photo preview)
    window._navigate(1)
    _setup_search_empty(window, app)
    desc = "matte grey subway tile"
    if hasattr(sv, "_description_input"):
        sv._description_input.setText(desc)
    results = _make_results(tile_paths)
    # Clear any lingering photo preview so this is clearly description search.
    if hasattr(sv, "_drop_zone") and hasattr(sv._drop_zone, "reset"):
        sv._drop_zone.reset()
    window._search_viewmodel._state = SearchState.RESULTS
    window._search_viewmodel.state_changed.emit(SearchState.RESULTS)
    window._search_viewmodel.results_ready.emit(results)
    window._search_viewmodel.search_stats_ready.emit(len(results), 0.22)
    sv._status_label.setText(f'Description search — {len(results)} matches for “{desc}”.')
    _process(app, 200)
    save(OUT_STEMS[5], ratio=0.80)

    # 7 Read results (+N copies)
    window._navigate(1)
    _setup_search_results(window, app, tile_paths)
    save(OUT_STEMS[6], ratio=None)

    # 8 Export (results with Export enabled)
    window._navigate(1)
    _setup_search_results(window, app, tile_paths)
    sv._status_label.setText("Click Export Catalogue to build a dealer PDF.")
    save(OUT_STEMS[7], ratio=0.80)

    # 9 Duplicates — real modal with a sample exact-duplicate group
    dup_uc = window._find_duplicates_use_case or FindDuplicatesUseCase(MagicMock())
    dup = DuplicatesView(use_case=dup_uc, parent=window, theme=theme)
    # Same visual twice (exact duplicate) — different folder paths on disk.
    src_tile = tile_paths[0]
    tiles_for_dup: list[TileImage] = []
    for i, folder in enumerate(("Collection A", "Collection B")):
        display = Path(f"/Showroom/{folder}") / src_tile.name
        tiles_for_dup.append(
            TileImage(
                file_path=str(src_tile),  # real path so thumbnail loads
                file_name=f"{folder.replace(' ', '_').lower()}_{src_tile.name}",
                file_size=src_tile.stat().st_size,
                dimensions="600x600",
                brand="Studio A",
                category="Floor",
                product_code="TV-2401",
                width=256,
                height=256,
                perceptual_hash="abcd" * 4,
                id=i + 1,
            )
        )
    dup.resize(1100, 720)
    dup._on_scan_completed([tiles_for_dup], [])
    _grab_widget(dup, out_dir / f"{OUT_STEMS[8]}.png", app)
    dup.close()
    dup.deleteLater()
    _process(app, 40)

    # 10 History & Activity
    window._navigate(7)
    save(OUT_STEMS[9], ratio=0.75)

    # 11 Settings — top of page (overview / preferences), distinct from step 2
    window._navigate(3)
    scroll = settings.findChild(QScrollArea, "SettingsGeneralScroll")
    if scroll is not None and scroll.verticalScrollBar() is not None:
        scroll.verticalScrollBar().setValue(0)
        _process(app, 60)
    save(OUT_STEMS[10], ratio=0.70)

    # 12 Licence activation dialog
    license_uc = MagicMock()
    license_uc.get_hardware_fingerprint.return_value = "ABCD-EF12-3456-7890-DEMO"
    license_dlg = LicenseView(
        validate_use_case=license_uc,
        theme=theme,
        show_back=True,
        parent=window,
    )
    # Fixed-size dialog — pad to Help image width via _grab_widget.
    _grab_widget(license_dlg, out_dir / f"{OUT_STEMS[11]}.png", app)
    license_dlg.close()
    license_dlg.deleteLater()
    _process(app, 40)

    # Full Help page render for PR evidence
    window._navigate(4)
    help_view = window._help_view
    help_view.set_theme(theme)
    _process(app, 150)
    artifacts = Path(
        os.environ.get("TILEVISION_ARTIFACT_DIR")
        or os.environ.get("CURSOR_ARTIFACT_DIR")
        or "/tmp/tilevision_help_renders"
    ) / "help_page_renders"
    artifacts.mkdir(parents=True, exist_ok=True)
    pix = help_view.grab()
    if pix.width() < 1400:
        # Force a wide offscreen size for the grab.
        help_view.resize(1440, max(1200, pix.height()))
        _process(app, 80)
        pix = help_view.grab()
    dest = artifacts / f"help_full_{theme}.png"
    pix.save(str(dest), "PNG")
    print(f"Wrote Help page render {dest} {pix.width()}x{pix.height()}", flush=True)

    window.close()
    window.deleteLater()
    _process(app, 50)


def main() -> int:
    HELP_DIR.mkdir(parents=True, exist_ok=True)
    app = QApplication.instance() or QApplication(sys.argv)
    tile_paths = _prepare_assets()

    for theme in ("light", "dark"):
        print(f"\n=== Capturing theme={theme} ===", flush=True)
        _capture_theme(app, tile_paths, theme)

    print("\nSummary:", flush=True)
    for theme in ("light", "dark"):
        for stem in OUT_STEMS:
            p = HELP_DIR / theme / f"{stem}.png"
            if p.is_file():
                print(
                    f"  {p.relative_to(ROOT)}  {p.stat().st_size:,} bytes",
                    flush=True,
                )
            else:
                print(f"  MISSING {p.relative_to(ROOT)}", flush=True)
    os._exit(0)


if __name__ == "__main__":
    raise SystemExit(main())
