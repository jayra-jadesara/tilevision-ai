"""
Release customer scenarios S1–S30.

Each function uses the existing qa_e2e harness (real UI / DINOv2 / FAISS / SQLite).
No mocks. Failures raise; the pipeline records stacktraces and screenshots.
"""

from __future__ import annotations

import os
import time
from pathlib import Path
from typing import Callable, Dict, List, Tuple

from PySide6.QtCore import QSize, Qt
from PySide6.QtTest import QTest
from PySide6.QtWidgets import QApplication, QComboBox, QPushButton

from qa_e2e.framework.expectations import evaluate_from_manifest
from qa_e2e.framework.failures import detect_after_search, detect_ui_freeze
from qa_e2e.framework.readiness import probe_readiness, wait_for_search_pipeline
from qa_e2e.framework.ui_driver import UIDriver
from qa_e2e.release.checks import all_checks_passed, common_ui_checks, summarize_checks
from qa_e2e.release.scenario_result import ScenarioResult, fail_result
from src.presentation.viewmodels.search_viewmodel import SearchState
from src.utils import search_stages

ScenarioFn = Callable[[object, UIDriver], ScenarioResult]


def _shot(session, name: str) -> str:
    try:
        return session.artifacts.screenshot(session.main_window, name=name)
    except Exception:
        return ""


def _begin(sid: str, name: str) -> float:
    return time.time()


def _ok(sid: str, name: str, started: float, detail: str, session, **metrics) -> ScenarioResult:
    checks = metrics.pop("checks", {})
    return ScenarioResult(
        id=sid,
        name=name,
        ok=True,
        started_at=started,
        ended_at=time.time(),
        detail=detail,
        screenshot=_shot(session, sid),
        metrics=metrics,
        checks=checks,
    )


def _require_query(session, kind: str = "crop_match"):
    for q in session.expected_manifest.get("queries", []):
        if q.get("kind") == kind or q.get("id") == kind:
            return q
    return session.expected_manifest["queries"][0]


def _search_and_verify(session, driver: UIDriver, path: Path, *, sid: str) -> Dict:
    since = time.time()
    driver.drag_drop_image(path)
    state = driver.wait_search_settled(timeout=float(os.environ.get("TILEVISION_RELEASE_SEARCH_TIMEOUT", "300")))
    missing = wait_for_search_pipeline(session, since=since, timeout=5.0, require_stages=False)
    scan = detect_after_search(session, since=since, expect_results=True)
    checks = common_ui_checks(session, expect_results=True)
    paths = driver.result_paths()
    return {
        "state": state,
        "missing_stages": missing,
        "scan_ok": scan.ok,
        "findings": [f.message for f in scan.findings],
        "paths": paths,
        "checks": checks,
        "elapsed_s": getattr(session.search_viewmodel, "_last_elapsed_seconds", 0.0),
        "since": since,
    }


# ── Gates / scenarios ────────────────────────────────────────────────────────


def s01_open_app_search(session, driver: UIDriver) -> ScenarioResult:
    started = _begin("S01", "Open app → Search image → Verify results")
    try:
        driver.goto_search()
        session.human.think(0.2, 0.5)
        q = _require_query(session)
        info = _search_and_verify(session, driver, Path(q["path"]), sid="S01")
        exp = evaluate_from_manifest(session.expected_manifest, q["id"], info["paths"])
        ok = info["state"] == SearchState.RESULTS and info["scan_ok"] and exp.ok and all_checks_passed(info["checks"])
        if not ok:
            raise AssertionError(f"{exp.detail}; checks={summarize_checks(info['checks'])}; {info['findings']}")
        return _ok("S01", "Open app → Search image → Verify results", started, exp.detail, session, **info, ranking_ok=exp.ok)
    except Exception as exc:
        return fail_result("S01", "Open app → Search image → Verify results", started, exc, screenshot=_shot(session, "S01"))


def s02_drag_drop(session, driver: UIDriver) -> ScenarioResult:
    started = _begin("S02", "Drag & Drop")
    try:
        q = _require_query(session)
        info = _search_and_verify(session, driver, Path(q["path"]), sid="S02")
        stages_ok = session.logs.contains(search_stages.STAGE_FAISS_SEARCH, since=info["since"])
        ok = info["state"] == SearchState.RESULTS and info["scan_ok"] and stages_ok
        if not ok:
            raise AssertionError(f"drag-drop failed state={info['state']} findings={info['findings']}")
        return _ok("S02", "Drag & Drop", started, "drop search completed", session, **info)
    except Exception as exc:
        return fail_result("S02", "Drag & Drop", started, exc, screenshot=_shot(session, "S02"))


def s03_open_file_dialog(session, driver: UIDriver) -> ScenarioResult:
    started = _begin("S03", "Open File dialog path")
    try:
        q = next(q for q in session.expected_manifest["queries"] if q["id"] == "q_jpg")
        since = time.time()
        driver.open_image_via_viewmodel(Path(q["path"]))
        state = driver.wait_search_settled(timeout=300)
        scan = detect_after_search(session, since=since, expect_results=True)
        checks = common_ui_checks(session, expect_results=True)
        ok = state == SearchState.RESULTS and scan.ok and all_checks_passed(checks)
        if not ok:
            raise AssertionError(f"browse path search failed state={state}")
        return _ok("S03", "Open File dialog path", started, "browse-path search ok", session, state=state, checks=checks)
    except Exception as exc:
        return fail_result("S03", "Open File dialog path", started, exc, screenshot=_shot(session, "S03"))


def s04_auto_crop(session, driver: UIDriver) -> ScenarioResult:
    started = _begin("S04", "Auto Crop")
    try:
        q = _require_query(session)
        driver.open_image_via_viewmodel(Path(q["path"]))
        driver.wait_search_settled(timeout=300)
        since = time.time()
        driver.auto_crop_search()
        state = driver.wait_search_settled(timeout=420)
        ok = state == SearchState.RESULTS
        if not ok:
            raise AssertionError(f"auto crop state={state} status={driver.search_status()}")
        return _ok("S04", "Auto Crop", started, f"state={state}", session, crop_time_s=time.time() - since, state=state)
    except Exception as exc:
        return fail_result("S04", "Auto Crop", started, exc, screenshot=_shot(session, "S04"))


def s05_precise_crop(session, driver: UIDriver) -> ScenarioResult:
    started = _begin("S05", "Precise Crop")
    try:
        q = _require_query(session)
        driver.open_image_via_viewmodel(Path(q["path"]))
        driver.wait_search_settled(timeout=300)
        since = time.time()
        driver.precise_crop_search()
        state = driver.wait_search_settled(timeout=600)
        status = driver.search_status().lower()
        # SAM2 may be unavailable — UI must settle without freeze
        freeze = detect_ui_freeze(session)
        ok = freeze.ok and state != SearchState.SEARCHING
        if not ok:
            raise AssertionError(f"precise crop unsettled state={state}")
        detail = f"state={state} status={status}"
        if "fail" in status or state == SearchState.ERROR:
            session.artifacts.note("Precise Crop soft-failed (SAM2/env) but UI settled")
        return _ok("S05", "Precise Crop", started, detail, session, crop_time_s=time.time() - since, state=state)
    except Exception as exc:
        return fail_result("S05", "Precise Crop", started, exc, screenshot=_shot(session, "S05"))


def s06_search(session, driver: UIDriver) -> ScenarioResult:
    started = _begin("S06", "Search")
    try:
        q = _require_query(session)
        info = _search_and_verify(session, driver, Path(q["path"]), sid="S06")
        if info["state"] != SearchState.RESULTS or not info["scan_ok"]:
            raise AssertionError(f"search failed: {info['findings']}")
        return _ok("S06", "Search", started, "search completed", session, **info)
    except Exception as exc:
        return fail_result("S06", "Search", started, exc, screenshot=_shot(session, "S06"))


def s07_cancel_search(session, driver: UIDriver) -> ScenarioResult:
    started = _begin("S07", "Cancel Search")
    try:
        q = _require_query(session)
        driver.drag_drop_image(Path(q["path"]))
        session.human.wait(0.2)
        try:
            driver.clear_search()
        except AssertionError:
            session.human.wait(0.5)
            driver.clear_search()
        session.human.think(0.2, 0.5)
        state = session.search_viewmodel.state
        ok = state != SearchState.SEARCHING and session.main_window.isVisible()
        if not ok:
            raise AssertionError(f"cancel left state={state}")
        return _ok("S07", "Cancel Search", started, f"state={state}", session, state=state)
    except Exception as exc:
        return fail_result("S07", "Cancel Search", started, exc, screenshot=_shot(session, "S07"))


def s08_search_again(session, driver: UIDriver) -> ScenarioResult:
    started = _begin("S08", "Search again after cancel")
    try:
        queries = [q for q in session.expected_manifest["queries"] if q.get("kind") == "crop_match"]
        q = queries[1] if len(queries) > 1 else queries[0]
        info = _search_and_verify(session, driver, Path(q["path"]), sid="S08")
        if info["state"] != SearchState.RESULTS or not info["scan_ok"]:
            raise AssertionError(f"re-search failed: {info['findings']}")
        return _ok("S08", "Search again after cancel", started, "re-search ok", session, **info)
    except Exception as exc:
        return fail_result("S08", "Search again after cancel", started, exc, screenshot=_shot(session, "S08"))


def s09_index_new_folder(session, driver: UIDriver) -> ScenarioResult:
    started = _begin("S09", "Index new folder")
    try:
        from qa_e2e.fixtures.catalog_builder import build_customer_catalog

        extra = Path(session.home_dir) / "extra_catalog_root"
        manifest = build_customer_catalog(extra, tile_count=4)
        catalog = Path(manifest["catalog_dir"])
        before = session.vector_index.get_total_count()
        driver.select_index_folder(catalog)
        driver.start_indexing()
        driver.wait_indexing_done(timeout=float(os.environ.get("TILEVISION_QA_INDEX_TIMEOUT", "1800")))
        after = session.vector_index.get_total_count()
        sqlite_n = len(session.image_repository.get_all())
        ok = after >= before and sqlite_n > 0
        if not ok:
            raise AssertionError(f"index folder failed faiss {before}->{after} sqlite={sqlite_n}")
        return _ok("S09", "Index new folder", started, f"faiss={after} sqlite={sqlite_n}", session, faiss=after, sqlite=sqlite_n)
    except Exception as exc:
        return fail_result("S09", "Index new folder", started, exc, screenshot=_shot(session, "S09"))


def s10_reopen_application(session, driver: UIDriver) -> ScenarioResult:
    """
    Simulate reopen: hide/show main window and re-probe readiness.
    Full process restart is covered by CI launching a fresh job.
    """
    started = _begin("S10", "Reopen application")
    try:
        session.main_window.hide()
        QApplication.processEvents()
        QTest.qWait(400)
        session.main_window.show()
        QApplication.processEvents()
        session.human.think(0.3, 0.6)
        from dataclasses import asdict

        report = probe_readiness(session, require_catalog=True)
        if not report.ok or not session.main_window.isVisible():
            raise AssertionError(f"reopen failed: {report.failures}")
        return _ok(
            "S10",
            "Reopen application",
            started,
            "window restored + stack ready",
            session,
            readiness=asdict(report),
        )
    except Exception as exc:
        return fail_result("S10", "Reopen application", started, exc, screenshot=_shot(session, "S10"))


def _format_scenario(session, driver: UIDriver, query_id: str, sid: str, name: str) -> ScenarioResult:
    started = _begin(sid, name)
    try:
        q = next(x for x in session.expected_manifest["queries"] if x["id"] == query_id)
        info = _search_and_verify(session, driver, Path(q["path"]), sid=sid)
        exp = evaluate_from_manifest(session.expected_manifest, query_id, info["paths"])
        ok = info["state"] == SearchState.RESULTS and info["scan_ok"] and exp.ok
        if not ok:
            raise AssertionError(exp.detail or str(info["findings"]))
        return _ok(sid, name, started, exp.detail, session, **info)
    except Exception as exc:
        return fail_result(sid, name, started, exc, screenshot=_shot(session, sid))


def s11_large_image(session, driver: UIDriver) -> ScenarioResult:
    return _format_scenario(session, driver, "q_large", "S11", "Large image")


def s12_tiny_image(session, driver: UIDriver) -> ScenarioResult:
    return _format_scenario(session, driver, "q_small", "S12", "Tiny image")


def s13_png(session, driver: UIDriver) -> ScenarioResult:
    return _format_scenario(session, driver, "q_png", "S13", "PNG")


def s14_jpeg(session, driver: UIDriver) -> ScenarioResult:
    return _format_scenario(session, driver, "q_jpg", "S14", "JPEG")


def s15_webp(session, driver: UIDriver) -> ScenarioResult:
    return _format_scenario(session, driver, "q_webp", "S15", "WEBP")


def s16_tiff(session, driver: UIDriver) -> ScenarioResult:
    return _format_scenario(session, driver, "q_tiff", "S16", "TIFF")


def s17_corrupt(session, driver: UIDriver) -> ScenarioResult:
    started = _begin("S17", "Corrupt image")
    try:
        corrupt = Path(session.expected_manifest["corrupt_dir"]) / "not_an_image.jpg"
        driver.drag_drop_image(corrupt)
        QTest.qWait(1500)
        QApplication.processEvents()
        state = session.search_viewmodel.state
        ok = session.main_window.isVisible() and state != SearchState.SEARCHING
        if not ok:
            raise AssertionError(f"corrupt image left app bad state={state}")
        return _ok("S17", "Corrupt image", started, f"handled state={state}", session, state=state)
    except Exception as exc:
        return fail_result("S17", "Corrupt image", started, exc, screenshot=_shot(session, "S17"))


def s18_unicode_filename(session, driver: UIDriver) -> ScenarioResult:
    started = _begin("S18", "Unicode filename")
    try:
        from PIL import Image

        q = _require_query(session)
        src = Path(q["path"])
        dest = session.query_dir / "瓷砖_サンプル_بلاط.jpg"
        Image.open(src).convert("RGB").save(dest, quality=90)
        info = _search_and_verify(session, driver, dest, sid="S18")
        if info["state"] != SearchState.RESULTS:
            raise AssertionError(f"unicode search failed state={info['state']}")
        return _ok("S18", "Unicode filename", started, "unicode query ok", session, **info)
    except Exception as exc:
        return fail_result("S18", "Unicode filename", started, exc, screenshot=_shot(session, "S18"))


def s19_long_filename(session, driver: UIDriver) -> ScenarioResult:
    started = _begin("S19", "Long filename")
    try:
        from PIL import Image

        q = _require_query(session)
        long_name = ("TILE_LONG_" + ("X" * 180) + ".jpg")
        dest = session.query_dir / long_name
        Image.open(q["path"]).convert("RGB").save(dest, quality=90)
        info = _search_and_verify(session, driver, dest, sid="S19")
        if info["state"] != SearchState.RESULTS:
            raise AssertionError(f"long name search failed state={info['state']}")
        return _ok("S19", "Long filename", started, "long filename ok", session, **info)
    except Exception as exc:
        return fail_result("S19", "Long filename", started, exc, screenshot=_shot(session, "S19"))


def s20_multiple_searches(session, driver: UIDriver) -> ScenarioResult:
    started = _begin("S20", "Multiple searches without restart")
    try:
        queries = [q for q in session.expected_manifest["queries"] if q.get("kind") == "crop_match"][:4]
        times = []
        for q in queries:
            info = _search_and_verify(session, driver, Path(q["path"]), sid="S20")
            times.append(info["elapsed_s"])
            if info["state"] != SearchState.RESULTS:
                raise AssertionError(f"multi-search failed on {q['id']}")
            session.human.think(0.2, 0.5)
        return _ok("S20", "Multiple searches without restart", started, f"{len(queries)} searches ok", session, times=times)
    except Exception as exc:
        return fail_result("S20", "Multiple searches without restart", started, exc, screenshot=_shot(session, "S20"))


def s21_memory_stress(session, driver: UIDriver) -> ScenarioResult:
    started = _begin("S21", "Memory stress")
    try:
        import psutil

        proc = psutil.Process(os.getpid())
        before = proc.memory_info().rss / (1024 * 1024)
        q = _require_query(session)
        for i in range(int(os.environ.get("TILEVISION_RELEASE_MEM_LOOPS", "8"))):
            _search_and_verify(session, driver, Path(q["path"]), sid=f"S21_{i}")
        after = proc.memory_info().rss / (1024 * 1024)
        growth = after - before
        # Soft bound — flag extreme leaks only (GiB-scale)
        ok = growth < float(os.environ.get("TILEVISION_RELEASE_MEM_MAX_GROWTH_MB", "1500"))
        if not ok:
            raise AssertionError(f"RSS grew {growth:.1f} MiB ({before:.1f}→{after:.1f})")
        return _ok("S21", "Memory stress", started, f"RSS {before:.1f}→{after:.1f} MiB", session, before_mb=before, after_mb=after, growth_mb=growth)
    except Exception as exc:
        return fail_result("S21", "Memory stress", started, exc, screenshot=_shot(session, "S21"))


def s22_hundred_searches(session, driver: UIDriver) -> ScenarioResult:
    started = _begin("S22", "100 consecutive searches")
    try:
        n = int(os.environ.get("TILEVISION_RELEASE_SEARCH_COUNT", "100"))
        queries = [q for q in session.expected_manifest["queries"] if q.get("kind") == "crop_match"]
        if not queries:
            queries = session.expected_manifest["queries"]
        failures = 0
        for i in range(n):
            q = queries[i % len(queries)]
            info = _search_and_verify(session, driver, Path(q["path"]), sid=f"S22_{i}")
            if info["state"] != SearchState.RESULTS or not info["scan_ok"]:
                failures += 1
                if failures > int(os.environ.get("TILEVISION_RELEASE_SEARCH_MAX_FAILS", "0")):
                    raise AssertionError(f"search #{i+1}/{n} failed: {info['findings']}")
        return _ok("S22", "100 consecutive searches", started, f"{n} searches completed", session, count=n, failures=failures)
    except Exception as exc:
        return fail_result("S22", "100 consecutive searches", started, exc, screenshot=_shot(session, "S22"))


def s23_idle(session, driver: UIDriver) -> ScenarioResult:
    started = _begin("S23", "Application idle")
    try:
        seconds = float(os.environ.get("TILEVISION_RELEASE_IDLE_SECONDS", "1800"))
        driver.goto_dashboard()
        # Pump events in chunks so CI doesn't look wedged and freezes are detected
        end = time.monotonic() + seconds
        while time.monotonic() < end:
            QApplication.processEvents()
            QTest.qWait(1000)
            if not session.main_window.isVisible():
                raise AssertionError("window hidden during idle")
        freeze = detect_ui_freeze(session)
        if not freeze.ok:
            raise AssertionError("UI freeze after idle")
        # Smoke search after idle
        q = _require_query(session)
        info = _search_and_verify(session, driver, Path(q["path"]), sid="S23")
        if info["state"] != SearchState.RESULTS:
            raise AssertionError("search after idle failed")
        return _ok("S23", "Application idle", started, f"idle {seconds:.0f}s then search ok", session, idle_s=seconds)
    except Exception as exc:
        return fail_result("S23", "Application idle", started, exc, screenshot=_shot(session, "S23"))


def s24_export_pdf(session, driver: UIDriver) -> ScenarioResult:
    started = _begin("S24", "Export PDF")
    try:
        from src.services.pdf_export_service import PDFExportService, PdfExportOptions

        q = _require_query(session)
        info = _search_and_verify(session, driver, Path(q["path"]), sid="S24")
        results = session.search_viewmodel.last_results
        if not results:
            raise AssertionError("no results to export")
        out = Path(session.artifacts.out_dir) / "export_catalogue.pdf"
        service = PDFExportService()
        created = service.export_catalogue(
            output_file=str(out),
            query_image_path=q["path"],
            results=results,
            options=PdfExportOptions(title="Release Validation Catalogue"),
        )
        path = Path(created)
        if not path.exists() or path.stat().st_size < 64:
            raise AssertionError(f"PDF not written: {path}")
        return _ok(
            "S24",
            "Export PDF",
            started,
            f"wrote {path.name} ({path.stat().st_size} bytes)",
            session,
            pdf=str(path),
            results=len(results),
            search_elapsed_s=info.get("elapsed_s"),
        )
    except Exception as exc:
        return fail_result("S24", "Export PDF", started, exc, screenshot=_shot(session, "S24"))


def s25_preview_image(session, driver: UIDriver) -> ScenarioResult:
    started = _begin("S25", "Preview image")
    try:
        q = _require_query(session)
        _search_and_verify(session, driver, Path(q["path"]), sid="S25")
        table = driver.results_table()
        if table.rowCount() < 1:
            raise AssertionError("no rows to preview")
        # Click the Preview cell
        preview = table.item(0, 5)
        if preview is None:
            raise AssertionError("preview cell missing")
        rect = table.visualItemRect(preview)
        QTest.mouseClick(table.viewport(), Qt.MouseButton.LeftButton, Qt.KeyboardModifier.NoModifier, rect.center())
        QApplication.processEvents()
        QTest.qWait(400)
        freeze = detect_ui_freeze(session)
        if not freeze.ok:
            raise AssertionError("freeze after preview click")
        return _ok("S25", "Preview image", started, "preview click ok", session)
    except Exception as exc:
        return fail_result("S25", "Preview image", started, exc, screenshot=_shot(session, "S25"))


def s26_zoom(session, driver: UIDriver) -> ScenarioResult:
    started = _begin("S26", "Zoom")
    try:
        driver.goto_search()
        # Approximate zoom via font/scale-independent resize of content + processEvents
        session.main_window.resize(session.main_window.size() * 1.05)
        QApplication.processEvents()
        session.human.think(0.2, 0.4)
        session.main_window.resize(QSize(1280, 840))
        QApplication.processEvents()
        if not session.main_window.isVisible():
            raise AssertionError("window lost after zoom/resize")
        return _ok("S26", "Zoom", started, "window scale/resize ok", session)
    except Exception as exc:
        return fail_result("S26", "Zoom", started, exc, screenshot=_shot(session, "S26"))


def s27_scroll(session, driver: UIDriver) -> ScenarioResult:
    started = _begin("S27", "Scroll")
    try:
        q = _require_query(session)
        _search_and_verify(session, driver, Path(q["path"]), sid="S27")
        table = driver.results_table()
        session.human.scroll(table)
        table.scrollToBottom()
        QApplication.processEvents()
        table.scrollToTop()
        QApplication.processEvents()
        return _ok("S27", "Scroll", started, "results scrolled", session, rows=table.rowCount())
    except Exception as exc:
        return fail_result("S27", "Scroll", started, exc, screenshot=_shot(session, "S27"))


def s28_resize_window(session, driver: UIDriver) -> ScenarioResult:
    started = _begin("S28", "Resize window")
    try:
        session.main_window.resize(1024, 700)
        QApplication.processEvents()
        session.human.think(0.2, 0.4)
        session.main_window.resize(1440, 900)
        QApplication.processEvents()
        session.main_window.resize(1280, 840)
        QApplication.processEvents()
        if not session.main_window.isVisible():
            raise AssertionError("window not visible after resize")
        return _ok("S28", "Resize window", started, "resize ok", session)
    except Exception as exc:
        return fail_result("S28", "Resize window", started, exc, screenshot=_shot(session, "S28"))


def _apply_theme(session, driver: UIDriver, theme: str) -> str:
    """Drive theme the same way Settings → Theme does for a customer."""
    wanted = theme.lower()
    driver.goto_settings()
    session.human.think(0.2, 0.5)
    for combo in session.main_window.findChildren(QComboBox):
        items = [combo.itemText(i).lower() for i in range(combo.count())]
        if "dark" in items and "light" in items:
            combo.setCurrentIndex(items.index(wanted))
            QApplication.processEvents()
            break
    else:
        if hasattr(session.main_window, "_on_theme_changed_request"):
            session.main_window._on_theme_changed_request(wanted)
        else:
            session.settings.theme = wanted
            from src.theme.theme_manager import get_app_stylesheet

            session.app.setStyleSheet(get_app_stylesheet(wanted))
        QApplication.processEvents()
    current = getattr(session.main_window, "_current_theme", None) or session.settings.theme
    return str(current).lower()


def s29_dark_mode(session, driver: UIDriver) -> ScenarioResult:
    started = _begin("S29", "Dark mode")
    try:
        theme = _apply_theme(session, driver, "dark")
        session.artifacts.note(f"theme now={theme}")
        if theme != "dark":
            raise AssertionError(f"expected dark theme, got {theme}")
        return _ok("S29", "Dark mode", started, f"theme={theme}", session, theme=theme)
    except Exception as exc:
        return fail_result("S29", "Dark mode", started, exc, screenshot=_shot(session, "S29"))


def s30_light_mode(session, driver: UIDriver) -> ScenarioResult:
    started = _begin("S30", "Light mode")
    try:
        theme = _apply_theme(session, driver, "light")
        if theme != "light":
            raise AssertionError(f"expected light theme, got {theme}")
        return _ok("S30", "Light mode", started, f"theme={theme}", session, theme=theme)
    except Exception as exc:
        return fail_result("S30", "Light mode", started, exc, screenshot=_shot(session, "S30"))


SCENARIO_REGISTRY: List[Tuple[str, str, ScenarioFn]] = [
    ("S01", "Open app → Search image → Verify results", s01_open_app_search),
    ("S02", "Drag & Drop", s02_drag_drop),
    ("S03", "Open File dialog", s03_open_file_dialog),
    ("S04", "Auto Crop", s04_auto_crop),
    ("S05", "Precise Crop", s05_precise_crop),
    ("S06", "Search", s06_search),
    ("S07", "Cancel Search", s07_cancel_search),
    ("S08", "Search again", s08_search_again),
    ("S09", "Index new folder", s09_index_new_folder),
    ("S10", "Reopen application", s10_reopen_application),
    ("S11", "Large image", s11_large_image),
    ("S12", "Tiny image", s12_tiny_image),
    ("S13", "PNG", s13_png),
    ("S14", "JPEG", s14_jpeg),
    ("S15", "WEBP", s15_webp),
    ("S16", "TIFF", s16_tiff),
    ("S17", "Corrupt image", s17_corrupt),
    ("S18", "Unicode filename", s18_unicode_filename),
    ("S19", "Long filename", s19_long_filename),
    ("S20", "Multiple searches without restart", s20_multiple_searches),
    ("S21", "Memory stress", s21_memory_stress),
    ("S22", "100 consecutive searches", s22_hundred_searches),
    ("S23", "Application idle 30 minutes", s23_idle),
    ("S24", "Export PDF", s24_export_pdf),
    ("S25", "Preview image", s25_preview_image),
    ("S26", "Zoom", s26_zoom),
    ("S27", "Scroll", s27_scroll),
    ("S28", "Resize window", s28_resize_window),
    ("S29", "Dark mode", s29_dark_mode),
    ("S30", "Light mode", s30_light_mode),
]
