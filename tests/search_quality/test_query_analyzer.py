"""Unit tests for query-side analyzer and view planning."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.ai.search_quality.query_analyzer import QueryKind, analyze_query
from src.ai.search_quality.query_views import collect_query_crop_pils, plan_query_views


def _marble(h=500, w=500, seed=0):
    rng = np.random.default_rng(seed)
    return Image.fromarray(rng.integers(180, 240, (h, w, 3), dtype=np.uint8))


def _room_with_tile(seed=3):
    room = np.full((900, 1400, 3), 55, dtype=np.uint8)
    room[:400, :] = (72, 80, 92)
    room[400:, :] = (118, 108, 96)
    tile = np.asarray(_marble(380, 480, seed=seed))
    room[470 : 470 + 380, 460 : 460 + 480] = tile
    return Image.fromarray(room)


def _catalog_sheet():
    sheet = Image.new("RGB", (1200, 900), (255, 255, 255))
    sheet.paste(_marble(880, 500, seed=9), (20, 10))
    draw = ImageDraw.Draw(sheet)
    draw.text((560, 40), "COLLECTION SKU", fill=(20, 20, 20))
    for i in range(6):
        x, y = 560 + (i % 3) * 100, 300 + (i // 3) * 160
        draw.rectangle((x, y, x + 80, y + 140), outline=(0, 0, 0), width=2)
    return sheet


def _phone_screenshot():
    canvas = Image.new("RGB", (780, 1280), (18, 18, 22))
    content = _marble(900, 600, seed=2).resize((720, 1100))
    canvas.paste(content, (30, 96))
    draw = ImageDraw.Draw(canvas)
    draw.rectangle((0, 0, 780, 72), fill=(8, 8, 10))
    return canvas


def test_clean_tile_single_view():
    a = analyze_query(_marble(1000, 1000, seed=1))
    assert a.kind == QueryKind.CLEAN_TILE
    plan = plan_query_views(a)
    assert plan.max_views == 1
    assert plan.isolate_scene is False


def test_room_scene_triggers_isolation_plan():
    a = analyze_query(_room_with_tile())
    assert a.kind == QueryKind.ROOM_SCENE
    assert a.band_color_delta >= 35.0
    plan = plan_query_views(a, max_views_cap=3)
    assert plan.isolate_scene is True
    assert plan.max_views >= 2
    _, crops = collect_query_crop_pils(_room_with_tile(), analysis=a, max_views_cap=3)
    assert len(crops) >= 1
    # Isolated crop should be smaller than full room
    assert crops[0].size[0] * crops[0].size[1] < 1400 * 900 * 0.5


def test_catalog_sheet_not_confused_with_room():
    sheet = _catalog_sheet()
    a = analyze_query(sheet)
    assert a.kind == QueryKind.CATALOG_SHEET
    assert a.white_border_ratio >= 0.15
    plan = plan_query_views(a)
    assert plan.isolate_scene is False
    assert plan.prefer_panel is True


def test_phone_screenshot_strips_ui_plan():
    a = analyze_query(_phone_screenshot())
    assert a.kind == QueryKind.PHONE_SCREENSHOT
    plan = plan_query_views(a)
    assert plan.strip_ui is True
