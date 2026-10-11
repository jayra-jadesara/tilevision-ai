"""Tests for Activity Log date-range presets."""

from __future__ import annotations

import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.utils.activity_date_filters import (
    ACTIVITY_DATE_PRESETS,
    format_dt_for_sqlite,
    resolve_activity_date_range,
)


def test_presets_include_requested_options():
    keys = {p.key for p in ACTIVITY_DATE_PRESETS}
    assert {
        "all",
        "today",
        "yesterday",
        "last_3_days",
        "last_7_days",
        "current_month",
        "last_month",
        "custom",
    } <= keys


def test_all_time_has_no_bounds():
    assert resolve_activity_date_range("all") == (None, None)


def test_today_starts_at_local_midnight():
    # Fixed instant: 2026-10-11 15:30 UTC
    now = datetime(2026, 10, 11, 15, 30, tzinfo=timezone.utc)
    since, until = resolve_activity_date_range("today", now=now)
    assert until is None
    assert since is not None
    local = since.astimezone()
    assert local.hour == 0 and local.minute == 0
    assert local.day == now.astimezone().day


def test_yesterday_is_closed_range():
    now = datetime(2026, 10, 11, 15, 30, tzinfo=timezone.utc)
    since, until = resolve_activity_date_range("yesterday", now=now)
    assert since is not None and until is not None
    assert until > since
    assert (until - since).days == 1


def test_last_7_days_covers_six_prior_midnights():
    now = datetime(2026, 10, 11, 12, 0, tzinfo=timezone.utc)
    since, until = resolve_activity_date_range("last_7_days", now=now)
    assert until is None
    assert since is not None
    today_start, _ = resolve_activity_date_range("today", now=now)
    assert (today_start - since).days == 6


def test_custom_inclusive_end_advances_midnight():
    since = datetime(2026, 10, 1, tzinfo=timezone.utc)
    until = datetime(2026, 10, 5, tzinfo=timezone.utc)
    s, u = resolve_activity_date_range(
        "custom", custom_since=since, custom_until=until
    )
    assert s == since
    assert u == datetime(2026, 10, 6, tzinfo=timezone.utc)


def test_format_dt_for_sqlite_strips_tz():
    dt = datetime(2026, 10, 11, 12, 0, 0, 123456, tzinfo=timezone.utc)
    assert format_dt_for_sqlite(dt) == "2026-10-11 12:00:00.123456"
