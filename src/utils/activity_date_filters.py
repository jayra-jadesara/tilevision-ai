"""Date-range presets for the Activity Log filter dropdown."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Optional, Tuple


@dataclass(frozen=True)
class ActivityDatePreset:
    """One selectable filter option."""

    key: str
    label: str


# Keys consumed by ActivityLogView / resolve_activity_date_range.
ACTIVITY_DATE_PRESETS: tuple[ActivityDatePreset, ...] = (
    ActivityDatePreset("all", "All time"),
    ActivityDatePreset("today", "Today"),
    ActivityDatePreset("yesterday", "Yesterday"),
    ActivityDatePreset("last_3_days", "Last 3 days"),
    ActivityDatePreset("last_7_days", "Last 7 days"),
    ActivityDatePreset("current_month", "Current month"),
    ActivityDatePreset("last_month", "Last month"),
    ActivityDatePreset("custom", "Custom range…"),
)


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


def _start_of_local_day(local_dt: datetime) -> datetime:
    """Midnight of the given local calendar day, as aware UTC."""
    local = local_dt.astimezone()
    start_local = local.replace(hour=0, minute=0, second=0, microsecond=0)
    return start_local.astimezone(timezone.utc)


def resolve_activity_date_range(
    preset_key: str,
    *,
    custom_since: Optional[datetime] = None,
    custom_until: Optional[datetime] = None,
    now: Optional[datetime] = None,
) -> Tuple[Optional[datetime], Optional[datetime]]:
    """
    Return ``(since, until)`` as aware UTC datetimes for SQL filtering.

    ``until`` is exclusive when set (start of the day after the inclusive end).
    ``(None, None)`` means no date filter (all time).
    """
    now_utc = now or _utc_now()
    if now_utc.tzinfo is None:
        now_utc = now_utc.replace(tzinfo=timezone.utc)
    local_now = now_utc.astimezone()

    key = (preset_key or "all").strip().lower()
    if key in ("all", "all_time", ""):
        return None, None

    if key == "today":
        since = _start_of_local_day(local_now)
        return since, None

    if key == "yesterday":
        today_start = _start_of_local_day(local_now)
        since = today_start - timedelta(days=1)
        return since, today_start

    if key == "last_3_days":
        since = _start_of_local_day(local_now) - timedelta(days=2)
        return since, None

    if key == "last_7_days":
        since = _start_of_local_day(local_now) - timedelta(days=6)
        return since, None

    if key == "current_month":
        local_month_start = local_now.replace(
            day=1, hour=0, minute=0, second=0, microsecond=0
        )
        return local_month_start.astimezone(timezone.utc), None

    if key == "last_month":
        local_month_start = local_now.replace(
            day=1, hour=0, minute=0, second=0, microsecond=0
        )
        # Day 0 of current month = last day of previous month.
        prev_month_last = local_month_start - timedelta(days=1)
        prev_month_start = prev_month_last.replace(
            day=1, hour=0, minute=0, second=0, microsecond=0
        )
        return (
            prev_month_start.astimezone(timezone.utc),
            local_month_start.astimezone(timezone.utc),
        )

    if key == "custom":
        since = custom_since
        until = custom_until
        if since is not None and since.tzinfo is None:
            since = since.replace(tzinfo=timezone.utc)
        if until is not None and until.tzinfo is None:
            until = until.replace(tzinfo=timezone.utc)
        # Treat a date-only "until" as inclusive end-of-day by advancing
        # to the next midnight when time is exactly 00:00:00.
        if until is not None and until.hour == 0 and until.minute == 0 and until.second == 0:
            until = until + timedelta(days=1)
        return since, until

    return None, None


def format_dt_for_sqlite(dt: datetime) -> str:
    """Format an aware/naive datetime as UTC text matching activity_log.created_at."""
    if dt.tzinfo is not None:
        dt = dt.astimezone(timezone.utc).replace(tzinfo=None)
    return dt.strftime("%Y-%m-%d %H:%M:%S.%f")
