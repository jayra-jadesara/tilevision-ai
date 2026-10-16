"""Human-readable duration formatting for rebuild ETA labels."""

from __future__ import annotations


def format_eta_seconds(seconds: float) -> str:
    """
    Format a remaining-time estimate for the rebuild dialog.

    Examples: ``~40 s``, ``~14 min``, ``~3 h 12 min``.
    Returns an empty string when ``seconds`` is not a positive estimate.
    """
    try:
        total = int(round(float(seconds)))
    except (TypeError, ValueError):
        return ""
    if total <= 0:
        return ""
    if total < 60:
        return f"~{total} s"
    if total < 3600:
        minutes, secs = divmod(total, 60)
        if secs >= 45:
            minutes += 1
            secs = 0
        if minutes >= 60:
            return format_eta_seconds(minutes * 60)
        if secs == 0 or minutes >= 10:
            return f"~{minutes} min"
        return f"~{minutes} min {secs} s"
    hours, rem = divmod(total, 3600)
    minutes = rem // 60
    if minutes == 0:
        return f"~{hours} h"
    return f"~{hours} h {minutes} min"
