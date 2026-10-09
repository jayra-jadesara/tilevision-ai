"""
Shared license activation gate for startup and mid-session expiry cutover.

Startup and the periodic re-check both route through ``show_license_activation``
so the license-only UI path stays identical.
"""

from __future__ import annotations

import logging
import os
from typing import Any, Dict, Optional

from PySide6.QtCore import QObject, QTimer, Signal

from src.core.use_cases.validate_license import ValidateLicenseUseCase
from src.presentation.views.license_view import LicenseView

logger = logging.getLogger("tilevision.presentation.license_gate")

# Default: every 15 minutes. Override with TILEVISION_LICENSE_RECHECK_MS for QA.
DEFAULT_LICENSE_RECHECK_INTERVAL_MS = 15 * 60 * 1000


def resolve_license_recheck_interval_ms(
    default_ms: int = DEFAULT_LICENSE_RECHECK_INTERVAL_MS,
) -> int:
    """Return the periodic license re-check interval (milliseconds)."""
    raw = os.environ.get("TILEVISION_LICENSE_RECHECK_MS", "").strip()
    if not raw:
        return default_ms
    try:
        value = int(raw)
    except ValueError:
        logger.warning("Ignoring invalid TILEVISION_LICENSE_RECHECK_MS=%r", raw)
        return default_ms
    if value < 1000:
        logger.warning(
            "TILEVISION_LICENSE_RECHECK_MS=%s is too small; using %s",
            value,
            default_ms,
        )
        return default_ms
    return value


def show_license_activation(
    validate_use_case: ValidateLicenseUseCase,
    theme: str = "light",
    *,
    show_back: bool = True,
    parent=None,
) -> Optional[Dict[str, Any]]:
    """
    Show the same LicenseView used at startup.

    Returns verified license details after successful activation, or None if
    the user skipped/failed activation.
    """
    logger.info("Showing license activation dialog.")
    license_dialog = LicenseView(
        validate_use_case=validate_use_case,
        theme=theme,
        show_back=show_back,
        parent=parent,
    )
    license_dialog.exec()

    if not license_dialog.is_activated:
        return None

    return validate_use_case.verify_existing_license()


class LicenseSessionGuard(QObject):
    """
    Periodically re-validates the installed license while the app is running.

    Emits ``license_invalidated`` when ``verify_existing_license()`` returns
    None (expired, revoked, hardware mismatch, or missing).
    """

    license_invalidated = Signal()

    def __init__(
        self,
        validate_use_case: ValidateLicenseUseCase,
        interval_ms: Optional[int] = None,
        parent: Optional[QObject] = None,
    ) -> None:
        super().__init__(parent)
        self._validate = validate_use_case
        self._interval_ms = (
            resolve_license_recheck_interval_ms()
            if interval_ms is None
            else max(1000, int(interval_ms))
        )
        self._timer = QTimer(self)
        self._timer.setInterval(self._interval_ms)
        self._timer.timeout.connect(self.recheck_now)
        self._cutover_in_progress = False

    @property
    def interval_ms(self) -> int:
        return self._interval_ms

    @property
    def is_active(self) -> bool:
        return self._timer.isActive()

    def start(self) -> None:
        if not self._timer.isActive():
            logger.info(
                "Starting mid-session license re-check every %d ms.",
                self._interval_ms,
            )
            self._timer.start()

    def stop(self) -> None:
        if self._timer.isActive():
            self._timer.stop()
            logger.info("Stopped mid-session license re-check.")

    def recheck_now(self) -> None:
        """Run one verification pass (also used by the timer)."""
        if self._cutover_in_progress:
            return
        details = self._validate.verify_existing_license()
        if details is None:
            logger.warning(
                "Mid-session license check failed — license missing or invalid."
            )
            self._cutover_in_progress = True
            self.stop()
            self.license_invalidated.emit()

    def mark_cutover_complete(self) -> None:
        """Allow future re-checks after a successful re-activation."""
        self._cutover_in_progress = False
