"""
Logging utility for TileVision AI.

Provides rotating file logging and console logging configuration.
"""

import logging
import os
import sys
from logging.handlers import RotatingFileHandler
from pathlib import Path


def resolve_log_level(requested: int | None = None) -> int:
    """
    Resolve effective log level.

    TILEVISION_LOG_LEVEL overrides everything (DEBUG/INFO/WARNING/ERROR).
    Packaged Release builds (``sys.frozen``) default to WARNING so console
    stays quiet; development defaults to INFO.
    """
    env = os.environ.get("TILEVISION_LOG_LEVEL", "").strip().upper()
    if env:
        return getattr(logging, env, logging.INFO)
    if requested is not None:
        return int(requested)
    if getattr(sys, "frozen", False):
        return logging.WARNING
    return logging.INFO


def get_log_file_path(log_file_name: str = "tilevision.log") -> Path:
    """
    Resolve the default log file path (Task D: Settings, Export Logs).

    Mirrors the location setup_logger() uses internally, without requiring
    a logger instance — used by the Settings page to locate the file to
    export/copy.
    """
    return Path.home() / ".tilevision_ai" / "logs" / log_file_name


def setup_logger(
    name: str = "tilevision",
    log_file_name: str = "tilevision.log",
    max_bytes: int = 10 * 1024 * 1024,  # 10MB
    backup_count: int = 5,
    log_level: int | None = None,
) -> logging.Logger:
    """
    Configure and return a logger instance with console and rotating file handlers.
    """
    resolved = resolve_log_level(log_level)
    logger = logging.getLogger(name)
    logger.setLevel(min(resolved, logging.INFO))

    # Avoid adding duplicate handlers if the logger is already configured
    if logger.handlers:
        return logger

    formatter = logging.Formatter(
        fmt="%(asctime)s [%(levelname)s] %(name)s (%(filename)s:%(lineno)d) - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # Console Handler — Release: warnings+ only
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    console_handler.setLevel(resolved)
    logger.addHandler(console_handler)

    # File Handler keeps INFO even in Release so support can export logs.
    file_level = logging.INFO if resolved > logging.INFO else resolved
    app_data_dir = Path.home() / ".tilevision_ai"
    logs_dir = app_data_dir / "logs"

    try:
        logs_dir.mkdir(parents=True, exist_ok=True)
        log_path = logs_dir / log_file_name
        file_handler = RotatingFileHandler(
            filename=str(log_path),
            maxBytes=max_bytes,
            backupCount=backup_count,
            encoding="utf-8",
        )
        file_handler.setFormatter(formatter)
        file_handler.setLevel(file_level)
        logger.addHandler(file_handler)
    except OSError as e:
        # Fallback to current directory logs if AppData is not writeable
        fallback_dir = Path("./logs")
        try:
            fallback_dir.mkdir(parents=True, exist_ok=True)
            log_path = fallback_dir / log_file_name
            file_handler = RotatingFileHandler(
                filename=str(log_path),
                maxBytes=max_bytes,
                backupCount=backup_count,
                encoding="utf-8",
            )
            file_handler.setFormatter(formatter)
            file_handler.setLevel(file_level)
            logger.addHandler(file_handler)
        except OSError:
            print(f"Failed to initialize file logger due to exception: {e}")

    return logger
