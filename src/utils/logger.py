"""
Logging utility for TileVision AI.

Provides rotating file logging and console logging configuration.
"""

import logging
import os
from logging.handlers import RotatingFileHandler
from pathlib import Path


def get_log_file_path(log_file_name: str = "tilevision.log") -> Path:
    """
    Resolve the default log file path (Task D: Settings, Export Logs).

    Mirrors the location setup_logger() uses internally, without requiring
    a logger instance — used by the Settings page to locate the file to
    export/copy.

    Args:
        log_file_name: The log file name, matching setup_logger()'s default.

    Returns:
        The expected absolute path to the current log file. Not guaranteed
        to exist (e.g. before the app has logged anything, or if the
        AppData fallback path was used instead — callers should check).
    """
    return Path.home() / ".tilevision_ai" / "logs" / log_file_name


def setup_logger(
    name: str = "tilevision",
    log_file_name: str = "tilevision.log",
    max_bytes: int = 10 * 1024 * 1024,  # 10MB
    backup_count: int = 5,
    log_level: int = logging.INFO,
) -> logging.Logger:
    """
    Configure and return a logger instance with console and rotating file handlers.

    Args:
        name: The name of the logger.
        log_file_name: The name of the log file to write to.
        max_bytes: The maximum size of a single log file before rotation.
        backup_count: The number of rotated backup log files to retain.
        log_level: The logging level (e.g. logging.INFO, logging.DEBUG).

    Returns:
        A configured logging.Logger instance.
    """
    logger = logging.getLogger(name)
    logger.setLevel(log_level)

    # Avoid adding duplicate handlers if the logger is already configured
    if logger.handlers:
        return logger

    formatter = logging.Formatter(
        fmt="%(asctime)s [%(levelname)s] %(name)s (%(filename)s:%(lineno)d) - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # Console Handler
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    console_handler.setLevel(log_level)
    logger.addHandler(console_handler)

    # File Handler - write logs relative to the user's local AppData or home directory
    # For local scratch run, place in a folder "logs" inside the user's home profile
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
        file_handler.setLevel(log_level)
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
            file_handler.setLevel(log_level)
            logger.addHandler(file_handler)
        except OSError:
            # If everything fails, write a warning to console
            print(f"Failed to initialize file logger due to exception: {e}")

    return logger
