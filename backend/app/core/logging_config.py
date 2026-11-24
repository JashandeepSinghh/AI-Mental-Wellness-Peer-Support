
"""
Backend Logging Configuration
==============================

Centralized logging system for the FastAPI backend.

Features:
- Console logging
- Rotating file logging
- Configurable log levels
- Consistent formatting
- Duplicate-handler prevention
- Safe exception logging

Privacy:
Never log raw student messages, conversation content,
passwords, authentication tokens or sensitive personal data.
"""

import logging
from logging.handlers import RotatingFileHandler
from pathlib import Path

from backend.app.core.config import get_settings


# ============================================================
# 1. PROJECT ROOT
# ============================================================

# logging_config.py is located at:
# project_root/backend/app/core/logging_config.py

PROJECT_ROOT = Path(__file__).resolve().parents[3]


# ============================================================
# 2. LOGGING FORMAT
# ============================================================

LOG_FORMAT = (
    "%(asctime)s | "
    "%(levelname)-8s | "
    "%(name)s | "
    "%(message)s"
)

DATE_FORMAT = "%Y-%m-%d %H:%M:%S"


# ============================================================
# 3. LOGGING CONSTANTS
# ============================================================

MAX_LOG_FILE_SIZE = 5 * 1024 * 1024

BACKUP_COUNT = 5


# ============================================================
# 4. LOG LEVEL VALIDATION
# ============================================================

def get_log_level(level_name):

    level = getattr(
        logging,
        level_name.upper(),
        None
    )

    if not isinstance(level, int):

        raise ValueError(
            f"Invalid logging level: {level_name}"
        )

    return level


# ============================================================
# 5. LOG DIRECTORY SETUP
# ============================================================

def get_log_file_path():

    settings = get_settings()

    log_path = Path(settings.LOG_FILE)

    # Resolve relative paths from the project root.
    if not log_path.is_absolute():

        log_path = PROJECT_ROOT / log_path

    log_path.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    return log_path


# ============================================================
# 6. LOGGER CONFIGURATION
# ============================================================

def configure_logging():

    settings = get_settings()

    log_level = get_log_level(
        settings.LOG_LEVEL
    )

    formatter = logging.Formatter(
        fmt=LOG_FORMAT,
        datefmt=DATE_FORMAT
    )

    # --------------------------------------------------------
    # Application logger
    # --------------------------------------------------------

    logger = logging.getLogger(
        "mental_wellness"
    )

    logger.setLevel(log_level)

    logger.propagate = False

    # --------------------------------------------------------
    # Remove existing handlers to prevent duplicates.
    # --------------------------------------------------------

    for handler in logger.handlers[:]:

        logger.removeHandler(handler)

        handler.close()

    # --------------------------------------------------------
    # Console handler
    # --------------------------------------------------------

    console_handler = logging.StreamHandler()

    console_handler.setLevel(
        log_level
    )

    console_handler.setFormatter(
        formatter
    )

    logger.addHandler(
        console_handler
    )

    # --------------------------------------------------------
    # Rotating file handler
    # --------------------------------------------------------

    log_file_path = get_log_file_path()

    file_handler = RotatingFileHandler(

        filename=log_file_path,

        maxBytes=MAX_LOG_FILE_SIZE,

        backupCount=BACKUP_COUNT,

        encoding="utf-8"
    )

    file_handler.setLevel(
        log_level
    )

    file_handler.setFormatter(
        formatter
    )

    logger.addHandler(
        file_handler
    )

    logger.info(
        "Backend logging initialized."
    )

    return logger


# ============================================================
# 7. LOGGER ACCESSOR
# ============================================================

def get_logger(name=None):

    """
    Returns a logger under the main application logger.

    Examples:
        get_logger("api")
        get_logger("database")
        get_logger("inference")
    """

    if name:

        return logging.getLogger(
            f"mental_wellness.{name}"
        )

    return logging.getLogger(
        "mental_wellness"
    )


# ============================================================
# 8. SAFE EXCEPTION LOGGING
# ============================================================

def log_exception(logger, message, error):

    """
    Logs the exception type without exposing arbitrary
    exception messages that may contain sensitive data.
    """

    logger.error(
        "%s | exception_type=%s",
        message,
        type(error).__name__
    )


# ============================================================
# 9. LOGGING VALIDATION
# ============================================================

def validate_logging():

    logger = configure_logging()

    if not logger.handlers:

        raise RuntimeError(
            "Logging handlers were not configured."
        )

    handler_types = {

        type(handler).__name__

        for handler in logger.handlers
    }

    required_handlers = {

        "StreamHandler",

        "RotatingFileHandler"
    }

    if not required_handlers.issubset(
        handler_types
    ):

        raise RuntimeError(
            "Required logging handlers are missing."
        )

    log_path = get_log_file_path()

    if not log_path.parent.exists():

        raise RuntimeError(
            "Log directory was not created."
        )

    if logger.propagate:

        raise RuntimeError(
            "Application logger propagation must be disabled."
        )

    return {

        "status": "PASSED",

        "logger_name": logger.name,

        "handler_types": sorted(
            handler_types
        ),

        "log_file": str(log_path),

        "log_level": logging.getLevelName(
            logger.level
        )
    }


# ============================================================
# 10. MAIN
# ============================================================

if __name__ == "__main__":

    result = validate_logging()

    print("=" * 70)

    print("BACKEND LOGGING VALIDATION: PASSED")

    print("=" * 70)

    for key, value in result.items():

        print(f"{key}: {value}")
