"""Logging configuration for the trading system."""

import logging
import logging.config
from pathlib import Path

from src.utils.paths import LOG_DIR


def setup_logging(level: str = "INFO", log_file: str | None = None) -> None:
    """Configure application-wide logging.

    Args:
        level: Logging level name (DEBUG, INFO, WARNING, ERROR).
        log_file: Optional log filename under logs/. Defaults to trading.log.
    """
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    filename = log_file or "trading.log"
    log_path = LOG_DIR / filename

    config = {
        "version": 1,
        "disable_existing_loggers": False,
        "formatters": {
            "standard": {
                "format": "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
                "datefmt": "%Y-%m-%d %H:%M:%S",
            },
        },
        "handlers": {
            "console": {
                "class": "logging.StreamHandler",
                "level": level,
                "formatter": "standard",
                "stream": "ext://sys.stdout",
            },
            "file": {
                "class": "logging.FileHandler",
                "level": level,
                "formatter": "standard",
                "filename": str(log_path),
                "encoding": "utf-8",
            },
        },
        "root": {
            "level": level,
            "handlers": ["console", "file"],
        },
    }
    logging.config.dictConfig(config)


def get_logger(name: str) -> logging.Logger:
    """Return a named logger instance.

    Args:
        name: Logger name, typically __name__ of the calling module.

    Returns:
        Configured logger instance.
    """
    return logging.getLogger(name)
