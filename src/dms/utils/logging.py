"""Logging utilities built on top of Loguru.

The :func:`configure_logging` function configures a root Loguru logger with a
standard format and optional file logging.  Individual modules should use
``from loguru import logger`` to log messages.

Example usage::

    from dms.utils.logging import configure_logging
    configure_logging(level="DEBUG")
    from loguru import logger
    logger.info("Application started")

"""
from __future__ import annotations

import sys
from typing import Optional

from loguru import logger

DEFAULT_FORMAT = (
    "<green>{time:YYYY-MM-DD HH:mm:ss}</green> | "
    "<level>{level: <8}</level> | "
    "<cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - "
    "<level>{message}</level>"
)

def configure_logging(level: str = "INFO", *, file_path: Optional[str] = None) -> None:
    """Configure the global logger.

    Parameters
    ----------
    level:
        Minimum log level to display (e.g. ``"DEBUG"``, ``"INFO"``, ``"WARNING"``).
    file_path:
        If provided, log messages will also be appended to this file.
    """
    # Remove any existing handlers
    logger.remove()
    # Console handler
    logger.add(sys.stderr, level=level.upper(), format=DEFAULT_FORMAT)
    # Optional file handler
    if file_path is not None:
        logger.add(file_path, level=level.upper(), format=DEFAULT_FORMAT, rotation="10 MB", retention="10 days")