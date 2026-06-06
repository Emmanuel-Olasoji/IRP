"""Logging helpers for consistent, research-friendly experiment logs."""

import logging
from typing import Optional

from config.settings import LOGGING_LEVEL


def setup_logging(level: Optional[str] = None) -> None:
    """Configure root logging for command-line scripts.

    Args:
        level: Optional logging level override.
    """
    chosen_level = (level or LOGGING_LEVEL).upper()
    logging.basicConfig(
        level=getattr(logging, chosen_level, logging.INFO),
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )


def get_logger(name: str) -> logging.Logger:
    """Create a module-specific logger.

    Args:
        name: Logger namespace name.

    Returns:
        Configured logger instance.
    """
    return logging.getLogger(name)
