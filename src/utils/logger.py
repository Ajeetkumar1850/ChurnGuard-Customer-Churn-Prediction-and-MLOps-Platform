"""Logging utilities for the customer churn platform."""

from __future__ import annotations

import logging


def get_logger(name: str) -> logging.Logger:
    """Create or retrieve a configured logger.

    Args:
        name: Name of the logger.

    Returns:
        A configured logger instance.
    """
    logger = logging.getLogger(name)

    if not logger.handlers:
        logger.setLevel(logging.INFO)

        handler = logging.StreamHandler()
        formatter = logging.Formatter(
            fmt="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)

    logger.propagate = False
    return logger