"""Logging setup helpers for the package."""
from __future__ import annotations

import logging
import sys
from pathlib import Path

_LOG_FMT = "%(asctime)s | %(levelname)-7s | %(name)s | %(message)s"
_DATE_FMT = "%Y-%m-%d %H:%M:%S"


def configure_logging(
    *,
    level: int | str = logging.INFO,
    log_file: str | Path | None = None,
) -> logging.Logger:
    """Configure the package's root logger.

    Idempotent: repeated calls reconfigure the handlers attached to the
    ``phot7ds`` logger (file handler swap, level change) rather than stacking
    duplicates.
    """
    logger = logging.getLogger("phot7ds")
    logger.setLevel(level)

    for h in list(logger.handlers):
        logger.removeHandler(h)

    fmt = logging.Formatter(_LOG_FMT, datefmt=_DATE_FMT)

    sh = logging.StreamHandler(sys.stderr)
    sh.setFormatter(fmt)
    logger.addHandler(sh)

    if log_file is not None:
        Path(log_file).parent.mkdir(parents=True, exist_ok=True)
        fh = logging.FileHandler(log_file, mode="a", encoding="utf-8")
        fh.setFormatter(fmt)
        logger.addHandler(fh)

    logger.propagate = False
    return logger


__all__ = ["configure_logging"]
