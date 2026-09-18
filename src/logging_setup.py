"""Central logging setup. Level is driven by LOG_LEVEL env via settings."""

from __future__ import annotations

import logging
import sys


_configured = False


def configure_logging(level: str = "INFO") -> logging.Logger:
    global _configured

    numeric = getattr(logging, level.upper(), logging.INFO)
    root = logging.getLogger()
    root.setLevel(numeric)

    if not _configured:
        handler = logging.StreamHandler(sys.stdout)
        handler.setLevel(numeric)

        fmt = logging.Formatter(
            "%(asctime)s | %(levelname)-7s | %(name)s | %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )

        handler.setFormatter(fmt)
        root.addHandler(handler)
        _configured = True

    return root


def get_logger(name: str) -> logging.Logger:
    return logging.getLogger(name)
