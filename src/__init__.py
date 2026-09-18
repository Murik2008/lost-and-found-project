"""
Smart Lost & Found — src package.

"""

from __future__ import annotations

from src.config import Settings, settings
from src.logging_setup import configure_logging, get_logger
from src.models import Item, ItemStatus, ItemType, MatchRecord
from validation import (
    ALLOWED_FORMATS,
    ALLOWED_SUFFIXES,
    MAX_TEXT_LEN,
    ValidationError,
    validate_image_bytes,
    validate_image_file,
    validate_user_text,
)

__version__ = "1.0.0"

__all__ = [
    "ALLOWED_FORMATS",
    "ALLOWED_SUFFIXES",
    "Item",
    "ItemStatus",
    "ItemType",
    "MAX_TEXT_LEN",
    "MatchRecord",
    "Settings",
    "ValidationError",
    "app",
    "configure_logging",
    "create_app",
    "get_logger",
    "settings",
    "validate_image_bytes",
    "validate_image_file",
    "validate_user_text",
]


def __getattr__(name: str):
    if name in ("create_app", "app"):
        from src import api as _api

        return getattr(_api, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
