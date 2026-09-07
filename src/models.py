from __future__ import annotations

import uuid
from datetime import datetime
from enum import Enum
from typing import Any

from pydantic import BaseModel, Field


class ItemType(str, Enum):
    LOST = "lost"
    FOUND = "found"


class ItemStatus(str, Enum):
    PENDING = "pending"    # no match found yet
    MATCHED = "matched"    # a match was found
    CLOSED = "closed"      # resolved / closed manually


class Item(BaseModel):
    """A lost or found item — this is what gets saved to PostgreSQL."""

    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    item_type: ItemType
    user_description: str = ""                # free text from the user
    image_path: str                           # where the image is stored on disk
    description_json: dict[str, Any]          # output of ItemDescription.to_dict()
    embedding: list[float]                    # output of ai.embed(), stored as JSON
    status: ItemStatus = ItemStatus.PENDING
    created_at: datetime = Field(default_factory=datetime.utcnow)


class MatchRecord(BaseModel):
    """A match between a lost item and a found item."""

    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    lost_item_id: str
    found_item_id: str
    score: float        # cosine similarity score, between -1 and 1
    reason: str = ""    # optional explanation
    created_at: datetime = Field(default_factory=datetime.utcnow)