"""Pydantic schemas and JSON schemas for Topic 1.

The Pydantic models below are the structured types the AI module emits and
the SE layer consumes. The companion JSON schema is the contract we pass to
the VLM so it returns structured data we can validate.

Students may extend either with extra fields if they want, but the required
fields below must remain.
"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict, Field, field_validator


# -- JSON schema we ask the VLM to honour ------------------------------------

ITEM_DESCRIPTION_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "object_class": {"type": "string"},
        "colors": {"type": "array", "items": {"type": "string"}},
        "brand": {"type": ["string", "null"]},
        "distinguishing_marks": {"type": "array", "items": {"type": "string"}},
        "location_hints": {"type": "array", "items": {"type": "string"}},
        "confidence": {"type": "number", "minimum": 0.0, "maximum": 1.0},
    },
    "required": ["object_class", "colors", "confidence"],
    "additionalProperties": False,
}


class ItemDescription(BaseModel):
    """Structured description returned by the VLM."""

    # `extra="forbid"` rejects unknown fields, matching the JSON schema's
    # `additionalProperties: False`. Students get a loud error if they
    # accidentally drift the contract.
    model_config = ConfigDict(extra="forbid")

    object_class: str
    colors: list[str]
    confidence: float = Field(ge=0.0, le=1.0)
    brand: str | None = None
    distinguishing_marks: list[str] = Field(default_factory=list)
    location_hints: list[str] = Field(default_factory=list)

    @field_validator("object_class")
    @classmethod
    def _object_class_nonempty(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("object_class must be non-empty")
        return v

    def to_search_text(self) -> str:
        """Flatten to a single string suitable for embedding.

        The order matters less than coverage: we want the embedding to capture
        every signal the VLM extracted.
        """
        parts: list[str] = [self.object_class]
        if self.colors:
            parts.append("colors: " + ", ".join(self.colors))
        if self.brand:
            parts.append(f"brand: {self.brand}")
        if self.distinguishing_marks:
            parts.append("marks: " + ", ".join(self.distinguishing_marks))
        if self.location_hints:
            parts.append("location: " + ", ".join(self.location_hints))
        return " | ".join(parts)

    def to_dict(self) -> dict[str, Any]:
        return self.model_dump()


class MatchResult(BaseModel):
    """A single match emitted by the similarity layer."""

    model_config = ConfigDict(extra="forbid")

    candidate_id: int
    score: float
    reason: str = ""
