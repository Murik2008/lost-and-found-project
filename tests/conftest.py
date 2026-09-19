"""Shared pytest fixtures for the AI smoke tests.

The fakes live here so both the AI smoke tests and any student-written
tests can reuse them without monkey-patching modules.
"""

from __future__ import annotations

import json
from typing import Any

import numpy as np
import pytest

from ai.providers.base import VLMProvider, EmbeddingProvider
from src.models import Item, ItemStatus, ItemType, MatchRecord


class FakeVLM(VLMProvider):
    """Returns a fixed JSON response. No network."""

    def __init__(self, payload: dict[str, Any] | None = None) -> None:
        self.payload = payload or {
            "object_class": "umbrella",
            "colors": ["black"],
            "brand": "Fulton",
            "distinguishing_marks": ["bent rib"],
            "location_hints": ["library entrance"],
            "confidence": 0.85,
        }
        self.calls: list[tuple[str, str]] = []

    def describe(
        self,
        image_path: str,
        prompt: str,
        *,
        json_schema: dict | None = None,
    ) -> str:
        self.calls.append((image_path, prompt))
        return json.dumps(self.payload)


class FakeEmbedder(EmbeddingProvider):
    """Deterministic toy embedder: 8-D unit vectors derived from a hash.

    Same input -> same output, different input -> different (but stable) output.
    Used by tests so we don't need network access.
    """

    def __init__(self, dim: int = 8) -> None:
        self._dim = dim

    @property
    def dimension(self) -> int:
        return self._dim

    def embed(self, text: str) -> np.ndarray:
        if not text.strip():
            raise ValueError("Cannot embed empty string.")
        rng = np.random.default_rng(seed=abs(hash(text)) % (2**31))
        v = rng.standard_normal(self._dim).astype(np.float32)
        v /= np.linalg.norm(v)
        return v


@pytest.fixture
def fake_vlm() -> FakeVLM:
    return FakeVLM()


@pytest.fixture
def fake_embedder() -> FakeEmbedder:
    return FakeEmbedder()


@pytest.fixture
def sample_image(tmp_path):
    """A tiny but valid PNG file. Enough to satisfy file-existence and
    extension checks; the FakeVLM ignores the contents."""
    # Minimal 1x1 PNG (89 bytes). Generated once and pasted here.
    png_bytes = bytes.fromhex(
        "89504e470d0a1a0a0000000d49484452000000010000000108020000"
        "00907753de0000000c4944415408d76360000000000004000146a13a"
        "020000000049454e44ae426082"
    )
    p = tmp_path / "tiny.png"
    p.write_bytes(png_bytes)
    return str(p)


@pytest.fixture
def sample_lost_item(sample_image: str) -> Item:
    """Creates sample the lost item for the tests"""
    item = Item(
        item_type=ItemType.LOST,
        user_description="Lost black Fulton umbrella",
        image_path=sample_image,
        description_json={"object_class": "umbrella", "colors": ["black"]},
        embedding=[0.2, 0.6, 0.8, 0.4, 0.1, 0.7, 0.4, 0.3],
        status=ItemStatus.PENDING,
    )
    return item

@pytest.fixture
def sample_found_item(sample_image: str) -> Item:
    """Creates sample found item for the tests"""
    item = Item(
        item_type=ItemType.FOUND,
        user_description="Found black Fulton umbrella",
        image_path=sample_image,
        description_json={"object_class": "umbrella", "colors": ["black"]},
        embedding=[0.9, 0.6, 0.7, 0.5, 0.7, 0.6, 0.1, 0.2],
        status=ItemStatus.PENDING,
    )
    return item

@pytest.fixture
def sample_match_record(sample_lost_item: Item, sample_found_item: Item) -> MatchRecord:
    """Creates sample match record for the tests"""
    return MatchRecord(
        lost_item_id=sample_lost_item.id,
        found_item_id=sample_found_item.id,
        score=0.92, #cosine similarity
        reason="High match of characteristics and text descriptions",
    )