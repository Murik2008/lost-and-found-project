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
