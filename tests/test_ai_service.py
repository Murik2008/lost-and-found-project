"""Tests for src/services/ai_service.py: retries, caching, logging hooks.

Uses the same FakeVLM / FakeEmbedder fixtures as the provided smoke tests
(tests/conftest.py) so nothing here touches the network.
"""

from __future__ import annotations

import numpy as np
import pytest

from ai.providers.base import ProviderError
from src.services.ai_service import AIService, EmbeddingCache


class FlakyEmbedder:
    """Fails the first `fail_times` calls, then succeeds. Used to prove
    the retry decorator actually retries instead of failing immediately."""

    def __init__(self, fail_times: int, dim: int = 8) -> None:
        self.fail_times = fail_times
        self.calls = 0
        self._dim = dim

    @property
    def dimension(self) -> int:
        return self._dim

    def embed(self, text: str) -> np.ndarray:
        self.calls += 1
        if self.calls <= self.fail_times:
            raise ProviderError("simulated transient failure")
        rng = np.random.default_rng(seed=abs(hash(text)) % (2**31))
        v = rng.standard_normal(self._dim).astype(np.float32)
        return v / np.linalg.norm(v)


class AlwaysFailsEmbedder:
    @property
    def dimension(self) -> int:
        return 8

    def embed(self, text: str) -> np.ndarray:
        raise ProviderError("permanently broken")


class FlakyVLM:
    def __init__(self, fail_times: int, payload: dict) -> None:
        self.fail_times = fail_times
        self.calls = 0
        self.payload = payload

    def describe(self, image_path, prompt, *, json_schema=None):
        self.calls += 1
        if self.calls <= self.fail_times:
            raise ProviderError("simulated transient VLM failure")
        import json

        return json.dumps(self.payload)


# --- describe_item ----------------------------------------------------


def test_describe_item_success(fake_vlm, sample_image):
    service = AIService(vlm=fake_vlm)
    result = service.describe_item(sample_image, "lost my umbrella")
    assert result.object_class == "umbrella"


def test_describe_item_retries_then_succeeds(sample_image, fake_vlm):
    flaky = FlakyVLM(fail_times=2, payload=fake_vlm.payload)
    service = AIService(vlm=flaky, max_retries=5)
    result = service.describe_item(sample_image, "x")
    assert result.object_class == "umbrella"
    assert flaky.calls == 3  # 2 failures + 1 success


def test_describe_item_gives_up_after_max_retries(sample_image, fake_vlm):
    flaky = FlakyVLM(fail_times=10, payload=fake_vlm.payload)
    service = AIService(vlm=flaky, max_retries=3)
    with pytest.raises(ProviderError):
        service.describe_item(sample_image, "x")
    assert flaky.calls == 3  # stopped after exactly max_retries attempts


# --- embed_text: retries -----------------------------------------------


def test_embed_text_retries_then_succeeds():
    flaky = FlakyEmbedder(fail_times=2)
    service = AIService(embedder=flaky, max_retries=5)
    vec = service.embed_text("hello world")
    assert vec.shape == (8,)
    assert flaky.calls == 3


def test_embed_text_gives_up_after_max_retries():
    service = AIService(embedder=AlwaysFailsEmbedder(), max_retries=2)
    with pytest.raises(ProviderError):
        service.embed_text("hello")


def test_embed_text_does_not_retry_on_value_error(fake_embedder):
    """Empty text raises ValueError from the ai package itself — that's a
    caller bug, not a transient failure, so it must not be retried/caught."""
    service = AIService(embedder=fake_embedder)
    with pytest.raises(ValueError):
        service.embed_text("   ")


# --- embed_text: caching -------------------------------------------------


def test_embed_text_cache_hit_skips_provider_call(fake_embedder):
    calls = {"n": 0}
    real_embed = fake_embedder.embed

    def counting_embed(text):
        calls["n"] += 1
        return real_embed(text)

    fake_embedder.embed = counting_embed
    service = AIService(embedder=fake_embedder)

    v1 = service.embed_text("same description")
    v2 = service.embed_text("same description")

    assert calls["n"] == 1
    assert np.allclose(v1, v2)
    assert service.cache.hits == 1
    assert service.cache.misses == 1


def test_embed_text_cache_differentiates_by_text(fake_embedder):
    service = AIService(embedder=fake_embedder)
    v1 = service.embed_text("alpha")
    v2 = service.embed_text("beta")
    assert not np.allclose(v1, v2)
    assert service.cache.misses == 2
    assert service.cache.hits == 0


def test_embed_description_uses_search_text(fake_vlm, fake_embedder, sample_image):
    service = AIService(vlm=fake_vlm, embedder=fake_embedder)
    description = service.describe_item(sample_image, "x")
    vec = service.embed_description(description)
    assert vec.shape == (fake_embedder.dimension,)


def test_describe_and_embed_combined(fake_vlm, fake_embedder, sample_image):
    service = AIService(vlm=fake_vlm, embedder=fake_embedder)
    description, vector = service.describe_and_embed(sample_image, "left at the gate")
    assert description.object_class == "umbrella"
    assert vector.shape == (fake_embedder.dimension,)


# --- EmbeddingCache standalone -------------------------------------------


def test_embedding_cache_eviction():
    cache = EmbeddingCache(max_size=2)
    cache.set("a", np.array([1.0], dtype=np.float32))
    cache.set("b", np.array([2.0], dtype=np.float32))
    cache.set("c", np.array([3.0], dtype=np.float32))  # evicts "a"
    assert cache.get("a") is None
    assert cache.get("b") is not None
    assert cache.get("c") is not None
    assert len(cache) == 2


def test_embedding_cache_strips_whitespace_for_key():
    cache = EmbeddingCache()
    cache.set("hello", np.array([1.0], dtype=np.float32))
    assert cache.get("  hello  ") is not None
