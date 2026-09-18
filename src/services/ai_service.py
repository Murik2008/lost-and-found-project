"""Wrapper around the provided `ai` package.

Everything the rest of the application needs from the AI layer goes through
this module. Nobody else is allowed to import `ai.providers.*` or a provider
SDK (OpenAI/Gemini/Anthropic) directly — see topic-1-lost-and-found/TOPIC.md,
"The contract (do not break)".

This module adds the three things the raw `ai` package intentionally leaves
out:

1. **Retries** — every call to `ai.describe_item` / `ai.embed` is wrapped in
   an exponential backoff retry (via `tenacity`), because provider calls are
   the one part of the system that can fail transiently (rate limits,
   timeouts, flaky network).
2. **Caching** — embedding the same description text twice is wasted work
   (and wasted money against a paid API), so repeated calls with the same
   text hit an in-memory cache instead of calling the provider again.
3. **Logging** — every call in and out is logged at DEBUG/INFO, and failures
   (including each retry attempt) are logged at WARNING/ERROR, so failures
   are traceable without needing a debugger.
"""

from __future__ import annotations

import logging
import threading
from typing import Optional

import numpy as np
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
    before_sleep_log,
)

from ai import describe_item as _ai_describe_item
from ai import embed as _ai_embed
from ai.providers.base import EmbeddingProvider, ProviderError, VLMProvider
from ai.schemas import ItemDescription

logger = logging.getLogger(__name__)


# Retryable failures: transient provider errors. Programmer errors (bad
# input, e.g. an empty string passed to embed) raise ValueError and are
# NOT retried, since retrying won't fix them.
_RETRYABLE_EXCEPTIONS = (ProviderError,)


def _retry_policy(max_attempts: int):
    return retry(
        retry=retry_if_exception_type(_RETRYABLE_EXCEPTIONS),
        stop=stop_after_attempt(max_attempts),
        wait=wait_exponential(multiplier=0.5, min=0.5, max=8),
        before_sleep=before_sleep_log(logger, logging.WARNING),
        reraise=True,
    )


class EmbeddingCache:
    """Thread-safe in-memory cache: text -> embedding vector.

    A simple dict is enough here (per the TOPIC.md requirement: "re-embedding
    the same description within a session must hit a cache"). Bounded with a
    max size + FIFO eviction so a long-running process can't grow unbounded.
    """

    def __init__(self, max_size: int = 2048) -> None:
        self._max_size = max_size
        self._store: dict[str, np.ndarray] = {}
        self._order: list[str] = []
        self._lock = threading.Lock()
        self.hits = 0
        self.misses = 0

    @staticmethod
    def _key(text: str) -> str:
        return text.strip()

    def get(self, text: str) -> Optional[np.ndarray]:
        key = self._key(text)
        with self._lock:
            vec = self._store.get(key)
            if vec is not None:
                self.hits += 1
                return vec.copy()
            self.misses += 1
            return None

    def set(self, text: str, vector: np.ndarray) -> None:
        key = self._key(text)
        with self._lock:
            if key in self._store:
                self._store[key] = vector.copy()
                return
            if len(self._order) >= self._max_size:
                oldest = self._order.pop(0)
                self._store.pop(oldest, None)
            self._store[key] = vector.copy()
            self._order.append(key)

    def clear(self) -> None:
        with self._lock:
            self._store.clear()
            self._order.clear()
            self.hits = 0
            self.misses = 0

    def __len__(self) -> int:
        return len(self._store)


class AIService:
    """Retry + cache + logging wrapper around `ai.describe_item` and `ai.embed`.

    Parameters
    ----------
    vlm, embedder:
        Optional provider overrides, forwarded straight to `ai.describe_item`
        / `ai.embed`. Used by tests (with the `FakeVLM` / `FakeEmbedder` from
        tests/conftest.py) and to pin a specific provider instance in
        production without touching environment variables.
    max_retries:
        Max attempts (including the first) for each provider call.
    cache:
        Optional pre-built `EmbeddingCache`, so callers can share one cache
        across multiple `AIService` instances or across a whole batch job.
    """

    def __init__(
        self,
        *,
        vlm: VLMProvider | None = None,
        embedder: EmbeddingProvider | None = None,
        max_retries: int = 3,
        cache: EmbeddingCache | None = None,
        cache_size: int = 2048,
    ) -> None:
        self._vlm = vlm
        self._embedder = embedder
        self._max_retries = max_retries
        self._cache = cache if cache is not None else EmbeddingCache(max_size=cache_size)

    @property
    def cache(self) -> EmbeddingCache:
        return self._cache

    # -- describe_item -------------------------------------------------

    def describe_item(self, image_path: str, user_text: str) -> ItemDescription:
        """Describe an item from an image + free text. Retried on transient
        provider failures. Never cached: images are (almost) always unique,
        so caching descriptions by image path would be a correctness trap if
        a path were ever reused for a different photo."""

        @_retry_policy(self._max_retries)
        def _call() -> ItemDescription:
            logger.debug("describe_item: calling VLM for %s", image_path)
            result = _ai_describe_item(image_path, user_text, vlm=self._vlm)
            logger.info(
                "describe_item: %s -> %s (confidence=%.2f)",
                image_path,
                result.object_class,
                result.confidence,
            )
            return result

        try:
            return _call()
        except ProviderError:
            logger.error("describe_item: giving up on %s after retries", image_path)
            raise

    # -- embed -----------------------------------------------------------

    def embed_text(self, text: str) -> np.ndarray:
        """Embed `text`, serving from cache when the same text was embedded
        before. Cache lookups happen on the *stripped* text so trivial
        whitespace differences still hit the cache."""

        cached = self._cache.get(text)
        if cached is not None:
            logger.debug("embed_text: cache hit (%d chars)", len(text))
            return cached

        @_retry_policy(self._max_retries)
        def _call() -> np.ndarray:
            logger.debug("embed_text: cache miss, calling embedder (%d chars)", len(text))
            vec = _ai_embed(text, embedder=self._embedder)
            logger.info("embed_text: embedded %d chars -> dim %d", len(text), vec.shape[0])
            return vec

        try:
            vec = _call()
        except ProviderError:
            logger.error("embed_text: giving up after retries")
            raise
        self._cache.set(text, vec)
        return vec

    def embed_description(self, description: ItemDescription) -> np.ndarray:
        """Convenience: flatten an `ItemDescription` and embed it."""
        return self.embed_text(description.to_search_text())

    # -- combined --------------------------------------------------------

    def describe_and_embed(self, image_path: str, user_text: str) -> tuple[ItemDescription, np.ndarray]:
        """Describe an item, then embed its flattened description.

        This is the operation the concurrency pipeline batches over.
        """
        description = self.describe_item(image_path, user_text)
        vector = self.embed_description(description)
        return description, vector
