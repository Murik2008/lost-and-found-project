"""Async orchestration: batch describe/embed/match with concurrency.

`ai.describe_item` and `ai.embed` (and therefore `AIService`) are synchronous,
blocking calls — a real VLM/embedding provider call is a network round trip.
To process a batch of items without waiting on them one at a time, each
blocking call is pushed to a worker thread via `asyncio.to_thread`, and the
resulting coroutines are run concurrently with `asyncio.gather`.

A semaphore caps how many provider calls are in flight at once, so a batch
of 200 items doesn't try to open 200 simultaneous connections to the
provider (and blow through rate limits, which just triggers `AIService`'s
retry logic in a thundering herd).
"""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass
from typing import Sequence

from ai.schemas import ItemDescription, MatchResult
from src.core.matcher import Candidate, Matcher
from src.services.ai_service import AIService

logger = logging.getLogger(__name__)


@dataclass
class ItemInput:
    """One item waiting to be processed: an image + free text + an id the
    caller assigns (e.g. a DB primary key or a temp id for a not-yet-saved
    row)."""

    item_id: str
    image_path: str
    user_text: str


@dataclass
class ProcessedItem:
    """Result of running one `ItemInput` through describe + embed."""

    item_id: str
    description: ItemDescription | None
    candidate: Candidate | None
    error: str | None = None

    @property
    def ok(self) -> bool:
        return self.error is None


class Pipeline:
    """Batch processing on top of `AIService` and `Matcher`.

    Parameters
    ----------
    ai_service:
        Shared `AIService` instance. Sharing one instance across a batch is
        what makes the embedding cache actually useful across the batch.
    matcher:
        Shared `Matcher` instance used for the scoring step.
    max_concurrency:
        Max number of describe/embed calls in flight at once.
    """

    def __init__(
        self,
        ai_service: AIService,
        matcher: Matcher | None = None,
        *,
        max_concurrency: int = 8,
    ) -> None:
        self._ai_service = ai_service
        self._matcher = matcher or Matcher(ai_service=ai_service)
        self._semaphore = asyncio.Semaphore(max_concurrency)

    async def _process_one(self, item: ItemInput) -> ProcessedItem:
        async with self._semaphore:
            try:
                description, vector = await asyncio.to_thread(
                    self._ai_service.describe_and_embed, item.image_path, item.user_text
                )
            except Exception as exc:  # noqa: BLE001 - surfaced per-item, not raised
                logger.error("pipeline: item %s failed: %s", item.item_id, exc)
                return ProcessedItem(
                    item_id=item.item_id, description=None, candidate=None, error=str(exc)
                )
            candidate = Candidate(item_id=item.item_id, embedding=vector, description=description)
            return ProcessedItem(item_id=item.item_id, description=description, candidate=candidate)

    async def process_batch(self, items: Sequence[ItemInput]) -> list[ProcessedItem]:
        """Describe + embed every item in `items` concurrently.

        Failures are isolated per-item (one bad image doesn't sink the whole
        batch): a failed item comes back as a `ProcessedItem` with `ok=False`
        and `error` set, rather than raising and losing the rest of the batch.
        """
        if not items:
            return []
        logger.info("pipeline: processing batch of %d items (max_concurrency=%d)",
                    len(items), self._semaphore._value)
        results = await asyncio.gather(*(self._process_one(item) for item in items))
        ok = sum(1 for r in results if r.ok)
        logger.info("pipeline: batch done, %d/%d succeeded", ok, len(results))
        return list(results)

    async def match_batch(
        self,
        lost_items: Sequence[ItemInput],
        found_pool: Sequence[Candidate],
        k: int = 3,
    ) -> dict[str, list[MatchResult]]:
        """Describe + embed a batch of lost items concurrently, then score
        each one (synchronously, in-process — scoring is pure NumPy and
        fast enough not to need its own thread) against `found_pool`.

        Returns a dict keyed by `item_id` -> its top-k matches. Items whose
        AI processing failed are simply omitted from the result.
        """
        processed = await self.process_batch(lost_items)
        matches: dict[str, list[MatchResult]] = {}
        for result in processed:
            if not result.ok or result.candidate is None:
                continue
            matches[result.item_id] = self._matcher.match(result.candidate, found_pool, k=k)
        return matches

    async def embed_pool_concurrently(self, items: Sequence[ItemInput]) -> list[Candidate]:
        """Convenience: process a batch and return only the successful
        `Candidate`s, e.g. to build a found-item pool concurrently before
        matching lost items against it."""
        processed = await self.process_batch(items)
        return [r.candidate for r in processed if r.ok and r.candidate is not None]
