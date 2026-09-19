"""Business logic: match lost items against found items.

This module never talks to a provider SDK directly and never calls
`ai.describe_item` / `ai.embed` directly either — all AI calls go through
`src.services.ai_service.AIService`, which is what adds retries/caching/
logging around the raw `ai` package. This module only calls the pure-NumPy
primitives (`ai.cosine`, `ai.top_k`) directly, since those have no
provider lock-in and no network dependency (see ai/similarity.py docstring).

`Candidate` is a deliberately minimal, provider-agnostic shape (an id, an
embedding vector, and an optional description) rather than importing a
pydantic model from `src.models` — that module belongs to the API/validation
layer and matching should not have to know about HTTP-facing types. The API
layer is free to adapt its `Item` model into a `Candidate` before calling in
here.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Iterable, Sequence

import numpy as np

from ai import cosine, top_k
from ai.schemas import ItemDescription, MatchResult
from src.services.ai_service import AIService

logger = logging.getLogger(__name__)


@dataclass
class Candidate:
    """One item (lost or found) as seen by the matcher."""

    item_id: str
    embedding: np.ndarray
    description: ItemDescription | None = None


def _shared_terms(a: ItemDescription | None, b: ItemDescription | None) -> str:
    """Build a short human-readable reason string from two descriptions,
    e.g. 'object: umbrella | colors: black'. Best-effort only — if either
    description is missing, returns an empty string.
    """
    if a is None or b is None:
        return ""
    parts: list[str] = []
    if a.object_class and a.object_class.lower() == b.object_class.lower():
        parts.append(f"object: {a.object_class}")
    shared_colors = {c.lower() for c in a.colors} & {c.lower() for c in b.colors}
    if shared_colors:
        parts.append("colors: " + ", ".join(sorted(shared_colors)))
    if a.brand and b.brand and a.brand.lower() == b.brand.lower():
        parts.append(f"brand: {a.brand}")
    return " | ".join(parts)


class Matcher:
    """Compares lost items against a pool of found items (and vice versa).

    Stateless with respect to storage: callers pass in the candidate pool
    (typically fetched from the repository) rather than the matcher owning
    a database connection.
    """

    def __init__(self, ai_service: AIService | None = None) -> None:
        # Only needed by the convenience methods that embed text/images on
        # the fly (`match_new_item`). Pure vector-in matching doesn't need it.
        self._ai_service = ai_service

    def match(
        self,
        query: Candidate,
        pool: Sequence[Candidate],
        k: int = 3,
    ) -> list[MatchResult]:
        """Return the top-`k` items in `pool` most similar to `query`.

        `candidate_id` in the returned `MatchResult`s is the *index into
        `pool`*, per `ai.top_k`'s contract — this method translates that back
        to `pool[i].item_id` in the `reason` field is NOT done here, since
        `MatchResult.candidate_id` is typed as `int` by the AI layer's
        schema. Use `match_with_ids` if you want item ids back directly.
        """
        if not pool:
            return []
        vectors = [c.embedding for c in pool]
        raw = top_k(query.embedding, vectors, k=k)
        enriched = []
        for m in raw:
            reason = _shared_terms(query.description, pool[m.candidate_id].description)
            enriched.append(MatchResult(candidate_id=m.candidate_id, score=m.score, reason=reason))
        logger.info(
            "match: query=%s pool_size=%d -> %d matches (top score=%.3f)",
            query.item_id,
            len(pool),
            len(enriched),
            enriched[0].score if enriched else 0.0,
        )
        return enriched

    def match_with_ids(
        self,
        query: Candidate,
        pool: Sequence[Candidate],
        k: int = 3,
    ) -> list[tuple[str, float, str]]:
        """Same as `match`, but returns `(item_id, score, reason)` tuples
        using the pool's real item ids instead of positional indices —
        the shape the API/CLI layers actually want to serialize."""
        results = self.match(query, pool, k=k)
        return [(pool[m.candidate_id].item_id, m.score, m.reason) for m in results]

    def pairwise_score(self, a: Candidate, b: Candidate) -> float:
        """Cosine similarity between two items' embeddings."""
        return cosine(a.embedding, b.embedding)

    # -- convenience: embed-then-match in one call ------------------------

    def match_new_item(
        self,
        image_path: str,
        user_text: str,
        item_id: str,
        pool: Sequence[Candidate],
        k: int = 3,
    ) -> list[tuple[str, float, str]]:
        """Describe + embed a brand-new item (via `AIService`), then match
        it against `pool`. Requires the matcher to have been constructed
        with an `AIService`.
        """
        if self._ai_service is None:
            raise RuntimeError(
                "Matcher.match_new_item requires an AIService; pass one to "
                "the constructor, or embed the item yourself and call match()."
            )
        description, vector = self._ai_service.describe_and_embed(image_path, user_text)
        query = Candidate(item_id=item_id, embedding=vector, description=description)
        return self.match_with_ids(query, pool, k=k)


def build_candidate_pool(
    items: Iterable[tuple[str, np.ndarray, ItemDescription | None]],
) -> list[Candidate]:
    """Small helper for callers (repository/pipeline) that hold items as
    plain tuples of `(item_id, embedding, description)`."""
    return [Candidate(item_id=i, embedding=v, description=d) for i, v, d in items]
