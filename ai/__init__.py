"""
AI module for Topic 1 — Smart Lost & Found.

This package is provided complete; do not modify its public interface.
You may read the source freely.

Public surface
--------------
describe_item(image_path, user_text, *, vlm=None) -> ItemDescription
    Use a vision-language model to produce a structured description of an item.

embed(text, *, embedder=None) -> np.ndarray
    Return a unit-normalized embedding vector for the given text.

cosine(a, b) -> float
    Cosine similarity between two 1-D vectors.

top_k(query_vec, candidates, k=3) -> list[MatchResult]
    Top-k most similar candidates, each as a MatchResult(candidate_id, score, reason).

ItemDescription, MatchResult — pydantic models defined in ai.schemas.
"""

from ai.schemas import ItemDescription, MatchResult
from ai.vlm import describe_item
from ai.embedding import embed
from ai.similarity import cosine, top_k

__all__ = [
    "ItemDescription",
    "MatchResult",
    "describe_item",
    "embed",
    "cosine",
    "top_k",
]
