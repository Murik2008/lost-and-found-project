"""High-level embedding helper."""

from __future__ import annotations

import numpy as np

from ai.providers.base import EmbeddingProvider
from ai.providers.factory import get_embedder


def embed(text: str, *, embedder: EmbeddingProvider | None = None) -> np.ndarray:
    """Return a unit-normalized embedding vector for `text`.

    Parameters
    ----------
    text : str
        The text to embed. Must be non-empty after stripping.
    embedder : EmbeddingProvider | None
        Optional override. Useful for tests or to pin a specific provider.

    Returns
    -------
    np.ndarray
        A 1-D float32 vector with unit L2 norm.
    """
    if not text or not text.strip():
        raise ValueError("Cannot embed empty text.")
    embedder = embedder or get_embedder()
    return embedder.embed(text)
