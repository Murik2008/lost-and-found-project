"""Pure-NumPy similarity primitives.

These functions are deterministic and have no external dependencies beyond
NumPy. They are safe to call from any context (sync, async, or in a worker
process) and have no provider lock-in.
"""

from __future__ import annotations

import numpy as np

from ai.schemas import MatchResult


def cosine(a: np.ndarray, b: np.ndarray) -> float:
    """Cosine similarity between two 1-D vectors.

    For pre-normalized vectors (which `embed` always returns) this is just
    the dot product, but we recompute the norm to be defensive against
    callers passing in raw vectors.
    """
    a = np.asarray(a, dtype=np.float32).ravel()
    b = np.asarray(b, dtype=np.float32).ravel()
    if a.shape != b.shape:
        raise ValueError(f"Shape mismatch: {a.shape} vs {b.shape}")
    na = float(np.linalg.norm(a))
    nb = float(np.linalg.norm(b))
    if na == 0.0 or nb == 0.0:
        return 0.0
    return float(np.dot(a, b) / (na * nb))


def top_k(
    query: np.ndarray,
    candidates: list[np.ndarray] | np.ndarray,
    k: int = 3,
) -> list[MatchResult]:
    """Return the indices and scores of the k most similar candidates.

    Parameters
    ----------
    query : np.ndarray
        A 1-D vector.
    candidates : list[np.ndarray] | np.ndarray
        Either a list of 1-D vectors, or a 2-D matrix of shape (N, dim).
    k : int
        Maximum number of matches to return. If fewer candidates exist,
        all of them are returned.

    Returns
    -------
    list[MatchResult]
        Sorted descending by score. `candidate_id` is the row index into
        `candidates`. `reason` is empty here; the SE layer is welcome to
        populate it (e.g. by joining colour matches between query and match).
    """
    if k <= 0:
        raise ValueError("k must be positive")
    if isinstance(candidates, list):
        if not candidates:
            return []
        mat = np.stack([np.asarray(c, dtype=np.float32).ravel() for c in candidates])
    else:
        mat = np.asarray(candidates, dtype=np.float32)
        if mat.ndim != 2:
            raise ValueError("candidates matrix must be 2-D")
        if mat.shape[0] == 0:
            return []

    q = np.asarray(query, dtype=np.float32).ravel()
    if q.shape[0] != mat.shape[1]:
        raise ValueError(
            f"Query dim {q.shape[0]} does not match candidate dim {mat.shape[1]}"
        )

    # Assume vectors are already unit-normalized (our `embed` guarantees this);
    # if not, normalize defensively.
    q_norm = float(np.linalg.norm(q))
    if q_norm == 0.0:
        return []
    q = q / q_norm
    mat_norms = np.linalg.norm(mat, axis=1, keepdims=True)
    mat_norms[mat_norms == 0.0] = 1.0
    mat = mat / mat_norms

    scores = mat @ q  # (N,)
    # argsort descending; clip k to available rows
    k = min(k, scores.shape[0])
    idx = np.argpartition(-scores, kth=k - 1)[:k]
    idx = idx[np.argsort(-scores[idx])]
    return [
        MatchResult(candidate_id=int(i), score=float(scores[i]))
        for i in idx
    ]
