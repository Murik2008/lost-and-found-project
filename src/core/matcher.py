# business logic

from __future__ import annotations
import logging

import numpy as np
from ai import top_k

logger = logging.getLogger(__name__)

def compute_matches(
        query_embedding: np.ndarray,
        candidate_embeddings: list[np.ndarray],
        candidate_ids: list[str],
        k: int = 3 ) -> list[tuple[str, float, str]]:
    if not candidate_embeddings:
        return []

    matches = top_k(query_embedding, candidate_embeddings, k)
    logger.info("compute_matches: %d candidates → %d matches",
                len(candidate_embeddings), len(matches))

    return [(candidate_ids[m.candidate_id], m.score, m.reason) for m in matches]
