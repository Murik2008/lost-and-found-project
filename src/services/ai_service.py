import logging
from functools import lru_cache
from typing import List
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type

import ai

logger = logging.getLogger(__name__)


class AIService:
    """Wrapper service around the provided 'ai' module, offering retry logic,

    caching, and error handling.
    """

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=10),
        reraise=True,
    )
    def describe_item(self, image_path: str) -> str:
        """Generates a text description for an item from its image path.

        Applies retry logic in case of temporary network or API failures.
        """
        logger.info(f"AI: Generating description for image '{image_path}'...")
        try:
            description = ai.describe_item(image_path)
            logger.info("AI: Description generated successfully.")
            return description
        except Exception as e:
            logger.error(
                f"AI: Error describing item from image '{image_path}': {e}"
            )
            raise e

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=1, max=8),
        reraise=True,
    )
    def _fetch_embedding(self, text: str) -> List[float]:
        """Internal helper to fetch text embedding with retry mechanism."""
        return ai.embed(text)

    @lru_cache(maxsize=1024)
    def embed(self, text: str) -> List[float]:
        """Converts text into a vector embedding.

        Results are cached in-memory to avoid redundant API calls for identical
        texts.
        """
        logger.info(
            f"AI: Generating embedding (Cache Miss) for text: '{text[:20]}...'"
        )
        return self._fetch_embedding(text)

    def cosine_similarity(
        self, vec1: List[float], vec2: List[float]
    ) -> float:
        """Calculates cosine similarity between two vector embeddings."""
        return ai.cosine(vec1, vec2)

    def top_k(
        self,
        query_vec: List[float],
        candidate_vecs: List[List[float]],
        k: int,
    ) -> List[int]:
        """Returns the indices of top-K matches based on vector similarity."""
        return ai.top_k(query_vec, candidate_vecs, k)