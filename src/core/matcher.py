from typing import Any, Dict, List
from src import AIService


class Matcher:
    """Core matching engine responsible for scoring lost and found items."""

    def __init__(self, ai_service: AIService):
        self.ai_service = ai_service

    def find_matches(
        self,
        target_item: Dict[str, Any],
        candidates: List[Dict[str, Any]],
        k: int = 5,
    ) -> List[Dict[str, Any]]:
        """Finds top-K candidate matches for a target item based on embedding

        similarity.
        """
        if not candidates:
            return []

        # Obtain embedding for the target item
        target_emb = target_item.get("embedding")
        if not target_emb:
            description = target_item.get("description", "")
            if not description:
                return []
            target_emb = self.ai_service.embed(description)

        # Process candidate embeddings
        candidate_embs = []
        valid_candidates = []

        for candidate in candidates:
            emb = candidate.get("embedding")
            if not emb:
                cand_desc = candidate.get("description", "")
                if not cand_desc:
                    continue
                emb = self.ai_service.embed(cand_desc)

            candidate_embs.append(emb)
            valid_candidates.append(candidate)

        if not valid_candidates:
            return []

        # Get top K matching indices
        top_indices = self.ai_service.top_k(target_emb, candidate_embs, k)

        # Build scored results
        results = []
        for idx in top_indices:
            cand = valid_candidates[idx]
            score = self.ai_service.cosine_similarity(
                target_emb, candidate_embs[idx]
            )
            results.append({"item": cand, "score": score})

        return results