from typing import Any

from src.services.ai_service import AIService


class Matcher:
    def __init__(self, ai_service: AIService):
        self.ai_service = ai_service

    def _embedding_of(self, item: dict[str, Any]) -> list[float] | None:
        emb = item.get("embedding")
        if emb:
            return list(emb)
        description = item.get("description") or ""
        if not description:
            return None
        return self.ai_service.embed(description)

    def find_matches(
        self,
        target_item: dict[str, Any],
        candidates: list[dict[str, Any]],
        k: int = 5,
    ) -> list[dict[str, Any]]:
        if not candidates:
            return []
        target_emb = self._embedding_of(target_item)
        if not target_emb:
            return []

        candidate_embs: list[list[float]] = []
        valid: list[dict[str, Any]] = []
        for candidate in candidates:
            emb = self._embedding_of(candidate)
            if not emb:
                continue
            candidate_embs.append(emb)
            valid.append(candidate)

        if not valid:
            return []

        top_indices = self.ai_service.top_k(target_emb, candidate_embs, k)
        results = []
        for idx in top_indices:
            score = self.ai_service.cosine_similarity(target_emb, candidate_embs[idx])
            results.append({"item": valid[idx], "score": score})
        return results
