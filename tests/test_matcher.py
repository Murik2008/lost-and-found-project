"""Tests for src.core.matcher using fakes from conftest."""

import pytest
from src.core.matcher import Matcher


class FakeAIService:
    def embed(self, text: str) -> list[float]:
        return [0.1, 0.2, 0.3, 0.4]

    def top_k(self, target_emb: list[float], candidate_embs: list[list[float]], k: int) -> list[int]:
        return list(range(min(k, len(candidate_embs))))

    def cosine_similarity(self, emb1: list[float], emb2: list[float]) -> float:
        return 0.95


def test_matcher_find_matches_with_embeddings():
    ai = FakeAIService()
    matcher = Matcher(ai_service=ai)

    target_item = {
        "id": "lost-1",
        "description": "Black leather wallet",
        "embedding": [0.1, 0.2, 0.3, 0.4],
    }

    candidates = [
        {
            "id": "found-1",
            "description": "Black wallet found near park",
            "embedding": [0.11, 0.21, 0.31, 0.41],
        }
    ]

    results = matcher.find_matches(target_item, candidates, k=3)
    assert len(results) == 1
    assert results[0]["item"]["id"] == "found-1"
    assert results[0]["score"] == 0.95


def test_matcher_empty_candidates():
    ai = FakeAIService()
    matcher = Matcher(ai_service=ai)

    target_item = {"id": "lost-1", "description": "Keys"}
    results = matcher.find_matches(target_item, [], k=5)
    assert results == []