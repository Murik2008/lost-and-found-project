"""Tests for caching logic if present, or simple in-memory cache verification."""

import numpy as np


def test_simple_cache_behavior():
    cache = {}

    def get_embedding_with_cache(text: str):
        if text in cache:
            return cache[text], True  # (val, hit)
        emb = [len(text) * 0.1]
        cache[text] = emb
        return emb, False

    #first call - cache miss
    emb1, hit1 = get_embedding_with_cache("umbrella")
    assert not hit1
    assert emb1 == [0.8]

    #second call - cache hit
    emb2, hit2 = get_embedding_with_cache("umbrella")
    assert hit2
    assert emb1 == emb2


def test_ai_service_embed_cache(monkeypatch):
    from src.services.ai_service import AIService

    calls = []

    def fake_embed(text, *, embedder=None):
        calls.append(text)
        return np.array([0.1, 0.2, 0.3], dtype=np.float32)

    monkeypatch.setattr("src.services.ai_service.ai.embed", fake_embed)
    svc = AIService(offline=True)
    svc._embedder = None
    monkeypatch.setattr(svc, "_ensure_providers", lambda: None)
    first = svc.embed("umbrella")
    second = svc.embed("umbrella")
    assert first == second
    assert len(calls) == 1