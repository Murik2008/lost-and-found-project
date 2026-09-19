"""Tests for caching logic if present, or simple in-memory cache verification."""

import pytest


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