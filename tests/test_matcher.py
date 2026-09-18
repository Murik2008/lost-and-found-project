"""Tests for src/core/matcher.py."""

from __future__ import annotations

import numpy as np
import pytest

from ai.schemas import ItemDescription
from src.core.matcher import Candidate, Matcher, build_candidate_pool
from src.services.ai_service import AIService


def _vec(*xs) -> np.ndarray:
    return np.array(xs, dtype=np.float32)


def test_match_returns_sorted_matches():
    query = Candidate(item_id="lost-1", embedding=_vec(1.0, 0.0))
    pool = [
        Candidate(item_id="found-a", embedding=_vec(0.0, 1.0)),
        Candidate(item_id="found-b", embedding=_vec(1.0, 0.0)),
        Candidate(item_id="found-c", embedding=_vec(0.7, 0.7)),
    ]
    matcher = Matcher()
    results = matcher.match(query, pool, k=3)
    assert [r.candidate_id for r in results] == [1, 2, 0]
    assert results[0].score > results[1].score > results[2].score


def test_match_with_ids_maps_back_to_item_ids():
    query = Candidate(item_id="lost-1", embedding=_vec(1.0, 0.0))
    pool = [
        Candidate(item_id="found-a", embedding=_vec(0.0, 1.0)),
        Candidate(item_id="found-b", embedding=_vec(1.0, 0.0)),
    ]
    matcher = Matcher()
    results = matcher.match_with_ids(query, pool, k=2)
    ids = [r[0] for r in results]
    assert ids[0] == "found-b"
    assert "found-a" in ids


def test_match_empty_pool_returns_empty():
    query = Candidate(item_id="lost-1", embedding=_vec(1.0, 0.0))
    matcher = Matcher()
    assert matcher.match(query, [], k=3) == []


def test_match_reason_includes_shared_object_class_and_color():
    lost_desc = ItemDescription(object_class="umbrella", colors=["black"], confidence=0.9)
    found_desc = ItemDescription(object_class="umbrella", colors=["black", "grey"], confidence=0.8)
    query = Candidate(item_id="lost-1", embedding=_vec(1.0, 0.0), description=lost_desc)
    pool = [Candidate(item_id="found-a", embedding=_vec(1.0, 0.0), description=found_desc)]
    matcher = Matcher()
    results = matcher.match(query, pool, k=1)
    assert "object: umbrella" in results[0].reason
    assert "colors: black" in results[0].reason


def test_match_reason_empty_when_no_descriptions():
    query = Candidate(item_id="lost-1", embedding=_vec(1.0, 0.0))
    pool = [Candidate(item_id="found-a", embedding=_vec(1.0, 0.0))]
    matcher = Matcher()
    results = matcher.match(query, pool, k=1)
    assert results[0].reason == ""


def test_pairwise_score():
    a = Candidate(item_id="a", embedding=_vec(1.0, 0.0))
    b = Candidate(item_id="b", embedding=_vec(1.0, 0.0))
    matcher = Matcher()
    assert matcher.pairwise_score(a, b) == pytest.approx(1.0)


def test_build_candidate_pool():
    desc = ItemDescription(object_class="keys", colors=["silver"], confidence=0.7)
    pool = build_candidate_pool([("id-1", _vec(1.0, 0.0), desc)])
    assert len(pool) == 1
    assert pool[0].item_id == "id-1"
    assert pool[0].description.object_class == "keys"


def test_match_new_item_requires_ai_service():
    matcher = Matcher()  # no ai_service
    with pytest.raises(RuntimeError):
        matcher.match_new_item("some.png", "text", "lost-1", pool=[], k=3)


def test_match_new_item_uses_ai_service(fake_vlm, fake_embedder, sample_image):
    service = AIService(vlm=fake_vlm, embedder=fake_embedder)
    matcher = Matcher(ai_service=service)
    # Build a found pool via the same service so vectors are comparable.
    found_desc, found_vec = service.describe_and_embed(sample_image, "found near door")
    pool = [Candidate(item_id="found-1", embedding=found_vec, description=found_desc)]

    results = matcher.match_new_item(sample_image, "lost near door", "lost-1", pool, k=1)
    assert len(results) == 1
    item_id, score, reason = results[0]
    assert item_id == "found-1"
    assert score == pytest.approx(1.0, abs=1e-4)  # same image/text -> cached identical vector
