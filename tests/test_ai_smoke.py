"""Smoke tests for the provided AI module.

These tests exercise the AI module's public interface end-to-end using fake
providers that do NOT touch the network. Students MUST NOT delete or weaken
these tests; they are part of the grading contract.

Add your own tests in tests/test_*.py — these stay as-is.
"""

from __future__ import annotations

import json

import numpy as np
import pytest

from ai import describe_item, embed, cosine, top_k, ItemDescription, MatchResult
from ai.providers.base import ProviderError
from ai.vlm import _parse_json
from ai.schemas import ITEM_DESCRIPTION_SCHEMA


# --- describe_item ---------------------------------------------------------


def test_describe_item_returns_dataclass(fake_vlm, sample_image):
    result = describe_item(sample_image, "lost my umbrella", vlm=fake_vlm)
    assert isinstance(result, ItemDescription)
    assert result.object_class == "umbrella"
    assert result.colors == ["black"]
    assert result.brand == "Fulton"
    assert 0.0 <= result.confidence <= 1.0


def test_describe_item_empty_user_text_is_allowed(fake_vlm, sample_image):
    """The user might not provide free-text. The VLM is still expected to work."""
    result = describe_item(sample_image, "", vlm=fake_vlm)
    assert isinstance(result, ItemDescription)


def test_describe_item_passes_image_path_to_provider(fake_vlm, sample_image):
    describe_item(sample_image, "anything", vlm=fake_vlm)
    assert len(fake_vlm.calls) == 1
    assert fake_vlm.calls[0][0] == sample_image


def test_describe_item_includes_user_text_in_prompt(fake_vlm, sample_image):
    describe_item(sample_image, "left it on the bench", vlm=fake_vlm)
    prompt = fake_vlm.calls[0][1]
    assert "left it on the bench" in prompt


def test_describe_item_rejects_invalid_confidence(fake_vlm, sample_image):
    fake_vlm.payload = {**fake_vlm.payload, "confidence": 1.5}
    with pytest.raises(ProviderError):
        describe_item(sample_image, "x", vlm=fake_vlm)


def test_describe_item_rejects_missing_required_field(fake_vlm, sample_image):
    bad = dict(fake_vlm.payload)
    del bad["object_class"]
    fake_vlm.payload = bad
    with pytest.raises(ProviderError):
        describe_item(sample_image, "x", vlm=fake_vlm)


def test_describe_item_to_search_text_is_non_empty(fake_vlm, sample_image):
    result = describe_item(sample_image, "x", vlm=fake_vlm)
    text = result.to_search_text()
    assert "umbrella" in text
    assert "black" in text


# --- JSON parsing forgiveness ---------------------------------------------


def test_parse_json_strips_markdown_fences():
    raw = '```json\n{"a": 1}\n```'
    assert _parse_json(raw) == {"a": 1}


def test_parse_json_strips_bare_fences():
    raw = '```\n{"a": 1}\n```'
    assert _parse_json(raw) == {"a": 1}


def test_parse_json_rejects_non_object():
    with pytest.raises(ProviderError):
        _parse_json("[1, 2, 3]")


def test_parse_json_rejects_garbage():
    with pytest.raises(ProviderError):
        _parse_json("not json at all")


# --- embed -----------------------------------------------------------------


def test_embed_returns_unit_vector(fake_embedder):
    v = embed("hello world", embedder=fake_embedder)
    assert v.shape == (fake_embedder.dimension,)
    assert v.dtype == np.float32
    assert abs(float(np.linalg.norm(v)) - 1.0) < 1e-5


def test_embed_is_deterministic(fake_embedder):
    a = embed("same text", embedder=fake_embedder)
    b = embed("same text", embedder=fake_embedder)
    assert np.allclose(a, b)


def test_embed_different_text_different_vector(fake_embedder):
    a = embed("alpha", embedder=fake_embedder)
    b = embed("beta", embedder=fake_embedder)
    assert not np.allclose(a, b)


def test_embed_rejects_empty(fake_embedder):
    with pytest.raises(ValueError):
        embed("", embedder=fake_embedder)
    with pytest.raises(ValueError):
        embed("   ", embedder=fake_embedder)


# --- cosine ---------------------------------------------------------------


def test_cosine_identity_is_one():
    v = np.array([1.0, 0.0, 0.0])
    assert abs(cosine(v, v) - 1.0) < 1e-6


def test_cosine_orthogonal_is_zero():
    a = np.array([1.0, 0.0])
    b = np.array([0.0, 1.0])
    assert abs(cosine(a, b)) < 1e-6


def test_cosine_zero_vector_returns_zero():
    a = np.array([0.0, 0.0])
    b = np.array([1.0, 0.0])
    assert cosine(a, b) == 0.0


def test_cosine_shape_mismatch_raises():
    with pytest.raises(ValueError):
        cosine(np.array([1.0, 0.0]), np.array([1.0, 0.0, 0.0]))


# --- top_k ----------------------------------------------------------------


def test_top_k_returns_sorted_descending():
    q = np.array([1.0, 0.0])
    cands = [
        np.array([0.0, 1.0]),  # 0.0
        np.array([1.0, 0.0]),  # 1.0
        np.array([0.7, 0.7]),  # ~0.707
    ]
    res = top_k(q, cands, k=3)
    assert [r.candidate_id for r in res] == [1, 2, 0]
    assert res[0].score > res[1].score > res[2].score


def test_top_k_clips_k_to_available():
    q = np.array([1.0, 0.0])
    cands = [np.array([1.0, 0.0])]
    res = top_k(q, cands, k=5)
    assert len(res) == 1


def test_top_k_empty_candidates_returns_empty():
    q = np.array([1.0, 0.0])
    assert top_k(q, [], k=3) == []


def test_top_k_returns_match_results():
    q = np.array([1.0, 0.0])
    res = top_k(q, [np.array([1.0, 0.0])], k=1)
    assert isinstance(res[0], MatchResult)
    assert res[0].candidate_id == 0


def test_top_k_k_must_be_positive():
    with pytest.raises(ValueError):
        top_k(np.array([1.0, 0.0]), [np.array([1.0, 0.0])], k=0)


def test_top_k_dimension_mismatch_raises():
    q = np.array([1.0, 0.0])
    with pytest.raises(ValueError):
        top_k(q, [np.array([1.0, 0.0, 0.0])], k=1)


# --- schema sanity --------------------------------------------------------


def test_schema_has_required_fields():
    assert ITEM_DESCRIPTION_SCHEMA["required"] == ["object_class", "colors", "confidence"]


def test_item_description_rejects_extra_fields():
    """Pydantic ConfigDict(extra='forbid') enforces the schema contract."""
    with pytest.raises(Exception):  # pydantic.ValidationError
        ItemDescription(
            object_class="phone", colors=["black"], confidence=0.8,
            totally_unknown_field=42,  # type: ignore[call-arg]
        )
