"""Tests for src/concurrency/pipeline.py."""

from __future__ import annotations

import asyncio
import time

import pytest

from ai.providers.base import EmbeddingProvider, VLMProvider
from src.concurrency.pipeline import ItemInput, Pipeline
from src.core.matcher import Candidate
from src.services.ai_service import AIService


class SlowFakeVLM(VLMProvider):
    """Same payload shape as FakeVLM, but sleeps to simulate a network call
    so we can prove batches run concurrently rather than sequentially."""

    def __init__(self, delay: float = 0.05) -> None:
        self.delay = delay

    def describe(self, image_path, prompt, *, json_schema=None):
        import json
        from pathlib import Path

        time.sleep(self.delay)
        name = Path(image_path).stem
        return json.dumps(
            {
                "object_class": name,
                "colors": ["black"],
                "brand": None,
                "distinguishing_marks": [],
                "location_hints": [],
                "confidence": 0.8,
            }
        )


class SlowFakeEmbedder(EmbeddingProvider):
    def __init__(self, delay: float = 0.05, dim: int = 8) -> None:
        self.delay = delay
        self._dim = dim

    @property
    def dimension(self) -> int:
        return self._dim

    def embed(self, text: str):
        import numpy as np

        time.sleep(self.delay)
        if not text.strip():
            raise ValueError("empty text")
        rng = np.random.default_rng(seed=abs(hash(text)) % (2**31))
        v = rng.standard_normal(self._dim).astype(np.float32)
        return v / np.linalg.norm(v)


class FailingOnceVLM(VLMProvider):
    """Fails for one specific image path, succeeds for all others — used to
    prove per-item failure isolation in a batch."""

    def __init__(self, fail_path: str) -> None:
        self.fail_path = fail_path

    def describe(self, image_path, prompt, *, json_schema=None):
        import json
        from pathlib import Path

        if image_path == self.fail_path:
            raise RuntimeError("boom")
        return json.dumps(
            {
                "object_class": Path(image_path).stem,
                "colors": [],
                "confidence": 0.5,
            }
        )


@pytest.fixture
def images(tmp_path):
    paths = []
    for name in ["umbrella", "backpack", "wallet"]:
        p = tmp_path / f"{name}.png"
        p.write_bytes(b"\x89PNG\r\n\x1a\nfake")
        paths.append(str(p))
    return paths


@pytest.mark.asyncio
async def test_process_batch_runs_concurrently(images):
    service = AIService(vlm=SlowFakeVLM(delay=0.1), embedder=SlowFakeEmbedder(delay=0.1))
    pipeline = Pipeline(service, max_concurrency=len(images))
    items = [ItemInput(item_id=f"item-{i}", image_path=p, user_text="") for i, p in enumerate(images)]

    start = time.monotonic()
    results = await pipeline.process_batch(items)
    elapsed = time.monotonic() - start

    assert len(results) == 3
    assert all(r.ok for r in results)
    # Sequential would take ~3 * (0.1 + 0.1) = 0.6s; concurrent should be well under that.
    assert elapsed < 0.45


@pytest.mark.asyncio
async def test_process_batch_empty_input():
    service = AIService(vlm=SlowFakeVLM(), embedder=SlowFakeEmbedder())
    pipeline = Pipeline(service)
    assert await pipeline.process_batch([]) == []


@pytest.mark.asyncio
async def test_process_batch_isolates_failures(images):
    failing_path = images[1]
    service = AIService(vlm=FailingOnceVLM(fail_path=failing_path), embedder=SlowFakeEmbedder(delay=0.0))
    pipeline = Pipeline(service)
    items = [ItemInput(item_id=f"item-{i}", image_path=p, user_text="") for i, p in enumerate(images)]

    results = await pipeline.process_batch(items)

    by_id = {r.item_id: r for r in results}
    assert by_id["item-0"].ok
    assert by_id["item-2"].ok
    assert not by_id["item-1"].ok
    assert by_id["item-1"].error is not None


@pytest.mark.asyncio
async def test_match_batch_scores_against_pool(images, fake_embedder):
    service = AIService(vlm=SlowFakeVLM(delay=0.0), embedder=fake_embedder)
    pipeline = Pipeline(service)
    lost_items = [ItemInput(item_id="lost-1", image_path=images[0], user_text="")]

    # Build a found pool "by hand" using the same embedder/service for comparability.
    found_desc, found_vec = service.describe_and_embed(images[0], "")
    found_pool = [Candidate(item_id="found-1", embedding=found_vec, description=found_desc)]

    matches = await pipeline.match_batch(lost_items, found_pool, k=1)

    assert "lost-1" in matches
    assert len(matches["lost-1"]) == 1
    assert matches["lost-1"][0].candidate_id == 0


@pytest.mark.asyncio
async def test_embed_pool_concurrently_returns_only_successes(images):
    failing_path = images[0]
    service = AIService(vlm=FailingOnceVLM(fail_path=failing_path), embedder=SlowFakeEmbedder(delay=0.0))
    pipeline = Pipeline(service)
    items = [ItemInput(item_id=f"item-{i}", image_path=p, user_text="") for i, p in enumerate(images)]

    pool = await pipeline.embed_pool_concurrently(items)

    assert len(pool) == 2
    assert all(isinstance(c, Candidate) for c in pool)
