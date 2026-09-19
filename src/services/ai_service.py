import asyncio
import json
import logging
import time
from pathlib import Path

import numpy as np
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

import ai
from ai.providers.base import EmbeddingProvider, ProviderError, VLMProvider
from src.config import settings

logger = logging.getLogger(__name__)


class _OfflineVLM(VLMProvider):
    def describe(self, image_path: str, prompt: str, *, json_schema=None) -> str:
        name = Path(image_path).stem.lower()
        if "umbrella" in name:
            payload = {"object_class": "umbrella", "colors": ["black"],
                       "brand": None, "distinguishing_marks": [],
                       "location_hints": [], "confidence": 0.8}
        elif "backpack" in name:
            payload = {"object_class": "backpack", "colors": ["navy"],
                       "brand": "JanSport",
                       "distinguishing_marks": ["worn front zipper"],
                       "location_hints": [], "confidence": 0.85}
        elif "phone" in name:
            payload = {"object_class": "phone", "colors": ["black"],
                       "brand": "Apple",
                       "distinguishing_marks": ["cracked screen corner"],
                       "location_hints": [], "confidence": 0.9}
        elif "wallet" in name:
            payload = {"object_class": "wallet", "colors": ["brown"],
                       "brand": None, "distinguishing_marks": ["leather, tri-fold"],
                       "location_hints": [], "confidence": 0.75}
        elif "keys" in name:
            payload = {"object_class": "key ring", "colors": ["silver"],
                       "brand": None, "distinguishing_marks": ["3 keys, blue keychain"],
                       "location_hints": [], "confidence": 0.8}
        else:
            payload = {"object_class": "unknown object", "colors": [],
                       "brand": None, "distinguishing_marks": [],
                       "location_hints": [], "confidence": 0.3}
        return json.dumps(payload)


class _OfflineEmbedder(EmbeddingProvider):
    @property
    def dimension(self) -> int:
        return 64

    def embed(self, text: str) -> np.ndarray:
        if not text.strip():
            raise ValueError("Cannot embed empty string.")
        rng = np.random.default_rng(seed=abs(hash(text)) % (2**31))
        v = rng.standard_normal(64).astype(np.float32)
        v /= np.linalg.norm(v)
        return v


class AIService:
    def __init__(self, max_retries: int | None = None, offline: bool | None = None) -> None:
        self.max_retries = max_retries or settings.max_retries
        self.offline = settings.use_offline if offline is None else offline
        self._embed_cache: dict[str, list[float]] = {}
        if self.offline:
            self._vlm = _OfflineVLM()
            self._embedder = _OfflineEmbedder()
            logger.info("AIService running in offline demo mode (no API keys needed)")
        else:
            self._vlm = None
            self._embedder = None
            logger.info(
                "AIService using %s / %s (providers connect on first use)",
                settings.llm_provider, settings.embedding_provider,
            )

    def _ensure_providers(self) -> None:
        if self.offline:
            return
        if self._vlm is None:
            self._vlm = self._build_vlm()
        if self._embedder is None:
            self._embedder = self._build_embedder()

    def _build_vlm(self):
        provider = settings.llm_provider.lower().strip()
        model = settings.llm_model
        if provider == "anthropic":
            from ai.providers.anthropic import AnthropicVLM

            return AnthropicVLM(model=model, api_key=settings.anthropic_api_key or None)
        if provider == "openai":
            from ai.providers.openai import OpenAIVLM

            return OpenAIVLM(model=model, api_key=settings.openai_api_key or None)
        from ai.providers.google import GeminiVLM

        return GeminiVLM(model=model, api_key=settings.google_api_key or None)

    def _build_embedder(self):
        provider = settings.embedding_provider.lower().strip()
        model = settings.embedding_model
        if provider in ("google", "gemini"):
            from ai.providers.google import GeminiEmbedding

            return GeminiEmbedding(model=model, api_key=settings.google_api_key or None)
        from ai.providers.openai import OpenAIEmbedding

        return OpenAIEmbedding(model=model, api_key=settings.openai_api_key or None)

    def _retry_decorator(self):
        return retry(
            stop=stop_after_attempt(self.max_retries),
            wait=wait_exponential(
                multiplier=settings.retry_min_wait,
                min=settings.retry_min_wait,
                max=settings.retry_max_wait,
            ),
            retry=retry_if_exception_type((ProviderError, TimeoutError, OSError)),
            reraise=True,
        )

    def describe_item(self, image_path: str, user_text: str = "") -> object:
        decorator = self._retry_decorator()

        @decorator
        def _call():
            self._ensure_providers()
            logger.info("describing item image=%s", image_path)
            started = time.monotonic()
            desc = ai.describe_item(image_path, user_text, vlm=self._vlm)
            logger.info(
                "described item image=%s class=%s elapsed=%.2fs",
                image_path,
                getattr(desc, "object_class", "?"),
                time.monotonic() - started,
            )
            return desc

        return _call()

    def _fetch_embedding(self, text: str) -> list[float]:
        decorator = self._retry_decorator()

        @decorator
        def _call():
            self._ensure_providers()
            vec = ai.embed(text, embedder=self._embedder)
            return np.asarray(vec, dtype=float).tolist()

        return _call()

    def embed(self, text: str) -> list[float]:
        if text in self._embed_cache:
            logger.debug("embedding cache hit text=%.30r", text)
            return self._embed_cache[text]
        logger.info("embedding cache miss text=%.30r", text)
        vec = self._fetch_embedding(text)
        self._embed_cache[text] = vec
        return vec

    def clear_cache(self) -> None:
        self._embed_cache.clear()

    def cosine_similarity(self, vec1, vec2) -> float:
        return float(ai.cosine(np.asarray(vec1), np.asarray(vec2)))

    def top_k(self, query_vec, candidate_vecs, k: int) -> list[int]:
        if not candidate_vecs:
            return []
        results = ai.top_k(
            np.asarray(query_vec, dtype=np.float32),
            [np.asarray(c, dtype=np.float32) for c in candidate_vecs],
            k=k,
        )
        return [r.candidate_id for r in results]

    async def adescribe_and_embed(
        self, image_path: str, user_text: str
    ) -> tuple[object, list[float]]:
        loop = asyncio.get_running_loop()
        timeout = settings.ai_timeout_seconds
        desc = await asyncio.wait_for(
            loop.run_in_executor(None, self.describe_item, image_path, user_text),
            timeout=timeout,
        )
        search_text = desc.to_search_text() if hasattr(desc, "to_search_text") else user_text
        vec = await asyncio.wait_for(
            loop.run_in_executor(None, self.embed, search_text),
            timeout=timeout,
        )
        return desc, vec
