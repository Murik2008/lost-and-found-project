"""FastAPI HTTP layer.

Endpoints (per TOPIC.md):
- POST /items/lost   (multipart: image + user_description)
- POST /items/found  (multipart: image + user_description)
- GET  /items?status=...&type=...
- GET  /items/{id}/matches?k=N

Validation: non-JPEG/PNG, oversize, missing fields -> 400/422.
AI failures (ProviderError) -> 502.
"""

from __future__ import annotations

from typing import Any, Protocol

import numpy as np
from fastapi import FastAPI, File, Form, HTTPException, Query, UploadFile

from ai.providers.base import ProviderError
from src.config import settings
from src.logging_setup import configure_logging, get_logger
from src.models import Item, ItemStatus, ItemType
from validation import ValidationError, validate_image_bytes, validate_user_text

configure_logging(settings.log_level)
logger = get_logger(__name__)


class ItemRepositoryProto(Protocol):
    def save(self, item: Item) -> Item: ...
    def get(self, item_id: str) -> Item | None: ...
    def list(
        self,
        item_type: ItemType | None = None,
        status: ItemStatus | None = None,
    ) -> list[Item]: ...


class AIServiceProto(Protocol):
    async def adescribe_and_embed(
        self, image_path: str, user_text: str
    ) -> tuple[Any, Any]: ...


def _default_repo() -> ItemRepositoryProto | None:
    """Lazy default repo."""
    try:
        from src.storage.repository import get_repository

        return get_repository()
    except (ImportError, AttributeError) as e:
        logger.warning("ItemRepository not available yet: %s", e)
        return None


def _default_service() -> AIServiceProto | None:
    """Lazy default AI service."""
    try:
        from src.services.ai_service import AIService

        return AIService()
    except (ImportError, AttributeError) as e:
        logger.warning("AIService not available yet: %s", e)
        return None


def _store_image_bytes(data: bytes, filename: str, item_type: ItemType) -> str:
    try:
        from src.storage.repository import store_image_bytes
    except (ImportError, AttributeError) as e:
        raise HTTPException(
            status_code=501, detail=f"Image storage not implemented yet: {e}"
        ) from e
    return store_image_bytes(data, filename, item_type)


def create_app(
    repo: ItemRepositoryProto | None = None,
    service: AIServiceProto | None = None,
) -> FastAPI:
    # NOTE: defaults are resolved lazily (not at import time) so that
    # `import src.api` works even before teammates land
    # repository.py / ai_service.py / pipeline.py.
    # Tests inject fakes via repo=/service=, production gets real ones when ready.
    if repo is None:
        repo = _default_repo()
    if service is None:
        service = _default_service()

    app = FastAPI(title="Smart Lost & Found", version="1.0.0")
    app.state.repo = repo
    app.state.service = service

    @app.get("/health")
    def health() -> dict[str, str]:
        return {"status": "ok"}

    async def _register(
        image: UploadFile, user_description: str, item_type: ItemType
    ) -> Item:
        if repo is None:
            raise HTTPException(
                status_code=501, detail="ItemRepository not implemented yet."
            )
        if service is None:
            raise HTTPException(
                status_code=501, detail="AIService not implemented yet."
            )
        try:
            text = validate_user_text(user_description)
        except ValidationError as e:
            raise HTTPException(status_code=422, detail=str(e)) from e
        data = await image.read()
        try:
            validate_image_bytes(
                data, image.filename or "upload.png", settings.max_file_size_bytes
            )
        except ValidationError as e:
            raise HTTPException(status_code=400, detail=str(e)) from e
        try:
            stored = _store_image_bytes(
                data, image.filename or "upload.png", item_type
            )
        except ValidationError as e:
            raise HTTPException(status_code=400, detail=str(e)) from e
        try:
            desc, vec = await service.adescribe_and_embed(stored, text)
        except ProviderError as e:
            raise HTTPException(status_code=502, detail=f"AI provider error: {e}") from e
        item = Item(
            item_type=item_type,
            user_description=text,
            image_path=stored,
            description_json=desc.to_dict(),
            embedding=np.asarray(vec, dtype=float).tolist(),
        )
        repo.save(item)
        logger.info("API registered %s id=%s", item_type.value, item.id)
        return item

    @app.post("/items/lost", response_model=Item, status_code=201)
    async def register_lost(
        image: UploadFile = File(...),
        user_description: str = Form(...),
    ) -> Item:
        return await _register(image, user_description, ItemType.LOST)

    @app.post("/items/found", response_model=Item, status_code=201)
    async def register_found(
        image: UploadFile = File(...),
        user_description: str = Form(...),
    ) -> Item:
        return await _register(image, user_description, ItemType.FOUND)

    @app.get("/items", response_model=list[Item])
    def list_items(
        status: ItemStatus | None = Query(default=None),
        item_type: ItemType | None = Query(default=None, alias="type"),
        item_status: ItemStatus | None = Query(default=None, alias="item_status"),
    ) -> list[Item]:
        # Accept both ?status= and ?item_status= for convenience.
        if repo is None:
            raise HTTPException(
                status_code=501, detail="ItemRepository not implemented yet."
            )
        effective = status or item_status
        return repo.list(item_type=item_type, status=effective)

    @app.get("/items/{item_id}")
    def get_item(item_id: str) -> Item:
        if repo is None:
            raise HTTPException(
                status_code=501, detail="ItemRepository not implemented yet."
            )
        item = repo.get(item_id)
        if item is None:
            raise HTTPException(status_code=404, detail="Item not found")
        return item

    @app.get("/items/{item_id}/matches")
    async def get_matches(item_id: str, k: int = Query(default=3, ge=1, le=20)) -> dict:
        if repo is None or service is None:
            raise HTTPException(
                status_code=501, detail="Matching pipeline not implemented yet."
            )
        try:
            from src.concurrency.pipeline import find_matches_for_item

            matches = await find_matches_for_item(item_id, k, service, repo)
        except (ImportError, AttributeError) as e:
            raise HTTPException(
                status_code=501, detail=f"Matching pipeline not implemented yet: {e}"
            ) from e
        except KeyError as e:
            raise HTTPException(status_code=404, detail=str(e)) from e
        return {
            "query_id": item_id,
            "matches": [
                {"item_id": mid, "score": score, "reason": reason}
                for mid, score, reason in matches
            ],
        }

    return app


app = create_app()
