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

from contextlib import asynccontextmanager
from inspect import isawaitable
from typing import Any, Protocol

import numpy as np
from fastapi import FastAPI, File, Form, HTTPException, Query, UploadFile
from pydantic import BaseModel

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


class StatusUpdate(BaseModel):
    status: ItemStatus


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


@asynccontextmanager
async def _lifespan(app: FastAPI):
    from src.storage import repository as repo_mod

    if await repo_mod._ensure_ready():
        logger.info("database ready")
    else:
        logger.info("running with in-memory store")
    yield
    try:
        await repo_mod.close_db()
    except Exception:
        pass


def create_app(
    repo: ItemRepositoryProto | None = None,
    service: AIServiceProto | None = None,
) -> FastAPI:
    # Defaults resolve lazily so `import src.api` works standalone.
    # Tests inject fakes via repo=/service=.
    if repo is None:
        repo = _default_repo()
    if service is None:
        service = _default_service()

    app = FastAPI(
        title="Smart Lost & Found",
        version="1.0.0",
        lifespan=_lifespan,
    )
    app.state.repo = repo
    app.state.service = service

    from pathlib import Path

    from fastapi.responses import FileResponse
    from fastapi.staticfiles import StaticFiles

    static_dir = Path(__file__).parent / "static"
    if static_dir.is_dir():
        app.mount("/static", StaticFiles(directory=str(static_dir)), name="static")

        @app.get("/", include_in_schema=False)
        def index() -> FileResponse:
            return FileResponse(str(static_dir / "index.html"))

    @app.get("/health")
    def health() -> dict[str, str]:
        mode = "offline" if settings.use_offline else "online"
        return {"status": "ok", "mode": mode}

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
        saved = repo.save(item)
        if isawaitable(saved):
            saved = await saved
        logger.info("API registered %s id=%s", item_type.value, saved.id)
        return saved

    @app.post(
        "/items/lost",
        response_model=Item,
        status_code=201,
        response_model_exclude={"owner_token"},
    )
    async def register_lost(
        image: UploadFile = File(...),
        user_description: str = Form(...),
    ) -> Item:
        return await _register(image, user_description, ItemType.LOST)

    @app.post(
        "/items/found",
        response_model=Item,
        status_code=201,
        response_model_exclude={"owner_token"},
    )
    async def register_found(
        image: UploadFile = File(...),
        user_description: str = Form(...),
    ) -> Item:
        return await _register(image, user_description, ItemType.FOUND)

    @app.get(
        "/items",
        response_model=list[Item],
        response_model_exclude={"__all__": {"owner_token"}},
    )
    async def list_items(
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
        items = repo.list(item_type=item_type, status=effective)
        if isawaitable(items):
            items = await items
        return items

    @app.get("/items/{item_id}", response_model_exclude={"owner_token"})
    async def get_item(item_id: str) -> Item:
        if repo is None:
            raise HTTPException(
                status_code=501, detail="ItemRepository not implemented yet."
            )
        item = repo.get(item_id)
        if isawaitable(item):
            item = await item
        if item is None:
            raise HTTPException(status_code=404, detail="Item not found")
        return item

    @app.get("/items/{item_id}/image", include_in_schema=False)
    async def get_image(item_id: str):
        from pathlib import Path

        from fastapi.responses import FileResponse

        item = repo.get(item_id) if repo is not None else None
        if isawaitable(item):
            item = await item
        if item is None:
            if repo is None:
                raise HTTPException(
                    status_code=501, detail="ItemRepository not implemented yet."
                )
            try:
                from src.storage import repository as repo_mod

                item = await repo_mod.get_item(item_id)
            except Exception:
                item = None
        if item is None:
            raise HTTPException(status_code=404, detail="Item not found")
        path = Path(item.image_path)
        if not path.is_file():
            raise HTTPException(status_code=404, detail="Image file not found")
        try:
            base = settings.data_dir.resolve()
            if base not in path.resolve().parents:
                raise HTTPException(status_code=404, detail="Image file not found")
        except HTTPException:
            raise
        except Exception:
            pass
        suffix = path.suffix.lower()
        media = "image/jpeg" if suffix in (".jpg", ".jpeg") else "image/png"
        return FileResponse(str(path), media_type=media)

    @app.patch(
        "/items/{item_id}/status",
        response_model=Item,
        response_model_exclude={"owner_token"},
    )
    async def update_status(item_id: str, payload: StatusUpdate) -> Item:
        if repo is None:
            raise HTTPException(
                status_code=501, detail="ItemRepository not implemented yet."
            )
        item = repo.get(item_id)
        if isawaitable(item):
            item = await item
        if item is None:
            raise HTTPException(status_code=404, detail="Item not found")
        if hasattr(repo, "update_item_status"):
            updated = repo.update_item_status(item_id, payload.status)
        else:
            from src.storage import repository as repo_mod

            updated = repo_mod.update_item_status(item_id, payload.status)
        if isawaitable(updated):
            updated = await updated
        if updated is None:
            raise HTTPException(status_code=404, detail="Item not found")
        logger.info("status change %s -> %s", item_id, payload.status.value)
        return updated

    @app.delete("/items/{item_id}", status_code=200)
    async def delete_item(item_id: str) -> dict:
        if repo is None:
            raise HTTPException(
                status_code=501, detail="ItemRepository not implemented yet."
            )
        item = repo.get(item_id)
        if isawaitable(item):
            item = await item
        if item is None:
            raise HTTPException(status_code=404, detail="Item not found")
        if hasattr(repo, "delete_item"):
            deleted = repo.delete_item(item_id)
        else:
            from src.storage import repository as repo_mod

            deleted = repo_mod.delete_item(item_id)
        if isawaitable(deleted):
            deleted = await deleted
        if not deleted:
            raise HTTPException(status_code=404, detail="Item not found")
        logger.info("deleted item %s", item_id)
        return {"deleted": item_id}

    @app.get("/items/{item_id}/matches")
    async def get_matches(
        item_id: str,
        k: int = Query(default=3, ge=1, le=20),
        min_score: float = Query(default=0.0, ge=-1.0, le=1.0),
    ) -> dict:
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
        enriched = []
        for mid, score, reason in matches:
            if score < min_score:
                continue
            entry: dict[str, Any] = {
                "item_id": mid,
                "score": score,
                "reason": reason,
                "image_url": f"/items/{mid}/image",
            }
            cand = repo.get(mid)
            if isawaitable(cand):
                cand = await cand
            if cand is not None:
                entry["item_type"] = cand.item_type.value
                entry["status"] = cand.status.value
                entry["user_description"] = cand.user_description
                entry["description"] = cand.description_json
            enriched.append(entry)
        return {"query_id": item_id, "matches": enriched}

    return app


app = create_app()
