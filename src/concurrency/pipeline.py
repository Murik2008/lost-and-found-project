import asyncio
from typing import Any

from src.config import settings
from src.models import ItemType


class BatchProcessingPipeline:
    def __init__(self, ai_service, limit: int | None = None):
        self.ai_service = ai_service
        self.limit = limit or settings.concurrency_limit
        self._sem = asyncio.Semaphore(self.limit)

    async def _process_single_item(self, item: dict[str, Any]) -> dict[str, Any]:
        async with self._sem:
            loop = asyncio.get_running_loop()
            if not item.get("description") and item.get("image_path"):
                user_text = item.get("user_text", "")
                item["description"] = await loop.run_in_executor(
                    None, self.ai_service.describe_item,
                    item["image_path"], user_text,
                )
            if not item.get("embedding"):
                text = ""
                desc = item.get("description")
                if hasattr(desc, "to_search_text"):
                    text = desc.to_search_text()
                elif isinstance(desc, str):
                    text = desc
                else:
                    text = item.get("user_text", "")
                if text:
                    item["embedding"] = await loop.run_in_executor(
                        None, self.ai_service.embed, text
                    )
            return item

    async def process_batch(self, items: list[dict[str, Any]]) -> list[dict[str, Any]]:
        self._sem = asyncio.Semaphore(self.limit)
        tasks = [self._process_single_item(item) for item in items]
        return list(await asyncio.gather(*tasks))


def _match_reason(target_desc: dict, cand_desc: dict) -> str:
    parts: list[str] = []
    t_cls = str(target_desc.get("object_class", "")).strip().lower()
    c_cls = str(cand_desc.get("object_class", "")).strip().lower()
    if t_cls and t_cls == c_cls and t_cls != "unknown object":
        parts.append(f"same object: {target_desc.get('object_class')}")
    t_colors = {str(c).lower() for c in target_desc.get("colors", []) or []}
    c_colors = {str(c).lower() for c in cand_desc.get("colors", []) or []}
    shared = sorted(t_colors & c_colors)
    if shared:
        parts.append("shared color: " + ", ".join(shared))
    t_brand = str(target_desc.get("brand") or "").strip().lower()
    c_brand = str(cand_desc.get("brand") or "").strip().lower()
    if t_brand and t_brand == c_brand:
        parts.append(f"same brand: {target_desc.get('brand')}")
    t_marks = {str(m).lower() for m in target_desc.get("distinguishing_marks", []) or []}
    c_marks = {str(m).lower() for m in cand_desc.get("distinguishing_marks", []) or []}
    if t_marks & c_marks:
        parts.append("matching marks")
    if not parts:
        parts.append("similar description")
    return " · ".join(parts)


async def _get_any(item_id: str, repo):
    import inspect

    from src.storage import repository as repo_mod

    if repo is not None and hasattr(repo, "get"):
        maybe = repo.get(item_id)
        item = await maybe if inspect.isawaitable(maybe) else maybe
        if item is not None:
            return item
    return await repo_mod.get_item(item_id)


async def find_matches_for_item(item_id: str, k: int, service, repo) -> list[tuple[str, float, str]]:
    import inspect

    import numpy as np

    from src.models import ItemType
    from src.storage import repository as repo_mod

    target = await _get_any(item_id, repo)
    if target is None:
        raise KeyError(f"Item not found: {item_id}")

    opposite = ItemType.FOUND if target.item_type == ItemType.LOST else ItemType.LOST
    ids, vecs = await repo_mod.get_candidates(opposite)
    if not ids and repo is not None and hasattr(repo, "list"):
        maybe = repo.list()
        all_items = await maybe if inspect.isawaitable(maybe) else maybe
        ids = [i.id for i in all_items if i.item_type == opposite and len(i.embedding) > 0]
        vecs = [np.asarray(i.embedding, dtype=np.float32) for i in all_items
                if i.item_type == opposite and len(i.embedding) > 0]
    if not ids:
        return []

    query = np.asarray(target.embedding, dtype=np.float32)
    results = service.top_k(query.tolist(), [v.tolist() for v in vecs], k)
    out: list[tuple[str, float, str]] = []
    for idx in results:
        score = service.cosine_similarity(query.tolist(), vecs[idx].tolist())
        cand = await _get_any(ids[idx], repo)
        cand_desc = cand.description_json if cand is not None else {}
        reason = _match_reason(target.description_json or {}, cand_desc or {})
        out.append((ids[idx], float(score), reason))
    return out
