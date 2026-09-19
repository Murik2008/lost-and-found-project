"""PostgreSQL persistence + filesystem image storage.

Uses Postgres when DATABASE_URL is set, otherwise falls back to an
in-memory store so the API, CLI and web UI work locally without a database.
"""

from __future__ import annotations

import json
import logging
import uuid
from pathlib import Path

import asyncpg
import numpy as np

from src.config import settings
from src.models import Item, ItemStatus, ItemType, MatchRecord

logger = logging.getLogger(__name__)

_pool: asyncpg.Pool | None = None

# In-memory fallback for local runs without DATABASE_URL.
_memory_items: dict[str, Item] = {}
_memory_matches: dict[str, MatchRecord] = {}


_db_failed: bool = False


def _use_memory() -> bool:
    return not settings.database_url or _db_failed


async def _ensure_ready() -> bool:
    """Make sure the Postgres pool is up. Returns False when the memory
    fallback should be used (no URL configured or DB unreachable)."""
    global _db_failed
    if not settings.database_url or _db_failed:
        return False
    if _pool is not None:
        return True
    try:
        await init_db()
        return True
    except Exception:
        logger.exception("database unavailable, falling back to in-memory store")
        _db_failed = True
        return False

ALLOWED_IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png" }

# Schema:

SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS items (
    id               TEXT PRIMARY KEY,
    item_type        TEXT        NOT NULL CHECK (item_type IN ('lost', 'found')),
    user_description TEXT        NOT NULL DEFAULT '',
    image_path       TEXT        NOT NULL,
    description_json JSONB       NOT NULL,
    embedding        DOUBLE PRECISION[] NOT NULL,
    status           TEXT        NOT NULL DEFAULT 'pending'
                     CHECK (status IN ('pending', 'matched', 'closed')),
    owner_token      TEXT        NOT NULL DEFAULT '',
    created_at       TIMESTAMPTZ NOT NULL DEFAULT now()
);
-- Migration for databases created before owner_token existed.
ALTER TABLE items ADD COLUMN IF NOT EXISTS owner_token TEXT NOT NULL DEFAULT '';
 
-- The hot query is "give me every pending item of the opposite type",
-- so index exactly that pair.
CREATE INDEX IF NOT EXISTS idx_items_type_status ON items (item_type, status);
CREATE INDEX IF NOT EXISTS idx_items_created_at  ON items (created_at DESC);
 
CREATE TABLE IF NOT EXISTS matches (
    id            TEXT PRIMARY KEY,
    lost_item_id  TEXT        NOT NULL REFERENCES items(id) ON DELETE CASCADE,
    found_item_id TEXT        NOT NULL REFERENCES items(id) ON DELETE CASCADE,
    score         DOUBLE PRECISION NOT NULL,
    reason        TEXT        NOT NULL DEFAULT '',
    created_at    TIMESTAMPTZ NOT NULL DEFAULT now(),
    -- Re-running the matcher shouldn't create duplicate rows for the
    -- same pair; ON CONFLICT DO UPDATE in create_match relies on this.
    UNIQUE (lost_item_id, found_item_id)
);
 
CREATE INDEX IF NOT EXISTS idx_matches_lost  ON matches (lost_item_id);
CREATE INDEX IF NOT EXISTS idx_matches_found ON matches (found_item_id);
"""

# Lifecycle:

async def init_db(create_schema : bool = True) -> asyncpg.Pool:
    global _pool

    if _pool is not None:
        return _pool

    if not settings.database_url:
        raise RuntimeError("DATABASE_URL is not set. Copy .env.example to .env and fill it in.")

    logger.info("connecting to PostgreSQL")
    _pool = await asyncpg.create_pool(
        dsn=settings.database_url,
        min_size=settings.db_pool_min_size,
        max_size=settings.db_pool_max_size,
        command_timeout=settings.db_command_timeout,
    )

    if create_schema:
        async with _pool.acquire() as conn:
            await conn.execute(SCHEMA_SQL)
        logger.info("schema ready")

    settings.lost_dir.mkdir(parents=True, exist_ok=True)
    settings.found_dir.mkdir(parents=True, exist_ok=True)

    return _pool

async def close_db() -> None:
    global _pool
    if _pool is not None:
        await _pool.close()
        _pool = None
        logger.info("PostgreSQL pool closed")

def get_pool() -> asyncpg.Pool:
    if _pool is None:
        raise RuntimeError("Database not initialised - call init_db() first.")
    return _pool

async def health_check() -> bool:
    try:
        async with get_pool().acquire() as conn:
            await conn.execute("SELECT 1")
        return True
    except Exception:
        logger.exception("database health check failed")
        return False

# Filesystem: Image storage

def store_image_bytes(data: bytes, original_filename: str, item_type: ItemType) -> str:
    suffix = Path(original_filename).suffix.lower()

    if suffix not in ALLOWED_IMAGE_SUFFIXES:
        raise ValueError(f"Unsupported image suffix: {suffix} "
                         f"allowed: {', '.join(sorted(ALLOWED_IMAGE_SUFFIXES))}")

    if len(data) > settings.max_file_size_bytes:
        raise ValueError(f"Image too large: {len(data)} "
                         f"(max {settings.max_file_size_bytes} bytes)")

    target_dir = settings.lost_dir if item_type == ItemType.LOST else settings.found_dir
    target_dir.mkdir(parents=True, exist_ok=True)

    safe_name = f"{uuid.uuid4().hex}{suffix}"
    path = (target_dir / safe_name).resolve()
    base = target_dir.resolve()
    if base not in path.parents and path != base:
        raise ValueError("path escapes storage dir")
    path.write_bytes(data)
    logger.info("saved image %s (%d bytes)", path, len(data))
    return str(path)


def save_image(data: bytes, original_filename: str, item_type: ItemType) -> Path:
    return Path(store_image_bytes(data, original_filename, item_type))

def delete_image(image_path : str | Path) -> None:
    try:
        Path(image_path).unlink(missing_ok=True)
    except OSError:
        logger.warning("could not delete image %s", image_path, exc_info=True)

# Items:

def _row_to_item(row : asyncpg.Record) -> Item:
    return Item(
        id=row["id"],
        item_type=ItemType(row["item_type"]),
        user_description=row["user_description"],
        image_path=row["image_path"],
        description_json=(json.loads(row["description_json"])
        if isinstance(row["description_json"], str)
        else row["description_json"]),
        embedding=list(row["embedding"]),
        status=ItemStatus(row["status"]),
        owner_token=row.get("owner_token", "") or "",
        created_at=row["created_at"],
    )

async def create_item(item : Item) -> Item:
    if not await _ensure_ready():
        _memory_items[item.id] = item
        logger.info("created item %s (%s) in memory", item.id, item.item_type.value)
        return item
    async with get_pool().acquire() as conn:
        row = await conn.fetchrow(
            """
            INSERT INTO items (
                id, item_type, user_description, image_path,
                description_json, embedding, status, owner_token, created_at
            )
            VALUES ($1, $2, $3, $4, $5::jsonb, $6, $7, $8, $9)
            RETURNING *
            """,
            item.id,
            item.item_type.value,
            item.user_description,
            item.image_path,
            json.dumps(item.description_json),
            item.embedding,
            item.status.value,
            item.owner_token,
            item.created_at,
        )
    logger.info("created item %s (%s)", item.id, item.item_type.value)
    return _row_to_item(row)

async def get_item(item_id : str) -> Item | None:
    if not await _ensure_ready():
        return _memory_items.get(item_id)
    async with get_pool().acquire() as conn:
        row = await conn.fetchrow("SELECT * FROM items WHERE id = $1", item_id)
    return _row_to_item(row) if row else None

async def list_items(item_type: ItemType | None = None, status : ItemStatus | None = None, limit : int = 50, offset : int = 0) -> list[Item]:
    if not await _ensure_ready():
        items = list(_memory_items.values())
        if item_type is not None:
            items = [i for i in items if i.item_type == item_type]
        if status is not None:
            items = [i for i in items if i.status == status]
        items.sort(key=lambda i: i.created_at, reverse=True)
        return items[offset:offset + limit]
    clauses: list[str] = []
    params: list[object] = []

    if item_type is not None:
        params.append(item_type.value)
        clauses.append(f"item_type = ${len(params)}")
    if status is not None:
        params.append(status.value)
        clauses.append(f"status = ${len(params)}")

    where = f"WHERE {' AND '.join(clauses)} " if clauses else ""
    params.extend([limit, offset])

    sql = (
        f"SELECT * FROM items {where}"
        f"ORDER BY created_at DESC LIMIT ${len(params) - 1} OFFSET ${len(params)}"
    )

    async with get_pool().acquire() as conn:
        rows = await conn.fetch(sql, *params)
    return [_row_to_item(r) for r in rows]

async def get_candidates(item_type: ItemType, status: ItemStatus | None = ItemStatus.PENDING) -> tuple[list[str], list[np.ndarray]]:
    if not await _ensure_ready():
        ids: list[str] = []
        embeddings: list[np.ndarray] = []
        for item in _memory_items.values():
            if item.item_type != item_type:
                continue
            if status is not None and item.status != status:
                continue
            ids.append(item.id)
            embeddings.append(np.asarray(item.embedding, dtype=np.float32))
        return ids, embeddings
    sql = "SELECT id, embedding FROM items WHERE item_type = $1"
    params: list[object] = [item_type.value]

    if status is not None:
        params.append(status.value)
        sql += f" AND status = ${len(params)}"

    async with get_pool().acquire() as conn:
        rows = await conn.fetch(sql, *params)

    ids = [r["id"] for r in rows]
    embeddings = [np.asarray(r["embedding"], dtype=np.float32) for r in rows]
    logger.info("loaded %d %s candidates", len(ids), item_type.value)
    return ids, embeddings

async def update_item_status(item_id: str, status: ItemStatus) -> Item | None:
    if not await _ensure_ready():
        item = _memory_items.get(item_id)
        if item is None:
            return None
        item.status = status
        return item
    async with get_pool().acquire() as conn:
        row = await conn.fetchrow("UPDATE items SET status = $2 WHERE id = $1 RETURNING *", item_id, status.value)
    return _row_to_item(row) if row else None

async def delete_item(item_id: str, remove_file: bool = True) -> bool:
    if not await _ensure_ready():
        item = _memory_items.pop(item_id, None)
        if item is None:
            return False
        if remove_file:
            delete_image(item.image_path)
        return True
    async with get_pool().acquire() as conn:
        row = await conn.fetchrow("DELETE FROM items WHERE id = $1 RETURNING image_path", item_id)

    if row is None:
        return False
    if remove_file:
        delete_image(row["image_path"])
    return True

# Matches:

def _row_to_match(row : asyncpg.Record) -> MatchRecord:
    return MatchRecord(
        id=row["id"],
        lost_item_id=row["lost_item_id"],
        found_item_id=row["found_item_id"],
        score=row["score"],
        reason=row["reason"],
        created_at=row["created_at"],
    )

async def create_match(match: MatchRecord) -> MatchRecord:
    if not await _ensure_ready():
        for existing in list(_memory_matches.values()):
            if (existing.lost_item_id == match.lost_item_id
                    and existing.found_item_id == match.found_item_id):
                existing.score = match.score
                existing.reason = match.reason
                return existing
        _memory_matches[match.id] = match
        return match
    async with get_pool().acquire() as conn:
        row = await conn.fetchrow(
            """
            INSERT INTO matches (
                id, lost_item_id, found_item_id, score, reason, created_at
            )
            VALUES ($1, $2, $3, $4, $5, $6)
            ON CONFLICT (lost_item_id, found_item_id)
            DO UPDATE SET score = EXCLUDED.score, reason = EXCLUDED.reason
            RETURNING *
            """,
            match.id,
            match.lost_item_id,
            match.found_item_id,
            match.score,
            match.reason,
            match.created_at,
        )
    return _row_to_match(row)

async def create_matches(matches : list[MatchRecord]) -> list[MatchRecord]:
    if not matches:
        return []
    if not await _ensure_ready():
        return [await create_match(m) for m in matches]

    async with get_pool().acquire() as conn, conn.transaction():
        rows = [await conn.fetchrow(
            """
            INSERT INTO matches (
                id, lost_item_id, found_item_id, score, reason, created_at
            )
            VALUES ($1, $2, $3, $4, $5, $6)
            ON CONFLICT (lost_item_id, found_item_id)
            DO UPDATE SET score = EXCLUDED.score, reason = EXCLUDED.reason
            RETURNING *
            """,
            m.id,
            m.lost_item_id,
            m.found_item_id,
            m.score,
            m.reason,
            m.created_at,
        ) for m in matches]
    logger.info("stored %d matches", len(rows))
    return [_row_to_match(r) for r in rows]

async def get_matches_for_item(item_id : str, min_score : float = 0.0) -> list[MatchRecord]:
    if not await _ensure_ready():
        out = [m for m in _memory_matches.values()
               if (m.lost_item_id == item_id or m.found_item_id == item_id)
               and m.score >= min_score]
        out.sort(key=lambda m: m.score, reverse=True)
        return out
    async with get_pool().acquire() as conn:
        row = await conn.fetch(
            """
            SELECT * FROM matches
            WHERE (lost_item_id = $1 OR found_item_id = $1)
                AND score >= $2
            ORDER BY score DESC
            """,
            item_id,
            min_score,
        )
        return [_row_to_match(r) for r in row]


def clear_memory() -> None:
    _memory_items.clear()
    _memory_matches.clear()


class SyncRepository:
    """Repository facade used by the HTTP layer.

    Methods are async and run on the caller's event loop, which the
    asyncpg pool requires. Callers accept both sync fakes (tests) and
    this async implementation via maybe-await.
    """

    async def save(self, item: Item) -> Item:
        return await create_item(item)

    async def get(self, item_id: str) -> Item | None:
        return await get_item(item_id)

    async def list(
        self,
        item_type: ItemType | None = None,
        status: ItemStatus | None = None,
    ) -> list[Item]:
        return await list_items(item_type=item_type, status=status)


def get_repository() -> SyncRepository:
    return SyncRepository()