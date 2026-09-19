"""PostgreSQL persistence + filesystem image storage.

Owns the connection pool, the schema, and every SQL statement in the project.
No other module should import asyncpg or write raw SQL.

Lifecycle
---------
Call `init_db()` once at application startup (FastAPI lifespan / CLI entry)
and `close_db()` at shutdown.
"""

from __future__ import annotations

import json
import logging
import uuid
# from datetime import datetime
from pathlib import Path

import asyncpg
import numpy as np

# from src import ItemType
from src.config import settings
# from src.core.matcher import Matcher
from src.models import Item, ItemStatus, ItemType, MatchRecord

logger = logging.getLogger(__name__)

_pool: asyncpg.Pool | None = None

ALLOWED_IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png" }

# Schema:

SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS schema (
    id               TEXT PRIMARY KEY,
    item_type        TEXT        NOT NULL CHECK (item_type IN ('lost', 'found')),
    user_description TEXT        NOT NULL DEFAULT '',
    image_path       TEXT        NOT NULL,
    description_json JSONB       NOT NULL,
    embedding        DOUBLE PRECISION[] NOT NULL,
    status           TEXT        NOT NULL DEFAULT 'pending'
                     CHECK (status IN ('pending', 'matched', 'closed')),
    created_at       TIMESTAMPTZ NOT NULL DEFAULT now()
);
 
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

def save_image(data: bytes, original_filename: str, item_type: ItemType) -> Path:
    suffix = Path(original_filename).suffix.lower()

    if suffix not in ALLOWED_IMAGE_SUFFIXES:
        raise ValueError(f"Unsupported image suffix: {suffix}"
                         f"allowed: {', '.join(sorted(ALLOWED_IMAGE_SUFFIXES))}")

    if len(data) > settings.max_file_size:
        raise ValueError(f"Image too large: {len(data)}"
                         f"(max {settings.max_file_size} bytes)")

    target_dir = settings.log_dir if item_type == ItemType.LOST else settings.founf_dir
    target_dir.mkdir(parents=True, exist_ok=True)

    path = target_dir / f"{uuid.uuid4()}{suffix}"
    path.write_bytes(data)
    logger.info(f"saved image %s (%d bytes)", path, len(data))
    return path

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
        created_at=row["created_at"],
    )

async def create_item(item : Item) -> Item:
    async with get_pool() as conn:
        row = await conn.fetchrow(
            """
            INSERT INTO items (
                id, item_type, user_description, image_path,
                description_json, embedding, status, created_at
            )
            VALUES ($1, $2, $3, $4, $5::jsonb, $6, $7, $8)
            RETURNING *
            """,
            item.id,
            item.item_type.value,
            item.user_description,
            item.image_path,
            json.dumps(item.description_json),
            item.embedding,
            item.status.value,
            item.created_at,
        )
    logger.info("created item %s (%s)", item.id, item.item_type.value)
    return _row_to_item(row)

async def get_item(item_id : str) -> Item | None:
    async with get_pool().acquire() as conn:
        row = await conn.fetchrow("SELECT * FROM items WHERE id = $1", item_id)
    return _row_to_item(row) if row else None

async def list_items(item_type: ItemType | None = None, status : ItemStatus | None = None, limit : int = 50, offset : int = 0) -> list[Item]:
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
        f"ORDER BY created_at DESC LIMIT ${len(params) - 1} OFFSET {len(params)}"
    )

    async with get_pool().acquire() as conn:
        rows = await conn.fetchrow(sql, *params)
    return [_row_to_item(r) for r in rows]

async def get_candidates(item_type: ItemType, status: ItemStatus | None = ItemStatus.PENDING) -> tuple[list[str], list[np.ndarray]]:
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
    async with get_pool().acquire() as conn:
        row = await conn.fetchrow("UPDATE items SET status = $2 WHERE id = $1 RETURNING *", item_id, status.value)
    return _row_to_item(row) if row else None

async def delete_item(item_id: str, remove_file: bool = True) -> bool:
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