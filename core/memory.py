"""Vector memory management with sqlite-vec and fastembed (CPU).

Implements:
- vec0 virtual table for 384-dim dense vectors
- Embedding serialization and cosine/L2 distance search
- Ingestion of user messages and conversation summaries
- Formatting of retrieved memories for system prompt injection
"""

import datetime
import struct
from typing import Any

import aiosqlite
import structlog

from core.config import settings
from core.embeddings import embed_one

logger = structlog.get_logger()

DIMENSIONS: int = settings.embedding_dimensions  # 384 for bge-small-en-v1.5


def serialize_vector(vec: list[float]) -> bytes:
    """Serialize float list to binary bytes for sqlite-vec."""
    return struct.pack(f"{len(vec)}f", *vec)


async def init_memory_table(db: aiosqlite.Connection) -> None:
    """Create the vector memory table if it does not exist, and ensure session_id column is present."""
    rows = await db.execute_fetchall("SELECT sql FROM sqlite_master WHERE name='memories'")
    if rows:
        sql = rows[0]["sql"] or ""
        if "+session_id" not in sql and "session_id" not in sql:
            await db.execute("DROP TABLE memories")
            await db.commit()

    await db.execute(
        f"""
        CREATE VIRTUAL TABLE IF NOT EXISTS memories
        USING vec0(
            embedding float[{DIMENSIONS}],
            +content TEXT,
            +timestamp TEXT,
            +session_id TEXT
        )
        """
    )
    await db.commit()
    await logger.ainfo("memory_table_initialized", dimensions=DIMENSIONS)



async def is_duplicate_memory(
    db: aiosqlite.Connection,
    vector: list[float],
    threshold: float = 0.15,
) -> bool:
    """Check if a semantically identical memory already exists within distance threshold."""
    try:
        rows = await db.execute_fetchall(
            """
            SELECT distance
            FROM memories
            WHERE embedding MATCH ?
            ORDER BY distance
            LIMIT 1
            """,
            [serialize_vector(vector)],
        )
        if rows and len(rows) > 0:
            dist = float(rows[0][0])
            return dist < threshold
        return False
    except Exception:
        return False


async def store_memory(
    db: aiosqlite.Connection,
    content: str,
    session_id: str,
    dedup: bool = True,
    dedup_threshold: float = 0.15,
) -> bool:
    """Sanitize, embed, and store a text memory into sqlite-vec with deduplication."""
    if not content or not content.strip():
        return False

    from core.audit import sanitize_secrets

    sanitized_content = sanitize_secrets(content.strip())
    vector = embed_one(sanitized_content)

    if dedup and await is_duplicate_memory(db, vector, threshold=dedup_threshold):
        await logger.ainfo("memory_deduplicated", session_id=session_id, length=len(sanitized_content))
        return False

    now_iso = datetime.datetime.now(datetime.timezone.utc).isoformat()

    await db.execute(
        """
        INSERT INTO memories(embedding, content, timestamp, session_id)
        VALUES (?, ?, ?, ?)
        """,
        [serialize_vector(vector), sanitized_content, now_iso, session_id],
    )
    await db.commit()
    await logger.ainfo("memory_stored", session_id=session_id, length=len(sanitized_content))
    return True



async def recall(
    db: aiosqlite.Connection,
    query: str,
    top_k: int = 5,
    max_distance: float = 1.3,
) -> list[dict[str, Any]]:
    """Retrieve top-K most semantically relevant memories with distance threshold filtering."""
    if not query or not query.strip():
        return []

    import time
    from core.metrics import MEMORY_RECALL_DURATION_SECONDS, MEMORY_RECALLS_TOTAL

    start_time = time.perf_counter()
    try:
        vector = embed_one(query)
        rows = await db.execute_fetchall(
            """
            SELECT rowid, content, timestamp, distance, session_id
            FROM memories
            WHERE embedding MATCH ?
            ORDER BY distance
            LIMIT ?
            """,
            [serialize_vector(vector), top_k],
        )

        results = [
            {
                "id": r[0],
                "content": r[1],
                "timestamp": r[2],
                "distance": float(r[3]),
                "session_id": r[4] if len(r) > 4 else "",
            }
            for r in rows
            if float(r[3]) <= max_distance
        ]
        duration = time.perf_counter() - start_time
        MEMORY_RECALL_DURATION_SECONDS.observe(duration)
        status = "success" if results else "empty"
        MEMORY_RECALLS_TOTAL.labels(status=status).inc()

        await logger.ainfo("memory_recalled", query_len=len(query), count=len(results), duration_s=round(duration, 3))
        return results
    except Exception as e:
        duration = time.perf_counter() - start_time
        MEMORY_RECALL_DURATION_SECONDS.observe(duration)
        MEMORY_RECALLS_TOTAL.labels(status="error").inc()
        raise


async def delete_memory(db: aiosqlite.Connection, memory_id: int) -> bool:
    """Delete a memory record by rowid."""
    try:
        await db.execute("DELETE FROM memories WHERE rowid = ?", [memory_id])
        await db.commit()
        await logger.ainfo("memory_deleted", memory_id=memory_id)
        return True
    except Exception as e:
        await logger.aerror("memory_delete_failed", memory_id=memory_id, error=str(e))
        return False


async def list_all_memories(db: aiosqlite.Connection, limit: int = 50) -> list[dict[str, Any]]:
    """List stored semantic memories."""
    try:
        rows = await db.execute_fetchall(
            "SELECT rowid, content, timestamp, session_id FROM memories ORDER BY rowid DESC LIMIT ?",
            [limit],
        )
        return [
            {"id": r[0], "content": r[1], "timestamp": r[2], "session_id": r[3]}
            for r in rows
        ]
    except Exception as e:
        await logger.aerror("list_memories_failed", error=str(e))
        return []


def format_memories(memories: list[dict[str, Any]]) -> str:
    """Format memory dictionaries into a prompt-ready context string."""
    if not memories:
        return "No relevant memories found."

    lines = []
    for m in memories:
        lines.append(f"- {m['content']} (saved at {m['timestamp'][:19]})")
    return "\n".join(lines)

