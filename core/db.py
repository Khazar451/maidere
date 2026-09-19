"""Async SQLite database with WAL mode and sqlite-vec extension.

Every connection from this module has:
- WAL journal mode (concurrent readers + single writer)
- busy_timeout = 5000ms (prevents 'database is locked' errors)
- synchronous = NORMAL (safe + fast)
- sqlite-vec loaded for vector similarity search
"""

import aiosqlite
import sqlite_vec
import structlog

logger = structlog.get_logger()

PRAGMAS = [
    "PRAGMA journal_mode = WAL;",
    "PRAGMA busy_timeout = 5000;",
    "PRAGMA synchronous = NORMAL;",
    "PRAGMA foreign_keys = ON;",
]

INIT_TABLES_SQL = [
    """
    CREATE TABLE IF NOT EXISTS users (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        external_id TEXT UNIQUE NOT NULL,
        platform TEXT NOT NULL DEFAULT 'api',
        display_name TEXT,
        is_allowed INTEGER NOT NULL DEFAULT 0,
        created_at TEXT NOT NULL DEFAULT (datetime('now'))
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS audit_log (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        timestamp TEXT NOT NULL DEFAULT (datetime('now')),
        thread_id TEXT,
        tool_name TEXT NOT NULL,
        tool_args TEXT,
        tool_result TEXT,
        duration_ms INTEGER,
        success INTEGER NOT NULL DEFAULT 1
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS threads_meta (
        thread_id TEXT PRIMARY KEY,
        title TEXT NOT NULL,
        summary TEXT,
        message_count INTEGER NOT NULL DEFAULT 1,
        created_at TEXT NOT NULL DEFAULT (datetime('now')),
        updated_at TEXT NOT NULL DEFAULT (datetime('now'))
    )
    """,
]



from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager


@asynccontextmanager
async def get_db_context(db_path: str = "db/maidere.db") -> AsyncGenerator[aiosqlite.Connection, None]:
    """Async context manager that acquires and safely closes an SQLite connection."""
    db = await get_db(db_path)
    try:
        yield db
    finally:
        await db.close()


async def get_db(db_path: str = "db/maidere.db") -> aiosqlite.Connection:
    """Create a properly configured async SQLite connection.

    Enables WAL mode, loads sqlite-vec, and applies all PRAGMAs.
    The caller is responsible for closing the connection.
    """
    db = await aiosqlite.connect(db_path)
    db.row_factory = aiosqlite.Row

    # Ensure connection is ready before loading extensions
    await db.execute("SELECT 1")

    # Load sqlite-vec C extension for vector similarity search
    await db.enable_load_extension(True)
    await db.load_extension(sqlite_vec.loadable_path())
    await db.enable_load_extension(False)  # re-disable for security

    # Apply all PRAGMAs
    for pragma in PRAGMAS:
        await db.execute(pragma)

    await logger.ainfo("database_connected", db_path=db_path, journal_mode="WAL")
    return db



async def init_tables(db: aiosqlite.Connection) -> None:
    """Create application tables if they don't exist."""
    for sql in INIT_TABLES_SQL:
        await db.execute(sql)
    await db.commit()

    # Initialize sqlite-vec virtual table for memories
    from core.memory import init_memory_table

    await init_memory_table(db)
    await logger.ainfo("tables_initialized")

