"""Async SQLite database with WAL mode and sqlite-vec extension.

Every connection from this module has:
- WAL journal mode (concurrent readers + single writer)
- busy_timeout = 5000ms (prevents 'database is locked' errors)
- synchronous = NORMAL (safe + fast)
- sqlite-vec loaded for vector similarity search

`get_db_context` reuses one connection per database file on this event loop
so the sqlite-vec extension is not loaded on every query. `get_db` still
returns a private connection for callers that close it themselves.
"""

import asyncio
import atexit
import time
from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager
from pathlib import Path

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


# Shared connections keyed by resolved path. :memory: is never shared.
_shared: dict[str, aiosqlite.Connection] = {}
_shared_refs: dict[str, int] = {}
_SHARED_LIMIT = 16

SCHEMA_VERSION = 2


def _resolve(db_path: str) -> str:
    if db_path == ":memory:":
        return db_path
    from maidere.paths import resolve_db_path

    return resolve_db_path(db_path)


async def backup_database(db: aiosqlite.Connection, db_path: str) -> Path | None:
    """Copy the database with VACUUM INTO before a destructive migration."""
    if not db_path or db_path == ":memory:":
        return None
    source = Path(db_path)
    if not source.exists():
        return None
    dest = source.with_name(f"{source.name}.bak-{int(time.time())}")
    escaped = str(dest).replace("'", "''")
    await db.commit()
    await db.execute(f"VACUUM INTO '{escaped}'")
    await logger.ainfo("database_backup_created", path=str(dest))
    return dest


async def _evict_idle_connections() -> None:
    while len(_shared) > _SHARED_LIMIT:
        idle = next((key for key, refs in _shared_refs.items() if refs == 0 and key in _shared), None)
        if idle is None:
            return
        conn = _shared.pop(idle)
        _shared_refs.pop(idle, None)
        await conn.close()


@asynccontextmanager
async def get_db_context(db_path: str = "db/maidere.db") -> AsyncGenerator[aiosqlite.Connection, None]:
    """Borrow a connection. File databases are reused; :memory: is private."""
    resolved = _resolve(db_path)
    if resolved == ":memory:":
        db = await get_db(resolved)
        try:
            yield db
        finally:
            await db.close()
        return

    pooled = _shared.get(resolved)
    if pooled is None:
        pooled = await get_db(resolved)
        _shared[resolved] = pooled
        _shared_refs.setdefault(resolved, 0)
    db = pooled
    _shared_refs[resolved] = _shared_refs.get(resolved, 0) + 1
    try:
        yield db
    finally:
        _shared_refs[resolved] = max(0, _shared_refs.get(resolved, 1) - 1)
        # Keep only the application database open. Every other file (tests,
        # one-off paths) is closed when the last borrower finishes, so the
        # process can exit.
        if _shared_refs[resolved] == 0 and not _is_app_database(resolved):
            conn = _shared.pop(resolved, None)
            _shared_refs.pop(resolved, None)
            if conn is not None:
                await conn.close()
        else:
            await _evict_idle_connections()


def _is_app_database(resolved: str) -> bool:
    try:
        from maidere.core.config import settings

        return resolved == _resolve(settings.db_path)
    except Exception:
        return False


async def close_shared_connections() -> None:
    """Close pooled connections. Used on shutdown and by tests."""
    for conn in list(_shared.values()):
        await conn.close()
    _shared.clear()
    _shared_refs.clear()


def stop_shared_connections() -> None:
    """Stop pooled worker threads without an event loop.

    aiosqlite's threads are not daemons, so a leftover pool keeps the process
    alive after the tests finish. `stop()` is safe once the loop is gone.
    """
    for conn in list(_shared.values()):
        try:
            conn.stop()
            thread = getattr(conn, "_thread", None)
            if thread is not None and thread.is_alive():
                thread.join(timeout=2)
        except Exception as exc:
            logger.debug("shared_connection_stop_failed", error=str(exc))
    _shared.clear()
    _shared_refs.clear()


atexit.register(stop_shared_connections)


_CHECKPOINT_MODES = ("PASSIVE", "FULL", "RESTART", "TRUNCATE")


async def checkpoint_wal(db_path: str = "db/maidere.db", mode: str = "PASSIVE") -> dict[str, int]:
    """Fold the write-ahead log into the main database file.

    PASSIVE never blocks readers or writers and is safe to run periodically.
    TRUNCATE also resets the ``-wal`` file to zero bytes, but waits for readers
    (bounded by ``busy_timeout``), so reserve it for idle moments and shutdown.
    Returns SQLite's ``busy``, ``log`` and ``checkpointed`` counters.
    """
    mode = mode.upper()
    if mode not in _CHECKPOINT_MODES:
        raise ValueError(f"Unsupported WAL checkpoint mode: {mode!r}")
    async with get_db_context(db_path) as db:
        cursor = await db.execute(f"PRAGMA wal_checkpoint({mode});")
        row = await cursor.fetchone()
    busy, log_frames, checkpointed = (row[0], row[1], row[2]) if row else (0, 0, 0)
    return {"busy": busy, "log": log_frames, "checkpointed": checkpointed}


async def optimize_database(db_path: str = "db/maidere.db") -> None:
    """Refresh query-planner statistics. Cheap; recommended before closing."""
    async with get_db_context(db_path) as db:
        await db.execute("PRAGMA optimize;")


async def run_wal_maintenance(db_path: str, interval_seconds: float) -> None:
    """Checkpoint the WAL on a schedule until cancelled."""
    while True:
        await asyncio.sleep(interval_seconds)
        try:
            stats = await checkpoint_wal(db_path, "PASSIVE")
            await logger.adebug("wal_checkpoint", mode="PASSIVE", **stats)
        except Exception as e:
            await logger.awarn("wal_checkpoint_failed", error=str(e))


async def shutdown_maintenance(db_path: str) -> None:
    """Final checkpoint so no ``-wal``/``-shm`` files linger after a clean exit."""
    try:
        await optimize_database(db_path)
        stats = await checkpoint_wal(db_path, "TRUNCATE")
        await logger.ainfo("wal_checkpoint", mode="TRUNCATE", **stats)
    except Exception as e:
        await logger.awarn("wal_shutdown_checkpoint_failed", error=str(e))


async def get_setting(db: aiosqlite.Connection, key: str) -> str | None:
    rows = list(await db.execute_fetchall("SELECT value FROM app_settings WHERE key = ?", [key]))
    if not rows or rows[0][0] is None:
        return None
    return str(rows[0][0])


async def set_setting(db: aiosqlite.Connection, key: str, value: str | None) -> None:
    if value is None:
        await db.execute("DELETE FROM app_settings WHERE key = ?", [key])
    else:
        await db.execute(
            """
            INSERT INTO app_settings (key, value) VALUES (?, ?)
            ON CONFLICT(key) DO UPDATE SET value = excluded.value
            """,
            [key, value],
        )
    await db.commit()


async def get_db(db_path: str = "db/maidere.db") -> aiosqlite.Connection:
    """Create a properly configured async SQLite connection.

    Enables WAL mode, loads sqlite-vec, and applies all PRAGMAs.
    The caller is responsible for closing the connection.
    """
    db_path = _resolve(db_path)

    db = await aiosqlite.connect(db_path)
    try:
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
    except BaseException:
        await db.close()
        raise

    await logger.ainfo("database_connected", db_path=db_path, journal_mode="WAL")
    return db


async def _migration_1(db: aiosqlite.Connection, db_path: str) -> None:
    await db.execute("CREATE TABLE IF NOT EXISTS schema_version (version INTEGER NOT NULL)")
    await db.execute(
        """
        CREATE TABLE IF NOT EXISTS app_settings (
            key TEXT PRIMARY KEY,
            value TEXT NOT NULL
        )
        """
    )
    await db.commit()


async def _migration_2(db: aiosqlite.Connection, db_path: str) -> None:
    from maidere.core.memory import init_memory_table

    await init_memory_table(db, db_path)


_MIGRATIONS = {
    1: _migration_1,
    2: _migration_2,
}


async def schema_version(db: aiosqlite.Connection) -> int:
    rows = await db.execute_fetchall("SELECT name FROM sqlite_master WHERE type='table' AND name='schema_version'")
    if not rows:
        return 0
    version_rows = list(await db.execute_fetchall("SELECT version FROM schema_version LIMIT 1"))
    if not version_rows:
        return 0
    return int(version_rows[0][0])


async def _set_schema_version(db: aiosqlite.Connection, version: int) -> None:
    await db.execute("DELETE FROM schema_version")
    await db.execute("INSERT INTO schema_version (version) VALUES (?)", [version])
    await db.commit()


async def apply_migrations(db: aiosqlite.Connection, db_path: str = "") -> int:
    """Apply ordered migrations. Returns the version now stored."""
    version = await schema_version(db)
    for number in range(version + 1, SCHEMA_VERSION + 1):
        if version > 0 and db_path and db_path != ":memory:":
            await backup_database(db, db_path)
        await _MIGRATIONS[number](db, db_path)
        await _set_schema_version(db, number)
        version = number
        await logger.ainfo("schema_migrated", version=number)
    return version


async def init_tables(db: aiosqlite.Connection, db_path: str = "") -> None:
    """Create application tables and apply migrations."""
    for sql in INIT_TABLES_SQL:
        await db.execute(sql)
    await db.commit()
    await apply_migrations(db, db_path)
    await logger.ainfo("tables_initialized", schema_version=await schema_version(db))
