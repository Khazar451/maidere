"""Liveness and readiness checks.

`/live` only proves the process can answer HTTP.
`/ready` proves a chat turn can be stored and a model can be called.
SearXNG is reported but does not block readiness: local chat still works
without web search.
"""

from __future__ import annotations

import shutil

import httpx

from maidere.core import llm
from maidere.core.config import settings
from maidere.core.db import get_db_context
from maidere.paths import resolve_config_path


async def database_accepts_writes() -> bool:
    """Create a one-row probe table and write it."""
    try:
        async with get_db_context(settings.db_path) as db:
            await db.execute("CREATE TABLE IF NOT EXISTS readiness_probe (id INTEGER PRIMARY KEY)")
            await db.execute("INSERT INTO readiness_probe (id) VALUES (1) ON CONFLICT(id) DO UPDATE SET id = 1")
            await db.commit()
            rows = await db.execute_fetchall("SELECT id FROM readiness_probe WHERE id = 1")
        return bool(rows)
    except Exception:
        return False


def disk_has_space() -> bool:
    """True when the database directory has at least `min_free_disk_mb` free."""
    try:
        target = resolve_config_path(settings.db_path)
        target.parent.mkdir(parents=True, exist_ok=True)
        free = shutil.disk_usage(target.parent).free
        return free >= settings.min_free_disk_mb * 1024 * 1024
    except OSError:
        return False


async def searxng_reachable() -> bool:
    """True when the local metasearch endpoint answers."""
    url = settings.searxng_url.rstrip("/") + "/"
    try:
        async with httpx.AsyncClient(timeout=2.0) as client:
            response = await client.get(url)
        return response.status_code < 500
    except httpx.HTTPError:
        return False


async def collect_checks() -> dict[str, bool]:
    return {
        "database": await database_accepts_writes(),
        "disk": disk_has_space(),
        "ollama": await llm.is_available(),
        "searxng": await searxng_reachable(),
    }


def readiness_status(checks: dict[str, bool]) -> tuple[bool, str]:
    """Required checks are database, disk, and ollama. SearXNG is optional."""
    required = checks["database"] and checks["disk"] and checks["ollama"]
    if required and checks["searxng"]:
        return True, "ok"
    if required:
        return True, "degraded"
    return False, "unavailable"
