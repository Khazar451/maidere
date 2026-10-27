"""Schema migrations, shared connections, per-thread subagent events, and personal-default removal."""

import asyncio
import tempfile
import unittest
from unittest.mock import AsyncMock, patch

from fastapi.testclient import TestClient

from maidere.api.app import create_app
from maidere.api.routes import set_graph
from maidere.core.agent import get_system_prompt, set_custom_system_prompt
from maidere.core.config import settings
from maidere.core.db import (
    SCHEMA_VERSION,
    close_shared_connections,
    get_db,
    get_db_context,
    get_setting,
    init_tables,
    schema_version,
    set_setting,
)
from maidere.core.memory import init_memory_table
from maidere.core.subagent import (
    clear_progress_callback,
    register_progress_callback,
    run_subagent,
    unregister_progress_callback,
)
from maidere.tools.github import GitHubTool
from tests.authhelpers import AUTH_HEADERS, patch_api_token


class TestMigrationsAndConnections(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as temp:
            self.path = temp.name

    async def asyncTearDown(self):
        import os

        await close_shared_connections()
        if os.path.exists(self.path):
            os.remove(self.path)
        set_custom_system_prompt(None)

    async def test_fresh_database_reaches_current_schema(self):
        db = await get_db(self.path)
        try:
            await init_tables(db, self.path)
            self.assertEqual(await schema_version(db), SCHEMA_VERSION)
            await set_setting(db, "system_prompt", "Be brief.")
            self.assertEqual(await get_setting(db, "system_prompt"), "Be brief.")
        finally:
            await db.close()

    async def test_reinit_does_not_drop_memories(self):
        db = await get_db(self.path)
        try:
            await init_tables(db, self.path)
            await db.execute(
                "INSERT INTO memories(embedding, content, timestamp, session_id) VALUES (?, ?, ?, ?)",
                [b"\x00" * (384 * 4), "keep me", "2026-01-01T00:00:00", "s1"],
            )
            await db.commit()
            await init_tables(db, self.path)
            rows = await db.execute_fetchall("SELECT content FROM memories")
            self.assertEqual([r[0] for r in rows], ["keep me"])
        finally:
            await db.close()

    async def test_missing_session_id_is_rebuilt_without_losing_rows(self):
        db = await get_db(self.path)
        try:
            await db.execute("CREATE TABLE memories (content TEXT, timestamp TEXT)")
            await db.execute("INSERT INTO memories(content, timestamp) VALUES ('saved fact', 't0')")
            await db.commit()
            with patch("maidere.core.memory.embed_one", return_value=[0.1] * 384):
                await init_memory_table(db, self.path)
            rows = await db.execute_fetchall("SELECT content, session_id FROM memories")
            self.assertEqual(rows[0][0], "saved fact")
            self.assertEqual(rows[0][1], "")
        finally:
            await db.close()

    async def test_file_connections_are_reused_and_memory_is_not(self):
        async with get_db_context(self.path) as first, get_db_context(self.path) as second:
            self.assertIs(first, second)
        async with get_db_context(":memory:") as left, get_db_context(":memory:") as right:
            self.assertIsNot(left, right)

    async def test_system_prompt_round_trip(self):
        db = await get_db(self.path)
        try:
            await init_tables(db, self.path)
            await set_setting(db, "system_prompt", "Persist this.")
            set_custom_system_prompt(await get_setting(db, "system_prompt"))
            self.assertIn("Persist this.", get_system_prompt())
        finally:
            await db.close()


class TestProgressIsolation(unittest.IsolatedAsyncioTestCase):
    async def asyncTearDown(self):
        clear_progress_callback()
        unregister_progress_callback("thread-a")
        unregister_progress_callback("thread-b")

    async def test_concurrent_subagents_do_not_share_callbacks(self):
        events_a: list[dict] = []
        events_b: list[dict] = []

        async def callback_a(event):
            events_a.append(event)

        async def callback_b(event):
            events_b.append(event)

        register_progress_callback("thread-a", callback_a)
        register_progress_callback("thread-b", callback_b)
        reply = {"message": {"role": "assistant", "content": "done", "tool_calls": []}}

        with (
            patch("maidere.core.llm.chat", new_callable=AsyncMock, return_value=reply),
            patch("maidere.core.subagent._store_subagent_memory", new_callable=AsyncMock),
        ):
            await asyncio.gather(
                run_subagent(task="alpha task", max_turns=1, thread_id="thread-a"),
                run_subagent(task="beta task", max_turns=1, thread_id="thread-b"),
            )

        tasks_a = " ".join(event.get("task", "") for event in events_a)
        tasks_b = " ".join(event.get("task", "") for event in events_b)
        self.assertIn("alpha task", tasks_a)
        self.assertNotIn("beta task", tasks_a)
        self.assertIn("beta task", tasks_b)
        self.assertNotIn("alpha task", tasks_b)


class TestPersonalDefaultsAndLimits(unittest.TestCase):
    def test_github_tool_does_not_assume_a_repository(self):
        with patch.object(settings, "github_default_repo", ""):
            self.assertEqual(GitHubTool()._resolve_repo(None), "")

    def test_profile_query_has_no_hardcoded_name(self):
        from pathlib import Path

        source = Path("src/maidere/api/routes.py").read_text(encoding="utf-8")
        self.assertNotIn("khazar", source.lower())

    def test_request_id_header_and_generation_cap(self):
        token = patch_api_token()
        try:
            app = create_app()
            client = TestClient(app, headers=AUTH_HEADERS)
            health = client.get("/health")
            self.assertEqual(health.status_code, 200)
            self.assertEqual(len(health.headers.get("x-request-id", "")), 16)

            with patch.object(settings, "max_concurrent_generations", 0):
                set_graph(object())
                busy = client.post("/chat", json={"message": "hello there friend"})
                self.assertEqual(busy.status_code, 429)
                self.assertEqual(busy.json()["detail"]["code"], "busy")
        finally:
            token.stop()
            set_graph(None)
            set_custom_system_prompt(None)
