"""Comprehensive tests for Maidere v0.1 enhancements."""

import asyncio
import os
import tempfile
import unittest
from unittest.mock import AsyncMock, MagicMock, patch

from fastapi.testclient import TestClient
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

from api.app import create_app
from api.routes import set_graph
from core.agent import MAX_TOOL_LOOPS, act_node, create_graph, get_system_prompt, set_custom_system_prompt, should_continue
from core.db import get_db, get_db_context, init_tables
from core.llm import chat_stream
from core.memory import delete_memory, is_duplicate_memory, list_all_memories, recall, store_memory
from core.state import AgentState


class TestV01Enhancements(unittest.IsolatedAsyncioTestCase):
    """Test v0.1 architectural and feature enhancements."""

    async def asyncSetUp(self):
        self.temp_db = tempfile.NamedTemporaryFile(suffix=".db", delete=False)
        self.temp_db_path = self.temp_db.name
        self.temp_db.close()

        # Init DB tables
        async with get_db_context(self.temp_db_path) as db:
            await init_tables(db)

        # Build graph and test client
        self.graph, self.checkpointer_ctx = await create_graph(self.temp_db_path)
        set_graph(self.graph)
        self.app = create_app()
        self.client = TestClient(self.app)

    async def asyncTearDown(self):
        await self.checkpointer_ctx.__aexit__(None, None, None)
        if os.path.exists(self.temp_db_path):
            os.remove(self.temp_db_path)
        set_custom_system_prompt(None)

    async def test_chat_stream_generator(self):
        """Test chat_stream yields chunks from Ollama streaming line protocol."""
        mock_chunks = [
            '{"message": {"content": "Hello "}, "done": false}',
            '{"message": {"content": "world!"}, "done": true, "eval_count": 2}',
        ]

        async def mock_aiter_lines():
            for line in mock_chunks:
                yield line

        mock_resp = MagicMock()
        mock_resp.aiter_lines = mock_aiter_lines
        mock_resp.raise_for_status = MagicMock()

        mock_stream_ctx = AsyncMock()
        mock_stream_ctx.__aenter__.return_value = mock_resp

        with patch("httpx.AsyncClient.stream", return_value=mock_stream_ctx):
            collected = []
            async for chunk in chat_stream([{"role": "user", "content": "Hi"}]):
                collected.append(chunk)

            self.assertEqual(len(collected), 2)
            self.assertEqual(collected[0]["message"]["content"], "Hello ")
            self.assertEqual(collected[1]["message"]["content"], "world!")
            self.assertTrue(collected[1]["done"])

    async def test_parallel_tool_execution(self):
        """Test act_node executes multiple tool calls concurrently using asyncio.gather."""
        call_times = []

        async def mock_exec_tool(name, args, thread_id):
            call_times.append(asyncio.get_running_loop().time())
            await asyncio.sleep(0.05)
            return f"Result for {name}"

        with patch("core.agent.execute_tool", side_effect=mock_exec_tool):
            state = {
                "messages": [
                    AIMessage(
                        content="",
                        tool_calls=[
                            {"name": "web_search", "args": {"query": "A"}, "id": "call_1"},
                            {"name": "web_search", "args": {"query": "B"}, "id": "call_2"},
                            {"name": "web_search", "args": {"query": "C"}, "id": "call_3"},
                        ],
                    )
                ],
                "thread_id": "test-parallel-thread",
                "tool_loop_count": 0,
            }

            t0 = asyncio.get_running_loop().time()
            result = await act_node(state)
            duration = asyncio.get_running_loop().time() - t0

            # 3 tools each sleeping 0.05s concurrently should take ~0.05s-0.08s (not 0.15s sequential)
            self.assertLess(duration, 0.12)
            self.assertEqual(len(result["messages"]), 3)
            self.assertEqual(result["tool_loop_count"], 1)

    def test_max_tool_loops_guard(self):
        """Test should_continue terminates when tool_loop_count exceeds MAX_TOOL_LOOPS."""
        state_under_limit = {
            "messages": [AIMessage(content="", tool_calls=[{"name": "test", "args": {}, "id": "1"}])],
            "tool_loop_count": 5,
        }
        self.assertEqual(should_continue(state_under_limit), "act")

        state_at_limit = {
            "messages": [AIMessage(content="", tool_calls=[{"name": "test", "args": {}, "id": "1"}])],
            "tool_loop_count": MAX_TOOL_LOOPS,
        }
        self.assertEqual(should_continue(state_at_limit), "__end__")

    async def test_db_context_manager(self):
        """Test get_db_context yields working connection and closes on exit."""
        async with get_db_context(self.temp_db_path) as db:
            rows = await db.execute_fetchall("SELECT 1")
            self.assertEqual(rows[0][0], 1)

    async def test_memory_deduplication_and_filtering(self):
        """Test semantic memory deduplication and distance threshold filtering."""
        async with get_db_context(self.temp_db_path) as db:
            # 1. Insert first memory
            res1 = await store_memory(db, "User prefers dark mode UI and Python.", session_id="t1", dedup=True)
            self.assertTrue(res1)

            # 2. Insert near-duplicate
            res2 = await store_memory(db, "User prefers dark mode UI and Python.", session_id="t1", dedup=True)
            self.assertFalse(res2)

            # 3. Recall with relevance threshold
            matches = await recall(db, "dark mode preference", top_k=5, max_distance=1.3)
            self.assertGreaterEqual(len(matches), 1)

            # 4. List memories
            all_mems = await list_all_memories(db)
            self.assertEqual(len(all_mems), 1)

            # 5. Delete memory
            mem_id = all_mems[0]["id"]
            del_res = await delete_memory(db, mem_id)
            self.assertTrue(del_res)

            # 6. Verify deleted
            after_del = await list_all_memories(db)
            self.assertEqual(len(after_del), 0)

    def test_export_endpoints(self):
        """Test GET /threads/{id}/export returns Markdown and JSON."""
        # Pre-seed state
        config = {"configurable": {"thread_id": "test-export-thread"}}
        asyncio.run(
            self.graph.aupdate_state(
                config,
                {"messages": [HumanMessage(content="Hello"), AIMessage(content="World!")]},
            )
        )

        # 1. Markdown Export
        res_md = self.client.get("/threads/test-export-thread/export?format=md")
        self.assertEqual(res_md.status_code, 200)
        self.assertIn("# Maidere Conversation Export", res_md.text)
        self.assertIn("Hello", res_md.text)
        self.assertIn("World!", res_md.text)

        # 2. JSON Export
        res_json = self.client.get("/threads/test-export-thread/export?format=json")
        self.assertEqual(res_json.status_code, 200)
        data = res_json.json()
        self.assertEqual(data["thread_id"], "test-export-thread")
        self.assertEqual(len(data["messages"]), 2)

    def test_fork_thread_endpoint(self):
        """Test POST /threads/{id}/fork forks conversation into a new thread."""
        config = {"configurable": {"thread_id": "orig-thread"}}
        asyncio.run(
            self.graph.aupdate_state(
                config,
                {"messages": [HumanMessage(content="Turn 1"), AIMessage(content="Ans 1"), HumanMessage(content="Turn 2")]},
            )
        )

        res = self.client.post("/threads/orig-thread/fork?from_message=2")
        self.assertEqual(res.status_code, 200)
        data = res.json()
        self.assertEqual(data["status"], "forked")
        self.assertEqual(data["message_count"], 2)
        self.assertNotEqual(data["new_thread_id"], "orig-thread")

    def test_upload_endpoint(self):
        """Test POST /upload uploads a file into workspace/uploads/."""
        file_content = b"Sample upload content for testing."
        response = self.client.post(
            "/upload",
            files={"file": ("test_doc.txt", file_content, "text/plain")},
        )
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertEqual(data["status"], "uploaded")
        self.assertEqual(data["filename"], "test_doc.txt")
        self.assertEqual(data["size"], len(file_content))

    def test_system_prompt_settings_endpoints(self):
        """Test GET and PUT /settings/system-prompt."""
        # 1. Get default
        res1 = self.client.get("/settings/system-prompt")
        self.assertEqual(res1.status_code, 200)
        self.assertIn("system_prompt", res1.json())

        # 2. Update custom prompt
        custom = "You are a custom AI agent persona."
        res2 = self.client.put("/settings/system-prompt", json={"system_prompt": custom})
        self.assertEqual(res2.status_code, 200)
        self.assertEqual(res2.json()["system_prompt"], custom)
        self.assertEqual(get_system_prompt(), custom)

        # 3. Reset
        res3 = self.client.put("/settings/system-prompt", json={"system_prompt": None})
        self.assertEqual(res3.status_code, 200)

    def test_skills_endpoint(self):
        """Test GET /skills returns registered skills."""
        res = self.client.get("/skills")
        self.assertEqual(res.status_code, 200)
        skills = res.json().get("skills", [])
        self.assertGreaterEqual(len(skills), 1)

    def test_chat_cancel_endpoint(self):
        """Test POST /chat/cancel."""
        res = self.client.post("/chat/cancel", json={"thread_id": "non-existent-task"})
        self.assertEqual(res.status_code, 200)
        self.assertEqual(res.json()["status"], "not_running")


if __name__ == "__main__":
    unittest.main()
