"""Tests for FastAPI endpoints: /chat, /history/{thread_id}, /memories, and /health."""

import asyncio
import tempfile
import unittest
from unittest.mock import AsyncMock, patch

from fastapi.testclient import TestClient
from langchain_core.messages import AIMessage, HumanMessage

from api.app import create_app
from api.routes import set_graph
from core.agent import create_graph
from core.db import get_db, init_tables
from core.memory import store_memory


class TestAPIEndpoints(unittest.IsolatedAsyncioTestCase):
    """Test API routes end-to-end."""

    @classmethod
    def setUpClass(cls):
        import numpy as np
        from unittest.mock import MagicMock
        cls.mock_emb_model = MagicMock()
        cls.mock_emb_model.embed.side_effect = lambda texts: [np.array([0.05] * 384) for _ in texts]
        cls.patcher1 = patch("core.embeddings._get_model", return_value=cls.mock_emb_model)
        cls.patcher2 = patch("core.embeddings.embed_one", side_effect=lambda t: [0.05] * 384)
        cls.patcher3 = patch("core.embeddings.embed", side_effect=lambda texts: [[0.05] * 384 for _ in texts])
        cls.patcher4 = patch("core.memory.embed_one", side_effect=lambda t: [0.05] * 384)
        cls.patcher1.start()
        cls.patcher2.start()
        cls.patcher3.start()
        cls.patcher4.start()

    @classmethod
    def tearDownClass(cls):
        cls.patcher1.stop()
        cls.patcher2.stop()
        cls.patcher3.stop()
        cls.patcher4.stop()

    async def asyncSetUp(self):
        self.temp_db = tempfile.NamedTemporaryFile(suffix=".db", delete=False)
        self.temp_db_path = self.temp_db.name
        self.temp_db.close()

        # Init DB tables
        db = await get_db(self.temp_db_path)
        await init_tables(db)
        await db.close()

        # Build graph and test client
        self.graph, self.checkpointer_ctx = await create_graph(self.temp_db_path)
        set_graph(self.graph)
        self.app = create_app()
        self.client = TestClient(self.app)

    async def asyncTearDown(self):
        import os
        await self.checkpointer_ctx.__aexit__(None, None, None)
        if os.path.exists(self.temp_db_path):
            os.remove(self.temp_db_path)

    def test_health_endpoint(self):
        """Test GET /health returns status and model info."""
        with patch("core.llm.is_available", new_callable=AsyncMock, return_value=True):
            response = self.client.get("/health")
            self.assertEqual(response.status_code, 200)
            data = response.json()
            self.assertEqual(data["status"], "ok")
            self.assertTrue(data["ollama_available"])

    def test_chat_and_history_endpoints(self):
        """Test POST /chat persists history retrievable via GET /history/{thread_id}."""
        mock_response = {
            "message": {
                "role": "assistant",
                "content": "Hello there! How can I assist you today?",
                "tool_calls": [],
            }
        }

        with patch("core.llm.chat", new_callable=AsyncMock, return_value=mock_response):
            # Send chat message
            thread_id = "test-history-thread"
            res = self.client.post("/chat", json={"message": "Hi Maidere!", "thread_id": thread_id})
            self.assertEqual(res.status_code, 200)
            chat_data = res.json()
            self.assertEqual(chat_data["response"], "Hello there! How can I assist you today?")
            self.assertEqual(chat_data["thread_id"], thread_id)

            # Retrieve history
            hist_res = self.client.get(f"/history/{thread_id}")
            self.assertEqual(hist_res.status_code, 200)
            hist_data = hist_res.json()
            self.assertEqual(hist_data["thread_id"], thread_id)
            self.assertGreaterEqual(hist_data["count"], 2)
            self.assertEqual(hist_data["messages"][0]["role"], "user")
            self.assertEqual(hist_data["messages"][0]["content"], "Hi Maidere!")
            self.assertEqual(hist_data["messages"][1]["role"], "assistant")
            self.assertEqual(hist_data["messages"][1]["content"], "Hello there! How can I assist you today?")

    def test_memories_endpoint(self):
        """Test GET /memories semantic search endpoint."""
        # Insert a memory directly
        async def _insert():
            db = await get_db(self.temp_db_path)
            try:
                await store_memory(db, "Antigravity agents use fastembed and sqlite-vec.", session_id="test-thread")
            finally:
                await db.close()

        asyncio.run(_insert())

        with patch("core.config.settings.db_path", self.temp_db_path):
            res = self.client.get("/memories", params={"query": "fastembed vector search"})
            self.assertEqual(res.status_code, 200)
            data = res.json()
            self.assertIn("memories", data)
            self.assertGreaterEqual(len(data["memories"]), 1)
            self.assertIn("Antigravity", data["memories"][0]["content"])


    def test_chat_endpoint_with_deep_reasoning(self):
        """Test POST /chat with deep_reasoning=True triggers deliberation."""
        mock_responses = [
            {
                "message": {
                    "role": "assistant",
                    "content": "<think>Deconstructing distributed consensus.</think>Draft on Paxos.",
                    "tool_calls": [],
                }
            },
            {
                "message": {
                    "role": "assistant",
                    "content": "Critique: Must explicitly evaluate Multi-Paxos vs Raft log compaction.",
                }
            },
            {
                "message": {
                    "role": "assistant",
                    "content": "Hardened analysis evaluating Paxos, Multi-Paxos, and Raft consensus.",
                }
            },
        ]

        with patch("core.llm.chat", new_callable=AsyncMock, side_effect=mock_responses) as mock_chat:
            res = self.client.post(
                "/chat",
                json={
                    "message": "Analyze Paxos and Raft consensus mechanisms",
                    "thread_id": "test-deep-reason-api",
                    "deep_reasoning": True,
                },
            )
            self.assertEqual(res.status_code, 200)
            data = res.json()
            self.assertIn("Raft", data["response"])
            # 3 calls for deliberation (think -> critique -> refine) + 2 for thread metadata/title
            self.assertGreaterEqual(mock_chat.call_count, 3)

    def test_user_login_and_profile_endpoints(self):
        """Test POST /user/login and GET /user/profile without passwords."""
        # Initial profile
        res = self.client.get("/user/profile")
        self.assertEqual(res.status_code, 200)

        # Login with unique test nickname
        login_res = self.client.post("/user/login", json={"username": "testuser_unique"})
        self.assertEqual(login_res.status_code, 200)
        login_data = login_res.json()
        self.assertEqual(login_data["username"], "testuser_unique")
        self.assertIn("testuser_unique", login_data["known_users"])

        # Fetch profile
        profile_res = self.client.get("/user/profile")
        self.assertEqual(profile_res.status_code, 200)
        profile_data = profile_res.json()
        self.assertEqual(profile_data["username"], "testuser_unique")

    def test_chat_with_custom_username(self):
        """Test POST /chat with custom nickname and verify system prompt."""
        mock_response = {
            "message": {
                "role": "assistant",
                "content": "Greetings alex, I am ready to assist you.",
                "tool_calls": [],
            }
        }
        with patch("core.llm.chat", new_callable=AsyncMock, return_value=mock_response) as mock_chat:
            res = self.client.post(
                "/chat",
                json={
                    "message": "Hello Maidere!",
                    "thread_id": "test-username-thread",
                    "username": "alex",
                },
            )
            self.assertEqual(res.status_code, 200)
            self.assertIn("alex", res.json()["response"])
            # Verify system prompt contained the username in agent turn
            first_call_args = mock_chat.call_args_list[0][0][0]
            sys_msg = next(m for m in first_call_args if m.get("role") == "system")
            self.assertIn("alex", sys_msg["content"])


if __name__ == "__main__":
    unittest.main()
