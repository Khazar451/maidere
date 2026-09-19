import asyncio
import tempfile
import unittest
from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import numpy as np
from httpx import ASGITransport
from langchain_core.messages import AIMessage, HumanMessage

from api.app import create_app
from api.routes import set_graph
from core.agent import create_graph
from core.db import get_db, init_tables


class TestWebUIAndThreads(unittest.IsolatedAsyncioTestCase):
    """Test Web UI static serving and thread management."""

    @classmethod
    def setUpClass(cls):
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

        # Build graph
        self.graph, self.checkpointer_ctx = await create_graph(self.temp_db_path)
        set_graph(self.graph)
        self.app = create_app()
        self.transport = ASGITransport(app=self.app)
        self.client = httpx.AsyncClient(transport=self.transport, base_url="http://test")

    async def asyncTearDown(self):
        import os
        await self.client.aclose()
        await self.checkpointer_ctx.__aexit__(None, None, None)
        if os.path.exists(self.temp_db_path):
            os.remove(self.temp_db_path)

    async def test_serve_index_html(self):
        """Test GET / serves static/index.html with HTML content and sidebar reopen buttons."""
        response = await self.client.get("/")
        self.assertEqual(response.status_code, 200)
        self.assertIn("text/html", response.headers["content-type"])
        self.assertIn("Maidere", response.text)
        self.assertIn('data-theme="dark"', response.text)
        self.assertIn('id="btn-toggle-sidebar"', response.text)
        self.assertIn('id="btn-toggle-sidebar-main"', response.text)
        self.assertIn('id="btn-sidebar-edge"', response.text)

    async def test_threads_lifecycle(self):
        """Test listing and deleting conversation threads."""
        mock_response = {
            "message": {
                "role": "assistant",
                "content": "Conversation turn response.",
                "tool_calls": [],
            }
        }

        with patch("core.llm.chat", new_callable=AsyncMock, return_value=mock_response), \
             patch("core.config.settings.db_path", self.temp_db_path):

            # Create 2 threads via chat
            r1 = await self.client.post("/chat", json={"message": "First thread message", "thread_id": "thread-alpha"})
            self.assertEqual(r1.status_code, 200)
            r2 = await self.client.post("/chat", json={"message": "Second thread message", "thread_id": "thread-beta"})
            self.assertEqual(r2.status_code, 200)

            # List threads
            res = await self.client.get("/threads")
            self.assertEqual(res.status_code, 200)
            threads_data = res.json()["threads"]
            tids = [t["thread_id"] for t in threads_data]
            self.assertIn("thread-alpha", tids)
            self.assertIn("thread-beta", tids)

            # Delete thread-alpha
            del_res = await self.client.delete("/threads/thread-alpha")
            self.assertEqual(del_res.status_code, 200)
            self.assertEqual(del_res.json()["status"], "deleted")

            # Verify thread-alpha is removed
            res2 = await self.client.get("/threads")
            tids2 = [t["thread_id"] for t in res2.json()["threads"]]
            self.assertNotIn("thread-alpha", tids2)
            self.assertIn("thread-beta", tids2)

    async def test_list_models_and_model_switching(self):
        """Test GET /models and passing model in chat."""
        # 1. Test GET /models returns list of models
        res = await self.client.get("/models")
        self.assertEqual(res.status_code, 200)
        self.assertIn("models", res.json())
        self.assertTrue(len(res.json()["models"]) > 0)

        # 2. Test chat with explicit model
        mock_response = {
            "message": {
                "role": "assistant",
                "content": "Model switch response.",
                "tool_calls": [],
            }
        }
        with patch("core.llm.chat", new_callable=AsyncMock, return_value=mock_response) as mock_chat, \
             patch("core.config.settings.db_path", self.temp_db_path):
            r = await self.client.post("/chat", json={
                "message": "Hello with specific model",
                "thread_id": "thread-model-test",
                "model": "qwen2.5:7b-instruct",
            })
            self.assertEqual(r.status_code, 200)
            passed_models = [call[1].get("model") for call in mock_chat.call_args_list if "model" in call[1]]
            self.assertIn("qwen2.5:7b-instruct", passed_models)



if __name__ == "__main__":
    unittest.main()


