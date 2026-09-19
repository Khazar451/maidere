"""Tests for Prometheus metrics, FastAPI instrumentator, and agent telemetry."""

import tempfile
import unittest
from unittest.mock import AsyncMock, patch

import httpx
from httpx import ASGITransport
from prometheus_client import REGISTRY

from api.app import create_app
from api.routes import set_graph
from core.agent import create_graph
from core.db import get_db, init_tables
from core.llm import chat as llm_chat
from core.memory import recall, store_memory
from core.metrics import (
    ACTIVE_SESSIONS,
    LLM_DURATION_SECONDS,
    LLM_REQUESTS_TOTAL,
    LLM_TOKENS_TOTAL,
    MEMORY_RECALLS_TOTAL,
    TOOL_CALLS_TOTAL,
    TOOL_DURATION_SECONDS,
)
from tools.registry import execute_tool


class TestPrometheusMetrics(unittest.IsolatedAsyncioTestCase):
    """Test metrics exposition and custom telemetry instrumentation."""

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

    async def test_metrics_endpoint_exposition(self):
        """Test GET /metrics returns valid Prometheus exposition text."""
        # Hit a route first
        await self.client.get("/health")
        
        response = await self.client.get("/metrics")
        self.assertEqual(response.status_code, 200)
        self.assertIn("text/plain", response.headers.get("content-type", ""))
        self.assertIn("http_requests_total", response.text)
        self.assertIn("maidere_tool_calls_total", response.text)
        self.assertIn("maidere_llm_requests_total", response.text)

    async def test_tool_metrics_recorded(self):
        """Test tool execution increments Prometheus tool counters and histograms."""
        initial_val = TOOL_CALLS_TOTAL.labels(tool_name="list_dir", status="success")._value.get()

        res = await execute_tool("list_dir", {"path": "."}, thread_id="test-thread-metrics")
        self.assertFalse(res.startswith("Error:"))

        new_val = TOOL_CALLS_TOTAL.labels(tool_name="list_dir", status="success")._value.get()
        self.assertEqual(new_val, initial_val + 1)

    async def test_llm_metrics_recorded(self):
        """Test LLM chat function records request, duration, and token metrics."""
        mock_response = {
            "model": "qwen2.5:7b-instruct",
            "eval_count": 42,
            "prompt_eval_count": 18,
            "eval_duration": 500000000,
            "message": {
                "role": "assistant",
                "content": "Hello telemetry test",
                "tool_calls": [],
            },
        }

        with patch("httpx.AsyncClient.post") as mock_post:
            mock_resp = AsyncMock()
            mock_resp.status_code = 200
            mock_resp.json = lambda: mock_response
            mock_resp.raise_for_status = lambda: None
            mock_post.return_value = mock_resp

            initial_reqs = LLM_REQUESTS_TOTAL.labels(model="qwen2.5:7b-instruct", status="success")._value.get()
            initial_prompt_tokens = LLM_TOKENS_TOTAL.labels(model="qwen2.5:7b-instruct", token_type="prompt")._value.get()
            initial_eval_tokens = LLM_TOKENS_TOTAL.labels(model="qwen2.5:7b-instruct", token_type="completion")._value.get()

            res = await llm_chat([{"role": "user", "content": "Ping"}], model="qwen2.5:7b-instruct")

            new_reqs = LLM_REQUESTS_TOTAL.labels(model="qwen2.5:7b-instruct", status="success")._value.get()
            new_prompt_tokens = LLM_TOKENS_TOTAL.labels(model="qwen2.5:7b-instruct", token_type="prompt")._value.get()
            new_eval_tokens = LLM_TOKENS_TOTAL.labels(model="qwen2.5:7b-instruct", token_type="completion")._value.get()

            self.assertEqual(new_reqs, initial_reqs + 1)
            self.assertEqual(new_prompt_tokens, initial_prompt_tokens + 18)
            self.assertEqual(new_eval_tokens, initial_eval_tokens + 42)

    async def test_memory_recall_metrics_recorded(self):
        """Test memory recall records Prometheus recall counters."""
        db = await get_db(self.temp_db_path)
        try:
            await store_memory(db, content="Telemetry metric testing snippet", session_id="session-metrics")

            initial_recalls = MEMORY_RECALLS_TOTAL.labels(status="success")._value.get()
            memories = await recall(db, query="telemetry metric testing", top_k=2)

            new_recalls = MEMORY_RECALLS_TOTAL.labels(status="success")._value.get()
            self.assertTrue(len(memories) > 0)
            self.assertEqual(new_recalls, initial_recalls + 1)
        finally:
            await db.close()


if __name__ == "__main__":
    unittest.main()
