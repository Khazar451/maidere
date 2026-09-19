"""Tests for LLM client integration and payload configuration."""

import unittest
from unittest.mock import AsyncMock, patch
import httpx

from core import llm
from core.config import settings
from tools.registry import get_tools_ollama_schemas


class TestLLMClient(unittest.IsolatedAsyncioTestCase):
    """Test LLM interaction and schema forwarding."""

    @patch("httpx.AsyncClient.post")
    async def test_chat_payload_structure(self, mock_post):
        """Test chat payload contains num_ctx=8192, stream=False, model, messages, and tools."""
        mock_response = httpx.Response(
            status_code=200,
            json={
                "model": settings.ollama_model,
                "message": {"role": "assistant", "content": "Hello!"},
                "eval_count": 10,
                "eval_duration": 1000000,
            },
            request=httpx.Request("POST", f"{settings.ollama_url}/api/chat"),
        )
        mock_post.return_value = mock_response

        messages = [{"role": "user", "content": "Hello"}]
        tools = get_tools_ollama_schemas()

        result = await llm.chat(messages=messages, tools=tools)

        self.assertIn("message", result)
        self.assertEqual(result["message"]["content"], "Hello!")

        # Verify POST request payload
        mock_post.assert_called_once()
        _, kwargs = mock_post.call_args
        json_payload = kwargs.get("json", {})
        self.assertEqual(json_payload["model"], settings.ollama_model)
        self.assertEqual(json_payload["options"]["num_ctx"], 8192)
        self.assertFalse(json_payload["stream"])
        self.assertEqual(json_payload["messages"], messages)
        self.assertEqual(json_payload["tools"], tools)

    @patch("httpx.AsyncClient.post")
    async def test_chat_custom_num_ctx_switching(self, mock_post):
        """Test chat payload respects custom num_ctx (e.g. 16K, 32K)."""
        mock_response = httpx.Response(
            status_code=200,
            json={
                "model": settings.ollama_model,
                "message": {"role": "assistant", "content": "Context test"},
            },
            request=httpx.Request("POST", f"{settings.ollama_url}/api/chat"),
        )
        mock_post.return_value = mock_response

        messages = [{"role": "user", "content": "Hi"}]

        # 16K test
        await llm.chat(messages=messages, num_ctx=16384)
        _, kwargs16 = mock_post.call_args
        self.assertEqual(kwargs16["json"]["options"]["num_ctx"], 16384)

        # 32K test
        await llm.chat(messages=messages, num_ctx=32768)
        _, kwargs32 = mock_post.call_args
        self.assertEqual(kwargs32["json"]["options"]["num_ctx"], 32768)


if __name__ == "__main__":
    unittest.main()
