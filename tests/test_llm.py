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

    @patch("httpx.AsyncClient.stream")
    async def test_chat_streaming_on_token(self, mock_stream):
        """Test chat with on_token streams chunks in real-time."""
        class MockStreamResponse:
            async def __aenter__(self):
                return self

            async def __aexit__(self, exc_type, exc_val, exc_tb):
                pass

            def raise_for_status(self):
                pass

            async def aiter_lines(self):
                yield '{"message": {"role": "assistant", "content": "Hello "}}'
                yield '{"message": {"role": "assistant", "content": "world!"}}'
                yield '{"done": true, "eval_count": 2}'

        mock_stream.return_value = MockStreamResponse()

        tokens = []
        async def token_cb(tok):
            tokens.append(tok)

        res = await llm.chat(messages=[{"role": "user", "content": "Hi"}], on_token=token_cb)
        self.assertEqual(tokens, ["Hello ", "world!"])
        self.assertEqual(res["message"]["content"], "Hello world!")
        self.assertTrue(res["done"])


if __name__ == "__main__":
    unittest.main()

