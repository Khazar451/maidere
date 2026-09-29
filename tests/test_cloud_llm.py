"""Tests for Cloud AI provider integration (NVIDIA NIM & OpenAI-compatible endpoints)."""

import json
import unittest
from unittest.mock import AsyncMock, patch
import httpx

from core import llm
from core.config import settings
from core.router import get_available_models


class TestCloudLLM(unittest.IsolatedAsyncioTestCase):
    """Test OpenAI-compatible cloud provider routing, streaming, and reasoning extraction."""

    def test_is_cloud_model_classification(self):
        """Verify model prefix and custom name detection for cloud routing."""
        self.assertTrue(llm.is_cloud_model("nvidia/nemotron-3-ultra-550b-a55b"))
        self.assertTrue(llm.is_cloud_model("meta/llama-3.3-70b-instruct"))
        self.assertTrue(llm.is_cloud_model("openai/gpt-4o"))
        self.assertTrue(llm.is_cloud_model("deepseek-ai/deepseek-r1"))
        self.assertTrue(llm.is_cloud_model("mistralai/mistral-large"))

        self.assertFalse(llm.is_cloud_model("qwen2.5:7b-instruct"))
        self.assertFalse(llm.is_cloud_model("deepseek-r1:7b"))
        self.assertFalse(llm.is_cloud_model("llama3.2:3b"))
        self.assertFalse(llm.is_cloud_model(None))
        self.assertFalse(llm.is_cloud_model(""))

    async def test_chat_cloud_missing_key_raises_error(self):
        """Verify clear ValueError when attempting cloud dispatch without API key."""
        with patch.object(settings, "cloud_api_key", ""), \
             patch.object(settings, "nvidia_api_key", ""), \
             patch.object(settings, "openai_api_key", ""):
            with self.assertRaises(ValueError) as ctx:
                await llm.chat(
                    messages=[{"role": "user", "content": "Hi"}],
                    model="nvidia/nemotron-3-ultra-550b-a55b",
                )
            self.assertIn("no NVIDIA_API_KEY or CLOUD_API_KEY", str(ctx.exception))

    @patch("httpx.AsyncClient.post")
    async def test_chat_cloud_non_streaming_success(self, mock_post):
        """Test non-streaming cloud chat with reasoning content formatting."""
        mock_response = httpx.Response(
            status_code=200,
            json={
                "id": "chatcmpl-test",
                "model": "nvidia/nemotron-3-ultra-550b-a55b",
                "choices": [
                    {
                        "message": {
                            "role": "assistant",
                            "content": "GPU computing excels at massive parallelism.",
                            "reasoning_content": "The user is asking about GPU computing. Let's craft an insightful response.",
                        },
                        "finish_reason": "stop",
                    }
                ],
            },
            request=httpx.Request("POST", "https://integrate.api.nvidia.com/v1/chat/completions"),
        )
        mock_post.return_value = mock_response

        with patch.object(settings, "nvidia_api_key", "nvapi-dummy-test-key"), \
             patch.object(settings, "nvidia_base_url", "https://integrate.api.nvidia.com/v1"):
            result = await llm.chat(
                messages=[{"role": "user", "content": "Tell me about GPUs"}],
                model="nvidia/nemotron-3-ultra-550b-a55b",
            )

        self.assertIn("message", result)
        content = result["message"]["content"]
        # Verify reasoning content is wrapped inside <think> tags
        self.assertIn("<think>", content)
        self.assertIn("Let's craft an insightful response.", content)
        self.assertIn("</think>", content)
        self.assertIn("GPU computing excels at massive parallelism.", content)

        # Verify POST payload and headers
        mock_post.assert_called_once()
        _, kwargs = mock_post.call_args
        self.assertEqual(kwargs["headers"]["Authorization"], "Bearer nvapi-dummy-test-key")
        self.assertEqual(kwargs["json"]["model"], "nvidia/nemotron-3-ultra-550b-a55b")
        self.assertTrue(kwargs["json"]["chat_template_kwargs"]["enable_thinking"])

    @patch("httpx.AsyncClient.stream")
    async def test_chat_cloud_streaming_with_reasoning_and_content(self, mock_stream):
        """Test streaming cloud chat emits reasoning tokens in <think> tags and content in real-time."""
        class MockCloudStream:
            async def __aenter__(self):
                return self

            async def __aexit__(self, exc_type, exc_val, exc_tb):
                pass

            def raise_for_status(self):
                pass

            async def aiter_lines(self):
                # Simulated SSE stream lines
                yield 'data: {"choices": [{"delta": {"reasoning_content": "Pondering "}}]}'
                yield 'data: {"choices": [{"delta": {"reasoning_content": "GPU wonders..."}}]}'
                yield 'data: {"choices": [{"delta": {"content": "Fast "}}]}'
                yield 'data: {"choices": [{"delta": {"content": "parallel chips!"}}]}'
                yield "data: [DONE]"

        mock_stream.return_value = MockCloudStream()

        streamed_tokens = []
        async def on_token_cb(token: str):
            streamed_tokens.append(token)

        with patch.object(settings, "nvidia_api_key", "nvapi-dummy-test-key"):
            result = await llm.chat(
                messages=[{"role": "user", "content": "Write poem"}],
                model="nvidia/nemotron-3-ultra-550b-a55b",
                on_token=on_token_cb,
            )

        # Verify streamed sequence contains <think> block and tokens
        self.assertIn("<think>\n", streamed_tokens)
        self.assertIn("Pondering ", streamed_tokens)
        self.assertIn("GPU wonders...", streamed_tokens)
        self.assertIn("\n</think>\n", streamed_tokens)
        self.assertIn("Fast ", streamed_tokens)
        self.assertIn("parallel chips!", streamed_tokens)

        # Verify full assembled content
        full_content = result["message"]["content"]
        self.assertTrue(full_content.startswith("<think>\nPondering GPU wonders...\n</think>\nFast parallel chips!"))

    @patch("httpx.AsyncClient.stream")
    async def test_chat_cloud_streaming_tool_call_assembly(self, mock_stream):
        """Test streaming cloud chat accumulates fragmented delta tool calls."""
        class MockToolStream:
            async def __aenter__(self):
                return self

            async def __aexit__(self, exc_type, exc_val, exc_tb):
                pass

            def raise_for_status(self):
                pass

            async def aiter_lines(self):
                yield 'data: {"choices": [{"delta": {"tool_calls": [{"index": 0, "id": "call_1", "function": {"name": "shell", "arguments": "{\\"command\\": "}}]}}]}'
                yield 'data: {"choices": [{"delta": {"tool_calls": [{"index": 0, "function": {"arguments": "\\"nvidia-smi\\"}"}}]}}]}'
                yield "data: [DONE]"

        mock_stream.return_value = MockToolStream()

        dummy_tokens = []
        async def on_tok(t):
            dummy_tokens.append(t)

        with patch.object(settings, "nvidia_api_key", "nvapi-dummy-test-key"):
            result = await llm.chat(
                messages=[{"role": "user", "content": "Check GPUs"}],
                model="nvidia/nemotron-3-ultra-550b-a55b",
                on_token=on_tok,
            )

        self.assertIn("tool_calls", result["message"])
        tcs = result["message"]["tool_calls"]
        self.assertEqual(len(tcs), 1)
        self.assertEqual(tcs[0]["function"]["name"], "shell")
        self.assertEqual(tcs[0]["function"]["arguments"], '{"command": "nvidia-smi"}')

    async def test_router_lists_cloud_models_when_configured(self):
        """Test get_available_models includes configured cloud models when key is present."""
        with patch.object(settings, "nvidia_api_key", "nvapi-test-active"):
            models = await get_available_models(force_refresh=True)
            self.assertIn("nvidia/nemotron-3-ultra-550b-a55b", models)
            self.assertIn("meta/llama-3.3-70b-instruct", models)

    async def test_is_available_returns_true_for_configured_cloud(self):
        """Test llm.is_available returns True if cloud model is configured with an active key."""
        with patch.object(settings, "ollama_model", "nvidia/nemotron-3-ultra-550b-a55b"), \
             patch.object(settings, "nvidia_api_key", "nvapi-test-active"):
            available = await llm.is_available()
            self.assertTrue(available)


    def test_context_window_resolution(self):
        """Verify dynamic context resolution: local defaults to 8K, cloud to 64K, explicit 128K respected."""
        from core.agent import _resolve_num_ctx

        # 1. Local model default
        state_local = {"model": "qwen2.5:7b-instruct"}
        self.assertEqual(_resolve_num_ctx(state_local), 8192)

        # 2. Cloud model default
        state_cloud = {"model": "nvidia/nemotron-3-ultra-550b-a55b"}
        self.assertEqual(_resolve_num_ctx(state_cloud), 65536)

        # 3. Explicit 64K user selection
        state_64k = {"model": "nvidia/nemotron-3-ultra-550b-a55b", "num_ctx": 65536}
        self.assertEqual(_resolve_num_ctx(state_64k), 65536)

        # 4. Explicit 128K user selection
        state_128k = {"model": "nvidia/nemotron-3-ultra-550b-a55b", "num_ctx": 131072}
        self.assertEqual(_resolve_num_ctx(state_128k), 131072)

        # 5. Local model explicit selection
        state_local_custom = {"model": "qwen2.5:7b-instruct", "num_ctx": 16384}
        self.assertEqual(_resolve_num_ctx(state_local_custom), 16384)

    def test_trim_messages_respects_massive_cloud_context(self):
        """Verify that 64K/128K context retains long context that would otherwise be trimmed under 8K."""
        from core.agent import trim_messages_node
        from langchain_core.messages import AIMessage, HumanMessage

        # Create message history with ~50,000 characters (~12,500 tokens)
        # In 8K context (limit: 8192 * 0.75 * 4 = 24,576 chars), this would be pruned.
        # In 64K context (limit: 65536 * 0.75 * 4 = 196,608 chars), this should NOT be pruned.
        messages = [
            HumanMessage(content="A" * 20000),
            AIMessage(content="B" * 20000),
            HumanMessage(content="C" * 10000),
        ]

        # Under 8K context
        state_8k = {
            "messages": messages,
            "model": "qwen2.5:7b-instruct",
            "num_ctx": 8192,
        }
        res_8k = trim_messages_node(state_8k)
        self.assertIn("messages", res_8k)
        self.assertTrue(len(res_8k["messages"]) > 0, "8K context should have pruned older messages")

        # Under 64K context (Nemotron default)
        state_64k = {
            "messages": messages,
            "model": "nvidia/nemotron-3-ultra-550b-a55b",
            "num_ctx": 65536,
        }
        res_64k = trim_messages_node(state_64k)
        self.assertEqual(res_64k, {}, "64K context should retain all messages without pruning")

        # Under 128K context
        state_128k = {
            "messages": messages,
            "model": "nvidia/nemotron-3-ultra-550b-a55b",
            "num_ctx": 131072,
        }
        res_128k = trim_messages_node(state_128k)
        self.assertEqual(res_128k, {}, "128K context should retain all messages without pruning")


    def test_normalize_messages_for_cloud_tool_compliance(self):
        """Verify _normalize_messages_for_cloud formats tool calls with id, type, and tool_call_id."""
        from core.llm import _normalize_messages_for_cloud

        raw_history = [
            {"role": "user", "content": "Run research"},
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [
                    {
                        "function": {
                            "name": "delegate_task",
                            "arguments": {"task": "Analyze AI"},
                        }
                    }
                ],
            },
            {
                "role": "tool",
                "content": "Research findings: AI is evolving rapidly.",
            },
        ]

        normalized = _normalize_messages_for_cloud(raw_history)
        self.assertEqual(len(normalized), 3)

        # Assistant turn
        asst = normalized[1]
        self.assertEqual(asst["role"], "assistant")
        self.assertEqual(len(asst["tool_calls"]), 1)
        tc = asst["tool_calls"][0]
        self.assertTrue(tc["id"].startswith("call_"))
        self.assertEqual(tc["type"], "function")
        self.assertEqual(tc["function"]["name"], "delegate_task")
        # Arguments must be a valid JSON string
        self.assertIsInstance(tc["function"]["arguments"], str)
        self.assertEqual(json.loads(tc["function"]["arguments"]), {"task": "Analyze AI"})

        # Tool turn
        tool_msg = normalized[2]
        self.assertEqual(tool_msg["role"], "tool")
        self.assertEqual(tool_msg["tool_call_id"], tc["id"])
        self.assertEqual(tool_msg["content"], "Research findings: AI is evolving rapidly.")

    async def test_router_selects_nemotron_for_thinking_when_configured(self):
        """Verify router selects Nemotron for thinking/deep reasoning when configured."""
        from core.router import select_model, TaskComplexity

        with patch.object(settings, "nvidia_api_key", "nvapi-test-key"), \
             patch.object(settings, "cloud_model", "nvidia/nemotron-3-ultra-550b-a55b"):
            # When requested_model is "auto" and thinking_mode is True, and no local deepseek exists:
            with patch("core.router.get_available_models", return_value=["nvidia/nemotron-3-ultra-550b-a55b", "qwen2.5:7b-instruct"]):
                model, reason, complexity = await select_model(
                    requested_model="auto",
                    query="Prove Pythagorean theorem",
                    thinking_mode=True,
                )
                self.assertEqual(model, "nvidia/nemotron-3-ultra-550b-a55b")
                self.assertEqual(complexity, TaskComplexity.REASONING)

                # And for deep reasoning
                model2, reason2, complexity2 = await select_model(
                    requested_model="auto",
                    query="Analyze logic paradox",
                    deep_reasoning=True,
                )
                self.assertEqual(model2, "nvidia/nemotron-3-ultra-550b-a55b")
                self.assertEqual(complexity2, TaskComplexity.REASONING)


if __name__ == "__main__":
    unittest.main()


