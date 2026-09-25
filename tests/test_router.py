"""Tests for Model Router, Task Complexity Classification, and Dynamic Model Routing."""

import tempfile
import unittest
from unittest.mock import AsyncMock, patch

from langchain_core.messages import AIMessage, HumanMessage

from core.agent import create_graph
from core.config import settings
from core.router import (
    TaskComplexity,
    classify_complexity,
    get_available_models,
    select_model,
)


class TestModelRouter(unittest.IsolatedAsyncioTestCase):
    """Test model routing logic and complexity classification."""

    def test_classify_complexity_simple(self):
        """Test simple queries classified as SIMPLE."""
        simple_queries = [
            "hello",
            "Hi there",
            "Good morning!",
            "How are you?",
            "Who are you?",
            "What can you do?",
            "Thanks a lot",
            "ping",
        ]
        for q in simple_queries:
            with self.subTest(query=q):
                self.assertEqual(classify_complexity(q), TaskComplexity.SIMPLE)

    def test_classify_complexity_complex(self):
        """Test tool, coding, and research queries classified as COMPLEX."""
        complex_queries = [
            "Write a Python script to calculate Fibonacci sequence",
            "Run bash command git status",
            "Search the web for latest LLM architectures",
            "Browse https://example.com and extract titles",
            "Schedule a task to backup database every day",
            "Analyze this csv dataset with pandas",
            "Conduct deep research on quantum computing and write obsidian note",
            "```python\ndef foo(): pass\n```",
        ]
        for q in complex_queries:
            with self.subTest(query=q):
                self.assertEqual(classify_complexity(q), TaskComplexity.COMPLEX)

    async def test_select_model_user_override(self):
        """Test explicit model override takes precedence."""
        model, reason, complexity = await select_model(
            requested_model="qwen2.5:14b",
            query="hello",
        )
        self.assertEqual(model, "qwen2.5:14b")
        self.assertEqual(reason, "user_override")
        self.assertEqual(complexity, TaskComplexity.SIMPLE)

    async def test_select_model_auto_simple_fast_available(self):
        """Test auto routing for simple task uses fast model when available."""
        with patch("core.router.get_available_models", new_callable=AsyncMock) as mock_models:
            mock_models.return_value = ["qwen2.5:7b-instruct", "qwen2.5:3b"]
            model, reason, complexity = await select_model(
                requested_model="auto",
                query="hello",
            )
            self.assertEqual(model, "qwen2.5:3b")
            self.assertEqual(reason, "auto_simple")
            self.assertEqual(complexity, TaskComplexity.SIMPLE)

    async def test_select_model_auto_simple_fast_unavailable_fallback(self):
        """Test auto routing falls back to primary model if fast model not installed."""
        with patch("core.router.get_available_models", new_callable=AsyncMock) as mock_models:
            mock_models.return_value = ["qwen2.5:7b-instruct"]  # only 7b installed
            model, reason, complexity = await select_model(
                requested_model="auto",
                query="hello",
            )
            self.assertEqual(model, settings.ollama_model)
            self.assertEqual(reason, "fallback_primary")
            self.assertEqual(complexity, TaskComplexity.SIMPLE)

    def test_classify_complexity_reasoning(self):
        """Test deep reasoning, proofs, and logic puzzles classified as REASONING."""
        reasoning_queries = [
            "Prove that the square root of 2 is irrational",
            "Solve for x in this calculus differential equation",
            "Give me a step-by-step proof for Fermat's little theorem",
            "Here is a logic puzzle with 3 doors and two guards",
            "Think deeply through the game theory implications of this decision",
            "Formal logic deduction: if P then Q, not Q, deduce P",
        ]
        for q in reasoning_queries:
            with self.subTest(query=q):
                self.assertEqual(classify_complexity(q), TaskComplexity.REASONING)

    async def test_select_model_auto_reasoning_available(self):
        """Test auto routing for reasoning task routes to thinking model when available."""
        with patch("core.router.get_available_models", new_callable=AsyncMock) as mock_models:
            mock_models.return_value = ["qwen2.5:7b-instruct", "qwen2.5:3b", "deepseek-r1:7b"]
            model, reason, complexity = await select_model(
                requested_model="auto",
                query="Prove that there are infinitely many primes",
            )
            self.assertEqual(model, "deepseek-r1:7b")
            self.assertEqual(reason, "auto_reasoning")
            self.assertEqual(complexity, TaskComplexity.REASONING)

    async def test_select_model_auto_reasoning_fallback(self):
        """Test auto routing for reasoning falls back to primary 7B if thinking model not installed."""
        with patch("core.router.get_available_models", new_callable=AsyncMock) as mock_models:
            mock_models.return_value = ["qwen2.5:7b-instruct", "qwen2.5:3b"]
            model, reason, complexity = await select_model(
                requested_model="auto",
                query="Prove that there are infinitely many primes",
            )
            self.assertEqual(model, settings.ollama_model)
            self.assertEqual(reason, "fallback_primary")
            self.assertEqual(complexity, TaskComplexity.REASONING)

    async def test_select_model_auto_complex(self):
        """Test auto routing for complex task uses primary 7B model."""
        with patch("core.router.get_available_models", new_callable=AsyncMock) as mock_models:
            mock_models.return_value = ["qwen2.5:7b-instruct", "qwen2.5:3b"]
            model, reason, complexity = await select_model(
                requested_model="auto",
                query="Write a python script fib.py and execute it",
            )
            self.assertEqual(model, settings.ollama_model)
            self.assertEqual(reason, "auto_complex")
            self.assertEqual(complexity, TaskComplexity.COMPLEX)

    async def test_select_model_forced_thinking_available(self):
        """Test thinking_mode=True routes to thinking model when available."""
        with patch("core.router.get_available_models", new_callable=AsyncMock) as mock_models:
            mock_models.return_value = ["qwen2.5:7b-instruct", "deepseek-r1:7b"]
            model, reason, complexity = await select_model(
                requested_model="auto",
                query="hello",
                thinking_mode=True,
            )
            self.assertEqual(model, "deepseek-r1:7b")
            self.assertEqual(reason, "forced_thinking")
            self.assertEqual(complexity, TaskComplexity.REASONING)

    async def test_select_model_forced_thinking_fallback(self):
        """Test thinking_mode=True falls back to primary model if no thinking model installed."""
        with patch("core.router.get_available_models", new_callable=AsyncMock) as mock_models:
            mock_models.return_value = ["qwen2.5:7b-instruct", "qwen2.5:3b"]
            model, reason, complexity = await select_model(
                requested_model="auto",
                query="hello",
                thinking_mode=True,
            )
            self.assertEqual(model, settings.ollama_model)
            self.assertEqual(reason, "thinking_mode_primary")
            self.assertEqual(complexity, TaskComplexity.REASONING)

    async def test_select_model_forced_thinking_user_override(self):
        """Test explicit model override is respected even when thinking_mode=True."""
        model, reason, complexity = await select_model(
            requested_model="qwen2.5:14b",
            query="hello",
            thinking_mode=True,
        )
        self.assertEqual(model, "qwen2.5:14b")
        self.assertEqual(reason, "user_override_thinking")
        self.assertEqual(complexity, TaskComplexity.REASONING)


class TestAgentRouterIntegration(unittest.IsolatedAsyncioTestCase):
    """Test dynamic routing inside the LangGraph agent think node."""

    async def asyncSetUp(self):
        self.temp_db = tempfile.NamedTemporaryFile(suffix=".db", delete=False)
        self.temp_db_path = self.temp_db.name
        self.temp_db.close()
        self.graph, self.checkpointer_ctx = await create_graph(self.temp_db_path)

    async def asyncTearDown(self):
        import os
        await self.checkpointer_ctx.__aexit__(None, None, None)
        if os.path.exists(self.temp_db_path):
            os.remove(self.temp_db_path)

    async def test_agent_routes_simple_greeting_to_fast_model(self):
        """Test agent graph routes simple greeting to fast model."""
        mock_response = {
            "model": "qwen2.5:3b",
            "message": {
                "role": "assistant",
                "content": "Hello! How can I help you today?",
                "tool_calls": [],
            },
        }

        with patch("core.router.get_available_models", new_callable=AsyncMock) as mock_models, \
             patch("core.agent.recall", new_callable=AsyncMock, return_value=[]), \
             patch("core.llm.chat", new_callable=AsyncMock, return_value=mock_response) as mock_chat:
            mock_models.return_value = ["qwen2.5:7b-instruct", "qwen2.5:3b"]

            config = {"configurable": {"thread_id": "test-router-simple"}}
            result = await self.graph.ainvoke(
                {
                    "messages": [HumanMessage(content="Hello")],
                    "memory_context": "No relevant memories found.",
                    "thread_id": "test-router-simple",
                    "model": "auto",
                },
                config=config,
            )

            self.assertEqual(mock_chat.call_args[1]["model"], "qwen2.5:3b")
            final_msg = result["messages"][-1]
            self.assertIn("Hello! How can I help", final_msg.content)

    async def test_agent_routes_coding_task_to_primary_model(self):
        """Test agent graph routes coding task to primary model."""
        mock_response = {
            "model": "qwen2.5:7b-instruct",
            "message": {
                "role": "assistant",
                "content": "I will write a python script to solve this.",
                "tool_calls": [],
            },
        }

        with patch("core.router.get_available_models", new_callable=AsyncMock) as mock_models, \
             patch("core.agent.recall", new_callable=AsyncMock, return_value=[]), \
             patch("core.llm.chat", new_callable=AsyncMock, return_value=mock_response) as mock_chat:
            mock_models.return_value = ["qwen2.5:7b-instruct", "qwen2.5:3b"]

            config = {"configurable": {"thread_id": "test-router-complex"}}
            result = await self.graph.ainvoke(
                {
                    "messages": [HumanMessage(content="Write a python script to analyze logs and run tests")],
                    "memory_context": "No relevant memories found.",
                    "thread_id": "test-router-complex",
                    "model": "auto",
                },
                config=config,
            )

            self.assertEqual(mock_chat.call_args[1]["model"], "qwen2.5:7b-instruct")
            final_msg = result["messages"][-1]
            self.assertIn("write a python script", final_msg.content)

    async def test_agent_routes_reasoning_task_to_thinking_model(self):
        """Test agent graph routes math proof / reasoning task to thinking model."""
        mock_response = {
            "model": "deepseek-r1:7b",
            "message": {
                "role": "assistant",
                "content": "<think>Let us prove sqrt(2) is irrational by contradiction.</think>Here is the proof.",
                "tool_calls": [],
            },
        }

        with patch("core.router.get_available_models", new_callable=AsyncMock) as mock_models, \
             patch("core.agent.recall", new_callable=AsyncMock, return_value=[]), \
             patch("core.llm.chat", new_callable=AsyncMock, return_value=mock_response) as mock_chat:
            mock_models.return_value = ["qwen2.5:7b-instruct", "qwen2.5:3b", "deepseek-r1:7b"]

            config = {"configurable": {"thread_id": "test-router-reasoning"}}
            result = await self.graph.ainvoke(
                {
                    "messages": [HumanMessage(content="Prove that the square root of 2 is irrational")],
                    "memory_context": "No relevant memories found.",
                    "thread_id": "test-router-reasoning",
                    "model": "auto",
                },
                config=config,
            )

            self.assertEqual(mock_chat.call_args[1]["model"], "deepseek-r1:7b")
            final_msg = result["messages"][-1]
            self.assertIn("Here is the proof", final_msg.content)

    async def test_agent_forced_thinking_mode_directive_injection(self):
        """Test agent graph injects thinking scratchpad directive into LLM messages when thinking_mode=True."""
        mock_response = {
            "model": "deepseek-r1:7b",
            "message": {
                "role": "assistant",
                "content": "<think>Step 1 analysis</think>Final thought response.",
                "tool_calls": [],
            },
        }

        with patch("core.router.get_available_models", new_callable=AsyncMock) as mock_models, \
             patch("core.agent.recall", new_callable=AsyncMock, return_value=[]), \
             patch("core.llm.chat", new_callable=AsyncMock, return_value=mock_response) as mock_chat:
            mock_models.return_value = ["qwen2.5:7b-instruct", "deepseek-r1:7b"]

            config = {"configurable": {"thread_id": "test-router-thinking-flag"}}
            result = await self.graph.ainvoke(
                {
                    "messages": [HumanMessage(content="Hello simple message")],
                    "memory_context": "No relevant memories found.",
                    "thread_id": "test-router-thinking-flag",
                    "model": "auto",
                    "thinking_mode": True,
                },
                config=config,
            )

            # Check that thinking directive was injected into ollama messages
            called_messages = mock_chat.call_args[0][0]
            system_directives = [m["content"] for m in called_messages if m.get("role") == "system"]
            has_directive = any("THINKING SCRATCHPAD: CHAIN-OF-THOUGHT DIRECTIVE" in d for d in system_directives)
            self.assertTrue(has_directive, "Expected THINKING SCRATCHPAD: CHAIN-OF-THOUGHT DIRECTIVE in system messages")

    async def test_agent_forced_deep_reasoning_mode_directive_injection(self):
        """Test agent graph injects System 2 divergent directive into LLM messages when deep_reasoning=True."""
        mock_response = {
            "model": "deepseek-r1:7b",
            "message": {
                "role": "assistant",
                "content": "<think>Divergent exploration</think>Provisional synthesis.",
                "tool_calls": [],
            },
        }

        with patch("core.router.get_available_models", new_callable=AsyncMock) as mock_models, \
             patch("core.agent.recall", new_callable=AsyncMock, return_value=[]), \
             patch("core.llm.chat", new_callable=AsyncMock, return_value=mock_response) as mock_chat:
            mock_models.return_value = ["qwen2.5:7b-instruct", "deepseek-r1:7b"]

            config = {"configurable": {"thread_id": "test-router-deep-reason-flag"}}
            result = await self.graph.ainvoke(
                {
                    "messages": [HumanMessage(content="Prove the theorem")],
                    "memory_context": "No relevant memories found.",
                    "thread_id": "test-router-deep-reason-flag",
                    "model": "auto",
                    "deep_reasoning": True,
                },
                config=config,
            )

            # Think node was the first call in the pipeline
            first_called_messages = mock_chat.call_args_list[0][0][0]
            system_directives = [m["content"] for m in first_called_messages if m.get("role") == "system"]
            has_directive = any("SYSTEM 2: DIVERGENT COGNITIVE EXPLORATION & DECONSTRUCTION" in d for d in system_directives)
            self.assertTrue(has_directive, "Expected SYSTEM 2 directive in system messages for deep_reasoning")
            # Verify that deep_reasoning triggered verification/convergence (multiple LLM calls)
            self.assertGreater(mock_chat.call_count, 1)


if __name__ == "__main__":
    unittest.main()
