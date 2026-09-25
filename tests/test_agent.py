"""Unit and integration tests for LangGraph agentic loop."""

import asyncio
import tempfile
import unittest
from unittest.mock import AsyncMock, patch

from langchain_core.messages import AIMessage, HumanMessage, RemoveMessage, SystemMessage, ToolMessage
from langgraph.graph import END

from core.agent import (
    create_graph,
    should_continue,
    think_node,
    trim_messages_node,
    SYSTEM_PROMPT,
)
from tools.filesystem import WORKSPACE


class TestAgentTrimmingAndRouting(unittest.TestCase):
    """Test message trimming and conditional routing logic."""

    def test_trim_messages_under_limit(self):
        """Test that short conversations are not trimmed."""
        state = {
            "messages": [
                SystemMessage(content="You are Maidere."),
                HumanMessage(content="Hello!"),
            ],
            "memory_context": "No memories",
            "thread_id": "test-1",
        }
        result = trim_messages_node(state)
        self.assertEqual(result, {})

    def test_trim_messages_over_limit_preserves_system(self):
        """Test that long conversations are trimmed while preserving system prompt."""
        huge_text = "x" * 10000
        state = {
            "messages": [
                SystemMessage(content="System prompt", id="sys-1"),
                HumanMessage(content=huge_text, id="human-1"),
                AIMessage(content=huge_text, id="ai-1"),
                HumanMessage(content="Recent user message", id="human-2"),
            ],
            "memory_context": "No memories",
            "thread_id": "test-2",
        }
        # With CHAR_LIMIT = 24,000, total chars ~ 20,000 + 20 chars.
        # Let's adjust state with > 25,000 chars
        state["messages"].append(AIMessage(content=huge_text, id="ai-2"))

        result = trim_messages_node(state)
        self.assertIn("messages", result)
        removals = result["messages"]
        self.assertTrue(len(removals) > 0)
        # Ensure system message was never removed
        removed_ids = [r.id for r in removals]
        self.assertNotIn("sys-1", removed_ids)
        self.assertIn("human-1", removed_ids)

    def test_should_continue_routes_to_act(self):
        """Test should_continue routes to act when tool calls exist."""
        state = {
            "messages": [
                AIMessage(
                    content="",
                    tool_calls=[{"name": "read_file", "args": {"path": "test.txt"}, "id": "call-1"}],
                )
            ]
        }
        self.assertEqual(should_continue(state), "act")

    def test_should_continue_routes_to_end(self):
        """Test should_continue routes to END when no tool calls exist."""
        state = {
            "messages": [
                AIMessage(content="Final answer for user.")
            ]
        }
        self.assertEqual(should_continue(state), END)


class TestAgenticLoopIntegration(unittest.IsolatedAsyncioTestCase):
    """Test full LangGraph agentic loop execution with mocked LLM."""

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

    async def asyncTearDown(self):
        import os
        if os.path.exists(self.temp_db_path):
            os.remove(self.temp_db_path)
        # Clean workspace hello.py if created
        target = WORKSPACE / "hello_generated.py"
        if target.exists():
            target.unlink()

    async def test_full_multistep_tool_calling_loop(self):
        """Test multi-step agent flow: think -> write_file -> think -> shell -> think -> answer."""
        graph, checkpointer_ctx = await create_graph(self.temp_db_path)

        # Mock sequence of LLM responses:
        # 1. First think: call write_file to create hello_generated.py
        # 2. Second think: call shell to run python3 hello_generated.py
        # 3. Third think: provide final answer to user
        mock_responses = [
            {
                "message": {
                    "role": "assistant",
                    "content": "",
                    "tool_calls": [
                        {
                            "function": {
                                "name": "write_file",
                                "arguments": {
                                    "path": "hello_generated.py",
                                    "content": "print('Agentic loop working!')",
                                },
                            }
                        }
                    ],
                }
            },
            {
                "message": {
                    "role": "assistant",
                    "content": "",
                    "tool_calls": [
                        {
                            "function": {
                                "name": "shell",
                                "arguments": {
                                    "command": "python3 hello_generated.py",
                                },
                            }
                        }
                    ],
                }
            },
            {
                "message": {
                    "role": "assistant",
                    "content": "I created and executed hello_generated.py. Output: Agentic loop working!",
                    "tool_calls": [],
                }
            },
        ]

        with patch("core.llm.chat", side_effect=mock_responses) as mock_chat:
            config = {"configurable": {"thread_id": "test-multistep-thread"}}
            result = await graph.ainvoke(
                {
                    "messages": [HumanMessage(content="Create hello_generated.py and run it")],
                    "memory_context": "No relevant memories.",
                    "thread_id": "test-multistep-thread",
                },
                config=config,
            )

            # Verify that llm.chat was called 3 times (multi-step loop)
            self.assertEqual(mock_chat.call_count, 3)

            # Verify the final message from the agent
            final_message = result["messages"][-1]
            self.assertIsInstance(final_message, AIMessage)
            self.assertIn("Agentic loop working!", final_message.content)

            # Verify the file was actually written to workspace
            self.assertTrue((WORKSPACE / "hello_generated.py").exists())

        await checkpointer_ctx.__aexit__(None, None, None)

    async def test_remember_node_recalls_and_injects_memory(self):
        """Test that remember node retrieves stored memory and injects into LLM prompt."""
        from core.db import get_db, init_tables
        from core.memory import store_memory

        with patch("core.config.settings.db_path", self.temp_db_path):
            db = await get_db(self.temp_db_path)
            await init_tables(db)
            await store_memory(db, "User favorite beverage is Japanese Matcha Latte.", session_id="thread-pref")
            await db.close()

            graph, checkpointer_ctx = await create_graph(self.temp_db_path)

            mock_response = {
                "message": {
                    "role": "assistant",
                    "content": "Your favorite beverage is Japanese Matcha Latte!",
                    "tool_calls": [],
                }
            }

            with patch("core.llm.chat", return_value=mock_response) as mock_chat:
                config = {"configurable": {"thread_id": "thread-pref"}}
                result = await graph.ainvoke(
                    {
                        "messages": [HumanMessage(content="What is my favorite beverage?")],
                        "thread_id": "thread-pref",
                    },
                    config=config,
                )

                # Verify Ollama was invoked and system prompt contained the recalled memory
                self.assertEqual(mock_chat.call_count, 1)
                called_messages = mock_chat.call_args[0][0]
                system_msg = called_messages[0]["content"]
                self.assertIn("Japanese Matcha Latte", system_msg)

            await checkpointer_ctx.__aexit__(None, None, None)

    def test_trim_messages_scales_with_active_num_ctx(self):
        """Test that trim_messages_node dynamically accommodates larger context windows."""
        huge_text = "x" * 10000
        # 3 messages of 10,000 chars = 30,000 chars
        state_8k = {
            "messages": [
                SystemMessage(content="System", id="sys"),
                HumanMessage(content=huge_text, id="h1"),
                AIMessage(content=huge_text, id="a1"),
                HumanMessage(content=huge_text, id="h2"),
            ],
            "num_ctx": 8192,
        }
        # Under 8K, char limit is ~24,576 chars -> must trim
        res_8k = trim_messages_node(state_8k)
        self.assertTrue(len(res_8k.get("messages", [])) > 0)

        # Under 16K, char limit is int(16384 * 0.75) * 4 = 49,152 chars -> 30,000 chars fits without trimming!
        state_16k = {
            "messages": [
                SystemMessage(content="System", id="sys"),
                HumanMessage(content=huge_text, id="h1"),
                AIMessage(content=huge_text, id="a1"),
                HumanMessage(content=huge_text, id="h2"),
            ],
            "num_ctx": 16384,
        }
        res_16k = trim_messages_node(state_16k)
        self.assertEqual(res_16k, {})

    def test_think_node_passes_custom_num_ctx(self):
        """Test that think_node forwards active num_ctx to llm.chat."""
        from unittest.mock import AsyncMock
        from core.agent import think_node

        state = {
            "messages": [HumanMessage(content="Hello context test")],
            "thread_id": "test-num-ctx",
            "num_ctx": 32768,
        }

        captured_num_ctx = []

        async def fake_chat(messages, tools=None, model=None, num_ctx=None):
            captured_num_ctx.append(num_ctx)
            return {"message": {"role": "assistant", "content": "Context test ok"}}

        with patch("core.llm.chat", side_effect=fake_chat), \
             patch("core.agent.emit_agent_event", new_callable=AsyncMock):
            asyncio.run(think_node(state))

        self.assertEqual(captured_num_ctx, [32768])

    def test_think_tag_and_tool_call_extraction_with_thinking_model(self):
        """Test extract_text_tool_calls extracts tool calls when model outputs <think> tags."""
        from core.agent import extract_text_tool_calls

        ai_response = (
            "<think>\n"
            "The user needs research on DeepSeek R1.\n"
            "I should use the Agent tool to delegate this task.\n"
            "</think>\n"
            "Agent(prompt=\"Research DeepSeek R1 Distill Qwen 7B\", subagent_type=\"researcher\")"
        )
        calls = extract_text_tool_calls(ai_response, ["Agent", "delegate_task", "shell"])
        self.assertEqual(len(calls), 1)
        self.assertEqual(calls[0]["name"], "Agent")
        self.assertIn("DeepSeek R1", calls[0]["args"]["prompt"])
        self.assertEqual(calls[0]["args"]["subagent_type"], "researcher")

    def test_think_node_emits_think_logs_when_thinking_tags_present(self):
        """Test that think_node emits [THINK] agent logs when thinking tags are present."""
        from unittest.mock import AsyncMock
        from core.agent import think_node

        state = {
            "messages": [HumanMessage(content="Reason about this and use tools")],
            "thread_id": "test-think-emit",
            "thinking_mode": True,
        }

        mock_resp = {
            "message": {
                "role": "assistant",
                "content": "<think>Evaluating tool options for complex analysis.</think>",
                "tool_calls": [
                    {
                        "function": {
                            "name": "read_file",
                            "arguments": {"path": "main.py"},
                        },
                        "id": "tc-123",
                    }
                ],
            }
        }

        emitted_events = []

        async def fake_emit(thread_id, event):
            emitted_events.append(event)

        with patch("core.llm.chat", new_callable=AsyncMock, return_value=mock_resp), \
             patch("core.agent.emit_agent_event", side_effect=fake_emit):
            asyncio.run(think_node(state))

        think_logs = [e for e in emitted_events if e.get("type") == "agent_log" and "[THINK]" in e.get("text", "")]
        self.assertTrue(len(think_logs) > 0)
        self.assertIn("Evaluating tool options", think_logs[0]["text"])


class TestAgenticDeliberation(unittest.IsolatedAsyncioTestCase):
    """Test Socratic Critic-Refiner deliberation loop and deep reasoning mode."""

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

    def test_should_deliberate_greetings_return_false(self):
        """Test that simple greetings bypass deliberation."""
        from core.agent import should_deliberate
        state = {"deep_reasoning": False, "refinement_count": 0}
        self.assertFalse(should_deliberate(state, "hello", AIMessage(content="Hello!")))
        self.assertFalse(should_deliberate(state, "hi", AIMessage(content="Hi there!")))

    def test_should_deliberate_deep_reasoning_returns_true(self):
        """Test that explicit deep reasoning flag triggers deliberation."""
        from core.agent import should_deliberate
        state = {"deep_reasoning": True, "refinement_count": 0}
        self.assertTrue(should_deliberate(state, "What are the trade-offs of microservices?", AIMessage(content="Draft answer")))

    def test_should_deliberate_already_refined_returns_false(self):
        """Test that already refined state avoids infinite deliberation loops."""
        from core.agent import should_deliberate
        state = {"deep_reasoning": True, "refinement_count": 1}
        self.assertFalse(should_deliberate(state, "What are the trade-offs of microservices?", AIMessage(content="Refined answer")))

    def test_should_continue_routes_to_critique_when_deep_reasoning(self):
        """Test should_continue routes to verify/critique when deep reasoning is active."""
        state = {
            "messages": [
                HumanMessage(content="Explain the Byzantine Generals Problem in distributed systems"),
                AIMessage(content="Preliminary draft regarding consensus..."),
            ],
            "deep_reasoning": True,
            "refinement_count": 0,
        }
        self.assertIn(should_continue(state), ("verify", "critique"))

    def test_should_continue_thinking_mode_does_not_deliberate(self):
        """Test that single-pass thinking_mode does NOT route to verify and terminates directly."""
        state = {
            "messages": [
                HumanMessage(content="Explain the Byzantine Generals Problem in distributed systems"),
                AIMessage(content="<think>Step-by-step thinking.</think>Byzantine fault tolerance is..."),
            ],
            "thinking_mode": True,
            "deep_reasoning": False,
            "refinement_count": 0,
        }
        self.assertEqual(should_continue(state), END)

    async def test_critique_and_refine_nodes_flow(self):
        """Test critique_node and refine_node execution with mocked LLM."""
        from core.agent import critique_node, refine_node

        state = {
            "messages": [
                HumanMessage(content="Explain causality in distributed systems"),
                AIMessage(content="Event A happened before Event B because timestamps were ordered."),
            ],
            "thread_id": "test-delib-thread",
            "model": "qwen2.5:7b-instruct",
            "num_ctx": 8192,
        }

        mock_critique_resp = {
            "message": {
                "role": "assistant",
                "content": "1. Wall clock timestamps are not monotonic in distributed systems. Must mention Lamport timestamps or vector clocks.",
            }
        }
        mock_refine_resp = {
            "message": {
                "role": "assistant",
                "content": "In distributed systems, physical wall clock timestamps cannot guarantee causality due to clock drift. Lamport logical clocks or vector clocks must be used to establish partial ordering.",
            }
        }

        with patch("core.llm.chat", new_callable=AsyncMock, return_value=mock_critique_resp):
            critique_res = await critique_node(state)
            self.assertIn("critique", critique_res)
            self.assertIn("Lamport", critique_res["critique"])

        state["critique"] = critique_res["critique"]

        with patch("core.llm.chat", new_callable=AsyncMock, return_value=mock_refine_resp):
            refine_res = await refine_node(state)
            self.assertIn("messages", refine_res)
            final_msg = refine_res["messages"][0]
            self.assertIn("Lamport logical clocks", final_msg.content)
            self.assertEqual(refine_res.get("refinement_count"), 1)

    async def test_full_deliberation_integration_in_graph(self):
        """Test full LangGraph execution: think -> critique -> refine -> END when deep_reasoning is active."""
        temp_db = tempfile.NamedTemporaryFile(suffix=".db", delete=False)
        temp_db_path = temp_db.name
        temp_db.close()

        try:
            with patch("core.config.settings.db_path", temp_db_path):
                graph, checkpointer_ctx = await create_graph(temp_db_path)

                mock_responses = [
                    {
                        "message": {
                            "role": "assistant",
                            "content": "<think>Deconstructing problem.</think>Draft on CAP theorem.",
                            "tool_calls": [],
                        }
                    },
                    {
                        "message": {
                            "role": "assistant",
                            "content": "Critique: Distinguish PACELC theorem for network partitions vs latency.",
                        }
                    },
                    {
                        "message": {
                            "role": "assistant",
                            "content": "Comprehensive analysis of CAP and PACELC theorem trade-offs.",
                        }
                    },
                ]

                with patch("core.llm.chat", side_effect=mock_responses) as mock_chat:
                    config = {"configurable": {"thread_id": "test-delib-graph"}}
                    result = await graph.ainvoke(
                        {
                            "messages": [HumanMessage(content="Explain CAP theorem trade-offs")],
                            "thread_id": "test-delib-graph",
                            "deep_reasoning": True,
                            "refinement_count": 0,
                        },
                        config=config,
                    )

                    self.assertEqual(mock_chat.call_count, 3)
                    final_ai = result["messages"][-1]
                    self.assertIn("PACELC", final_ai.content)

                await checkpointer_ctx.__aexit__(None, None, None)
        finally:
            import os
            if os.path.exists(temp_db_path):
                os.remove(temp_db_path)


class TestTopicChangeDetection(unittest.TestCase):
    """Test topic-change detection and pruning."""

    def test_detect_topic_change_same_topic(self):
        """Same topic should NOT trigger topic change."""
        from core.agent import detect_topic_change
        query = "Explain the tie-breaker rule in the inventory system"
        context = "The inventory system has three containers X, Y, Z. Tie-breaker rules apply when containers have equal items."
        self.assertFalse(detect_topic_change(query, context))

    def test_detect_topic_change_different_topic(self):
        """Completely unrelated topics should trigger topic change."""
        from core.agent import detect_topic_change
        query = "Who was the first president of the United States?"
        context = "The inventory system has three containers X, Y, Z with 10 items each. Apply the tie-breaker rule."
        self.assertTrue(detect_topic_change(query, context))

    def test_detect_topic_change_empty_inputs(self):
        """Empty inputs should not trigger topic change."""
        from core.agent import detect_topic_change
        self.assertFalse(detect_topic_change("", "some context"))
        self.assertFalse(detect_topic_change("some query", ""))
        self.assertFalse(detect_topic_change("", ""))

    def test_detect_topic_change_short_words_only(self):
        """Queries with only stop words should not trigger topic change."""
        from core.agent import detect_topic_change
        self.assertFalse(detect_topic_change("is it", "the"))

    def test_trim_messages_prunes_on_topic_change(self):
        """trim_messages_node should prune old messages when topic changes."""
        state = {
            "messages": [
                SystemMessage(content="System prompt", id="sys-1"),
                HumanMessage(content="Simulate the inventory system with containers X Y Z and tie-breaker rules", id="h1"),
                AIMessage(content="The inventory system simulation shows container X=10 Y=10 Z=10 with tie-breaker", id="a1"),
                HumanMessage(content="Now apply the decay phase to container X with 5 tokens removed", id="h2"),
                AIMessage(content="After decay phase container X has 5 tokens remaining in the inventory", id="a2"),
                HumanMessage(content="Who was Napoleon Bonaparte and what battles did he fight?", id="h3"),
            ],
            "memory_context": "No memories",
            "thread_id": "test-topic",
        }
        result = trim_messages_node(state)
        self.assertIn("messages", result)
        removals = result["messages"]
        removed_ids = [r.id for r in removals]
        # System message must never be pruned
        self.assertNotIn("sys-1", removed_ids)
        # Old inventory messages should be pruned
        self.assertIn("h1", removed_ids)
        self.assertIn("a1", removed_ids)
        # The latest 2 non-system messages (a2, h3) should be kept
        self.assertNotIn("h3", removed_ids)

    def test_trim_messages_no_prune_same_topic(self):
        """trim_messages_node should NOT prune when topic stays the same."""
        state = {
            "messages": [
                SystemMessage(content="System prompt", id="sys-1"),
                HumanMessage(content="Simulate inventory system containers X Y Z", id="h1"),
                AIMessage(content="Container X=10 Y=10 Z=10 inventory system", id="a1"),
                HumanMessage(content="Apply tie-breaker rule to containers", id="h2"),
                AIMessage(content="Tie-breaker applied to container X first", id="a2"),
                HumanMessage(content="Now show the inventory after decay phase", id="h3"),
            ],
            "memory_context": "No memories",
            "thread_id": "test-same",
        }
        result = trim_messages_node(state)
        # Should return empty (no pruning) since topic is the same and under char limit
        self.assertEqual(result, {})

    def test_trim_messages_skip_with_few_messages(self):
        """Topic detection should not activate with fewer than 4 non-system messages."""
        state = {
            "messages": [
                SystemMessage(content="System prompt", id="sys-1"),
                HumanMessage(content="What is quantum computing?", id="h1"),
                AIMessage(content="Quantum computing uses qubits.", id="a1"),
                HumanMessage(content="Who was Napoleon?", id="h2"),
            ],
            "memory_context": "No memories",
            "thread_id": "test-few",
        }
        # Only 3 non-system messages, below threshold
        result = trim_messages_node(state)
        self.assertEqual(result, {})


class TestRouterWarningLogging(unittest.TestCase):
    """Test router terminal logging when DeepSeek is missing."""

    def test_thinking_mode_logs_warning_when_deepseek_missing(self):
        """select_model should log a warning when thinking_mode=True but DeepSeek is not installed."""
        from core.router import select_model

        async def run():
            with patch("core.router.get_available_models", new_callable=AsyncMock) as mock_models, \
                 patch("core.router.logger") as mock_logger:
                mock_models.return_value = ["qwen2.5:7b-instruct", "qwen2.5:3b"]
                mock_logger.awarn = AsyncMock()

                model, reason, _ = await select_model(
                    requested_model="auto",
                    query="solve this logic puzzle",
                    thinking_mode=True,
                )
                # Should fall back to primary model
                self.assertEqual(reason, "thinking_mode_primary")
                # Should have logged a warning
                mock_logger.awarn.assert_called_once()
                call_args = mock_logger.awarn.call_args
                self.assertEqual(call_args[0][0], "deepseek_not_available")

        asyncio.run(run())

    def test_auto_reasoning_logs_warning_when_deepseek_missing(self):
        """select_model should log a warning for REASONING queries when DeepSeek is not installed."""
        from core.router import select_model

        async def run():
            with patch("core.router.get_available_models", new_callable=AsyncMock) as mock_models, \
                 patch("core.router.logger") as mock_logger:
                mock_models.return_value = ["qwen2.5:7b-instruct", "qwen2.5:3b"]
                mock_logger.awarn = AsyncMock()

                model, reason, _ = await select_model(
                    requested_model=None,
                    query="prove the theorem step-by-step using formal logic",
                    thinking_mode=False,
                )
                self.assertEqual(reason, "fallback_primary")
                mock_logger.awarn.assert_called_once()
                call_args = mock_logger.awarn.call_args
                self.assertEqual(call_args[0][0], "deepseek_not_available")

        asyncio.run(run())


if __name__ == "__main__":
    unittest.main()
