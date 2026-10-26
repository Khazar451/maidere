"""Unit and integration tests for LangGraph agentic loop."""

import asyncio
import tempfile
import unittest
from unittest.mock import AsyncMock, patch

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langgraph.graph import END

from maidere.core.agent import (
    create_graph,
    should_continue,
    trim_messages_node,
)
from maidere.tools.filesystem import WORKSPACE


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
        state = {"messages": [AIMessage(content="Final answer for user.")]}
        self.assertEqual(should_continue(state), END)


class TestAgenticLoopIntegration(unittest.IsolatedAsyncioTestCase):
    """Test full LangGraph agentic loop execution with mocked LLM."""

    @classmethod
    def setUpClass(cls):
        from unittest.mock import MagicMock

        import numpy as np

        cls.mock_emb_model = MagicMock()
        cls.mock_emb_model.embed.side_effect = lambda texts: [np.array([0.05] * 384) for _ in texts]
        cls.patcher1 = patch("maidere.core.embeddings._get_model", return_value=cls.mock_emb_model)
        cls.patcher2 = patch("maidere.core.embeddings.embed_one", side_effect=lambda t: [0.05] * 384)
        cls.patcher3 = patch("maidere.core.embeddings.embed", side_effect=lambda texts: [[0.05] * 384 for _ in texts])
        cls.patcher4 = patch("maidere.core.memory.embed_one", side_effect=lambda t: [0.05] * 384)
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
        with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as temp_db:
            self.temp_db_path = temp_db.name

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

        with patch("maidere.core.llm.chat", side_effect=mock_responses) as mock_chat:
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
        from maidere.core.db import get_db, init_tables
        from maidere.core.memory import store_memory

        with patch("maidere.core.config.settings.db_path", self.temp_db_path):
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

            with patch("maidere.core.llm.chat", return_value=mock_response) as mock_chat:
                config = {"configurable": {"thread_id": "thread-pref"}}
                await graph.ainvoke(
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

        from maidere.core.agent import think_node

        state = {
            "messages": [HumanMessage(content="Hello context test")],
            "thread_id": "test-num-ctx",
            "num_ctx": 32768,
        }

        captured_num_ctx = []

        async def fake_chat(messages, tools=None, model=None, num_ctx=None):
            captured_num_ctx.append(num_ctx)
            return {"message": {"role": "assistant", "content": "Context test ok"}}

        with (
            patch("maidere.core.llm.chat", side_effect=fake_chat),
            patch("maidere.core.agent.emit_agent_event", new_callable=AsyncMock),
        ):
            asyncio.run(think_node(state))

        self.assertEqual(captured_num_ctx, [32768])

    def test_think_tag_and_tool_call_extraction_with_thinking_model(self):
        """Test extract_text_tool_calls extracts tool calls when model outputs <think> tags."""
        from maidere.core.agent import extract_text_tool_calls

        ai_response = (
            "<think>\n"
            "The user needs research on DeepSeek R1.\n"
            "I should use the Agent tool to delegate this task.\n"
            "</think>\n"
            'Agent(prompt="Research DeepSeek R1 Distill Qwen 7B", subagent_type="researcher")'
        )
        calls = extract_text_tool_calls(ai_response, ["Agent", "delegate_task", "shell"])
        self.assertEqual(len(calls), 1)
        self.assertEqual(calls[0]["name"], "Agent")
        self.assertIn("DeepSeek R1", calls[0]["args"]["prompt"])
        self.assertEqual(calls[0]["args"]["subagent_type"], "researcher")

    def test_think_node_emits_think_logs_when_thinking_tags_present(self):
        """Test that think_node emits [THINK] agent logs when thinking tags are present."""
        from unittest.mock import AsyncMock

        from maidere.core.agent import think_node

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

        with (
            patch("maidere.core.llm.chat", new_callable=AsyncMock, return_value=mock_resp),
            patch("maidere.core.agent.emit_agent_event", side_effect=fake_emit),
        ):
            asyncio.run(think_node(state))

        think_logs = [e for e in emitted_events if e.get("type") == "agent_log" and "[THINK]" in e.get("text", "")]
        self.assertTrue(len(think_logs) > 0)
        self.assertIn("Evaluating tool options", think_logs[0]["text"])


class TestAgenticDeliberation(unittest.IsolatedAsyncioTestCase):
    """Test Socratic Critic-Refiner deliberation loop and deep reasoning mode."""

    @classmethod
    def setUpClass(cls):
        from unittest.mock import MagicMock

        import numpy as np

        cls.mock_emb_model = MagicMock()
        cls.mock_emb_model.embed.side_effect = lambda texts: [np.array([0.05] * 384) for _ in texts]
        cls.patcher1 = patch("maidere.core.embeddings._get_model", return_value=cls.mock_emb_model)
        cls.patcher2 = patch("maidere.core.embeddings.embed_one", side_effect=lambda t: [0.05] * 384)
        cls.patcher3 = patch("maidere.core.embeddings.embed", side_effect=lambda texts: [[0.05] * 384 for _ in texts])
        cls.patcher4 = patch("maidere.core.memory.embed_one", side_effect=lambda t: [0.05] * 384)
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
        from maidere.core.agent import should_deliberate

        state = {"deep_reasoning": False, "refinement_count": 0}
        self.assertFalse(should_deliberate(state, "hello", AIMessage(content="Hello!")))
        self.assertFalse(should_deliberate(state, "hi", AIMessage(content="Hi there!")))

    def test_should_deliberate_deep_reasoning_returns_true(self):
        """Test that explicit deep reasoning flag triggers deliberation."""
        from maidere.core.agent import should_deliberate

        state = {"deep_reasoning": True, "refinement_count": 0}
        self.assertTrue(
            should_deliberate(state, "What are the trade-offs of microservices?", AIMessage(content="Draft answer"))
        )

    def test_should_deliberate_already_refined_returns_false(self):
        """Test that already refined state avoids infinite deliberation loops."""
        from maidere.core.agent import should_deliberate

        state = {"deep_reasoning": True, "refinement_count": 1}
        self.assertFalse(
            should_deliberate(state, "What are the trade-offs of microservices?", AIMessage(content="Refined answer"))
        )

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
        from maidere.core.agent import critique_node, refine_node

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

        with patch("maidere.core.llm.chat", new_callable=AsyncMock, return_value=mock_critique_resp):
            critique_res = await critique_node(state)
            self.assertIn("critique", critique_res)
            self.assertIn("Lamport", critique_res["critique"])

        state["critique"] = critique_res["critique"]

        with patch("maidere.core.llm.chat", new_callable=AsyncMock, return_value=mock_refine_resp):
            refine_res = await refine_node(state)
            self.assertIn("messages", refine_res)
            final_msg = refine_res["messages"][0]
            self.assertIn("Lamport logical clocks", final_msg.content)
            self.assertEqual(refine_res.get("refinement_count"), 1)

    async def test_full_deliberation_integration_in_graph(self):
        """Test full LangGraph execution: think -> critique -> refine -> END when deep_reasoning is active."""
        with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as temp_db:
            temp_db_path = temp_db.name

        try:
            with patch("maidere.core.config.settings.db_path", temp_db_path):
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

                with patch("maidere.core.llm.chat", side_effect=mock_responses) as mock_chat:
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
        from maidere.core.agent import detect_topic_change

        query = "Explain the tie-breaker rule in the inventory system"
        context = "The inventory system has three containers X, Y, Z. Tie-breaker rules apply when containers have equal items."
        self.assertFalse(detect_topic_change(query, context))

    def test_detect_topic_change_different_topic(self):
        """Completely unrelated topics should trigger topic change."""
        from maidere.core.agent import detect_topic_change

        query = "Who was the first president of the United States?"
        context = "The inventory system has three containers X, Y, Z with 10 items each. Apply the tie-breaker rule."
        self.assertTrue(detect_topic_change(query, context))

    def test_detect_topic_change_empty_inputs(self):
        """Empty inputs should not trigger topic change."""
        from maidere.core.agent import detect_topic_change

        self.assertFalse(detect_topic_change("", "some context"))
        self.assertFalse(detect_topic_change("some query", ""))
        self.assertFalse(detect_topic_change("", ""))

    def test_detect_topic_change_short_words_only(self):
        """Queries with only stop words should not trigger topic change."""
        from maidere.core.agent import detect_topic_change

        self.assertFalse(detect_topic_change("is it", "the"))

    def test_trim_messages_prunes_on_topic_change(self):
        """trim_messages_node should prune old messages when topic changes."""
        state = {
            "messages": [
                SystemMessage(content="System prompt", id="sys-1"),
                HumanMessage(
                    content="Simulate the inventory system with containers X Y Z and tie-breaker rules", id="h1"
                ),
                AIMessage(
                    content="The inventory system simulation shows container X=10 Y=10 Z=10 with tie-breaker", id="a1"
                ),
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
        from maidere.core.router import select_model

        async def run():
            with (
                patch("maidere.core.router.get_available_models", new_callable=AsyncMock) as mock_models,
                patch("maidere.core.router.logger") as mock_logger,
            ):
                mock_models.return_value = ["qwen2.5:7b-instruct", "qwen2.5:3b"]
                mock_logger.awarn = AsyncMock()

                _model, reason, _ = await select_model(
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
        from maidere.core.router import select_model

        async def run():
            with (
                patch("maidere.core.router.get_available_models", new_callable=AsyncMock) as mock_models,
                patch("maidere.core.router.logger") as mock_logger,
            ):
                mock_models.return_value = ["qwen2.5:7b-instruct", "qwen2.5:3b"]
                mock_logger.awarn = AsyncMock()

                _model, reason, _ = await select_model(
                    requested_model=None,
                    query="prove the theorem step-by-step using formal logic",
                    thinking_mode=False,
                )
                self.assertEqual(reason, "fallback_primary")
                mock_logger.awarn.assert_called_once()
                call_args = mock_logger.awarn.call_args
                self.assertEqual(call_args[0][0], "deepseek_not_available")


class TestActNodeConcurrencyPartitioning(unittest.IsolatedAsyncioTestCase):
    """Test per-call concurrency classification and act_node execution partitioning."""

    def test_is_tool_call_concurrent_safe(self):
        """Test classification of concurrent-safe vs serial maidere.tools."""
        from maidere.tools.registry import is_tool_call_concurrent_safe

        # Pure read-only tools
        self.assertTrue(is_tool_call_concurrent_safe("read_file"))
        self.assertTrue(is_tool_call_concurrent_safe("list_dir"))
        self.assertTrue(is_tool_call_concurrent_safe("web_search"))
        self.assertTrue(is_tool_call_concurrent_safe("browser"))
        self.assertTrue(is_tool_call_concurrent_safe("delegate_task"))
        self.assertTrue(is_tool_call_concurrent_safe("Agent"))

        # Mutating tools
        self.assertFalse(is_tool_call_concurrent_safe("write_file"))
        self.assertFalse(is_tool_call_concurrent_safe("replace_file_content"))
        self.assertFalse(is_tool_call_concurrent_safe("edit_file"))
        self.assertFalse(is_tool_call_concurrent_safe("rollback_file"))
        self.assertFalse(is_tool_call_concurrent_safe("shell"))
        self.assertFalse(is_tool_call_concurrent_safe("code_runner"))

        # Mixed action tools evaluated per-call
        self.assertTrue(is_tool_call_concurrent_safe("manage_memory", {"action": "read_topic"}))
        self.assertTrue(is_tool_call_concurrent_safe("manage_memory", {"action": "list_topics"}))
        self.assertTrue(is_tool_call_concurrent_safe("manage_memory", {"action": "view_rules"}))
        self.assertFalse(is_tool_call_concurrent_safe("manage_memory", {"action": "save_fact"}))

        self.assertTrue(is_tool_call_concurrent_safe("obsidian", {"action": "read_note"}))
        self.assertTrue(is_tool_call_concurrent_safe("obsidian", {"action": "search_notes"}))
        self.assertFalse(is_tool_call_concurrent_safe("obsidian", {"action": "write_note"}))

        self.assertTrue(is_tool_call_concurrent_safe("agy_staffer", {"persona": "researcher"}))
        self.assertTrue(is_tool_call_concurrent_safe("agy_staffer", {"persona": "reviewer"}))
        self.assertFalse(is_tool_call_concurrent_safe("agy_staffer", {"persona": "implementer"}))
        self.assertFalse(is_tool_call_concurrent_safe("agy_staffer", {"persona": "staffer"}))

    async def test_act_node_partitions_and_preserves_order(self):
        """Test act_node partitions mixed concurrent and serial tools while preserving result order."""
        from maidere.core.agent import act_node

        execution_log = []

        async def mock_execute_tool(name: str, args: dict, thread_id: str | None = None, **kwargs):
            execution_log.append(f"start:{name}:{args.get('id')}")
            # simulate slight delay to test concurrency
            await asyncio.sleep(0.01)
            execution_log.append(f"end:{name}:{args.get('id')}")
            return f"result_{name}_{args.get('id')}"

        state = {
            "messages": [
                AIMessage(
                    content="",
                    tool_calls=[
                        {"name": "read_file", "args": {"path": "a.txt", "id": 1}, "id": "c1"},
                        {"name": "read_file", "args": {"path": "b.txt", "id": 2}, "id": "c2"},
                        {"name": "write_file", "args": {"path": "c.txt", "id": 3, "content": "x"}, "id": "c3"},
                        {"name": "read_file", "args": {"path": "d.txt", "id": 4}, "id": "c4"},
                    ],
                )
            ],
            "thread_id": "test-partition",
        }

        with patch("maidere.core.agent.execute_tool", side_effect=mock_execute_tool):
            result = await act_node(state)

        tool_msgs = result.get("messages", [])
        self.assertEqual(len(tool_msgs), 4)

        # Verify results preserve original sequence
        self.assertEqual(tool_msgs[0].content, "result_read_file_1")
        self.assertEqual(tool_msgs[1].content, "result_read_file_2")
        self.assertEqual(tool_msgs[2].content, "result_write_file_3")
        self.assertEqual(tool_msgs[3].content, "result_read_file_4")

        # Verify write_file only started AFTER read_file 1 & 2 both ended
        c1_end = execution_log.index("end:read_file:1")
        c2_end = execution_log.index("end:read_file:2")
        c3_start = execution_log.index("start:write_file:3")
        c3_end = execution_log.index("end:write_file:3")
        c4_start = execution_log.index("start:read_file:4")

        self.assertGreater(c3_start, c1_end)
        self.assertGreater(c3_start, c2_end)
        self.assertGreater(c4_start, c3_end)


class TestThinkNodeStreaming(unittest.IsolatedAsyncioTestCase):
    """Test real-time token streaming from think_node."""

    async def test_think_node_streams_tokens_to_callback(self):
        """Verify think_node passes on_token callback to llm.chat when direct answer is produced."""
        from langchain_core.messages import HumanMessage

        from maidere.core.agent import think_node

        streamed_tokens = []

        async def dummy_callback(token: str):
            streamed_tokens.append(token)

        async def fake_streaming_chat(messages, tools=None, model=None, num_ctx=None, on_token=None):
            if on_token:
                await on_token("Hello ")
                await on_token("from ")
                await on_token("stream!")
            return {"message": {"role": "assistant", "content": "Hello from stream!"}}

        state = {
            "messages": [HumanMessage(content="Hello")],
            "thread_id": "test-stream-thread",
            "callback": dummy_callback,
            "skip_tools": True,  # Ensures tools_schemas is empty
        }

        with (
            patch("maidere.core.llm.chat", side_effect=fake_streaming_chat),
            patch("maidere.core.agent.emit_agent_event", new_callable=AsyncMock),
        ):
            result = await think_node(state)

        self.assertEqual(streamed_tokens, ["Hello ", "from ", "stream!"])
        self.assertEqual(result["messages"][0].content, "Hello from stream!")


class TestEmptyModelResponse(unittest.IsolatedAsyncioTestCase):
    """A reply with no text and no tool call must be retried, then explained, never returned blank."""

    EMPTY = {"message": {"role": "assistant", "content": "", "tool_calls": []}}

    @staticmethod
    def _answer(text: str) -> dict:
        return {"message": {"role": "assistant", "content": text, "tool_calls": []}}

    @staticmethod
    def _state(**extra) -> dict:
        from langchain_core.messages import HumanMessage

        return {
            "messages": [HumanMessage(content="Hello there my friend")],
            "thread_id": "empty-reply-thread",
            "skip_tools": True,
            **extra,
        }

    async def _think(self, responses, empty_response_retries=2):
        from maidere.core.agent import think_node
        from maidere.core.config import settings

        with (
            patch("maidere.core.llm.chat", side_effect=responses) as chat,
            patch("maidere.core.agent.emit_agent_event", new_callable=AsyncMock),
            patch.object(settings, "empty_response_retries", empty_response_retries),
        ):
            result = await think_node(self._state())
        return result, chat

    async def test_normal_reply_makes_a_single_call(self):
        result, chat = await self._think([self._answer("Hi!")])

        self.assertEqual(result["messages"][0].content, "Hi!")
        self.assertEqual(chat.call_count, 1)

    async def test_retry_recovers_from_an_empty_reply(self):
        result, chat = await self._think([self.EMPTY, self._answer("Recovered answer")])

        self.assertEqual(result["messages"][0].content, "Recovered answer")
        self.assertEqual(chat.call_count, 2)
        self.assertFalse(result["messages"][0].additional_kwargs)

    async def test_retry_tells_the_model_which_tools_exist(self):
        from maidere.core.agent import EMPTY_RESPONSE_RETRY_NOTE, EMPTY_RESPONSE_RETRY_NOTE_NO_TOOLS

        _, chat = await self._think([self.EMPTY, self._answer("ok")])
        first, retry = chat.call_args_list[0].args[0], chat.call_args_list[1].args[0]

        self.assertEqual(len(retry), len(first) + 1)
        self.assertEqual(retry[-1], {"role": "system", "content": EMPTY_RESPONSE_RETRY_NOTE_NO_TOOLS})
        self.assertIn("Agent tool", EMPTY_RESPONSE_RETRY_NOTE)
        self.assertIn("web_search", EMPTY_RESPONSE_RETRY_NOTE)

    async def test_retry_can_return_a_tool_call(self):
        from maidere.core.agent import think_node
        from maidere.core.config import settings

        tool_reply = {
            "message": {
                "role": "assistant",
                "content": "",
                "tool_calls": [{"function": {"name": "Agent", "arguments": {"prompt": "look it up"}}}],
            }
        }
        state = self._state()
        state.pop("skip_tools")
        with (
            patch("maidere.core.llm.chat", side_effect=[self.EMPTY, tool_reply]) as chat,
            patch("maidere.core.agent.emit_agent_event", new_callable=AsyncMock),
            patch.object(settings, "empty_response_retries", 2),
        ):
            result = await think_node(state)

        self.assertEqual(chat.call_count, 2)
        self.assertEqual(result["messages"][0].tool_calls[0]["name"], "Agent")

    async def test_gives_up_after_the_configured_retries_with_a_clear_message(self):
        from maidere.core.agent import EMPTY_RESPONSE_NOTICE

        result, chat = await self._think([self.EMPTY] * 5, empty_response_retries=2)

        message = result["messages"][0]
        self.assertEqual(chat.call_count, 3)  # first attempt + 2 retries
        self.assertEqual(message.content, EMPTY_RESPONSE_NOTICE)
        self.assertTrue(message.additional_kwargs["empty_response_fallback"])

    async def test_retries_can_be_disabled_but_reply_is_never_blank(self):
        from maidere.core.agent import EMPTY_RESPONSE_NOTICE

        result, chat = await self._think([self.EMPTY], empty_response_retries=0)

        self.assertEqual(chat.call_count, 1)
        self.assertEqual(result["messages"][0].content, EMPTY_RESPONSE_NOTICE)

    async def test_whitespace_only_reply_counts_as_empty(self):
        result, _ = await self._think([self._answer("  \n "), self._answer("Real answer")])

        self.assertEqual(result["messages"][0].content, "Real answer")

    async def test_fallback_notice_is_not_sent_to_the_verifier(self):
        from langchain_core.messages import AIMessage, HumanMessage
        from langgraph.graph import END

        from maidere.core.agent import EMPTY_RESPONSE_NOTICE, should_continue, should_deliberate

        notice = AIMessage(content=EMPTY_RESPONSE_NOTICE, additional_kwargs={"empty_response_fallback": True})
        state = {
            "messages": [HumanMessage(content="Prove that the square root of 2 is irrational"), notice],
            "deep_reasoning": True,
        }

        self.assertFalse(should_deliberate(state, "Prove that the square root of 2 is irrational", notice))
        self.assertEqual(should_continue(state), END)
        # The same state with a real draft still deliberates.
        real = AIMessage(content="Assume sqrt(2) = p/q in lowest terms...")
        self.assertTrue(should_deliberate(state, "Prove that the square root of 2 is irrational", real))


class TestDirectWebCallRedirect(unittest.IsolatedAsyncioTestCase):
    """The orchestrator never runs web_search/browser itself; direct calls become researcher delegations."""

    def test_web_search_becomes_a_researcher_delegation(self):
        from maidere.core.agent import redirect_direct_web_calls

        calls = [{"name": "web_search", "args": {"query": "python asyncio docs"}, "id": "c1"}]
        out = redirect_direct_web_calls(calls, "fallback")

        self.assertEqual(len(out), 1)
        self.assertEqual(out[0]["name"], "Agent")
        self.assertEqual(out[0]["id"], "c1")
        self.assertEqual(out[0]["args"]["subagent_type"], "researcher")
        self.assertEqual(out[0]["args"]["prompt"], "Search the web for: python asyncio docs")

    def test_browser_call_is_delegated_and_other_calls_pass_through(self):
        from maidere.core.agent import redirect_direct_web_calls

        shell = {"name": "shell", "args": {"command": "ls"}, "id": "s1"}
        out = redirect_direct_web_calls(
            [shell, {"name": "browser", "args": {"url": "https://example.org"}, "id": "b1"}], "fallback"
        )

        self.assertEqual(out[0], shell)
        self.assertEqual(out[1]["name"], "Agent")
        self.assertIn("https://example.org", out[1]["args"]["prompt"])

    def test_duplicate_and_empty_queries_are_handled(self):
        from maidere.core.agent import redirect_direct_web_calls

        calls = [
            {"name": "web_search", "args": {"query": "same"}, "id": "1"},
            {"name": "web_search", "args": {"query": "same"}, "id": "2"},
            {"name": "web_search", "args": {}, "id": "3"},
        ]
        out = redirect_direct_web_calls(calls, "what the user asked")

        self.assertEqual(
            [c["args"]["prompt"] for c in out], ["Search the web for: same", "Search the web for: what the user asked"]
        )

    async def test_model_is_offered_web_tools_only_while_it_can_delegate(self):
        from langchain_core.messages import HumanMessage

        from maidere.core.agent import think_node

        seen: list[set[str]] = []

        async def chat(messages, tools=None, model=None, num_ctx=None, on_token=None):
            seen.append({t["function"]["name"] for t in (tools or [])})
            return {"message": {"role": "assistant", "content": "ok", "tool_calls": []}}

        with (
            patch("maidere.core.llm.chat", side_effect=chat),
            patch("maidere.core.agent.emit_agent_event", new_callable=AsyncMock),
        ):
            await think_node({"messages": [HumanMessage(content="Hello there my friend")], "thread_id": "t1"})
            await think_node(
                {"messages": [HumanMessage(content="Hello there my friend")], "thread_id": "t2", "has_delegated": True}
            )

        self.assertTrue({"Agent", "web_search", "browser"} <= seen[0])
        self.assertNotIn("Agent", seen[1])
        self.assertNotIn("web_search", seen[1])


class TestToolsOmittedForUnsupportedModels(unittest.IsolatedAsyncioTestCase):
    """Vision models reject requests that carry tools, so the agent must not send any."""

    def setUp(self):
        from maidere.core import llm

        llm._TOOL_SUPPORT.clear()
        self.addCleanup(llm._TOOL_SUPPORT.clear)

    @staticmethod
    def _show(capabilities):
        import httpx

        return httpx.Response(
            200, json={"capabilities": capabilities}, request=httpx.Request("POST", "http://x/api/show")
        )

    async def test_capabilities_decide_tool_support_and_are_cached(self):
        from maidere.core import llm

        with patch(
            "httpx.AsyncClient.post", side_effect=[self._show(["completion", "vision"]), self._show(["tools"])]
        ) as post:
            self.assertFalse(await llm.model_supports_tools("qwen2.5vl:3b"))
            self.assertFalse(await llm.model_supports_tools("qwen2.5vl:3b"))  # cached
            self.assertTrue(await llm.model_supports_tools("qwen2.5:7b-instruct"))
        self.assertEqual(post.call_count, 2)

    async def test_unreachable_ollama_or_cloud_model_assumes_support_without_caching(self):
        import httpx

        from maidere.core import llm

        with patch("httpx.AsyncClient.post", side_effect=httpx.ConnectError("down")):
            self.assertTrue(await llm.model_supports_tools("qwen2.5vl:3b"))
        self.assertNotIn("qwen2.5vl:3b", llm._TOOL_SUPPORT)
        with patch("httpx.AsyncClient.post", side_effect=AssertionError("no request for cloud models")):
            self.assertTrue(await llm.model_supports_tools("nvidia/nemotron-3-ultra-550b-a55b"))

    async def test_think_node_sends_no_tools_to_a_model_without_tool_support(self):
        from langchain_core.messages import HumanMessage

        from maidere.core.agent import think_node

        seen = []

        async def chat(messages, tools=None, model=None, num_ctx=None, on_token=None):
            seen.append((model, tools))
            return {"message": {"role": "assistant", "content": "A red button.", "tool_calls": []}}

        state = {
            "messages": [HumanMessage(content="What color is the button?")],
            "thread_id": "vl",
            "model": "vision-x:3b",
        }
        for supports, expect_tools in ((False, False), (True, True)):
            seen.clear()
            with (
                patch("maidere.core.llm.chat", side_effect=chat),
                patch("maidere.core.llm.model_supports_tools", new=AsyncMock(return_value=supports)),
                patch("maidere.core.agent.emit_agent_event", new_callable=AsyncMock),
            ):
                await think_node(state)
            self.assertEqual(seen[0][0], "vision-x:3b")
            self.assertEqual(bool(seen[0][1]), expect_tools)


class TestTypingSensitiveEdges(unittest.IsolatedAsyncioTestCase):
    """Behaviour that the strict type check made explicit."""

    async def test_delegated_turn_without_tools_does_not_crash(self):
        """has_delegated filters the tool list; with tools disabled that list is None."""
        from langchain_core.messages import HumanMessage

        from maidere.core.agent import think_node

        async def chat(messages, tools=None, model=None, num_ctx=None, on_token=None):
            return {"message": {"role": "assistant", "content": "Done.", "tool_calls": []}}

        state = {
            "messages": [HumanMessage(content="Hello there my friend")],
            "thread_id": "delegated-no-tools",
            "has_delegated": True,
            "skip_tools": True,
        }
        with (
            patch("maidere.core.llm.chat", side_effect=chat),
            patch("maidere.core.agent.emit_agent_event", new_callable=AsyncMock),
        ):
            result = await think_node(state)

        self.assertEqual(result["messages"][0].content, "Done.")

    def test_message_text_handles_missing_and_non_string_content(self):
        from langchain_core.messages import AIMessage

        from maidere.core.agent import _message_text

        self.assertEqual(_message_text(None), "")
        self.assertEqual(_message_text(AIMessage(content="draft")), "draft")
        self.assertEqual(
            _message_text(AIMessage(content=[{"type": "text", "text": "x"}])), str([{"type": "text", "text": "x"}])
        )

    def test_restore_markdown_urls_leaves_short_brackets_and_citations_intact(self):
        from langchain_core.messages import ToolMessage

        from maidere.core.agent import restore_markdown_urls

        history = [
            ToolMessage(
                content="[Python 3.12 Release Notes](https://docs.python.org/3/whatsnew/3.12.html)",
                tool_call_id="tc1",
            )
        ]
        draft = "Index with arr[i] or items[1] and cite [Python 3.12 Release Notes]."
        restored = restore_markdown_urls(draft, history)
        self.assertIn("arr[i]", restored)
        self.assertNotIn("arr[i](", restored)
        self.assertNotIn("items[1](", restored)
        self.assertIn("https://docs.python.org/3/whatsnew/3.12.html", restored)

    def test_system_prompt_template_tolerates_json_braces(self):
        from maidere.core.agent import SystemPromptTemplate

        tpl = SystemPromptTemplate('Hello {username}, output JSON like {"status": "ok"} with {unknown_var}.')
        rendered = tpl.format(username="Khazar")
        self.assertEqual(rendered, 'Hello Khazar, output JSON like {"status": "ok"} with {unknown_var}.')


class TestEmptyReplyThroughGraph(unittest.IsolatedAsyncioTestCase):
    async def test_deep_reasoning_turn_with_silent_model_ends_with_notice_and_no_critic_calls(self):
        import os
        import tempfile

        from langchain_core.messages import HumanMessage

        from maidere.core.agent import EMPTY_RESPONSE_NOTICE, create_graph

        with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as db:
            db_name = db.name
        calls: list[int] = []

        async def silent_chat(messages, tools=None, model=None, num_ctx=None, on_token=None):
            calls.append(len(messages))
            return {"message": {"role": "assistant", "content": "", "tool_calls": []}}

        graph, ctx = await create_graph(db_name)
        try:
            with (
                patch("maidere.core.llm.chat", side_effect=silent_chat),
                patch("maidere.core.agent.recall", new_callable=AsyncMock, return_value=[]),
                patch("maidere.core.agent.emit_agent_event", new_callable=AsyncMock),
            ):
                result = await graph.ainvoke(
                    {
                        "messages": [HumanMessage(content="Prove that the square root of 2 is irrational")],
                        "memory_context": "No relevant memories.",
                        "thread_id": "silent-model-thread",
                        "deep_reasoning": True,
                    },
                    config={"configurable": {"thread_id": "silent-model-thread"}},
                )
        finally:
            await ctx.__aexit__(None, None, None)
            os.remove(db_name)

        self.assertEqual(result["messages"][-1].content, EMPTY_RESPONSE_NOTICE)
        self.assertEqual(len(calls), 3, "one attempt plus two retries, and no verify/converge calls")


if __name__ == "__main__":
    unittest.main()
