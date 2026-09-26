"""Unit tests for Maidere Sub-Agent System and DelegateTaskTool.

Tests cover:
- Sub-agent execution loop with mock LLM
- Enforcement of strictly read-only tools
- Prevention of destructive/write tools within sub-agent (Patch #1 defense)
- Anti-hallucination instruction in DelegateTaskTool (Patch #2)
- Infinite delegation loop prevention / tool filtering (Patch #3)
- Max sub-agents per query cap (MAX_SUBAGENTS_PER_QUERY = 5)
- Tagged vector memory persistence
- Act node delegation tracking
"""

import asyncio
import tempfile
import unittest
from unittest.mock import AsyncMock, patch

from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

from core.agent import act_node, think_node
from core.subagent import (
    MAX_SUBAGENTS_PER_QUERY,
    SUBAGENT_ALLOWED_TOOLS,
    SUBAGENT_SYSTEM_PROMPT,
    SubAgentResult,
    run_subagent,
    set_progress_callback,
    clear_progress_callback,
)
from tools.delegate import DelegateTaskTool


class TestSubAgentEngine(unittest.TestCase):
    """Test core subagent execution logic and boundaries."""

    def test_subagent_allowed_tools_strictly_readonly(self):
        """Ensure sub-agents only have read-only data-gathering tools."""
        self.assertIn("web_search", SUBAGENT_ALLOWED_TOOLS)
        self.assertIn("browser", SUBAGENT_ALLOWED_TOOLS)
        self.assertIn("read_file", SUBAGENT_ALLOWED_TOOLS)
        self.assertIn("list_dir", SUBAGENT_ALLOWED_TOOLS)

        # Ensure write and execution tools are strictly disallowed
        self.assertNotIn("shell", SUBAGENT_ALLOWED_TOOLS)
        self.assertNotIn("code_runner", SUBAGENT_ALLOWED_TOOLS)
        self.assertNotIn("write_file", SUBAGENT_ALLOWED_TOOLS)
        self.assertNotIn("obsidian", SUBAGENT_ALLOWED_TOOLS)
        self.assertNotIn("scheduler", SUBAGENT_ALLOWED_TOOLS)
        self.assertNotIn("delegate_task", SUBAGENT_ALLOWED_TOOLS)

    def test_dense_extraction_rules_in_system_prompt(self):
        """Verify strict tool-bound directives in subagent system prompt."""
        prompt = SUBAGENT_SYSTEM_PROMPT.format(task="Compare React and Vue benchmarks")
        self.assertIn("CRITICAL DIRECTIVES", prompt)
        self.assertIn("Do NOT write out your plan", prompt)
        self.assertIn("Your very first action MUST be an actual tool call", prompt)
        self.assertIn("DENSE DATA PAYLOAD", prompt)
        self.assertIn("benchmarks", prompt)
        self.assertIn("source URLs", prompt)

    def test_epistemic_grounding_in_subagent_system_prompt(self):
        """Verify Epistemic Grounding Directive is present in SUBAGENT_SYSTEM_PROMPT."""
        prompt = SUBAGENT_SYSTEM_PROMPT.format(task="Research Jaime Lannister")
        self.assertIn("EPISTEMIC GROUNDING (CANONICAL FACTS VS. SPECULATION)", prompt)
        self.assertIn(
            "When researching fictional characters or historical events, you MUST prioritize canonical facts and primary source data",
            prompt,
        )
        self.assertIn(
            "You are FORBIDDEN from treating fan theories, speculative blog posts, or forum discussions as factual canon",
            prompt,
        )

    def test_epistemic_grounding_in_researcher_markdown(self):
        """Verify agents/researcher.md contains Epistemic Grounding Directive."""
        from pathlib import Path
        researcher_path = Path(__file__).resolve().parent.parent / "agents" / "researcher.md"
        self.assertTrue(researcher_path.exists())
        content = researcher_path.read_text(encoding="utf-8")
        self.assertIn("EPISTEMIC GROUNDING (CANONICAL FACTS VS. SPECULATION)", content)
        self.assertIn(
            "When researching fictional characters or historical events, you MUST prioritize canonical facts and primary source data",
            content,
        )
        self.assertIn(
            "You are FORBIDDEN from treating fan theories, speculative blog posts, or forum discussions as factual canon",
            content,
        )

    def test_extract_research_subtasks_sanitizes_fan_theories_and_handles_temporal(self):
        """Verify extract_research_subtasks sanitizes theories and handles temporal anchoring."""
        from core.agent import extract_research_subtasks

        # Case 1: Fictional query with candidate text containing fan theories -> must be sanitized, no 2026
        model_text = (
            "1. Jaime Lannister character background and Kingsguard service\n"
            "2. Jamie Lannister fan theories critical analyses on reddit\n"
            "3. Jaime Lannister relationship with Cersei and Brienne"
        )
        subtasks = extract_research_subtasks(model_text, "Research Jaime Lannister")
        self.assertTrue(len(subtasks) >= 2)
        # Should NOT contain "fan theories"
        for st in subtasks:
            self.assertNotIn("fan theories", st.lower())
            self.assertNotIn("2026", st)  # No temporal year for fictional character

        # Case 2: Temporal/financial query -> MUST append current year
        fin_subtasks = extract_research_subtasks("", "Research Meta financial performance")
        self.assertTrue(len(fin_subtasks) >= 2)
        for st in fin_subtasks:
            self.assertIn("2026", st)

        # Case 3: Fictional fallback 3-way decomposition -> canonical lore, no 2026
        fict_subtasks = extract_research_subtasks("", "Research Jaime Lannister")
        self.assertEqual(len(fict_subtasks), 3)
        self.assertIn("canonical background", fict_subtasks[0])
        self.assertIn("canonical narrative arc", fict_subtasks[1])
        for st in fict_subtasks:
            self.assertNotIn("2026", st)

        # Case 4: User explicitly requests fan theories -> preserve theories
        theory_text = "1. Jaime Lannister fan theories about the valonqar\n2. Jaime Lannister fan theories on reddit"
        user_q = "What are popular fan theories about Jaime Lannister?"
        theory_subtasks = extract_research_subtasks(theory_text, user_q)
        has_theories = any("theor" in st.lower() for st in theory_subtasks)
        self.assertTrue(has_theories)

    def test_run_subagent_direct_response(self):
        """Test subagent completes when LLM provides direct text response without tool calls."""
        mock_response = {
            "message": {
                "role": "assistant",
                "content": "- React 19: LCP 920ms, Bundle 42kb\n- Vue 3.5: LCP 810ms, Bundle 33kb",
                "tool_calls": [],
            }
        }

        with patch("core.llm.chat", new_callable=AsyncMock, return_value=mock_response), \
             patch("core.subagent._store_subagent_memory", new_callable=AsyncMock) as mock_store:

            result = asyncio.run(run_subagent(task="Benchmark React vs Vue", max_turns=3))

            self.assertTrue(result.success)
            self.assertIn("React 19", result.summary)
            self.assertEqual(result.turns_used, 1)
            self.assertEqual(len(result.tools_used), 0)
            mock_store.assert_awaited_once()

    def test_run_subagent_executes_readonly_tool(self):
        """Test subagent handles tool calls with allowed read-only tools."""
        responses = [
            # Turn 1: Call web_search
            {
                "message": {
                    "role": "assistant",
                    "content": "",
                    "tool_calls": [
                        {
                            "id": "call_1",
                            "function": {
                                "name": "web_search",
                                "arguments": {"query": "Svelte 5 benchmarks 2026"},
                            },
                        }
                    ],
                }
            },
            # Turn 2: Synthesize findings
            {
                "message": {
                    "role": "assistant",
                    "content": "Svelte 5 runes achieve 650ms LCP on standard metrics (source: benchmark.dev).",
                    "tool_calls": [],
                }
            },
        ]

        with patch("core.llm.chat", new_callable=AsyncMock, side_effect=responses), \
             patch("tools.registry.get_tool") as mock_get_tool, \
             patch("core.subagent._store_subagent_memory", new_callable=AsyncMock):

            mock_search_tool = AsyncMock()
            mock_search_tool.execute = AsyncMock(return_value="1. Svelte 5 benchmarks: 650ms LCP")
            mock_search_tool.to_ollama_schema = lambda: {"type": "function", "function": {"name": "web_search"}}
            mock_get_tool.return_value = mock_search_tool

            result = asyncio.run(run_subagent(task="Research Svelte 5", max_turns=3))

            self.assertTrue(result.success)
            self.assertEqual(len(result.tools_used), 1)
            self.assertEqual(result.tools_used[0]["tool_name"], "web_search")
            self.assertTrue(result.tools_used[0]["success"])
            self.assertIn("Svelte 5 runes", result.summary)

    def test_run_subagent_blocks_disallowed_tool(self):
        """Test subagent denies execution of unauthorized tools (e.g. shell)."""
        responses = [
            # Turn 1: Try to call shell (disallowed)
            {
                "message": {
                    "role": "assistant",
                    "content": "",
                    "tool_calls": [
                        {
                            "id": "call_shell",
                            "function": {
                                "name": "shell",
                                "arguments": {"command": "rm -rf /"},
                            },
                        }
                    ],
                }
            },
            # Turn 2: Final response
            {
                "message": {
                    "role": "assistant",
                    "content": "Unable to execute shell commands. Researched using available tools.",
                    "tool_calls": [],
                }
            },
        ]

        with patch("core.llm.chat", new_callable=AsyncMock, side_effect=responses), \
             patch("core.subagent._store_subagent_memory", new_callable=AsyncMock):

            result = asyncio.run(run_subagent(task="Try shell command", max_turns=3))

            self.assertTrue(result.success)
            self.assertEqual(len(result.tools_used), 1)
            self.assertEqual(result.tools_used[0]["tool_name"], "shell")
            self.assertFalse(result.tools_used[0]["success"])

    def test_subagent_progress_callbacks(self):
        """Verify progress events are emitted during subagent execution."""
        events = []

        async def callback(event):
            events.append(event)

        set_progress_callback(callback)

        mock_response = {
            "message": {
                "role": "assistant",
                "content": "Research completed.",
                "tool_calls": [],
            }
        }

        with patch("core.llm.chat", new_callable=AsyncMock, return_value=mock_response), \
             patch("core.subagent._store_subagent_memory", new_callable=AsyncMock):

            asyncio.run(run_subagent(task="Test Progress", subagent_index=1, total_subagents=1))

        clear_progress_callback()

        event_types = [e["type"] for e in events]
        self.assertIn("subagent_started", event_types)
        self.assertIn("subagent_completed", event_types)


class TestDelegateTaskTool(unittest.TestCase):
    """Test DelegateTaskTool functionality and guardrails."""

    def setUp(self):
        self.tool = DelegateTaskTool()
        self.tool.reset_spawn_count()

    def test_anti_hallucination_instruction_in_description(self):
        """Verify Patch #2: Tool description instructs LLM to STOP and WAIT."""
        desc = self.tool.description
        self.assertIn("MUST stop generating text and WAIT", desc)
        self.assertIn("Do NOT attempt to answer", desc)
        self.assertIn("synthesize them into one comprehensive final report", desc)

    def test_spawn_counter_and_limit_enforcement(self):
        """Verify max 5 sub-agents per query limit."""
        mock_subagent_result = SubAgentResult(
            task="Test task",
            summary="Test summary data payload",
            tools_used=[],
            duration_ms=100,
            success=True,
            turns_used=1,
        )

        with patch("tools.delegate.run_subagent", new_callable=AsyncMock, return_value=mock_subagent_result):
            for i in range(MAX_SUBAGENTS_PER_QUERY):
                output = asyncio.run(self.tool.execute(task=f"Subtask {i+1}"))
                self.assertIn("[SUB-AGENT RESEARCH RESULT]", output)

            # 6th call should be rejected
            excess_output = asyncio.run(self.tool.execute(task="Excess Subtask 6"))
            self.assertIn("Error: Maximum sub-agent limit", excess_output)
            self.assertIn("synthesize your comprehensive answer", excess_output)

    def test_spawn_counter_reset(self):
        """Verify spawn counter resets cleanly for new query."""
        self.tool._spawn_count = MAX_SUBAGENTS_PER_QUERY
        self.tool.reset_spawn_count()
        self.assertEqual(self.tool._spawn_count, 0)
        self.assertEqual(self.tool._total_delegated, 0)

    def test_empty_task_rejected(self):
        """Verify empty task string returns error."""
        output = asyncio.run(self.tool.execute(task="   "))
        self.assertIn("Error: Empty task description", output)


class TestPatch3InfiniteDelegationPrevention(unittest.TestCase):
    """Test Patch #3: Removal of delegate_task tool after first delegation."""

    def test_think_node_removes_delegate_task_when_has_delegated(self):
        """Ensure delegate_task schema is stripped from LLM call if has_delegated is True."""
        state = {
            "messages": [HumanMessage(content="Synthesize the research.")],
            "memory_context": "",
            "thread_id": "test_thread",
            "model": "qwen2.5:7b-instruct",
            "has_delegated": True,
        }

        mock_response = {
            "message": {
                "role": "assistant",
                "content": "Final synthesized report.",
                "tool_calls": [],
            }
        }

        with patch("core.llm.chat", new_callable=AsyncMock, return_value=mock_response) as mock_chat:
            asyncio.run(think_node(state))

            # Inspect the tools passed to llm.chat
            passed_tools = mock_chat.call_args.kwargs.get("tools") or mock_chat.call_args[0][1]
            tool_names = [t.get("function", {}).get("name") for t in passed_tools]
            self.assertNotIn("delegate_task", tool_names)

    def test_think_node_includes_delegate_task_when_has_not_delegated(self):
        """Ensure delegate_task is available when has_delegated is False/absent."""
        state = {
            "messages": [HumanMessage(content="Perform deep research on LLMs.")],
            "memory_context": "",
            "thread_id": "test_thread",
            "model": "qwen2.5:7b-instruct",
            "has_delegated": False,
        }

        mock_response = {
            "message": {
                "role": "assistant",
                "content": "Delegating tasks.",
                "tool_calls": [],
            }
        }

        with patch("core.llm.chat", new_callable=AsyncMock, return_value=mock_response) as mock_chat:
            asyncio.run(think_node(state))

            passed_tools = mock_chat.call_args.kwargs.get("tools") or mock_chat.call_args[0][1]
            tool_names = [t.get("function", {}).get("name") for t in passed_tools]
            self.assertIn("delegate_task", tool_names)

    def test_act_node_sets_has_delegated_flag(self):
        """Ensure act_node detects delegate_task execution and sets has_delegated=True."""
        state = {
            "messages": [
                AIMessage(
                    content="",
                    tool_calls=[
                        {
                            "name": "delegate_task",
                            "args": {"task": "Research React"},
                            "id": "call_del_1",
                        }
                    ],
                )
            ],
            "thread_id": "test_thread",
        }

        with patch("core.agent.execute_tool", new_callable=AsyncMock, return_value="[SUB-AGENT RESEARCH RESULT]"):
            result = asyncio.run(act_node(state))
            self.assertTrue(result.get("has_delegated"))


class TestClaudeCodeSubagents(unittest.TestCase):
    """Test Claude Code subagent specification compatibility."""

    def test_load_subagent_definitions(self):
        """Verify discovery and loading of subagent files from agents/ directory."""
        from core.subagent import load_subagent_definitions

        defs = load_subagent_definitions()
        self.assertIn("researcher", defs)
        self.assertIn("plan", defs)
        self.assertIn("code-reviewer", defs)
        self.assertIn("general-purpose", defs)
        self.assertIn("verification", defs)
        self.assertIn("verifier", defs)
        self.assertIn("validation", defs)
        self.assertIn("validator", defs)

        plan_agent = defs["plan"]
        self.assertEqual(plan_agent.name, "plan")
        self.assertIn("read_file", plan_agent.tools)
        self.assertIn("web_search", plan_agent.tools)

        researcher = defs["researcher"]
        self.assertEqual(researcher.name, "researcher")
        self.assertIn("web_search", researcher.tools)
        self.assertIn("browser", researcher.tools)

        reviewer = defs["code-reviewer"]
        self.assertEqual(reviewer.name, "code-reviewer")
        self.assertIn("read_file", reviewer.tools)

        verification_agent = defs["verification"]
        self.assertEqual(verification_agent.name, "verification")
        self.assertIn("read_file", verification_agent.tools)
        self.assertIn("web_search", verification_agent.tools)
        self.assertIn("INDEPENDENT MATHEMATICAL & ALGORITHMIC RECALCULATION", verification_agent.system_prompt)

        validation_agent = defs["validation"]
        self.assertEqual(validation_agent.name, "validation")
        self.assertIn("web_search", validation_agent.tools)
        self.assertIn("browser", validation_agent.tools)
        self.assertIn("EMPIRICAL REAL-WORLD & VERSION VALIDATION", validation_agent.system_prompt)

    def test_verification_and_validation_agent_tool_execution(self):
        """Verify Agent/delegate_task tool execution with verification and validation types."""
        from tools.delegate import AgentTool

        tool = AgentTool()
        v_result = SubAgentResult(
            task="Verify mathematical invariant in proof",
            summary="[PASS: VERIFIED] Recalculated steps 1-5. Invariants hold.",
            subagent_type="verification",
            tools_used=[],
            duration_ms=45,
            success=True,
            turns_used=2,
        )

        with patch("tools.delegate.run_subagent", new_callable=AsyncMock, return_value=v_result) as mock_run:
            output = asyncio.run(tool.execute(prompt="Verify mathematical invariant in proof", subagent_type="verification"))
            self.assertIn("[SUB-AGENT RESEARCH RESULT]", output)
            self.assertIn("Agent Type: verification", output)
            self.assertIn("[PASS: VERIFIED]", output)
            mock_run.assert_awaited_once_with(
                task="Verify mathematical invariant in proof",
                subagent_type="verification",
                max_turns=6,
                model=None,
                subagent_index=1,
                total_subagents=1,
            )

        tool.reset_spawn_count()
        val_result = SubAgentResult(
            task="Validate PyTorch 2.6 CUDA 12.8 compatibility in 2026",
            summary="[PASS: VALIDATED] Confirmed compatibility and driver versions.",
            subagent_type="validation",
            tools_used=[],
            duration_ms=60,
            success=True,
            turns_used=3,
        )

        with patch("tools.delegate.run_subagent", new_callable=AsyncMock, return_value=val_result) as mock_run:
            output = asyncio.run(tool.execute(prompt="Validate PyTorch 2.6 CUDA 12.8 compatibility in 2026", subagent_type="validation"))
            self.assertIn("[SUB-AGENT RESEARCH RESULT]", output)
            self.assertIn("Agent Type: validation", output)
            self.assertIn("[PASS: VALIDATED]", output)
            mock_run.assert_awaited_once_with(
                task="Validate PyTorch 2.6 CUDA 12.8 compatibility in 2026",
                subagent_type="validation",
                max_turns=6,
                model=None,
                subagent_index=1,
                total_subagents=1,
            )

    def test_agent_tool_execution_with_prompt_and_type(self):
        """Verify Agent tool execution with prompt and subagent_type parameters."""
        from tools.delegate import AgentTool

        tool = AgentTool()
        mock_result = SubAgentResult(
            task="Review python code",
            summary="Found 0 bugs. Clean structure.",
            subagent_type="code-reviewer",
            tools_used=[],
            duration_ms=50,
            success=True,
            turns_used=1,
        )

        with patch("tools.delegate.run_subagent", new_callable=AsyncMock, return_value=mock_result) as mock_run:
            output = asyncio.run(tool.execute(prompt="Review python code", subagent_type="code-reviewer"))
            self.assertIn("[SUB-AGENT RESEARCH RESULT]", output)
            self.assertIn("Agent Type: code-reviewer", output)
            mock_run.assert_awaited_once_with(
                task="Review python code",
                subagent_type="code-reviewer",
                max_turns=6,
                model=None,
                subagent_index=1,
                total_subagents=1,
            )

    def test_extract_text_tool_calls_with_agent_syntax(self):
        """Verify extract_text_tool_calls parses Agent(...) and delegate_task(...)."""
        from core.agent import extract_text_tool_calls

        text = 'Sure! I will run Agent(subagent_type="researcher", prompt="Deep research on Elon Musk") now.'
        calls = extract_text_tool_calls(text, ["Agent", "delegate_task"])
        self.assertEqual(len(calls), 1)
        self.assertEqual(calls[0]["name"], "Agent")
        self.assertEqual(calls[0]["args"]["subagent_type"], "researcher")
        self.assertEqual(calls[0]["args"]["prompt"], "Deep research on Elon Musk")

    def test_agent_telemetry_event_callbacks(self):
        """Verify rich agent telemetry events are emitted to registered thread callback."""
        from core.agent import emit_agent_event, register_event_callback, unregister_event_callback

        received = []

        async def cb(event):
            received.append(event)

        register_event_callback("thread-telemetry", cb)
        try:
            asyncio.run(emit_agent_event("thread-telemetry", {"type": "agent_log", "text": "Testing log"}))
            self.assertEqual(len(received), 1)
            self.assertEqual(received[0]["text"], "Testing log")
        finally:
            unregister_event_callback("thread-telemetry")

    def test_subagent_tool_started_and_completed_telemetry(self):
        """Verify subagent emits tool started, tool completed with duration and preview, and memory log."""
        from core.subagent import clear_progress_callback, run_subagent, set_progress_callback

        events = []

        async def progress_handler(evt):
            events.append(evt)

        set_progress_callback(progress_handler)
        try:
            mock_turn1 = {
                "message": {
                    "role": "assistant",
                    "content": "",
                    "tool_calls": [
                        {
                            "id": "c1",
                            "function": {"name": "web_search", "arguments": {"query": "AMD 2026 specs"}},
                        }
                    ],
                }
            }
            mock_turn2 = {
                "message": {
                    "role": "assistant",
                    "content": "AMD specs: Zen 5 IPC +16%.",
                    "tool_calls": [],
                }
            }
            with patch("core.llm.chat", new_callable=AsyncMock, side_effect=[mock_turn1, mock_turn2]), \
                 patch("tools.registry.get_tool") as mock_get_tool, \
                 patch("core.subagent._store_subagent_memory", new_callable=AsyncMock):

                tool_inst = AsyncMock()
                tool_inst.execute.return_value = "Zen 5 features 16 cores and +16% IPC"
                tool_inst.to_ollama_schema.return_value = {"type": "function", "function": {"name": "web_search"}}
                mock_get_tool.return_value = tool_inst

                res = asyncio.run(run_subagent("Research AMD 2026", subagent_index=1, total_subagents=1))

            types = [e.get("type") for e in events]
            self.assertIn("subagent_started", types)
            self.assertIn("subagent_tool_started", types)
            self.assertIn("subagent_tool", types)
            self.assertIn("agent_log", types)
            self.assertIn("subagent_completed", types)

            tool_completed = next(e for e in events if e.get("type") == "subagent_tool")
            self.assertIn("Zen 5", tool_completed.get("result_preview", ""))
            self.assertTrue(tool_completed.get("success"))
        finally:
            clear_progress_callback()

    def test_temporal_anchor_in_system_prompt(self):
        """Verify dynamic date and current year (2026) are anchored in the system prompt."""
        from datetime import datetime
        from core.agent import get_system_prompt

        prompt = get_system_prompt()
        now = datetime.now()
        formatted = prompt.format(
            memory_context="No memories",
            current_date=now.strftime("%A, %B %d, %Y"),
            current_year=str(now.year),
        )
        self.assertIn(str(now.year), formatted)
        self.assertIn("Temporal Real-World Anchor", formatted)
        self.assertIn("ANTI-PASS-THROUGH", formatted)

    def test_turn_scoping_prevents_prior_toolmessage_leak(self):
        """Verify prior turn's raw ToolMessages are stripped when building LLM context for a new turn."""
        from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
        from core.agent import think_node

        state = {
            "messages": [
                HumanMessage(content="What was Meta's revenue?"),
                AIMessage(content="", tool_calls=[{"name": "Agent", "args": {"prompt": "Meta"}, "id": "call1"}]),
                ToolMessage(content="[LEAK TEST: Meta revenue $39B in 2026, Kraken market cap]", tool_call_id="call1", name="Agent"),
                AIMessage(content="Meta revenue in 2026 was over $39B."),
                HumanMessage(content="Find software engineering internships"),
            ],
            "memory_context": "No relevant memories",
            "thread_id": "test-leak",
            "model": "qwen2.5:7b-instruct",
        }

        captured_messages = []

        async def fake_chat(messages, tools=None, model=None, **kwargs):
            captured_messages.extend(messages)
            return {"message": {"role": "assistant", "content": "Here are 2026 internships."}}

        with patch("core.llm.chat", side_effect=fake_chat), \
             patch("core.agent.emit_agent_event", new_callable=AsyncMock):
            asyncio.run(think_node(state))

        # Check captured messages sent to Ollama
        roles = [m.get("role") for m in captured_messages]
        contents = [str(m.get("content", "")) for m in captured_messages]

        # Prior turn tool message must NOT be in the LLM context
        self.assertFalse(any("[LEAK TEST" in c for c in contents), "Prior turn ToolMessage leaked into new turn context!")
        # The new turn query must be present
        self.assertTrue(any("internships" in c for c in contents))
        # Prior turn human question and completed answer should be present
        self.assertTrue(any("Meta's revenue" in c for c in contents))
        self.assertTrue(any("Meta revenue in 2026 was over $39B." in c for c in contents))

    def test_reality_check_and_enterprise_grounding_directive_domain_isolation(self):
        """Verify Reality Check directive is isolated to tech-hardware subagent and stripped from root orchestrator."""
        from datetime import datetime
        from pathlib import Path
        from core.agent import get_system_prompt, is_hardware_query
        from core.subagent import TECH_HARDWARE_SYSTEM_PROMPT, SUBAGENT_SYSTEM_PROMPT

        # 1. Verify root Orchestrator prompt is clean of hardware directives (NO instruction bleed)
        prompt = get_system_prompt()
        now = datetime.now()
        formatted = prompt.format(
            memory_context="No memories",
            current_date=now.strftime("%A, %B %d, %Y"),
            current_year=str(now.year),
        )
        self.assertNotIn("Craigslist", formatted)
        self.assertNotIn("National AI Research Resource (NAIRR) Pilot", formatted)
        self.assertNotIn("DGX Cloud credits", formatted)

        # 2. Verify general researcher agent prompt is also clean
        self.assertNotIn("Craigslist", SUBAGENT_SYSTEM_PROMPT)
        self.assertNotIn("NAIRR", SUBAGENT_SYSTEM_PROMPT)

        # 3. Verify tech-hardware subagent prompt has the strict Reality Check & NAIRR directives
        hardware_path = Path(__file__).resolve().parent.parent / "agents" / "tech-hardware.md"
        self.assertTrue(hardware_path.exists())
        hardware_content = hardware_path.read_text(encoding="utf-8")

        for text in (hardware_content, TECH_HARDWARE_SYSTEM_PROMPT):
            self.assertIn("REALITY CHECK & ENTERPRISE HARDWARE GROUNDING", text)
            self.assertIn("Craigslist", text)
            self.assertIn("National AI Research Resource (NAIRR) Pilot", text)
            self.assertIn("NVIDIA Inception Program provides up to $100,000 in DGX Cloud credits", text)

        # 4. Verify hardware query detection routes to tech-hardware
        self.assertTrue(is_hardware_query("How to get a free NVIDIA DGX H100"))
        self.assertTrue(is_hardware_query("Research AI supercomputers and GPU clusters"))
        self.assertFalse(is_hardware_query("Research Paul Atreides from Dune"))
        self.assertFalse(is_hardware_query("Research Jaime Lannister from Game of Thrones"))

    def test_strict_url_citation_rules_in_prompt(self):
        """Verify strict inline index citation rules forbid bibliography sections and require [1]-style inline citations."""
        from datetime import datetime
        from core.agent import get_system_prompt

        prompt = get_system_prompt()
        now = datetime.now()
        formatted = prompt.format(
            memory_context="No memories",
            current_date=now.strftime("%A, %B %d, %Y"),
            current_year=str(now.year),
        )
        self.assertIn("Strict Inline Index Citation Formatting", formatted)
        self.assertIn("bracketed integers inline", formatted)
        self.assertIn("FORBIDDEN from generating", formatted)

    def test_restore_markdown_urls_revives_stripped_urls(self):
        """Verify restore_markdown_urls revives stripped titles into clickable markdown links."""
        from langchain_core.messages import ToolMessage
        from core.agent import restore_markdown_urls

        tool_msg = ToolMessage(
            content=(
                "1. [NVIDIA Inception Program](https://www.nvidia.com/en-us/startups/)\n"
                "   URL: https://www.nvidia.com/en-us/startups/\n"
                "   Snippet: Startup program offering cloud credits.\n\n"
                "2. [NVIDIA AI for Robotics Program](https://www.nvidia.com/en-us/deep-learning-ai/solutions/robotics/)\n"
                "   URL: https://www.nvidia.com/en-us/deep-learning-ai/solutions/robotics/\n"
                "   Snippet: Grants for robotics research.\n\n"
                "3. National AI Research Resource (NAIRR) Pilot: https://nairrpilot.org/"
            ),
            tool_call_id="call_res",
            name="Agent",
        )

        broken_ai_output = (
            "Here are the avenues for accessing DGX compute:\n"
            "- Program 1:\n"
            "  URL: NVIDIA Inception Program\n"
            "- Program 2:\n"
            "  * URL: [NVIDIA AI for Robotics Program]\n"
            "- Program 3:\n"
            "  Apply via [National AI Research Resource (NAIRR) Pilot] for allocations.\n"
            "- Existing valid link: [Google](https://google.com)"
        )

        restored = restore_markdown_urls(broken_ai_output, [tool_msg])

        self.assertIn("[NVIDIA Inception Program](https://www.nvidia.com/en-us/startups/)", restored)
        self.assertIn("[NVIDIA AI for Robotics Program](https://www.nvidia.com/en-us/deep-learning-ai/solutions/robotics/)", restored)
        self.assertIn("[National AI Research Resource (NAIRR) Pilot](https://nairrpilot.org/)", restored)
        self.assertIn("[Google](https://google.com)", restored)
        self.assertNotIn("URL: NVIDIA Inception Program", restored)

    def test_conditional_temporal_constraint_in_subagent_prompts(self):
        """Verify conditional temporal constraint in SUBAGENT_SYSTEM_PROMPT and researcher.md."""
        from pathlib import Path
        prompt = SUBAGENT_SYSTEM_PROMPT.format(task="Research Adolf Hitler")
        self.assertIn("CONDITIONAL TEMPORAL CONSTRAINT", prompt)
        self.assertIn("Append the current year (2026) to search queries ONLY for financial data", prompt)
        self.assertIn("Do NOT append the current year to queries about historical events, historical figures, or established lore", prompt)

        researcher_path = Path(__file__).resolve().parent.parent / "agents" / "researcher.md"
        content = researcher_path.read_text(encoding="utf-8")
        self.assertIn("CONDITIONAL TEMPORAL CONSTRAINT", content)
        self.assertIn("Append the current year (2026) to search queries ONLY for financial data", content)
        self.assertIn("Do NOT append the current year to queries about historical events", content)

    def test_chronological_timeline_directive_in_prompts(self):
        """Verify Chronological Timeline Directive in researcher.md, subagent prompt, and SYSTEM_PROMPT."""
        from pathlib import Path
        from core.agent import SYSTEM_PROMPT

        prompt = SUBAGENT_SYSTEM_PROMPT.format(task="Research WWII")
        self.assertIn("CHRONOLOGICAL TIMELINE & CAUSALITY VERIFICATION", prompt)
        self.assertIn("Verify the exact month, day, and year of consecutive events before establishing cause-and-effect relationships", prompt)

        researcher_path = Path(__file__).resolve().parent.parent / "agents" / "researcher.md"
        content = researcher_path.read_text(encoding="utf-8")
        self.assertIn("CHRONOLOGICAL TIMELINE & CAUSALITY VERIFICATION", content)

        self.assertIn("Chronological Timeline & Causality Verification", SYSTEM_PROMPT)
        self.assertIn("Verify the exact month, day, and year of consecutive events before establishing cause-and-effect relationships", SYSTEM_PROMPT)

    def test_search_diversity_rule_in_prompts(self):
        """Verify Search Diversity & Redundancy Prevention rule in researcher.md and subagent prompt."""
        from pathlib import Path
        prompt = SUBAGENT_SYSTEM_PROMPT.format(task="Research Nazi rearmament")
        self.assertIn("SEARCH DIVERSITY & REDUNDANCY PREVENTION", prompt)
        self.assertIn("prevent retrieving the same source URL multiple times", prompt)

        researcher_path = Path(__file__).resolve().parent.parent / "agents" / "researcher.md"
        content = researcher_path.read_text(encoding="utf-8")
        self.assertIn("SEARCH DIVERSITY & REDUNDANCY PREVENTION", content)
        self.assertIn("prevent retrieving the same source URL multiple times", content)

    def test_is_historical_or_lore_and_temporal_sensitivity(self):
        """Verify is_historical_or_lore detects historical/lore queries and suppresses current year."""
        from core.agent import is_historical_or_lore, is_temporally_sensitive

        # Historical figures and events -> True for historical, False for temporally sensitive
        self.assertTrue(is_historical_or_lore("Adolf Hitler biography key events", "Adolf Hitler"))
        self.assertTrue(is_historical_or_lore("Nazi rearmament programs Luftwaffe Wehrmacht military buildup 1930s", "Nazi rearmament"))
        self.assertTrue(is_historical_or_lore("Julius Caesar reign and assassination", "Julius Caesar"))
        self.assertTrue(is_historical_or_lore("Battle of Stalingrad 20th century", "Battle of Stalingrad"))

        self.assertFalse(is_temporally_sensitive("Adolf Hitler biography key events", "Adolf Hitler"))
        self.assertFalse(is_temporally_sensitive("Nazi rearmament programs Luftwaffe Wehrmacht", "Nazi rearmament"))
        self.assertFalse(is_temporally_sensitive("Julius Caesar reign", "Julius Caesar"))

        # Fictional lore -> True for historical_or_lore, False for temporally sensitive
        self.assertTrue(is_historical_or_lore("Jaime Lannister character arc in Game of Thrones", "Jaime Lannister"))
        self.assertFalse(is_temporally_sensitive("Jaime Lannister character arc", "Jaime Lannister"))

        # Financial & tech benchmark queries -> False for historical, True for temporally sensitive
        self.assertFalse(is_historical_or_lore("Meta financial performance quarterly revenue", "Meta"))
        self.assertTrue(is_temporally_sensitive("Meta financial performance quarterly revenue", "Meta"))
        self.assertTrue(is_temporally_sensitive("Compare React vs Vue benchmarks", "React vs Vue"))

        # Modern override for historical queries
        self.assertFalse(is_historical_or_lore("Adolf Hitler in modern day pop culture 2026", "Adolf Hitler"))
        self.assertTrue(is_temporally_sensitive("Adolf Hitler in modern day pop culture 2026", "Adolf Hitler"))

    def test_extract_research_subtasks_historical_no_2026(self):
        """Verify extract_research_subtasks produces historical decomposition without 2026."""
        from datetime import datetime
        from core.agent import extract_research_subtasks

        current_year = str(datetime.now().year)
        query = "Adolf Hitler biography key events ideologies impacts 20th-century Europe"

        # 1. Fallback default 3-way decomposition for historical query
        subtasks = extract_research_subtasks("", query)
        self.assertEqual(len(subtasks), 3)
        for st in subtasks:
            self.assertNotIn(current_year, st)
        self.assertIn("origins, early background", subtasks[0])
        self.assertIn("strict chronological timeline", subtasks[1])
        self.assertIn("historical impact, legacy", subtasks[2])

        # 2. Decomposed lines that accidentally contain (2026) are stripped for historical queries
        model_text = (
            f"1. Adolf Hitler rise to power in the Weimar Republic ({current_year})\n"
            f"2. Nazi foreign policy and expansionism 1933-1939 ({current_year})\n"
            f"3. World War II and the Holocaust key historical timeline ({current_year})"
        )
        cleaned = extract_research_subtasks(model_text, query)
        self.assertTrue(len(cleaned) >= 2)
        for st in cleaned:
            self.assertNotIn(f"({current_year})", st)
            self.assertNotIn(current_year, st)

    def test_subagent_url_normalization_and_redundancy_helpers(self):
        """Verify URL normalization and search query redundancy detection."""
        from core.subagent import normalize_subagent_url, is_search_query_redundant

        # URL normalization
        url1 = "https://en.wikipedia.org/wiki/German_military_technology_during_World_War_II#Luftwaffe/"
        url2 = "https://en.wikipedia.org/wiki/German_military_technology_during_World_War_II"
        self.assertEqual(normalize_subagent_url(url1), normalize_subagent_url(url2))

        # Query redundancy detection (The Wikipedia Loop prevention)
        past_queries = ["Nazi rearmament programs Luftwaffe Wehrmacht military buildup 1930s"]
        redundant_query = "Luftwaffe expansion and German rearmament key military programs 1930s"
        diverse_query = "Economic treaties of the Weimar Republic 1920s hyperinflation"

        self.assertTrue(is_search_query_redundant(redundant_query, past_queries))
        self.assertFalse(is_search_query_redundant(diverse_query, past_queries))

    def test_run_subagent_prevents_duplicate_browsing_and_flags_redundancy(self):
        """Verify subagent execution loop prevents duplicate URL browsing and flags redundant searches."""
        from unittest.mock import AsyncMock, patch
        from core.subagent import run_subagent

        wiki_url = "https://en.wikipedia.org/wiki/German_military_technology_during_World_War_II"

        responses = [
            # Turn 1: Search Nazi rearmament
            {
                "message": {
                    "role": "assistant",
                    "content": "",
                    "tool_calls": [{
                        "id": "call_1",
                        "function": {
                            "name": "web_search",
                            "arguments": {"query": "Nazi rearmament programs Luftwaffe Wehrmacht military buildup 1930s"},
                        },
                    }],
                }
            },
            # Turn 2: Redundant search (The Wikipedia Loop query)
            {
                "message": {
                    "role": "assistant",
                    "content": "",
                    "tool_calls": [{
                        "id": "call_2",
                        "function": {
                            "name": "web_search",
                            "arguments": {"query": "Luftwaffe expansion and German rearmament key military programs 1930s"},
                        },
                    }],
                }
            },
            # Turn 3: Browse Wikipedia article twice
            {
                "message": {
                    "role": "assistant",
                    "content": "",
                    "tool_calls": [{
                        "id": "call_3",
                        "function": {
                            "name": "browser",
                            "arguments": {"url": wiki_url},
                        },
                    }],
                }
            },
            {
                "message": {
                    "role": "assistant",
                    "content": "",
                    "tool_calls": [{
                        "id": "call_4",
                        "function": {
                            "name": "browser",
                            "arguments": {"url": f"{wiki_url}/#overview"},
                        },
                    }],
                }
            },
            # Turn 5: Final summary
            {
                "message": {
                    "role": "assistant",
                    "content": "Dense payload: Summary of German rearmament programs.",
                    "tool_calls": [],
                }
            },
        ]

        with patch("core.llm.chat", new_callable=AsyncMock, side_effect=responses), \
             patch("tools.registry.get_tool") as mock_get_tool, \
             patch("core.subagent._store_subagent_memory", new_callable=AsyncMock):

            mock_search = AsyncMock()
            mock_search.execute = AsyncMock(return_value=(
                f"1. German military technology - Wikipedia\n"
                f"   URL: {wiki_url}\n"
                f"   Snippet: Overview of Luftwaffe and Wehrmacht equipment."
            ))
            mock_search.to_ollama_schema = lambda: {"type": "function", "function": {"name": "web_search"}}

            mock_browser = AsyncMock()
            mock_browser.execute = AsyncMock(return_value="Detailed article text about military rearmament.")
            mock_browser.to_ollama_schema = lambda: {"type": "function", "function": {"name": "browser"}}

            def get_tool_side_effect(name):
                if name == "web_search":
                    return mock_search
                if name == "browser":
                    return mock_browser
                return None

            mock_get_tool.side_effect = get_tool_side_effect

            result = asyncio.run(run_subagent(task="Research German rearmament", max_turns=6))

            self.assertTrue(result.success)
            # Browser execute should only be called ONCE because the second call was intercepted as duplicate!
            self.assertEqual(mock_browser.execute.await_count, 1)
            self.assertEqual(len(result.tools_used), 4)
            # Second browser call in tools_used succeeded via cache/notice
            self.assertTrue(result.tools_used[3]["success"])





