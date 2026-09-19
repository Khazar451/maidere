"""Automated test suite for Agent Ethics, Secret Redaction, and Evaluation Loop."""

import tempfile
import unittest
from unittest.mock import AsyncMock, MagicMock, patch

import numpy as np
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

from core.agent import create_graph, evaluate_node
from core.audit import record_audit, sanitize_secrets
from core.db import get_db, init_tables
from core.memory import init_memory_table, store_memory
from tools.shell import ShellTool

_patchers = []


def setUpModule():
    mock_emb_model = MagicMock()
    mock_emb_model.embed.side_effect = lambda texts: [np.array([0.05] * 384) for _ in texts]
    p1 = patch("core.embeddings._get_model", return_value=mock_emb_model)
    p2 = patch("core.embeddings.embed_one", side_effect=lambda t: [0.05] * 384)
    p3 = patch("core.embeddings.embed", side_effect=lambda texts: [[0.05] * 384 for _ in texts])
    p4 = patch("core.memory.embed_one", side_effect=lambda t: [0.05] * 384)
    for p in (p1, p2, p3, p4):
        p.start()
        _patchers.append(p)


def tearDownModule():
    for p in _patchers:
        p.stop()
    _patchers.clear()


class TestSecretSanitization(unittest.TestCase):
    """Test regex-based secret detection and redaction."""

    def test_sanitize_openai_keys(self):
        text = "Here is my key: sk-abcdef1234567890abcdef1234567890 and another sk-1234567890123456789012345."
        sanitized = sanitize_secrets(text)
        self.assertNotIn("sk-abcdef", sanitized)
        self.assertIn("[REDACTED_SECRET]", sanitized)

    def test_sanitize_anthropic_and_hf_keys(self):
        text = "Anthropic: sk-ant-api03-abcdef1234567890abcdef1234567890 and HF: hf_abcdef1234567890abcdef1234."
        sanitized = sanitize_secrets(text)
        self.assertNotIn("sk-ant-api03", sanitized)
        self.assertNotIn("hf_abcdef", sanitized)
        self.assertEqual(sanitized.count("[REDACTED_SECRET]"), 2)

    def test_sanitize_github_and_aws_keys(self):
        text = "GH: ghp_1234567890abcdef1234567890abcdef12 and AWS: AKIAIOSFODNN7EXAMPLE."
        sanitized = sanitize_secrets(text)
        self.assertNotIn("ghp_123456", sanitized)
        self.assertNotIn("AKIAIOSFODNN7EXAMPLE", sanitized)
        self.assertIn("[REDACTED_SECRET]", sanitized)

    def test_sanitize_bearer_and_basic_tokens(self):
        text = "Authorization: Bearer secret_access_token_1234567890 and Basic dXNlcjpwYXNzd29yZDEyMw=="
        sanitized = sanitize_secrets(text)
        self.assertIn("Bearer [REDACTED_SECRET]", sanitized)
        self.assertIn("Basic [REDACTED_SECRET]", sanitized)

    def test_sanitize_private_keys(self):
        text = (
            "-----BEGIN RSA PRIVATE KEY-----\n"
            "MIIEowIBAAKCAQEA0Y123456789abcdefghijklmnopqrstuvwxyz...\n"
            "-----END RSA PRIVATE KEY-----"
        )
        sanitized = sanitize_secrets(text)
        self.assertEqual(sanitized, "[REDACTED_PRIVATE_KEY]")

    def test_sanitize_password_assignments(self):
        text = '{"username": "admin", "password": "supersecretpassword123", "api_key": "mysecretkey456"}'
        sanitized = sanitize_secrets(text)
        self.assertNotIn("supersecretpassword123", sanitized)
        self.assertNotIn("mysecretkey456", sanitized)
        self.assertIn("[REDACTED_SECRET]", sanitized)


class TestDatabaseRedaction(unittest.IsolatedAsyncioTestCase):
    """Test that SQLite memory and audit tables redact secrets before storage."""

    async def asyncSetUp(self):
        self.temp_db = tempfile.NamedTemporaryFile(suffix=".db", delete=False)
        self.temp_db_path = self.temp_db.name
        self.temp_db.close()
        db = await get_db(self.temp_db_path)
        try:
            await init_tables(db)
            await init_memory_table(db)
        finally:
            await db.close()

    async def asyncTearDown(self):
        import os
        if os.path.exists(self.temp_db_path):
            os.remove(self.temp_db_path)

    async def test_store_memory_redacts_secrets(self):
        """Verify secrets are redacted before inserting into memories table."""
        db = await get_db(self.temp_db_path)
        try:
            raw_content = "My AWS key is AKIAIOSFODNN7EXAMPLE and token is sk-1234567890123456789012345"
            await store_memory(db, content=raw_content, session_id="test-session-redact")

            rows = await db.execute_fetchall("SELECT content FROM memories")
            self.assertEqual(len(rows), 1)
            stored_content = rows[0][0]
            self.assertNotIn("AKIAIOSFODNN7EXAMPLE", stored_content)
            self.assertNotIn("sk-123456", stored_content)
            self.assertIn("[REDACTED_SECRET]", stored_content)
        finally:
            await db.close()

    async def test_record_audit_redacts_secrets(self):
        """Verify tool args and results are redacted before inserting into audit_log table."""
        db = await get_db(self.temp_db_path)
        try:
            raw_args = {"token": "ghp_1234567890abcdef1234567890abcdef12", "path": "secrets.env"}
            raw_result = "Loaded API_KEY=sk-ant-api03-abcdef1234567890abcdef1234567890"

            await record_audit(
                thread_id="test-thread",
                tool_name="read_file",
                tool_args=raw_args,
                tool_result=raw_result,
                duration_ms=15,
                success=True,
                db_path=self.temp_db_path,
            )

            rows = await db.execute_fetchall("SELECT tool_args, tool_result FROM audit_log")
            self.assertEqual(len(rows), 1)
            stored_args, stored_result = rows[0][0], rows[0][1]

            self.assertNotIn("ghp_123456", stored_args)
            self.assertIn("[REDACTED_SECRET]", stored_args)
            self.assertNotIn("sk-ant-api03", stored_result)
            self.assertIn("[REDACTED_SECRET]", stored_result)
        finally:
            await db.close()


class TestDestructiveCommandBlocking(unittest.IsolatedAsyncioTestCase):
    """Test deterministic rejection of destructive shell commands."""

    async def test_rm_rf_blocked(self):
        tool = ShellTool()
        result = await tool.execute(command="rm -rf workspace/notes")
        self.assertIn("Error: Destructive action blocked", result)

    async def test_rm_r_blocked(self):
        tool = ShellTool()
        result = await tool.execute(command="rm -r some_dir")
        self.assertIn("Error: Destructive action blocked", result)

    async def test_git_reset_hard_blocked(self):
        tool = ShellTool()
        result = await tool.execute(command="git reset --hard HEAD~1")
        self.assertIn("Error: Destructive action blocked", result)

    async def test_drop_table_blocked(self):
        tool = ShellTool()
        result = await tool.execute(command="python3 -c 'DROP TABLE users'")
        self.assertIn("Error: Destructive action blocked", result)

    async def test_safe_command_allowed(self):
        tool = ShellTool()
        result = await tool.execute(command="echo 'Maidere Safety Active'")
        self.assertIn("Maidere Safety Active", result)


class TestEvaluationNodeAndSelfCorrection(unittest.IsolatedAsyncioTestCase):
    """Test evaluate_node diagnostic guidance and consecutive error abort limit."""

    async def test_evaluate_node_no_error(self):
        """When tool succeeds, evaluate_node returns empty dict."""
        state = {
            "messages": [
                ToolMessage(content="File created successfully.", tool_call_id="call-1", name="write_file")
            ]
        }
        res = await evaluate_node(state)
        self.assertEqual(res, {})

    async def test_evaluate_node_first_error_injects_guidance(self):
        """On first tool failure, evaluate_node appends self-correction guidance."""
        state = {
            "messages": [
                ToolMessage(content="Error: File not found", tool_call_id="call-1", name="read_file")
            ]
        }
        res = await evaluate_node(state)
        self.assertIn("messages", res)
        updated_msg = res["messages"][0]
        self.assertIn("SYSTEM EVALUATION GUIDANCE", updated_msg.content)
        self.assertIn("Error: File not found", updated_msg.content)

    async def test_evaluate_node_consecutive_errors_triggers_abort_retries(self):
        """When multiple consecutive tool errors occur, evaluate_node injects [ABORT RETRIES]."""
        state = {
            "messages": [
                ToolMessage(content="Error: SearXNG container offline", tool_call_id="call-1", name="web_search"),
                AIMessage(content="", tool_calls=[{"name": "web_search", "args": {"query": "test"}, "id": "call-2"}]),
                ToolMessage(content="Error: SearXNG container offline", tool_call_id="call-2", name="web_search"),
            ]
        }
        res = await evaluate_node(state)
        self.assertIn("messages", res)
        updated_msg = res["messages"][0]
        self.assertIn("[ABORT RETRIES", updated_msg.content)
        self.assertIn("Do NOT retry calling this tool", updated_msg.content)


class TestAgentLoopWithEvaluation(unittest.IsolatedAsyncioTestCase):
    """Test full LangGraph agentic loop with evaluate_node."""

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

    async def test_graph_executes_evaluate_node_in_loop(self):
        """Verify full loop: think -> act -> evaluate -> think -> finish."""
        mock_tool_call_response = {
            "message": {
                "role": "assistant",
                "content": "",
                "tool_calls": [
                    {
                        "function": {
                            "name": "shell",
                            "arguments": {"command": "echo 'Testing loop'"},
                        }
                    }
                ],
            }
        }
        mock_final_response = {
            "message": {
                "role": "assistant",
                "content": "Command finished: Testing loop",
                "tool_calls": [],
            }
        }

        with patch("core.llm.chat", new_callable=AsyncMock, side_effect=[mock_tool_call_response, mock_final_response]):
            config = {"configurable": {"thread_id": "test-eval-loop"}}
            result = await self.graph.ainvoke(
                {
                    "messages": [HumanMessage(content="Run test command")],
                    "memory_context": "No memories",
                    "thread_id": "test-eval-loop",
                },
                config=config,
            )

            final_msg = result["messages"][-1]
            self.assertIn("Testing loop", final_msg.content)


class TestBrowserSanitizationAndCap(unittest.IsolatedAsyncioTestCase):
    """Test character cap (2500 chars) and whitespace sanitization in browser tool."""

    @patch("tools.browser.async_playwright")
    async def test_browse_url_caps_at_2500_and_sanitizes_whitespace(self, mock_playwright):
        from tools.browser import MAX_CONTENT_CHARS, browse_url

        self.assertEqual(MAX_CONTENT_CHARS, 2500)

        # Mock playwright page with massive text and excessive whitespace
        mock_browser = AsyncMock()
        mock_page = AsyncMock()
        mock_element = AsyncMock()
        
        huge_raw_text = "Headline\n\n\n\n\n   Lots   of   spaces   \n\n" + ("Technical content details. " * 300)
        mock_element.inner_text.return_value = huge_raw_text
        mock_page.query_selector.return_value = mock_element
        mock_page.inner_text.return_value = huge_raw_text
        mock_browser.new_page.return_value = mock_page

        mock_p_instance = AsyncMock()
        mock_p_instance.chromium.launch.return_value = mock_browser

        mock_cm = AsyncMock()
        mock_cm.__aenter__.return_value = mock_p_instance
        mock_playwright.return_value = mock_cm


        result = await browse_url("https://example.com/huge-spec")
        self.assertLessEqual(len(result), 2500)
        self.assertNotIn("   Lots   of   spaces   ", result)
        self.assertIn("Lots of spaces", result)
        self.assertNotIn("\n\n\n\n\n", result)


class TestOllamaTimeoutConfiguration(unittest.TestCase):
    """Test extended Ollama client timeout settings."""

    def test_ollama_timeout_is_180s(self):
        from core.config import settings

        self.assertEqual(settings.ollama_timeout, 180.0)
        self.assertEqual(settings.ollama_connect_timeout, 10.0)


if __name__ == "__main__":
    unittest.main()

