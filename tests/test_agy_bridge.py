"""Tests for Phase 2: External AGY Staffer Bridge (tools/agy_bridge.py)."""

import asyncio
from pathlib import Path
import tempfile
import unittest
from unittest.mock import AsyncMock, MagicMock, patch

from core.subagent import SubAgentResult
from tools.agy_bridge import (
    AgyStafferTool,
    find_agy_executable,
    get_workspace_dirty_files,
    is_rate_limited,
)
from tools.registry import get_tool


class TestAgyBridge(unittest.IsolatedAsyncioTestCase):
    """Test suite verifying AGY staffer bridge tool and resilience."""

    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.workspace = Path(self.temp_dir.name)
        self.tool = AgyStafferTool()

    def tearDown(self):
        self.temp_dir.cleanup()

    def test_tool_registration_and_aliases(self):
        """Verify that agy_staffer is registered in registry and aliased to 'agy'."""
        tool1 = get_tool("agy_staffer")
        tool2 = get_tool("agy")
        tool3 = get_tool("agy_agent")

        self.assertIsNotNone(tool1)
        self.assertIsInstance(tool1, AgyStafferTool)
        self.assertEqual(tool1, tool2)
        self.assertEqual(tool1, tool3)

    def test_empty_task_returns_error(self):
        """Verify empty task string is rejected."""
        loop = asyncio.get_event_loop()
        res = loop.run_until_complete(self.tool.execute(task="   "))
        self.assertIn("Error: Empty task", res)

    def test_rate_limit_detection(self):
        """Verify rate limit and quota exhaustion pattern matching."""
        self.assertTrue(is_rate_limited("Error: 429 Too Many Requests"))
        self.assertTrue(is_rate_limited("status: RESOURCE_EXHAUSTED"))
        self.assertTrue(is_rate_limited("Quota exceeded for project"))
        self.assertTrue(is_rate_limited("User rate limit reached"))
        self.assertFalse(is_rate_limited("Compilation successful with 0 errors"))

    def test_dirty_workspace_detection(self):
        """Verify that uncommitted git files are properly detected by get_workspace_dirty_files."""
        # Mock subprocess.run for git status --porcelain
        mock_proc = MagicMock()
        mock_proc.returncode = 0
        mock_proc.stdout = " M src/index.ts\n?? notes/new.md\n D old.txt\n"

        with patch("subprocess.run", return_value=mock_proc):
            dirty = get_workspace_dirty_files(self.workspace)

        self.assertEqual(dirty, ["src/index.ts", "notes/new.md", "old.txt"])

    @patch("tools.agy_bridge.find_agy_executable", return_value=None)
    @patch("tools.agy_bridge.run_subagent")
    async def test_fallback_when_agy_missing(self, mock_run_subagent, mock_find):
        """Verify that missing agy executable gracefully falls back to local subagent."""
        mock_run_subagent.return_value = SubAgentResult(
            task="Audit dependencies",
            summary="Checked dependencies: clean.",
            subagent_type="reviewer",
            tools_used=[],
            duration_ms=120,
            success=True,
            turns_used=1,
            artifact_path="/tmp/artifact_01.md",
            brief="• All 12 packages clean.",
        )

        res = await self.tool.execute(task="Audit dependencies", persona="reviewer")

        self.assertIn("[AGY STAFFER FALLBACK -> LOCAL REVIEWER]", res)
        self.assertIn("Google Antigravity CLI ('agy') is not installed", res)
        self.assertIn("curl -fsSL https://antigravity.google/cli/install.sh | bash", res)
        self.assertIn("All 12 packages clean", res)
        mock_run_subagent.assert_called_once()

    @patch("tools.agy_bridge.find_agy_executable", return_value="/usr/local/bin/agy")
    @patch("tools.agy_bridge.get_workspace_dirty_files", return_value=["core/db.py"])
    @patch("asyncio.create_subprocess_exec")
    async def test_successful_subprocess_execution(self, mock_exec, mock_dirty, mock_find):
        """Verify successful agy execution via safe argument array (shell=False)."""
        mock_proc = AsyncMock()
        mock_proc.returncode = 0
        mock_proc.communicate.return_value = (
            b"### AGY Flash Analysis\n- Finding 1: Refactor complete.\n- Finding 2: Zero regressions.",
            b"",
        )
        mock_exec.return_value = mock_proc

        with patch("tools.agy_bridge.settings.agent_workspace", self.temp_dir.name):
            res = await self.tool.execute(
                task="Refactor async session context",
                persona="implementer",
            )

        # Ensure create_subprocess_exec called with argument list
        self.assertTrue(mock_exec.called)
        args, kwargs = mock_exec.call_args
        self.assertEqual(args[0], "/usr/local/bin/agy")
        self.assertEqual(args[1], "-p")
        # Ensure prompt contains dirty state notice and directives
        self.assertIn("[WORKSPACE DIRTY STATE NOTICE]", args[2])
        self.assertIn("core/db.py", args[2])
        self.assertIn("Refactor async session context", args[2])

        # Verify returned result
        self.assertIn("[AGY STAFFER RESULT: IMPLEMENTER]", res)
        self.assertIn("Gemini 3.8 Flash (Google Antigravity CLI)", res)
        self.assertIn("Finding 1: Refactor complete", res)

    @patch("tools.agy_bridge.find_agy_executable", return_value="/usr/local/bin/agy")
    @patch("tools.agy_bridge.run_subagent")
    @patch("asyncio.create_subprocess_exec")
    async def test_rate_limit_fallback_to_local(self, mock_exec, mock_run_subagent, mock_find):
        """Verify that agy 429 / RESOURCE_EXHAUSTED triggers local fallback."""
        mock_proc = AsyncMock()
        mock_proc.returncode = 1
        mock_proc.communicate.return_value = (
            b"",
            b"Error: 429 RESOURCE_EXHAUSTED: User rate limit exceeded for model gemini-3.8-flash",
        )
        mock_exec.return_value = mock_proc

        mock_run_subagent.return_value = SubAgentResult(
            task="Heavy code scan",
            summary="Completed locally despite rate limit.",
            subagent_type="researcher",
            tools_used=[],
            duration_ms=250,
            success=True,
            turns_used=2,
            artifact_path="/tmp/local_artifact.md",
            brief="• Local researcher finished.",
        )

        with patch("tools.agy_bridge.settings.agent_workspace", self.temp_dir.name):
            res = await self.tool.execute(task="Heavy code scan", persona="researcher")

        self.assertIn("[AGY STAFFER FALLBACK -> LOCAL RESEARCHER]", res)
        self.assertIn("Rate-limited / quota exhausted", res)
        self.assertIn("Local researcher finished", res)
        mock_run_subagent.assert_called_once()

    @patch("tools.agy_bridge.find_agy_executable", return_value=None)
    @patch("tools.agy_bridge.run_subagent")
    async def test_verification_and_validation_fallback_to_local(self, mock_run_subagent, mock_find):
        """Verify that verification and validation personas fallback to correct local subagents."""
        mock_run_subagent.return_value = SubAgentResult(
            task="Verify logic",
            summary="[PASS: VERIFIED]",
            subagent_type="verification",
            tools_used=[],
            duration_ms=100,
            success=True,
            turns_used=2,
            artifact_path="/tmp/v_artifact.md",
            brief="• Verification passed.",
        )

        res_v = await self.tool.execute(task="Verify logic", persona="verification")
        self.assertIn("[AGY STAFFER FALLBACK -> LOCAL VERIFICATION]", res_v)
        self.assertIn("Verification passed", res_v)

        mock_run_subagent.return_value = SubAgentResult(
            task="Validate API",
            summary="[PASS: VALIDATED]",
            subagent_type="validation",
            tools_used=[],
            duration_ms=100,
            success=True,
            turns_used=2,
            artifact_path="/tmp/val_artifact.md",
            brief="• Validation passed.",
        )

        res_val = await self.tool.execute(task="Validate API", persona="validation")
        self.assertIn("[AGY STAFFER FALLBACK -> LOCAL VALIDATION]", res_val)
        self.assertIn("Validation passed", res_val)


if __name__ == "__main__":
    unittest.main()
