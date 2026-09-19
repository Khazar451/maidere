"""Tests for Phase 4 components: code_runner, browser, and scheduler."""

import asyncio
from datetime import datetime
import tempfile
import unittest
from unittest.mock import AsyncMock, patch

from core.scheduler import (
    add_scheduled_job,
    delete_scheduled_job,
    list_scheduled_jobs,
    shutdown_scheduler,
    start_scheduler,
)
from tools.browser import BrowserTool, browse_url
from tools.code_runner import CodeRunnerTool
from tools.registry import execute_tool, get_all_tools, get_tool
from tools.scheduler import SchedulerTool


class TestCodeRunner(unittest.IsolatedAsyncioTestCase):
    """Test sandboxed Python code runner."""

    async def asyncSetUp(self):
        self.runner = CodeRunnerTool()

    async def test_python_code_execution(self):
        """Test running basic Python script."""
        code = "print('Result is:', 100 * 25)"
        result = await self.runner.execute(code=code)
        self.assertIn("Result is: 2500", result)

    async def test_python_code_with_imports_and_multiline(self):
        """Test multiline code using standard library."""
        code = """
import json
data = {"agent": "Maidere", "version": "0.1.0"}
print(json.dumps(data))
"""
        result = await self.runner.execute(code=code)
        self.assertIn('"agent": "Maidere"', result)

    async def test_python_error_traceback_capture(self):
        """Test that runtime errors return stderr and exit code."""
        code = "raise ValueError('Custom test error')"
        result = await self.runner.execute(code=code)
        self.assertIn("ValueError: Custom test error", result)
        self.assertIn("[Exit code: 1]", result)

    async def test_empty_code_string(self):
        """Test empty code string validation."""
        result = await self.runner.execute(code="   ")
        self.assertIn("Error: No Python code provided", result)


class TestBrowserTool(unittest.IsolatedAsyncioTestCase):
    """Test browser tool with Playwright zombie process prevention."""

    async def asyncSetUp(self):
        self.browser_tool = BrowserTool()

    async def test_empty_url_handling(self):
        """Test empty URL validation."""
        result = await self.browser_tool.execute(url="")
        self.assertIn("Error: Empty URL provided.", result)

    @patch("tools.browser.async_playwright")
    async def test_browser_guaranteed_cleanup(self, mock_playwright):
        """Verify that browser.close() is ALWAYS called even if page fails."""
        mock_browser = AsyncMock()
        mock_page = AsyncMock()
        mock_page.goto.side_effect = TimeoutError("Navigation timeout")
        mock_browser.new_page.return_value = mock_page

        mock_p_instance = AsyncMock()
        mock_p_instance.chromium.launch.return_value = mock_browser

        # Setup async context manager
        mock_cm = AsyncMock()
        mock_cm.__aenter__.return_value = mock_p_instance
        mock_playwright.return_value = mock_cm

        result = await self.browser_tool.execute(url="https://slow-site.com")
        self.assertIn("Error browsing", result)
        # Verify browser.close() was executed in finally block
        mock_browser.close.assert_called_once()


class TestSchedulerSystem(unittest.IsolatedAsyncioTestCase):
    """Test APScheduler persistent jobs and scheduler tool."""

    async def asyncSetUp(self):
        self.temp_db = tempfile.NamedTemporaryFile(suffix=".db", delete=False)
        self.temp_db_path = self.temp_db.name
        self.temp_db.close()
        self.scheduler = start_scheduler(self.temp_db_path)
        self.tool = SchedulerTool()

    async def asyncTearDown(self):
        import os
        shutdown_scheduler()
        if os.path.exists(self.temp_db_path):
            os.remove(self.temp_db_path)

    async def test_create_cron_and_interval_jobs(self):
        """Test creating cron and interval scheduled jobs."""
        # Add cron job: Every Friday at 5pm
        job1 = add_scheduled_job(
            task_name="friday_goal_review",
            schedule_type="cron",
            schedule_value="0 17 * * 5",
            prompt="Review weekly development goals and summarize achievements.",
            db_path=self.temp_db_path,
        )
        self.assertEqual(job1["name"], "friday_goal_review")

        # Add interval job: Every 30 minutes
        job2 = add_scheduled_job(
            task_name="health_heartbeat",
            schedule_type="interval",
            schedule_value="minutes=30",
            prompt="Check system status and disk space.",
            db_path=self.temp_db_path,
        )
        self.assertEqual(job2["name"], "health_heartbeat")

        # List jobs
        jobs = list_scheduled_jobs(self.temp_db_path)
        job_names = [j["name"] for j in jobs]
        self.assertIn("friday_goal_review", job_names)
        self.assertIn("health_heartbeat", job_names)

        # Delete job
        deleted = delete_scheduled_job("health_heartbeat", self.temp_db_path)
        self.assertTrue(deleted)

        remaining_jobs = list_scheduled_jobs(self.temp_db_path)
        remaining_names = [j["name"] for j in remaining_jobs]
        self.assertNotIn("health_heartbeat", remaining_names)
        self.assertIn("friday_goal_review", remaining_names)

    async def test_scheduler_tool_actions(self):
        """Test scheduler tool execute dispatcher."""
        with patch("core.config.settings.db_path", self.temp_db_path):
            # Create
            res_create = await self.tool.execute(
                action="create",
                task_name="daily_standup_report",
                schedule_type="cron",
                schedule_value="0 9 * * 1-5",
                prompt="Prepare daily standup bullet points.",
            )
            self.assertIn("Successfully scheduled task 'daily_standup_report'", res_create)

            # List
            res_list = await self.tool.execute(action="list")
            self.assertIn("daily_standup_report", res_list)

            # Delete
            res_delete = await self.tool.execute(
                action="delete",
                task_name="daily_standup_report",
            )
            self.assertIn("Successfully deleted", res_delete)


class TestPhase4ToolRegistry(unittest.TestCase):
    """Test that all Phase 4 tools are registered and have valid schemas."""

    def test_registry_contains_phase4_tools(self):
        all_tools = get_all_tools()
        names = [t.name for t in all_tools]
        self.assertIn("code_runner", names)
        self.assertIn("browser", names)
        self.assertIn("scheduler", names)
        self.assertIn("read_file", names)
        self.assertIn("write_file", names)
        self.assertIn("list_dir", names)
        self.assertIn("shell", names)


if __name__ == "__main__":
    unittest.main()
