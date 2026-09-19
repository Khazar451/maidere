"""Unit tests for GitHub REST API tool."""

import base64
import unittest
from unittest.mock import AsyncMock, MagicMock, patch

import httpx

from tools.github import GitHubTool
from tools.registry import get_tool


def _make_response(status_code: int = 200, json_data: any = None, text: str = "") -> MagicMock:
    """Create a mock httpx.Response object with synchronous methods."""
    resp = MagicMock()
    resp.status_code = status_code
    resp.text = text
    if json_data is not None:
        resp.json.return_value = json_data
    resp.raise_for_status.return_value = None
    return resp


class TestGitHubTool(unittest.IsolatedAsyncioTestCase):
    """Test suite for GitHubTool."""

    def setUp(self):
        self.tool = GitHubTool()

    def test_tool_registration(self):
        """Verify GitHubTool is registered with appropriate aliases."""
        tool = get_tool("github")
        self.assertIsNotNone(tool)
        self.assertEqual(tool.name, "github")

        tool_alias = get_tool("gh")
        self.assertIsNotNone(tool_alias)
        self.assertEqual(tool_alias.name, "github")

        tool_action_alias = get_tool("github_action")
        self.assertIsNotNone(tool_action_alias)
        self.assertEqual(tool_action_alias.name, "github")

    def test_tool_ollama_schema(self):
        """Verify tool produces a valid Ollama function schema."""
        schema = self.tool.to_ollama_schema()
        self.assertEqual(schema["type"], "function")
        self.assertEqual(schema["function"]["name"], "github")
        self.assertIn("action", schema["function"]["parameters"]["properties"])

    async def test_unknown_action(self):
        """Test handling of unsupported action."""
        result = await self.tool.execute(action="non_existent_action")
        self.assertIn("Error: Unknown GitHub action", result)

    @patch("httpx.AsyncClient.get")
    async def test_get_workflow_runs(self, mock_get):
        """Test listing workflow runs."""
        mock_resp = _make_response(
            status_code=200,
            json_data={
                "workflow_runs": [
                    {
                        "id": 12345678,
                        "name": "CI",
                        "status": "completed",
                        "conclusion": "failure",
                        "head_branch": "main",
                        "head_commit": {"message": "fix: update workflow"},
                        "created_at": "2026-09-19T22:50:00Z",
                    }
                ]
            },
        )
        mock_get.return_value = mock_resp

        result = await self.tool.execute(action="get_workflow_runs", repo="testowner/testrepo")
        self.assertIn("Recent workflow runs for testowner/testrepo:", result)
        self.assertIn("12345678", result)
        self.assertIn("failure", result)
        self.assertIn("main", result)

    @patch("httpx.AsyncClient.get")
    async def test_get_workflow_logs(self, mock_get):
        """Test fetching workflow job diagnostics and failing logs."""
        jobs_resp = _make_response(
            status_code=200,
            json_data={
                "jobs": [
                    {
                        "id": 888999,
                        "name": "Unit & Integration Tests",
                        "conclusion": "failure",
                        "steps": [
                            {"name": "Checkout repository", "conclusion": "success"},
                            {"name": "Install dependencies", "conclusion": "failure"},
                        ],
                    }
                ]
            },
        )
        logs_resp = _make_response(
            status_code=200,
            text="Error: Failed to resolve dependencies\nTraceback line 42",
        )

        async def fake_get(url, *args, **kwargs):
            if "jobs" in url and "logs" not in url:
                return jobs_resp
            return logs_resp

        mock_get.side_effect = fake_get

        result = await self.tool.execute(
            action="get_workflow_logs",
            repo="testowner/testrepo",
            run_id="12345678",
        )
        self.assertIn("Diagnostic report for workflow run 12345678", result)
        self.assertIn("Unit & Integration Tests", result)
        self.assertIn("Failing Step: 'Install dependencies'", result)
        self.assertIn("Error: Failed to resolve dependencies", result)

    @patch("httpx.AsyncClient.get")
    async def test_list_issues(self, mock_get):
        """Test listing open issues."""
        mock_resp = _make_response(
            status_code=200,
            json_data=[
                {
                    "number": 5,
                    "title": "Bug in query router",
                    "user": {"login": "tester"},
                    "comments": 2,
                }
            ],
        )
        mock_get.return_value = mock_resp

        result = await self.tool.execute(action="list_issues", repo="testowner/testrepo")
        self.assertIn("Open issues for testowner/testrepo:", result)
        self.assertIn("#5 [Issue] 'Bug in query router'", result)

    @patch("httpx.AsyncClient.get")
    async def test_get_issue(self, mock_get):
        """Test retrieving issue details."""
        mock_resp = _make_response(
            status_code=200,
            json_data={
                "title": "Memory leak on heavy reasoning",
                "state": "open",
                "user": {"login": "developer"},
                "body": "Detailed description of memory issue.",
                "created_at": "2026-09-19T20:00:00Z",
                "labels": [{"name": "bug"}, {"name": "priority-high"}],
            },
        )
        mock_get.return_value = mock_resp

        result = await self.tool.execute(action="get_issue", repo="testowner/testrepo", issue_number=10)
        self.assertIn("Issue #10 (testowner/testrepo): Memory leak on heavy reasoning", result)
        self.assertIn("bug, priority-high", result)
        self.assertIn("Detailed description of memory issue.", result)

    @patch("httpx.AsyncClient.post")
    async def test_create_issue(self, mock_post):
        """Test creating a new issue."""
        mock_resp = _make_response(
            status_code=201,
            json_data={
                "number": 12,
                "html_url": "https://github.com/testowner/testrepo/issues/12",
            },
        )
        mock_post.return_value = mock_resp

        result = await self.tool.execute(
            action="create_issue",
            repo="testowner/testrepo",
            title="Implement GitHub tool",
            body="Feature request details",
        )
        self.assertIn("Successfully created issue #12", result)
        self.assertIn("https://github.com/testowner/testrepo/issues/12", result)

    @patch("httpx.AsyncClient.post")
    async def test_create_issue_comment(self, mock_post):
        """Test adding a comment to an issue."""
        mock_resp = _make_response(
            status_code=201,
            json_data={
                "id": 9911,
            },
        )
        mock_post.return_value = mock_resp

        result = await self.tool.execute(
            action="create_issue_comment",
            repo="testowner/testrepo",
            issue_number=12,
            body="This issue is resolved.",
        )
        self.assertIn("Successfully posted comment (ID: 9911)", result)

    @patch("httpx.AsyncClient.get")
    async def test_get_pull_request(self, mock_get):
        """Test retrieving PR details."""
        mock_resp = _make_response(
            status_code=200,
            json_data={
                "title": "Add feature",
                "state": "open",
                "user": {"login": "dev"},
                "head": {"ref": "feature/branch"},
                "base": {"ref": "main"},
                "mergeable": True,
                "body": "PR notes",
            },
        )
        mock_get.return_value = mock_resp

        result = await self.tool.execute(
            action="get_pull_request",
            repo="testowner/testrepo",
            issue_number=20,
        )
        self.assertIn("Pull Request #20 (testowner/testrepo): Add feature", result)
        self.assertIn("feature/branch -> main", result)

    @patch("httpx.AsyncClient.post")
    async def test_create_pull_request(self, mock_post):
        """Test creating a pull request."""
        mock_resp = _make_response(
            status_code=201,
            json_data={
                "number": 15,
                "html_url": "https://github.com/testowner/testrepo/pull/15",
            },
        )
        mock_post.return_value = mock_resp

        result = await self.tool.execute(
            action="create_pull_request",
            repo="testowner/testrepo",
            title="feat: add GitHub tool",
            body="PR description",
            head="feature/github-tool",
            base="main",
        )
        self.assertIn("Successfully created pull request #15", result)
        self.assertIn("feature/github-tool -> main", result)

    @patch("httpx.AsyncClient.get")
    async def test_get_file_content(self, mock_get):
        """Test reading and decoding base64 file content from GitHub."""
        raw_code = "print('Hello from GitHub')"
        b64_code = base64.b64encode(raw_code.encode("utf-8")).decode("utf-8")

        mock_resp = _make_response(
            status_code=200,
            json_data={
                "type": "file",
                "encoding": "base64",
                "content": b64_code,
            },
        )
        mock_get.return_value = mock_resp

        result = await self.tool.execute(
            action="get_file_content",
            repo="testowner/testrepo",
            path="scripts/test.py",
            ref="main",
        )
        self.assertIn("File 'scripts/test.py' (testowner/testrepo@main):", result)
        self.assertIn("print('Hello from GitHub')", result)

    @patch("httpx.AsyncClient.get")
    async def test_401_unauthorized(self, mock_get):
        """Test error handling on 401 Unauthorized."""
        req = httpx.Request("GET", "https://api.github.com/repos/test/test")
        resp = httpx.Response(status_code=401, request=req)
        mock_get.side_effect = httpx.HTTPStatusError("Unauthorized", request=req, response=resp)

        result = await self.tool.execute(action="get_workflow_runs", repo="test/test")
        self.assertIn("Error: GitHub API authentication failed (401 Unauthorized)", result)

    @patch("httpx.AsyncClient.get")
    async def test_404_not_found(self, mock_get):
        """Test error handling on 404 Not Found."""
        req = httpx.Request("GET", "https://api.github.com/repos/test/test")
        resp = httpx.Response(status_code=404, request=req)
        mock_get.side_effect = httpx.HTTPStatusError("Not Found", request=req, response=resp)

        result = await self.tool.execute(action="get_issue", repo="test/test", issue_number=999)
        self.assertIn("Error: Resource not found (404) for repository 'test/test'", result)


if __name__ == "__main__":
    unittest.main()
