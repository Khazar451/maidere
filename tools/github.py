"""GitHub REST API integration tool for Maidere.

Enables the agent to:
- Inspect GitHub Actions workflow runs and fetch failing CI/CD logs
- List and read GitHub issues and pull requests
- Create new issues, comments, and pull requests
- Read file contents from remote repositories
"""

import base64
import json
from typing import Any

import httpx
import structlog

from core.config import settings
from tools.base import BaseTool

logger = structlog.get_logger()

GITHUB_API_BASE = "https://api.github.com"
DEFAULT_TIMEOUT = 15.0


class GitHubTool(BaseTool):
    """Tool to interact with GitHub repositories, actions, issues, and pull requests."""

    name: str = "github"
    description: str = (
        "Interact with GitHub repositories: inspect CI/CD workflow runs and logs, "
        "read/create issues and pull requests, and view repository files."
    )
    parameters: dict[str, Any] = {
        "type": "object",
        "properties": {
            "action": {
                "type": "string",
                "enum": [
                    "get_workflow_runs",
                    "get_workflow_logs",
                    "list_issues",
                    "get_issue",
                    "create_issue",
                    "create_issue_comment",
                    "get_pull_request",
                    "create_pull_request",
                    "get_file_content",
                ],
                "description": "The GitHub action to perform.",
            },
            "repo": {
                "type": "string",
                "description": (
                    "Target repository in 'owner/repo' format (e.g., 'Khazar451/maidere'). "
                    "Defaults to configured GITHUB_DEFAULT_REPO."
                ),
            },
            "run_id": {
                "type": "string",
                "description": "Workflow run ID for inspecting runs or downloading logs.",
            },
            "issue_number": {
                "type": "integer",
                "description": "Issue or Pull Request number.",
            },
            "title": {
                "type": "string",
                "description": "Title for a new issue or pull request.",
            },
            "body": {
                "type": "string",
                "description": "Body text for a new issue, pull request, or comment.",
            },
            "head": {
                "type": "string",
                "description": "Head branch name when creating a pull request.",
            },
            "base": {
                "type": "string",
                "description": "Base branch name when creating a pull request (default: 'main').",
            },
            "path": {
                "type": "string",
                "description": "File path in repository for get_file_content.",
            },
            "ref": {
                "type": "string",
                "description": "Git commit, branch, or tag reference (default: 'main').",
            },
        },
        "required": ["action"],
    }

    def _get_headers(self) -> dict[str, str]:
        """Construct standard GitHub API headers."""
        headers = {
            "Accept": "application/vnd.github+json",
            "X-GitHub-Api-Version": "2022-11-28",
            "User-Agent": "Maidere-Agent",
        }
        token = settings.github_token
        if token and token.strip():
            headers["Authorization"] = f"Bearer {token.strip()}"
        return headers

    def _resolve_repo(self, repo: str | None) -> str:
        """Resolve repository name with fallback to default config."""
        if repo and repo.strip():
            return repo.strip()
        default = getattr(settings, "github_default_repo", "Khazar451/maidere")
        return default or "Khazar451/maidere"

    async def execute(self, **kwargs: Any) -> str:
        """Execute the specified GitHub action asynchronously."""
        action = kwargs.get("action", "")
        repo = self._resolve_repo(kwargs.get("repo"))

        try:
            if action == "get_workflow_runs":
                return await self._get_workflow_runs(repo)
            elif action == "get_workflow_logs":
                run_id = str(kwargs.get("run_id") or "").strip()
                return await self._get_workflow_logs(repo, run_id)
            elif action == "list_issues":
                return await self._list_issues(repo)
            elif action == "get_issue":
                issue_num = kwargs.get("issue_number")
                return await self._get_issue(repo, issue_num)
            elif action == "create_issue":
                title = kwargs.get("title", "")
                body = kwargs.get("body", "")
                return await self._create_issue(repo, title, body)
            elif action == "create_issue_comment":
                issue_num = kwargs.get("issue_number")
                body = kwargs.get("body", "")
                return await self._create_issue_comment(repo, issue_num, body)
            elif action == "get_pull_request":
                pr_num = kwargs.get("issue_number")
                return await self._get_pull_request(repo, pr_num)
            elif action == "create_pull_request":
                title = kwargs.get("title", "")
                body = kwargs.get("body", "")
                head = kwargs.get("head", "")
                base = kwargs.get("base", "main")
                return await self._create_pull_request(repo, title, body, head, base)
            elif action == "get_file_content":
                path = kwargs.get("path", "")
                ref = kwargs.get("ref", "main")
                return await self._get_file_content(repo, path, ref)
            else:
                return f"Error: Unknown GitHub action '{action}'."
        except httpx.HTTPStatusError as e:
            status = e.response.status_code
            if status == 401:
                return (
                    "Error: GitHub API authentication failed (401 Unauthorized). "
                    "Ensure GITHUB_TOKEN is set with valid credentials in .env."
                )
            elif status == 404:
                return (
                    f"Error: Resource not found (404) for repository '{repo}'. "
                    "Check the repository name or token permissions for private repositories."
                )
            elif status == 403:
                return (
                    "Error: GitHub API rate limit exceeded or access forbidden (403). "
                    "Check token permissions or API quota."
                )
            elif status == 422:
                try:
                    err_json = e.response.json()
                    msg = err_json.get("message", str(e))
                except Exception:
                    msg = e.response.text
                return f"Error: Validation failed (422 Unprocessable Entity): {msg}"
            return f"Error: GitHub API request failed with status {status}: {e.response.text[:300]}"
        except httpx.RequestError as e:
            return f"Error: Failed to connect to GitHub API: {str(e)}"
        except Exception as e:
            await logger.aerror("github_tool_failed", action=action, repo=repo, error=str(e))
            return f"Error executing GitHub action '{action}': {str(e)}"

    async def _get_workflow_runs(self, repo: str) -> str:
        """Fetch recent workflow runs."""
        url = f"{GITHUB_API_BASE}/repos/{repo}/actions/runs?per_page=5"
        async with httpx.AsyncClient(headers=self._get_headers(), timeout=DEFAULT_TIMEOUT) as client:
            resp = await client.get(url)
            resp.raise_for_status()
            data = resp.json()

        runs = data.get("workflow_runs", [])
        if not runs:
            return f"No workflow runs found for {repo}."

        lines = [f"Recent workflow runs for {repo}:"]
        for r in runs:
            run_id = r.get("id")
            name = r.get("name", "workflow")
            status = r.get("status", "unknown")
            conclusion = r.get("conclusion") or "in_progress"
            branch = r.get("head_branch", "unknown")
            commit_msg = (r.get("head_commit") or {}).get("message", "").split("\n")[0][:60]
            created = r.get("created_at", "")[:19]
            lines.append(
                f"- Run ID {run_id} | Name: '{name}' | Status: {status} | "
                f"Conclusion: {conclusion} | Branch: {branch} | Commit: \"{commit_msg}\" | Created: {created}"
            )
        return "\n".join(lines)

    async def _get_workflow_logs(self, repo: str, run_id: str) -> str:
        """Fetch failing step logs from a workflow run."""
        headers = self._get_headers()
        async with httpx.AsyncClient(headers=headers, timeout=DEFAULT_TIMEOUT, follow_redirects=True) as client:
            if not run_id:
                # Find the most recent failed run automatically
                runs_url = f"{GITHUB_API_BASE}/repos/{repo}/actions/runs?per_page=5"
                r_resp = await client.get(runs_url)
                r_resp.raise_for_status()
                runs = r_resp.json().get("workflow_runs", [])
                failed_run = next((r for r in runs if r.get("conclusion") == "failure"), None)
                if not failed_run:
                    latest = runs[0] if runs else None
                    if not latest:
                        return f"No workflow runs found for {repo}."
                    run_id = str(latest.get("id"))
                else:
                    run_id = str(failed_run.get("id"))

            # 1. Fetch jobs for the run
            jobs_url = f"{GITHUB_API_BASE}/repos/{repo}/actions/runs/{run_id}/jobs"
            j_resp = await client.get(jobs_url)
            j_resp.raise_for_status()
            jobs = j_resp.json().get("jobs", [])

            if not jobs:
                return f"No jobs found for workflow run {run_id}."

            output_lines = [f"Diagnostic report for workflow run {run_id} ({repo}):"]
            found_failure = False

            for job in jobs:
                job_id = job.get("id")
                job_name = job.get("name", "Job")
                conclusion = job.get("conclusion") or job.get("status")
                steps = job.get("steps", [])

                failed_step_name = None
                for step in steps:
                    if step.get("conclusion") == "failure":
                        failed_step_name = step.get("name")
                        break

                output_lines.append(f"\nJob '{job_name}' (ID: {job_id}) - Status: {conclusion}")
                if failed_step_name:
                    output_lines.append(f"  Failing Step: '{failed_step_name}'")

                # Fetch plaintext log if job failed
                if conclusion == "failure":
                    found_failure = True
                    try:
                        log_url = f"{GITHUB_API_BASE}/repos/{repo}/actions/jobs/{job_id}/logs"
                        log_resp = await client.get(log_url)
                        if log_resp.status_code == 200:
                            log_text = log_resp.text
                            # Extract the tail of the log (last 50 lines / 3000 chars)
                            log_lines = log_text.splitlines()
                            tail_lines = log_lines[-50:] if len(log_lines) > 50 else log_lines
                            output_lines.append("  Execution Log Excerpt:")
                            output_lines.append("  ----------------------------------------")
                            output_lines.extend(f"  {line}" for line in tail_lines)
                            output_lines.append("  ----------------------------------------")
                    except Exception as ex:
                        output_lines.append(f"  (Could not fetch log text: {str(ex)})")

            if not found_failure:
                output_lines.append("\nAll jobs in this run passed or are currently in progress.")

            return "\n".join(output_lines)

    async def _list_issues(self, repo: str) -> str:
        """List open issues in the repository."""
        url = f"{GITHUB_API_BASE}/repos/{repo}/issues?state=open&per_page=10"
        async with httpx.AsyncClient(headers=self._get_headers(), timeout=DEFAULT_TIMEOUT) as client:
            resp = await client.get(url)
            resp.raise_for_status()
            issues = resp.json()

        if not issues:
            return f"No open issues found in {repo}."

        lines = [f"Open issues for {repo}:"]
        for iss in issues:
            num = iss.get("number")
            title = iss.get("title", "")
            user = (iss.get("user") or {}).get("login", "unknown")
            is_pr = "pull_request" in iss
            kind = "PR" if is_pr else "Issue"
            comments = iss.get("comments", 0)
            lines.append(f"- #{num} [{kind}] '{title}' by @{user} ({comments} comments)")
        return "\n".join(lines)

    async def _get_issue(self, repo: str, issue_number: int | None) -> str:
        """Retrieve single issue details."""
        if not issue_number:
            return "Error: 'issue_number' parameter is required for get_issue."
        url = f"{GITHUB_API_BASE}/repos/{repo}/issues/{issue_number}"
        async with httpx.AsyncClient(headers=self._get_headers(), timeout=DEFAULT_TIMEOUT) as client:
            resp = await client.get(url)
            resp.raise_for_status()
            iss = resp.json()

        title = iss.get("title", "")
        state = iss.get("state", "")
        user = (iss.get("user") or {}).get("login", "unknown")
        body = iss.get("body") or "No description provided."
        created = iss.get("created_at", "")[:19]
        labels = [l.get("name", "") for l in iss.get("labels", []) if l.get("name")]

        res = [
            f"Issue #{issue_number} ({repo}): {title}",
            f"State: {state} | Author: @{user} | Created: {created}",
        ]
        if labels:
            res.append(f"Labels: {', '.join(labels)}")
        res.append(f"\nDescription:\n{body}")
        return "\n".join(res)

    async def _create_issue(self, repo: str, title: str, body: str) -> str:
        """Create a new issue."""
        if not title or not title.strip():
            return "Error: 'title' parameter is required to create an issue."
        url = f"{GITHUB_API_BASE}/repos/{repo}/issues"
        payload = {"title": title.strip(), "body": body or ""}
        async with httpx.AsyncClient(headers=self._get_headers(), timeout=DEFAULT_TIMEOUT) as client:
            resp = await client.post(url, json=payload)
            resp.raise_for_status()
            data = resp.json()
        num = data.get("number")
        html_url = data.get("html_url", "")
        return f"Successfully created issue #{num}: '{title}'\nURL: {html_url}"

    async def _create_issue_comment(self, repo: str, issue_number: int | None, body: str) -> str:
        """Add a comment to an issue or pull request."""
        if not issue_number:
            return "Error: 'issue_number' parameter is required to add a comment."
        if not body or not body.strip():
            return "Error: 'body' parameter is required for comment content."
        url = f"{GITHUB_API_BASE}/repos/{repo}/issues/{issue_number}/comments"
        payload = {"body": body.strip()}
        async with httpx.AsyncClient(headers=self._get_headers(), timeout=DEFAULT_TIMEOUT) as client:
            resp = await client.post(url, json=payload)
            resp.raise_for_status()
            data = resp.json()
        comment_id = data.get("id")
        return f"Successfully posted comment (ID: {comment_id}) on #{issue_number} in {repo}."

    async def _get_pull_request(self, repo: str, pr_number: int | None) -> str:
        """Get details and status of a pull request."""
        if not pr_number:
            return "Error: 'issue_number' parameter is required for get_pull_request."
        url = f"{GITHUB_API_BASE}/repos/{repo}/pulls/{pr_number}"
        async with httpx.AsyncClient(headers=self._get_headers(), timeout=DEFAULT_TIMEOUT) as client:
            resp = await client.get(url)
            resp.raise_for_status()
            pr = resp.json()

        title = pr.get("title", "")
        state = pr.get("state", "")
        user = (pr.get("user") or {}).get("login", "unknown")
        head = (pr.get("head") or {}).get("ref", "")
        base = (pr.get("base") or {}).get("ref", "")
        mergeable = pr.get("mergeable")
        body = pr.get("body") or "No description provided."

        return (
            f"Pull Request #{pr_number} ({repo}): {title}\n"
            f"State: {state} | Author: @{user} | Branch: {head} -> {base} | Mergeable: {mergeable}\n\n"
            f"Description:\n{body}"
        )

    async def _create_pull_request(self, repo: str, title: str, body: str, head: str, base: str) -> str:
        """Create a new pull request."""
        if not title or not title.strip():
            return "Error: 'title' is required to create a pull request."
        if not head or not head.strip():
            return "Error: 'head' branch name is required to create a pull request."
        url = f"{GITHUB_API_BASE}/repos/{repo}/pulls"
        payload = {
            "title": title.strip(),
            "body": body or "",
            "head": head.strip(),
            "base": (base or "main").strip(),
        }
        async with httpx.AsyncClient(headers=self._get_headers(), timeout=DEFAULT_TIMEOUT) as client:
            resp = await client.post(url, json=payload)
            resp.raise_for_status()
            data = resp.json()
        num = data.get("number")
        html_url = data.get("html_url", "")
        return f"Successfully created pull request #{num}: '{title}' ({head} -> {base})\nURL: {html_url}"

    async def _get_file_content(self, repo: str, path: str, ref: str) -> str:
        """Read a file's content from a GitHub repository."""
        if not path or not path.strip():
            return "Error: 'path' parameter is required for get_file_content."
        clean_path = path.strip().lstrip("/")
        url = f"{GITHUB_API_BASE}/repos/{repo}/contents/{clean_path}?ref={ref}"
        async with httpx.AsyncClient(headers=self._get_headers(), timeout=DEFAULT_TIMEOUT) as client:
            resp = await client.get(url)
            resp.raise_for_status()
            data = resp.json()

        if isinstance(data, list):
            # Directory listing
            entries = [f"- {item.get('type')}: {item.get('name')}" for item in data]
            return f"Directory listing for '{clean_path}' in {repo}@{ref}:\n" + "\n".join(entries)

        content_b64 = data.get("content", "")
        encoding = data.get("encoding", "")
        if encoding == "base64" and content_b64:
            decoded = base64.b64decode(content_b64).decode("utf-8", errors="replace")
            return f"File '{clean_path}' ({repo}@{ref}):\n\n{decoded}"
        return f"File '{clean_path}' in {repo}@{ref} (unsupported encoding '{encoding}')."
