"""AGY Staffer Bridge for Maidere.

Enables Maidere to hire Google's Antigravity CLI ('agy') as a fast cloud-accelerated
staffer running Gemini 3.8 Flash, based on Keli Wen's agy-staff pattern.

Supported Personas:
- researcher  : Rapid codebase survey, web search, and documentation research.
- reviewer    : Adversarial cross-model review of code, diffs, and implementation plans.
- implementer : Scoped code modifications with git dirty-state conflict prevention.
- ask         : Tool-free instant Q&A.
- staffer     : General autonomous worker.

Security & Resilience Guarantees:
- Safe subprocess execution: argument arrays only, shell=False unconditionally.
- Dirty-state git checkpointing: detects pre-existing uncommitted files and instructs agy not to clobber them.
- Rate-limit & Quota resilience: automatically detects 429/RESOURCE_EXHAUSTED and transparently falls back to local subagents.
- File-based artifact bus: results persisted to workspace/.maidere/artifacts/ with compact briefs returned to the orchestrator.
"""

import asyncio
from datetime import datetime, timezone
import os
from pathlib import Path
import re
import shutil
import time
from typing import Any
import uuid

import structlog

from core.config import settings
from core.subagent import (
    generate_handoff_brief,
    prune_artifacts,
    run_subagent,
    write_subagent_artifact,
)
from tools.base import BaseTool

logger = structlog.get_logger()

# Common search paths for the agy binary
AGY_CANDIDATE_PATHS = [
    Path.home() / ".local" / "bin" / "agy",
    Path.home() / ".gemini" / "antigravity" / "bin" / "agy",
    Path("/usr/local/bin/agy"),
    Path("/usr/bin/agy"),
]

PERSONA_DIRECTIVES = {
    "researcher": (
        "You are acting as an AGY Research Staffer running Gemini 3.8 Flash.\n"
        "Your task is rapid, comprehensive codebase and technical research.\n"
        "Provide dense facts, benchmarks, exact file paths, code snippets, and verified source URLs.\n"
        "Avoid conversational filler or narrative preamble."
    ),
    "reviewer": (
        "You are acting as an AGY Adversarial Reviewer running Gemini 3.8 Flash.\n"
        "Your task is to critically inspect code, git diffs, architectural plans, and system designs.\n"
        "You are an independent, second-opinion reviewer: look for regressions, missing tests, security vulnerabilities, and logic flaws.\n"
        "Format findings with numbered points categorized as [BLOCKER], [WARNING], or [SUGGESTION]."
    ),
    "implementer": (
        "You are acting as an AGY Implementation Staffer running Gemini 3.8 Flash.\n"
        "Your task is focused, precise code modification according to the given instructions.\n"
        "Follow project conventions, maintain documentation integrity, and do not introduce unrelated refactors."
    ),
    "ask": (
        "You are acting as an AGY Fast Q&A Staffer running Gemini 3.8 Flash.\n"
        "Answer the query directly, concisely, and accurately without tool use."
    ),
    "staffer": (
        "You are acting as an AGY Autonomous Staffer running Gemini 3.8 Flash.\n"
        "Execute the scoped task with precision and provide an actionable, structured report."
    ),
}

RATE_LIMIT_PATTERNS = [
    "429",
    "resource_exhausted",
    "rate limit",
    "rate-limit",
    "quota exceeded",
    "quota_exceeded",
    "user rate limit",
]


def find_agy_executable() -> str | None:
    """Locate the agy CLI binary on the host system."""
    # 1. Check system PATH
    found = shutil.which("agy")
    if found:
        return found

    # 2. Check candidate user paths
    for p in AGY_CANDIDATE_PATHS:
        if p.is_file() and os.access(p, os.X_OK):
            return str(p)

    return None


def get_workspace_dirty_files(workspace_dir: Path) -> list[str]:
    """Run git status --porcelain to detect existing uncommitted or modified files."""
    if not workspace_dir.exists():
        return []

    try:
        import subprocess

        res = subprocess.run(
            ["git", "status", "--porcelain"],
            cwd=str(workspace_dir),
            capture_output=True,
            text=True,
            check=False,
            timeout=5,
        )
        if res.returncode != 0:
            return []

        dirty_files = []
        for line in res.stdout.strip().splitlines():
            parts = line.strip().split(maxsplit=1)
            if len(parts) == 2:
                dirty_files.append(parts[1].strip())
        return dirty_files
    except Exception as e:
        logger.debug("git_dirty_check_failed", error=str(e))
        return []


def is_rate_limited(error_text: str) -> bool:
    """Check if output or error indicates quota exhaustion or rate limiting."""
    lower = error_text.lower()
    return any(p in lower for p in RATE_LIMIT_PATTERNS)


class AgyStafferTool(BaseTool):
    """Bridge tool allowing Maidere to hire Google Antigravity CLI ('agy') as a fast staffer."""

    name: str = "agy_staffer"
    description: str = (
        "Hire Google's Antigravity CLI ('agy') as a fast cloud-accelerated staffer running Gemini 3.8 Flash. "
        "Supports 5 personas: "
        "'researcher' (rapid codebase survey and web discovery), "
        "'reviewer' (adversarial cross-model review of code, plans, and diffs), "
        "'implementer' (code changes with git dirty-state awareness), "
        "'ask' (tool-free instant Q&A), and "
        "'staffer' (general autonomous worker). "
        "Automatically protects uncommitted user work, writes results to the artifact bus, "
        "and seamlessly falls back to local subagents if agy is not installed or quota-limited."
    )
    parameters: dict[str, Any] = {
        "type": "object",
        "properties": {
            "task": {
                "type": "string",
                "description": "Specific instruction, query, or review prompt for the AGY staffer.",
            },
            "persona": {
                "type": "string",
                "description": (
                    "The AGY persona to invoke: 'researcher', 'reviewer', 'implementer', 'ask', or 'staffer'. "
                    "Default: 'researcher'."
                ),
            },
            "extra_args": {
                "type": "array",
                "items": {"type": "string"},
                "description": "Optional list of additional CLI flags to pass to agy (e.g. ['--model', 'gemini-3.8-flash']).",
            },
        },
        "required": ["task"],
    }

    async def execute(
        self,
        task: str = "",
        persona: str = "researcher",
        extra_args: list[str] | None = None,
        **kwargs: Any,
    ) -> str:
        """Execute task via agy CLI or fallback to local subagent."""
        actual_task = task.strip()
        if not actual_task:
            return "Error: Empty task specified for agy_staffer."

        resolved_persona = persona.lower().strip() if persona else "researcher"
        if resolved_persona not in PERSONA_DIRECTIVES:
            resolved_persona = "researcher"

        start_time = time.perf_counter()
        workspace_path = Path(settings.agent_workspace).resolve()

        agy_binary = find_agy_executable()

        # Fallback 1: Binary not installed on host
        if not agy_binary:
            await logger.ainfo("agy_binary_not_found_fallback_to_local", persona=resolved_persona)
            return await self._fallback_local_subagent(
                actual_task,
                resolved_persona,
                reason=(
                    "Google Antigravity CLI ('agy') is not installed on this machine.\n"
                    "Install via: curl -fsSL https://antigravity.google/cli/install.sh | bash"
                ),
            )

        # Build persona directive
        persona_directive = PERSONA_DIRECTIVES[resolved_persona]

        # Dirty-state git awareness
        dirty_notice = ""
        dirty_files = get_workspace_dirty_files(workspace_path)
        if dirty_files:
            dirty_list_str = "\n".join(f"- {f}" for f in dirty_files[:20])
            dirty_notice = (
                f"\n\n[WORKSPACE DIRTY STATE NOTICE]\n"
                f"The following {len(dirty_files)} file(s) currently have uncommitted user modifications:\n"
                f"{dirty_list_str}\n"
                f"CRITICAL: Do NOT overwrite, delete, or touch these files unless explicitly instructed.\n"
                f"Leave your modifications as an uncommitted working tree diff."
            )

        full_prompt = (
            f"{persona_directive}\n"
            f"{dirty_notice}\n\n"
            f"Assigned Task:\n{actual_task}"
        )

        # Assemble safe argument array (shell=False)
        cmd = [agy_binary, "-p", full_prompt]
        if extra_args and isinstance(extra_args, list):
            # Only allow sanitized string flags
            for arg in extra_args:
                if isinstance(arg, str) and not arg.startswith(";"):
                    cmd.append(arg)

        await logger.ainfo(
            "agy_staffer_executing",
            persona=resolved_persona,
            cmd_binary=agy_binary,
            task_preview=actual_task[:100],
        )

        try:
            process = await asyncio.create_subprocess_exec(
                *cmd,
                cwd=str(workspace_path),
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )

            stdout_bytes, stderr_bytes = await asyncio.wait_for(
                process.communicate(),
                timeout=120.0,
            )

            stdout = stdout_bytes.decode("utf-8", errors="replace").strip()
            stderr = stderr_bytes.decode("utf-8", errors="replace").strip()
            duration_ms = int((time.perf_counter() - start_time) * 1000)

            # Check for non-zero exit or rate-limit
            combined_output = f"{stdout}\n{stderr}"
            if process.returncode != 0 or is_rate_limited(combined_output):
                reason = "Rate-limited / quota exhausted" if is_rate_limited(combined_output) else f"Process exited with code {process.returncode}: {stderr[:200]}"
                await logger.awarn(
                    "agy_staffer_rate_limited_or_error",
                    exit_code=process.returncode,
                    reason=reason,
                )
                return await self._fallback_local_subagent(
                    actual_task,
                    resolved_persona,
                    reason=f"AGY run failed ({reason}). Fallback to local subagent runner.",
                )

            # Execution succeeded! Write artifact
            prune_artifacts()
            artifact_file = write_subagent_artifact(
                task=actual_task,
                content=stdout or "Task completed with empty output.",
                subagent_type=f"agy_{resolved_persona}",
                tools_used=[{"tool_name": "agy_cli", "success": True}],
                duration_ms=duration_ms,
                turns_used=1,
                model="gemini-3.8-flash (via agy)",
            )

            brief = generate_handoff_brief(
                task=actual_task,
                full_content=stdout,
                artifact_path=artifact_file,
                tools_used=[{"tool_name": "agy_cli", "success": True}],
                duration_ms=duration_ms,
                turns_used=1,
                subagent_type=f"agy_{resolved_persona}",
            )

            return (
                f"[AGY STAFFER RESULT: {resolved_persona.upper()}]\n"
                f"Task: {actual_task}\n"
                f"Status: completed successfully\n"
                f"Model: Gemini 3.8 Flash (Google Antigravity CLI)\n"
                f"Artifact: {artifact_file}\n"
                f"Execution: Duration: {duration_ms}ms\n"
                f"{'─' * 60}\n"
                f"{brief}\n"
                f"{'─' * 60}\n"
                f"[END AGY STAFFER RESULT]"
            )

        except asyncio.TimeoutError:
            await logger.awarn("agy_staffer_timeout", task=actual_task[:100])
            return await self._fallback_local_subagent(
                actual_task,
                resolved_persona,
                reason="AGY execution timed out after 120s. Fallback to local subagent.",
            )
        except Exception as e:
            await logger.aerror("agy_staffer_exception", error=str(e))
            return await self._fallback_local_subagent(
                actual_task,
                resolved_persona,
                reason=f"Subprocess error: {str(e)}. Fallback to local subagent.",
            )

    async def _fallback_local_subagent(self, task: str, persona: str, reason: str) -> str:
        """Transparently execute task using Maidere's internal subagent engine."""
        # Map agy persona to local subagent definition
        local_mapping = {
            "researcher": "researcher",
            "reviewer": "reviewer",
            "implementer": "code-reviewer",
            "ask": "general-purpose",
            "staffer": "staffer",
        }
        target_local_type = local_mapping.get(persona, "researcher")

        res = await run_subagent(
            task=task,
            subagent_type=target_local_type,
            max_turns=6,
        )

        tools_summary = ", ".join(
            f"{t['tool_name']}({'OK' if t['success'] else 'FAIL'})"
            for t in res.tools_used
        ) or "none"

        return (
            f"[AGY STAFFER FALLBACK -> LOCAL {target_local_type.upper()}]\n"
            f"Notice: {reason}\n"
            f"Task: {res.task}\n"
            f"Status: {'completed successfully' if res.success else 'FAILED'}\n"
            f"Artifact: {res.artifact_path}\n"
            f"Execution: {res.turns_used} turns | Duration: {res.duration_ms}ms | Tools: {tools_summary}\n"
            f"{'─' * 60}\n"
            f"{res.brief or res.summary}\n"
            f"{'─' * 60}\n"
            f"[END AGY STAFFER RESULT]"
        )
