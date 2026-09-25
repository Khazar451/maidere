"""delegate_task / Agent tool — spawns specialized sub-agents.

Follows the Claude Code sub-agent architecture:
- Each sub-agent runs in an isolated context window with read-only tools.
- Scoped subagent types (researcher, code-reviewer, general-purpose).
- Returns dense verified payload to the parent context.
- Prevents context overflow and premature conversational completion.
"""

from typing import Any

import structlog

from core.subagent import MAX_SUBAGENTS_PER_QUERY, run_subagent
from tools.base import BaseTool

logger = structlog.get_logger()


class DelegateTaskTool(BaseTool):
    """Tool for the orchestrator to spawn focused sub-agents (Claude Code Agent tool)."""

    name: str = "delegate_task"
    description: str = (
        "Spawn an independent specialized subagent running in its own fresh 8K context window. "
        "MANDATORY for deep research, comprehensive analysis, detailed comparisons, technical overviews, "
        "or multi-faceted investigations. Decompose broad queries into 2-4 distinct sub-tasks and call "
        "Agent or delegate_task for each sub-task. Each sub-agent independently searches the web, browses pages, "
        "and extracts dense verified data. "
        "Available subagent types: "
        "'researcher' (fast read-only research: web search, documentation, facts, benchmarks), "
        "'tech-hardware' (enterprise hardware, GPU clusters, supercomputing, cloud compute grants), "
        "'plan' (software architecture, system design, technical implementation plans), "
        "'reviewer' or 'code-reviewer' (adversarial review of code, plans, and diffs), "
        "'staffer' or 'general-purpose' (scoped multi-step task execution). "
        "CRITICAL RULES: "
        "(1) When you call this tool, you MUST stop generating text and WAIT for results. "
        "Do NOT attempt to answer the user's query until ALL sub-agent calls have returned. "
        "(2) Decompose broad queries into 2-4 specific, non-overlapping sub-tasks. "
        "(3) After receiving ALL sub-agent summaries, synthesize them into one comprehensive final report."
    )
    parameters: dict[str, Any] = {
        "type": "object",
        "properties": {
            "subagent_type": {
                "type": "string",
                "description": (
                    "The type/name of subagent to spawn: 'researcher' (default), "
                    "'tech-hardware' (hardware, compute clusters, grants), "
                    "'plan' (architecture & planning), 'reviewer', 'code-reviewer', 'staffer', or 'general-purpose'."
                ),
            },
            "prompt": {
                "type": "string",
                "description": (
                    "Clear, specific instruction or research task for this sub-agent. "
                    "Be precise about what data points, metrics, comparisons, or facts to extract."
                ),
            },
            "task": {
                "type": "string",
                "description": "Alias for prompt.",
            },
            "max_turns": {
                "type": "integer",
                "description": (
                    "Maximum tool execution turns for this sub-agent (default: 6). "
                    "Higher values allow deeper research."
                ),
            },
            "model": {
                "type": "string",
                "description": "Optional model override (e.g. 'inherit').",
            },
        },
        "required": [],
    }

    # Per-query spawn counter — reset by route handler before each graph invocation.
    _spawn_count: int = 0
    _total_delegated: int = 0

    async def execute(
        self,
        prompt: str = "",
        task: str = "",
        subagent_type: str = "researcher",
        max_turns: int = 6,
        model: str | None = None,
        **kwargs: Any,
    ) -> str:
        """Execute a sub-agent for the given task.

        Returns a formatted result string that the orchestrator LLM
        uses to compose the final comprehensive answer.
        """
        actual_task = prompt.strip() or task.strip()
        if not actual_task:
            return "Error: Empty task description. Provide a specific subagent task or prompt."

        if self._spawn_count >= MAX_SUBAGENTS_PER_QUERY:
            return (
                f"Error: Maximum sub-agent limit ({MAX_SUBAGENTS_PER_QUERY}) reached. "
                "You must now synthesize your comprehensive answer from the sub-agent "
                "results you have already received. Do NOT call delegate_task again."
            )

        self._spawn_count += 1
        self._total_delegated += 1

        resolved_type = subagent_type.strip() if subagent_type else "researcher"

        await logger.ainfo(
            "subagent_spawning",
            subagent_type=resolved_type,
            task=actual_task[:100],
            spawn_index=self._spawn_count,
            max_turns=max_turns,
        )

        subagent_kwargs: dict[str, Any] = {
            "task": actual_task,
            "subagent_type": resolved_type,
            "max_turns": max_turns,
            "model": model,
            "subagent_index": self._spawn_count,
            "total_subagents": self._total_delegated,
        }
        if kwargs.get("num_ctx") is not None:
            subagent_kwargs["num_ctx"] = kwargs["num_ctx"]

        result = await run_subagent(**subagent_kwargs)

        # Format the result for the orchestrator to consume
        tools_summary = ", ".join(
            f"{t['tool_name']}({'[OK]' if t['success'] else '[FAIL]'})"
            for t in result.tools_used
        ) or "none"

        artifact_line = f"Artifact: {result.artifact_path}\n" if result.artifact_path else ""
        content_body = result.brief if result.brief else result.summary

        return (
            f"[SUB-AGENT RESEARCH RESULT]\n"
            f"Agent Type: {result.subagent_type}\n"
            f"Task: {result.task}\n"
            f"Status: {'completed successfully' if result.success else 'FAILED'}\n"
            f"{artifact_line}"
            f"Execution: {result.turns_used} turns | Duration: {result.duration_ms}ms | Tools: {tools_summary}\n"
            f"{'─' * 60}\n"
            f"{content_body}\n"
            f"{'─' * 60}\n"
            f"[END SUB-AGENT RESULT]"
        )

    def reset_spawn_count(self) -> None:
        """Reset per-query spawn counter. Called before each new user query."""
        self._spawn_count = 0
        self._total_delegated = 0


class AgentTool(DelegateTaskTool):
    """Claude Code canonical Agent tool alias."""

    name: str = "Agent"

