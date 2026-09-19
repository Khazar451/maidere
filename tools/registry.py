"""Tool registry and dispatcher with audit logging for Maidere."""

import json
import time
from typing import Any

import structlog

from core.db import get_db
from tools.base import BaseTool
from tools.browser import BrowserTool
from tools.code_runner import CodeRunnerTool
from tools.delegate import AgentTool, DelegateTaskTool
from tools.filesystem import ListDirTool, ReadFileTool, WriteFileTool
from tools.github import GitHubTool
from tools.obsidian import ObsidianTool
from tools.scheduler import SchedulerTool
from tools.shell import ShellTool
from tools.skills import ListSkillsTool, LoadSkillTool
from tools.web_search import WebSearchTool

logger = structlog.get_logger()

_delegate_instance = DelegateTaskTool()
_agent_instance = AgentTool()

# Register all built-in tools
_TOOLS: dict[str, BaseTool] = {
    "delegate_task": _delegate_instance,
    "Agent": _agent_instance,
    "web_search": WebSearchTool(),
    "browser": BrowserTool(),
    "github": GitHubTool(),
    "obsidian": ObsidianTool(),
    "read_file": ReadFileTool(),
    "write_file": WriteFileTool(),
    "list_dir": ListDirTool(),
    "shell": ShellTool(),
    "code_runner": CodeRunnerTool(),
    "scheduler": SchedulerTool(),
    "list_available_skills": ListSkillsTool(),
    "load_skill": LoadSkillTool(),
}

_ALIASES: dict[str, str] = {
    "agent": "Agent",
    "delegate": "Agent",
    "subagent": "Agent",
    "Task": "Agent",
    "task": "Agent",
    "spawn_subagent": "Agent",
    "deep_research": "Agent",
    "search_web": "web_search",
    "browse_url": "browser",
    "gh": "github",
    "github_action": "github",
    "github_repo": "github",
    "execute_command": "shell",
    "run_code": "code_runner",
    "obsidian_write_note": "obsidian",
    "obsidian_read_note": "obsidian",
    "obsidian_list_notes": "obsidian",
    "obsidian_open_note": "obsidian",
    "obsidian_search_notes": "obsidian",
}



def get_tool(name: str) -> BaseTool | None:
    """Retrieve a tool instance by name with alias support."""
    actual_name = _ALIASES.get(name, name)
    return _TOOLS.get(actual_name)



ORCHESTRATOR_TOOL_NAMES: tuple[str, ...] = (
    "delegate_task",
    "Agent",
    "obsidian",
    "read_file",
    "write_file",
    "list_dir",
    "shell",
    "code_runner",
    "scheduler",
    "list_available_skills",
    "load_skill",
)


def get_all_tools() -> list[BaseTool]:
    """Retrieve all registered tool instances."""
    return list(_TOOLS.values())


def get_tools_ollama_schemas() -> list[dict[str, Any]]:
    """Return list of all tool schemas formatted for Ollama API."""
    return [tool.to_ollama_schema() for tool in _TOOLS.values()]


def get_orchestrator_ollama_schemas() -> list[dict[str, Any]]:
    """Return tool schemas specifically for the orchestrator agent.

    Excludes raw web_search and browser to force all research and web data
    gathering through delegate_task sub-agents with dedicated 8K contexts.
    """
    schemas = []
    for name in ORCHESTRATOR_TOOL_NAMES:
        tool = _TOOLS.get(name)
        if tool:
            schemas.append(tool.to_ollama_schema())
    return schemas


from core.audit import log_audit_record, record_audit, sanitize_secrets



async def execute_tool(
    name: str,
    args: dict[str, Any],
    thread_id: str | None = None,
    db_path: str | None = None,
) -> str:
    """Execute a registered tool by name with parameter validation and audit logging."""
    tool = get_tool(name)
    if tool is None:
        return f"Error: Tool '{name}' is not recognized. Available tools: {', '.join(_TOOLS.keys())}"

    start_time = time.perf_counter()
    success = True
    result = ""

    await logger.ainfo("tool_executing", tool_name=name, thread_id=thread_id, args=args)

    try:
        result = await tool.execute(**args)
        if result.startswith("Error:"):
            success = False
    except Exception as e:
        success = False
        result = f"Error executing tool '{name}': {str(e)}"
        await logger.aerror("tool_execution_exception", tool_name=name, error=str(e))

    duration_s = time.perf_counter() - start_time
    duration_ms = int(duration_s * 1000)

    from core.metrics import TOOL_CALLS_TOTAL, TOOL_DURATION_SECONDS

    TOOL_CALLS_TOTAL.labels(
        tool_name=name,
        status="success" if success else "error",
    ).inc()
    TOOL_DURATION_SECONDS.labels(tool_name=name).observe(duration_s)

    await logger.ainfo(
        "tool_executed",
        tool_name=name,
        thread_id=thread_id,
        duration_ms=duration_ms,
        success=success,
    )

    # Persist audit log
    await log_audit_record(
        thread_id=thread_id,
        tool_name=name,
        tool_args=args,
        tool_result=result,
        duration_ms=duration_ms,
        success=success,
        db_path=db_path,
    )

    return result

