"""Tools package for Maidere agent."""

from maidere.tools.base import BaseTool
from maidere.tools.browser import BrowserTool, browse_url
from maidere.tools.code_runner import CodeRunnerTool
from maidere.tools.filesystem import ListDirTool, ReadFileTool, WriteFileTool, secure_path
from maidere.tools.registry import (
    execute_tool,
    get_all_tools,
    get_tool,
    get_tools_ollama_schemas,
)
from maidere.tools.scheduler import SchedulerTool
from maidere.tools.shell import ShellTool
from maidere.tools.skills import ListSkillsTool, LoadSkillTool, list_available_skills, load_skill
from maidere.tools.web_search import WebSearchTool, search_searxng

__all__ = [
    "BaseTool",
    "BrowserTool",
    "CodeRunnerTool",
    "ListDirTool",
    "ListSkillsTool",
    "LoadSkillTool",
    "ReadFileTool",
    "SchedulerTool",
    "ShellTool",
    "WebSearchTool",
    "WriteFileTool",
    "browse_url",
    "execute_tool",
    "get_all_tools",
    "get_tool",
    "get_tools_ollama_schemas",
    "list_available_skills",
    "load_skill",
    "search_searxng",
    "secure_path",
]
