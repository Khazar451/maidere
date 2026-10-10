"""Shell command execution tool for Maidere with security guardrails.

Policy layer (this module):
- shell=False with argument list execution
- Safe shlex.split() exception handling
- Strict Linux binary allowlist, read-only git subcommands
- Deterministic destructive-pattern rejection

Isolation layer (``maidere.tools.sandbox``):
- bubblewrap namespaces (read-only root, workspace-only writes, no network),
  or argument-level confinement when bubblewrap is unavailable
- Scrubbed environment, rlimits, timeout, output caps
"""

import asyncio
import shutil
from pathlib import Path
import shlex
from typing import Any

from maidere.core.config import settings
from maidere.tools.base import BaseTool
from maidere.tools.filesystem import get_workspace
from maidere.tools.sandbox import SandboxError, run_sandboxed


def __getattr__(name: str) -> Any:
    if name == "WORKSPACE":
        return get_workspace()
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

# Strict allowlist of permitted Linux binaries
ALLOWED_COMMANDS: set[str] = {
    "python3",
    "python",
    "node",
    "ls",
    "cat",
    "grep",
    "mkdir",
    "echo",
    "touch",
    "find",
    "head",
    "tail",
    "pwd",
    "cp",
    "mv",
    "rm",
    "wc",
    "git",
}

ALLOWED_GIT_SUBCOMMANDS: set[str] = {
    "status",
    "diff",
    "log",
    "branch",
    "show",
}

DISALLOWED_GIT_FLAGS: set[str] = {"-c", "--work-tree", "--git-dir", "--exec-path"}

DEFAULT_TIMEOUT_SECONDS: float = 30.0


class ShellTool(BaseTool):
    """Tool to execute allowed shell commands inside the workspace sandbox."""

    name: str = "shell"
    description: str = (
        "Execute a permitted shell command inside the workspace sandbox. "
        "Allowed binaries: python3, python, node, ls, cat, grep, mkdir, echo, touch, find, head, tail, pwd, cp, mv, rm, wc, "
        "git (read-only inspection: status, diff, log, branch, show)."
    )
    parameters: dict[str, Any] = {
        "type": "object",
        "properties": {
            "command": {
                "type": "string",
                "description": "The shell command to execute, e.g. 'python3 hello.py' or 'ls -la'.",
            }
        },
        "required": ["command"],
    }

    async def execute(self, command: str, **kwargs: Any) -> str:
        """Execute command securely and return output string."""
        if not command or not command.strip():
            return "Error: Empty command provided."

        # 1. Deterministic Destructive Action Check
        import re

        DESTRUCTIVE_PATTERNS = [
            r"\brm\s+-[a-zA-Z]*[rf][a-zA-Z]*\b",  # rm -rf, rm -r, rm -f, etc.
            r"\bgit\s+reset\s+--hard\b",
            r"\bgit\s+clean\s+-[a-zA-Z]*f\b",
            r"\b(drop|truncate)\s+(table|database)\b",
            r"\bmkfs(\.[a-zA-Z0-9]+)?\b",
            r"\bdd\s+if=",
        ]
        for pat in DESTRUCTIVE_PATTERNS:
            if re.search(pat, command.strip(), re.IGNORECASE):
                return "Error: Destructive action blocked. Explicit user confirmation required."

        # 2. Safe shlex.split() with exception handling for unclosed quotes
        try:
            args = shlex.split(command.strip())
        except ValueError as e:
            return f"Error parsing command syntax: {str(e)}"

        if not args:
            return "Error: No command arguments parsed."

        # 3. Allowlist verification

        binary_raw = args[0]
        binary_name = Path(binary_raw).name

        if binary_name not in ALLOWED_COMMANDS:
            return (
                f"Error: Command binary '{binary_name}' is not permitted by security policy. "
                f"Allowed binaries: {', '.join(sorted(ALLOWED_COMMANDS))}."
            )

        # 4. Git subcommand and boundary verification
        if binary_name == "git":
            for arg in args[1:]:
                arg_lower = arg.lower()
                if arg_lower in DISALLOWED_GIT_FLAGS or any(arg_lower.startswith(f"{f}=") for f in DISALLOWED_GIT_FLAGS):
                    return f"Error: Git flag '{arg}' is blocked by security policy to prevent workspace escaping."

            subcmd = None
            for arg in args[1:]:
                if not arg.startswith("-"):
                    subcmd = arg.lower()
                    break

            if not subcmd:
                return (
                    f"Error: Git command requires a permitted inspection subcommand. "
                    f"Allowed subcommands: {', '.join(sorted(ALLOWED_GIT_SUBCOMMANDS))}."
                )

            if subcmd not in ALLOWED_GIT_SUBCOMMANDS:
                return (
                    f"Error: Git subcommand '{subcmd}' is blocked by security policy. "
                    f"Only read-only inspection commands are permitted: {', '.join(sorted(ALLOWED_GIT_SUBCOMMANDS))}."
                )

        workspace = get_workspace()

        # 5. Resolve the binary on the host. The allowlist is matched on the
        # basename, so a relative/absolute argv[0] cannot smuggle in another
        # executable; the resolved real path is what actually runs.
        resolved = shutil.which(binary_name)
        if not resolved:
            return f"Error: Executable '{binary_name}' not found on host system."
        args[0] = str(Path(resolved).resolve())

        # 6. Execution inside the sandbox (bwrap namespaces or restricted mode)
        timeout = min(DEFAULT_TIMEOUT_SECONDS, float(settings.tool_sandbox_timeout_seconds))

        def _run() -> str:
            try:
                result = run_sandboxed(args, workspace=workspace, timeout=timeout)
            except SandboxError as e:
                return str(e)
            except Exception as e:
                return f"Error executing command: {str(e)}"
            if result.timed_out:
                return f"Error: Command execution timed out after {timeout:g} seconds."
            return result.format()

        return await asyncio.to_thread(_run)
