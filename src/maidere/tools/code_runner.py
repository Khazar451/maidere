"""Sandboxed Python code runner tool for Maidere.

Executes Python code in an isolated subprocess via ``maidere.tools.sandbox``:
- bubblewrap namespaces when available (read-only root, workspace-only writes,
  no network), otherwise restricted mode with a scrubbed environment
- ``python -I`` (isolated mode: ignores PYTHON* env vars and user site-packages)
- 30-second hard timeout, memory/file-size rlimits, output caps
- Guaranteed cleanup of the temporary script file
- Project virtualenv packages remain importable (mounted read-only)
"""

import asyncio
import sys
import uuid
from typing import Any

from maidere.core.config import settings
from maidere.tools.base import BaseTool
from maidere.tools.filesystem import get_workspace
from maidere.tools.sandbox import SandboxError, run_sandboxed

DEFAULT_TIMEOUT_SECONDS: float = 30.0


class CodeRunnerTool(BaseTool):
    """Tool to execute Python code in a dedicated sandboxed subprocess with timeout enforcement."""

    name: str = "code_runner"
    description: str = (
        "Execute Python code in an isolated subprocess with a 30-second timeout. "
        "Code runs within the workspace environment."
    )
    parameters: dict[str, Any] = {
        "type": "object",
        "properties": {
            "code": {
                "type": "string",
                "description": "The Python source code string to execute.",
            }
        },
        "required": ["code"],
    }

    async def execute(self, code: str, **kwargs: Any) -> str:
        """Execute Python code asynchronously in a sandboxed subprocess."""
        if not code or not code.strip():
            return "Error: No Python code provided to execute."

        workspace = get_workspace()
        tmp_dir = workspace / ".tmp"
        tmp_dir.mkdir(parents=True, exist_ok=True)
        script_file = tmp_dir / f"snippet_{uuid.uuid4().hex[:8]}.py"
        timeout = min(DEFAULT_TIMEOUT_SECONDS, float(settings.tool_sandbox_timeout_seconds))
        # Keep the unresolved interpreter path: a venv is detected from the
        # symlink location (pyvenv.cfg), resolving it would drop the venv.
        python_bin = sys.executable or "python3"

        def _run() -> str:
            try:
                script_file.write_text(code, encoding="utf-8")
                try:
                    result = run_sandboxed(
                        [python_bin, "-I", str(script_file)],
                        workspace=workspace,
                        timeout=timeout,
                    )
                except SandboxError as e:
                    return str(e)
                if result.timed_out:
                    return f"Error: Execution timed out after {timeout:g} seconds."

                output_parts = []
                if result.stdout:
                    output_parts.append(result.stdout)
                if result.stderr:
                    output_parts.append(f"[stderr]\n{result.stderr}")
                if result.truncated:
                    output_parts.append("[output truncated]")
                if result.returncode != 0:
                    output_parts.append(f"[Exit code: {result.returncode}]")
                return "\n".join(output_parts) if output_parts else "Code executed successfully (no output)."
            except Exception as e:
                return f"Error executing Python snippet: {e!s}"
            finally:
                try:
                    script_file.unlink(missing_ok=True)
                except Exception:
                    pass

        return await asyncio.to_thread(_run)
