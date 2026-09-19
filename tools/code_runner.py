"""Sandboxed Python code runner tool for Maidere.

Executes Python code in an isolated subprocess with:
- shell=False
- 30-second hard timeout
- Guaranteed cleanup of temporary script files
- Scoped inside workspace/
"""

import asyncio
from pathlib import Path
import subprocess
import sys
import uuid
from typing import Any

from core.config import settings
from tools.base import BaseTool

WORKSPACE: Path = (Path.cwd() / settings.agent_workspace).resolve()
DEFAULT_TIMEOUT_SECONDS: float = 30.0


class CodeRunnerTool(BaseTool):
    """Tool to execute Python code in a dedicated subprocess with timeout enforcement."""

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
        """Execute Python code asynchronously in a subprocess."""
        if not code or not code.strip():
            return "Error: No Python code provided to execute."

        tmp_dir = WORKSPACE / ".tmp"
        tmp_dir.mkdir(parents=True, exist_ok=True)
        script_file = tmp_dir / f"snippet_{uuid.uuid4().hex[:8]}.py"

        def _run() -> str:
            try:
                # Write script
                script_file.write_text(code, encoding="utf-8")

                # Execute using the active Python interpreter
                python_bin = sys.executable or "python3"
                result = subprocess.run(
                    [python_bin, str(script_file)],
                    cwd=str(WORKSPACE),
                    shell=False,
                    capture_output=True,
                    text=True,
                    timeout=DEFAULT_TIMEOUT_SECONDS,
                )

                stdout = result.stdout.strip()
                stderr = result.stderr.strip()

                output_parts = []
                if stdout:
                    output_parts.append(stdout)
                if stderr:
                    output_parts.append(f"[stderr]\n{stderr}")
                if result.returncode != 0:
                    output_parts.append(f"[Exit code: {result.returncode}]")

                return "\n".join(output_parts) if output_parts else "Code executed successfully (no output)."
            except subprocess.TimeoutExpired:
                return f"Error: Execution timed out after {DEFAULT_TIMEOUT_SECONDS} seconds."
            except Exception as e:
                return f"Error executing Python snippet: {str(e)}"
            finally:
                if script_file.exists():
                    try:
                        script_file.unlink()
                    except Exception:
                        pass

        return await asyncio.to_thread(_run)
