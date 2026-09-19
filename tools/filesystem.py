"""Sandboxed filesystem tools for Maidere.

Guarantees strict directory containment within the workspace/ folder,
preventing directory traversal attacks (e.g. ../../etc/passwd).
"""

import asyncio
import os
from pathlib import Path
from typing import Any

from core.config import settings
from tools.base import BaseTool

# Path resolution symmetry: WORKSPACE is explicitly resolved to absolute path at startup
WORKSPACE: Path = (Path.cwd() / settings.agent_workspace).resolve()


def secure_path(requested_path: str) -> Path:
    """Resolve and validate that requested path is within the workspace sandbox.

    Raises:
        ValueError: If path attempts to escape the workspace sandbox.
    """
    WORKSPACE.mkdir(parents=True, exist_ok=True)
    target = (WORKSPACE / requested_path).resolve()
    if not target.is_relative_to(WORKSPACE):
        raise ValueError("Access denied: Path is outside the workspace sandbox.")
    return target


class ReadFileTool(BaseTool):
    """Tool to read text content from a file in the workspace sandbox."""

    name: str = "read_file"
    description: str = "Read text content from a file in the workspace directory."
    parameters: dict[str, Any] = {
        "type": "object",
        "properties": {
            "path": {
                "type": "string",
                "description": "Relative path to file inside the workspace directory.",
            }
        },
        "required": ["path"],
    }

    async def execute(self, path: str, **kwargs: Any) -> str:
        if not path or not path.strip():
            return "Error: Empty path provided."

        clean_path = path.strip()
        if "?" in clean_path or len(clean_path.split()) > 4:
            return f"Error: Invalid file path '{path}'. File paths cannot be conversational phrases."

        try:
            target = secure_path(clean_path)
            if not target.exists():
                return f"Error: File '{clean_path}' does not exist."
            if not target.is_file():
                return f"Error: Path '{clean_path}' is not a file."


            content = await asyncio.to_thread(target.read_text, encoding="utf-8")
            return content
        except ValueError as e:
            return f"Error: {e}"
        except Exception as e:
            return f"Error reading file '{path}': {str(e)}"


class WriteFileTool(BaseTool):
    """Tool to write text content to a file in the workspace sandbox."""

    name: str = "write_file"
    description: str = "Write or overwrite text content to a file in the workspace directory."
    parameters: dict[str, Any] = {
        "type": "object",
        "properties": {
            "path": {
                "type": "string",
                "description": "Relative path to file inside the workspace directory.",
            },
            "content": {
                "type": "string",
                "description": "Text content to write into the file.",
            },
        },
        "required": ["path", "content"],
    }

    async def execute(self, path: str, content: str, **kwargs: Any) -> str:
        try:
            target = secure_path(path)
            target.parent.mkdir(parents=True, exist_ok=True)

            await asyncio.to_thread(target.write_text, content, encoding="utf-8")
            return f"Successfully wrote {len(content)} characters to '{path}'."
        except ValueError as e:
            return f"Error: {e}"
        except Exception as e:
            return f"Error writing file '{path}': {str(e)}"


class ListDirTool(BaseTool):
    """Tool to list files and directories within a workspace path."""

    name: str = "list_dir"
    description: str = "List files and directories inside the workspace directory."
    parameters: dict[str, Any] = {
        "type": "object",
        "properties": {
            "path": {
                "type": "string",
                "description": "Relative directory path inside workspace (use '.' for workspace root).",
                "default": ".",
            }
        },
        "required": [],
    }

    async def execute(self, path: str = ".", **kwargs: Any) -> str:
        try:
            target = secure_path(path)
            if not target.exists():
                return f"Error: Directory '{path}' does not exist."
            if not target.is_dir():
                return f"Error: Path '{path}' is not a directory."

            items = await asyncio.to_thread(lambda: sorted(os.listdir(target)))
            if not items:
                return f"Directory '{path}' is empty."

            formatted = []
            for item in items:
                item_path = target / item
                prefix = "[DIR] " if item_path.is_dir() else "[FILE]"
                formatted.append(f"{prefix} {item}")

            return "\n".join(formatted)
        except ValueError as e:
            return f"Error: {e}"
        except Exception as e:
            return f"Error listing directory '{path}': {str(e)}"
