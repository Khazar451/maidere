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
    is_concurrent_safe: bool = True
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

            backup_created = False
            if target.exists() and target.is_file():
                bak_path = target.with_name(f"{target.name}.bak")
                existing_text = await asyncio.to_thread(target.read_text, encoding="utf-8")
                await asyncio.to_thread(bak_path.write_text, existing_text, encoding="utf-8")
                backup_created = True

            await asyncio.to_thread(target.write_text, content, encoding="utf-8")
            bak_msg = f" (Backup snapshot saved to '{path}.bak')" if backup_created else ""
            return f"Successfully wrote {len(content)} characters to '{path}'.{bak_msg}"
        except ValueError as e:
            return f"Error: {e}"
        except Exception as e:
            return f"Error writing file '{path}': {str(e)}"


class ReplaceFileContentTool(BaseTool):
    """Tool to edit an existing file by replacing a unique target text block with replacement content."""

    name: str = "replace_file_content"
    description: str = (
        "Edit an existing file by replacing a contiguous block of text with replacement content. "
        "Automatically creates a .bak backup snapshot before making edits."
    )
    parameters: dict[str, Any] = {
        "type": "object",
        "properties": {
            "path": {
                "type": "string",
                "description": "Relative path to file inside the workspace directory.",
            },
            "target_content": {
                "type": "string",
                "description": "The exact existing text block to be replaced (must match whitespace/indentation exactly).",
            },
            "replacement_content": {
                "type": "string",
                "description": "The new replacement text to substitute for target_content.",
            },
            "allow_multiple": {
                "type": "boolean",
                "description": "If true, replaces all occurrences of target_content. If false (default), requires target_content to be unique.",
                "default": False,
            },
        },
        "required": ["path", "target_content", "replacement_content"],
    }

    async def execute(
        self,
        path: str,
        target_content: str,
        replacement_content: str,
        allow_multiple: bool = False,
        **kwargs: Any,
    ) -> str:
        if not path or not path.strip():
            return "Error: Empty path provided."
        if not target_content:
            return "Error: target_content cannot be empty."

        try:
            target = secure_path(path.strip())
            if not target.exists():
                return f"Error: File '{path}' does not exist. Use write_file to create new files."
            if not target.is_file():
                return f"Error: Path '{path}' is not a file."

            original_text = await asyncio.to_thread(target.read_text, encoding="utf-8")
            count = original_text.count(target_content)

            if count == 0:
                return (
                    f"Error: target_content not found in '{path}'. "
                    f"Verify exact indentation and line breaks, or call read_file to inspect the file."
                )

            if count > 1 and not allow_multiple:
                return (
                    f"Error: target_content matched {count} occurrences in '{path}'. "
                    f"Provide more surrounding lines of context to ensure a unique match, "
                    f"or set allow_multiple=True to replace all occurrences."
                )

            # Create .bak backup snapshot
            bak_path = target.with_name(f"{target.name}.bak")
            await asyncio.to_thread(bak_path.write_text, original_text, encoding="utf-8")

            if allow_multiple:
                new_text = original_text.replace(target_content, replacement_content)
                replaced_count = count
            else:
                new_text = original_text.replace(target_content, replacement_content, 1)
                replaced_count = 1

            await asyncio.to_thread(target.write_text, new_text, encoding="utf-8")
            return (
                f"Successfully replaced {replaced_count} occurrence(s) in '{path}'. "
                f"Backup snapshot saved to '{path}.bak'."
            )
        except ValueError as e:
            return f"Error: {e}"
        except Exception as e:
            return f"Error editing file '{path}': {str(e)}"


class RollbackFileTool(BaseTool):
    """Tool to restore a workspace file from its .bak backup snapshot."""

    name: str = "rollback_file"
    description: str = "Revert a workspace file to its previous .bak backup snapshot."
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

        try:
            target = secure_path(path.strip())
            bak_path = target.with_name(f"{target.name}.bak")

            if not bak_path.exists():
                return f"Error: No backup snapshot '{path}.bak' found to restore."

            backup_text = await asyncio.to_thread(bak_path.read_text, encoding="utf-8")
            await asyncio.to_thread(target.write_text, backup_text, encoding="utf-8")
            return f"Successfully rolled back '{path}' from backup snapshot '{path}.bak'."
        except ValueError as e:
            return f"Error: {e}"
        except Exception as e:
            return f"Error rolling back file '{path}': {str(e)}"



class ListDirTool(BaseTool):
    """Tool to list files and directories within a workspace path."""

    name: str = "list_dir"
    description: str = "List files and directories inside the workspace directory."
    is_concurrent_safe: bool = True
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
