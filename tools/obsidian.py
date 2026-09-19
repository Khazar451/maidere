"""Obsidian Vault integration tool for Maidere.

Enables the agent to:
- Write & format structured Markdown notes with YAML frontmatter, tags, and callouts into the user's Obsidian Vault
- Read existing notes from the vault
- List notes and subfolder structures
- Full-text search across the vault
- Open notes directly in the desktop Obsidian application via obsidian:// URI scheme
"""

import asyncio
import datetime
import os
from pathlib import Path
import re
import subprocess
import urllib.parse
from typing import Any

import structlog

from core.config import get_obsidian_vault_info
from tools.base import BaseTool

logger = structlog.get_logger()


def resolve_vault_path() -> tuple[Path, str]:
    """Resolve and ensure the active Obsidian Vault directory exists."""
    vault_str, vault_name = get_obsidian_vault_info()
    vault_path = Path(vault_str).expanduser().resolve()
    vault_path.mkdir(parents=True, exist_ok=True)
    return vault_path, vault_name


def format_frontmatter(title: str, tags: list[str] | None = None, extra_meta: dict | None = None) -> str:
    """Generate Obsidian-standard YAML frontmatter."""
    now_iso = datetime.datetime.now().strftime("%Y-%m-%d %H:%M")
    tag_list = tags or ["maidere", "note"]
    formatted_tags = "\n".join(f"  - {t.strip().lstrip('#')}" for t in tag_list if t.strip())

    meta_lines = [
        "---",
        f'title: "{title}"',
        f"date: {now_iso}",
        "tags:",
        formatted_tags,
        "created_by: Maidere",
    ]
    if extra_meta:
        for k, v in extra_meta.items():
            meta_lines.append(f'{k}: "{v}"')
    meta_lines.append("---\n")
    return "\n".join(meta_lines)


class ObsidianTool(BaseTool):
    """Tool to read, write, organize, search, and open notes in the Obsidian Vault."""

    name: str = "obsidian"
    description: str = (
        "Interact directly with the local Obsidian Vault. "
        "Actions: 'write_note' (create or update a note), 'read_note' (read content), "
        "'list_notes' (list all vault files), 'search_notes' (search text in vault), "
        "'open_note' (launch & display the note inside desktop Obsidian app)."
    )
    parameters: dict[str, Any] = {
        "type": "object",
        "properties": {
            "action": {
                "type": "string",
                "enum": ["write_note", "read_note", "list_notes", "search_notes", "open_note"],
                "description": "The action to perform: 'write_note', 'read_note', 'list_notes', 'search_notes', or 'open_note'.",
            },
            "title": {
                "type": "string",
                "description": "The note title or relative path within the vault (e.g. 'Hugging Face' or 'Research/AI Models.md').",
            },
            "content": {
                "type": "string",
                "description": "The Markdown content to write. Used with 'write_note'.",
            },
            "folder": {
                "type": "string",
                "description": "Optional subfolder inside the vault (e.g. 'Research', 'Daily', 'Projects').",
            },
            "tags": {
                "type": "array",
                "items": {"type": "string"},
                "description": "Optional tags for the note frontmatter, e.g. ['research', 'ai', 'huggingface'].",
            },
            "query": {
                "type": "string",
                "description": "Search term to look for across the vault. Used with 'search_notes'.",
            },
        },
        "required": ["action"],
    }

    async def execute(
        self,
        action: str,
        title: str | None = None,
        content: str | None = None,
        folder: str | None = None,
        tags: list[str] | None = None,
        query: str | None = None,
        **kwargs: Any,
    ) -> str:
        """Dispatch Obsidian vault actions."""
        try:
            vault_path, vault_name = resolve_vault_path()

            if action == "write_note":
                return await self._write_note(vault_path, vault_name, title, content, folder, tags)
            elif action == "read_note":
                return await self._read_note(vault_path, title)
            elif action == "list_notes":
                return await self._list_notes(vault_path, folder)
            elif action == "search_notes":
                return await self._search_notes(vault_path, query or title)
            elif action == "open_note":
                return await self._open_note(vault_path, vault_name, title)
            else:
                return f"Error: Unknown Obsidian action '{action}'. Valid actions: write_note, read_note, list_notes, search_notes, open_note."
        except Exception as e:
            await logger.aerror("obsidian_action_failed", action=action, error=str(e))
            return f"Error executing Obsidian action '{action}': {str(e)}"

    async def _write_note(
        self,
        vault_path: Path,
        vault_name: str,
        title: str | None,
        content: str | None,
        folder: str | None,
        tags: list[str] | None,
    ) -> str:
        """Write note into the Obsidian vault with frontmatter."""
        if not title or not title.strip():
            return "Error: Note 'title' is required for write_note."
        if content is None:
            return "Error: Note 'content' is required for write_note."

        raw_title = title.strip()
        filename = raw_title if raw_title.endswith(".md") else f"{raw_title}.md"
        
        # Resolve target directory
        target_dir = vault_path
        if folder and folder.strip():
            target_dir = (vault_path / folder.strip()).resolve()
            if not target_dir.is_relative_to(vault_path):
                return "Error: Target folder escapes the Obsidian Vault boundary."
        target_dir.mkdir(parents=True, exist_ok=True)

        target_file = (target_dir / filename).resolve()
        if not target_file.is_relative_to(vault_path):
            return "Error: Target file escapes the Obsidian Vault boundary."

        # If content doesn't already have YAML frontmatter, add it
        final_content = content.strip()
        clean_title_display = raw_title[:-3] if raw_title.endswith(".md") else raw_title
        if not final_content.startswith("---"):
            fm = format_frontmatter(clean_title_display, tags=tags)
            final_content = f"{fm}\n{final_content}\n"

        await asyncio.to_thread(target_file.write_text, final_content, encoding="utf-8")

        rel_path = target_file.relative_to(vault_path)
        encoded_vault = urllib.parse.quote(vault_name)
        encoded_file = urllib.parse.quote(str(rel_path))
        uri = f"obsidian://open?vault={encoded_vault}&file={encoded_file}"

        await logger.ainfo(
            "obsidian_note_written",
            vault=vault_name,
            file=str(rel_path),
            chars=len(final_content),
        )

        return (
            f"Successfully saved note to Obsidian Vault '{vault_name}' at '{rel_path}' ({len(final_content)} chars).\n"
            f"Obsidian URI: {uri}"
        )

    async def _read_note(self, vault_path: Path, title: str | None) -> str:
        """Read note content from the vault."""
        if not title or not title.strip():
            return "Error: Note 'title' or path is required for read_note."

        raw_title = title.strip()
        filename = raw_title if raw_title.endswith(".md") else f"{raw_title}.md"
        
        # Try direct relative path
        target_file = (vault_path / filename).resolve()
        if not target_file.exists() or not target_file.is_file():
            # Search recursively in vault for matching filename
            found = list(vault_path.rglob(filename))
            if found:
                target_file = found[0]
            else:
                return f"Error: Note '{title}' not found in Obsidian Vault at '{vault_path}'."

        if not target_file.is_relative_to(vault_path):
            return "Error: File path is outside the Obsidian Vault."

        content = await asyncio.to_thread(target_file.read_text, encoding="utf-8")
        rel_path = target_file.relative_to(vault_path)
        return f"=== OBSIDIAN NOTE: {rel_path} ===\n{content}"

    async def _list_notes(self, vault_path: Path, folder: str | None) -> str:
        """List notes and directories in the vault."""
        search_root = vault_path
        if folder and folder.strip():
            search_root = (vault_path / folder.strip()).resolve()
            if not search_root.exists():
                return f"Error: Folder '{folder}' does not exist in Obsidian Vault."
            if not search_root.is_relative_to(vault_path):
                return "Error: Folder is outside the Obsidian Vault."

        def _scan() -> list[str]:
            items = []
            for root, dirs, files in os.walk(search_root):
                # Ignore hidden directories like .obsidian
                dirs[:] = [d for d in dirs if not d.startswith(".")]
                rel_root = Path(root).relative_to(vault_path)
                for f in sorted(files):
                    if f.endswith(".md") and not f.startswith("."):
                        p = (rel_root / f) if str(rel_root) != "." else Path(f)
                        items.append(f"📄 {p}")
            return items

        files = await asyncio.to_thread(_scan)
        if not files:
            return f"Obsidian Vault '{vault_path.name}' contains no markdown notes in '{folder or '.'}'."

        return f"Obsidian Vault '{vault_path.name}' Notes ({len(files)} total):\n" + "\n".join(files[:60])

    async def _search_notes(self, vault_path: Path, query: str | None) -> str:
        """Search note titles and contents in the vault."""
        if not query or not query.strip():
            return "Error: Search 'query' is required."

        clean_query = query.strip().lower()

        def _search() -> list[str]:
            matches = []
            for root, dirs, files in os.walk(vault_path):
                dirs[:] = [d for d in dirs if not d.startswith(".")]
                for f in sorted(files):
                    if f.endswith(".md") and not f.startswith("."):
                        file_path = Path(root) / f
                        rel_path = file_path.relative_to(vault_path)
                        try:
                            text = file_path.read_text(encoding="utf-8")
                            if clean_query in str(rel_path).lower() or clean_query in text.lower():
                                snippet = ""
                                for line in text.splitlines():
                                    if clean_query in line.lower():
                                        snippet = line.strip()[:100]
                                        break
                                matches.append(f"• **{rel_path}**: {snippet or 'Title match'}")
                        except Exception:
                            continue
            return matches

        results = await asyncio.to_thread(_search)
        if not results:
            return f"No notes matching '{query}' found in Obsidian Vault."

        return f"Found {len(results)} matching notes for '{query}':\n" + "\n".join(results[:15])

    async def _open_note(self, vault_path: Path, vault_name: str, title: str | None) -> str:
        """Launch and open the note inside the Obsidian desktop application."""
        raw_title = (title or "").strip()
        filename = (raw_title if raw_title.endswith(".md") else f"{raw_title}.md") if raw_title else ""
        
        rel_path = filename
        if filename:
            target_file = (vault_path / filename).resolve()
            if not target_file.exists():
                found = list(vault_path.rglob(filename))
                if found:
                    rel_path = str(found[0].relative_to(vault_path))

        encoded_vault = urllib.parse.quote(vault_name)
        encoded_file = urllib.parse.quote(rel_path) if rel_path else ""
        uri = f"obsidian://open?vault={encoded_vault}" + (f"&file={encoded_file}" if encoded_file else "")

        # Non-blocking desktop launch via xdg-open or flatpak
        def _launch() -> None:
            try:
                # Try xdg-open first
                subprocess.Popen(["xdg-open", uri], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            except Exception:
                try:
                    subprocess.Popen(
                        ["flatpak", "run", "md.obsidian.Obsidian", uri],
                        stdout=subprocess.DEVNULL,
                        stderr=subprocess.DEVNULL,
                    )
                except Exception as ex:
                    logger.warn("obsidian_open_exec_failed", error=str(ex))

        await asyncio.to_thread(_launch)
        await logger.ainfo("obsidian_opened", uri=uri, note=rel_path)

        return f"Triggered desktop Obsidian application to open note '{rel_path or vault_name}'. URI: {uri}"
