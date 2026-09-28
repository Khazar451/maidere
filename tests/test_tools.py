"""Tests for sandboxed filesystem and shell tools."""

import asyncio
import os
import shutil
import tempfile
import unittest
from pathlib import Path

from tools.filesystem import (
    ListDirTool,
    ReadFileTool,
    ReplaceFileContentTool,
    RollbackFileTool,
    WriteFileTool,
    secure_path,
    WORKSPACE,
)
from tools.shell import ShellTool, ALLOWED_COMMANDS
from tools.registry import get_all_tools, get_tool, get_tools_ollama_schemas, execute_tool


class TestFilesystemTools(unittest.IsolatedAsyncioTestCase):
    """Test filesystem sandbox and operations."""

    async def asyncSetUp(self):
        # Ensure clean test workspace
        WORKSPACE.mkdir(parents=True, exist_ok=True)
        self.test_file = "test_sample.txt"
        self.test_content = "Hello, Maidere sandbox!"

    async def asyncTearDown(self):
        # Clean up test files in workspace
        target = WORKSPACE / self.test_file
        if target.exists():
            target.unlink()
        bak_target = WORKSPACE / f"{self.test_file}.bak"
        if bak_target.exists():
            bak_target.unlink()
        sub_dir = WORKSPACE / "sub_folder"
        if sub_dir.exists():
            shutil.rmtree(sub_dir, ignore_errors=True)

    def test_secure_path_valid(self):
        """Test valid relative path resolves safely inside workspace."""
        path = secure_path("sub/file.txt")
        self.assertTrue(path.is_relative_to(WORKSPACE))
        self.assertEqual(path, (WORKSPACE / "sub/file.txt").resolve())

    def test_secure_path_traversal_prevention(self):
        """Test directory traversal attempts raise ValueError."""
        traversal_attempts = [
            "../../../../etc/passwd",
            "../secret.txt",
            "/etc/shadow",
            "foo/../../../../root/.ssh/id_rsa",
        ]
        for bad_path in traversal_attempts:
            with self.assertRaises(ValueError, msg=f"Failed to block: {bad_path}"):
                secure_path(bad_path)

    async def test_write_and_read_file(self):
        """Test write_file and read_file tools."""
        write_tool = WriteFileTool()
        read_tool = ReadFileTool()

        write_res = await write_tool.execute(path=self.test_file, content=self.test_content)
        self.assertIn("Successfully wrote", write_res)

        read_res = await read_tool.execute(path=self.test_file)
        self.assertEqual(read_res, self.test_content)

    async def test_write_file_creates_backup_on_overwrite(self):
        """Test write_file creates .bak snapshot when overwriting existing file."""
        write_tool = WriteFileTool()
        read_tool = ReadFileTool()

        await write_tool.execute(path=self.test_file, content="Original version")
        write_res = await write_tool.execute(path=self.test_file, content="Overwritten version")

        self.assertIn("Backup snapshot saved", write_res)
        read_res = await read_tool.execute(path=self.test_file)
        self.assertEqual(read_res, "Overwritten version")

        bak_res = await read_tool.execute(path=f"{self.test_file}.bak")
        self.assertEqual(bak_res, "Original version")

    async def test_replace_file_content_success(self):
        """Test targeted block replacement in existing file."""
        write_tool = WriteFileTool()
        read_tool = ReadFileTool()
        edit_tool = ReplaceFileContentTool()

        initial_code = (
            "def calculate(a, b):\n"
            "    # TODO: fix formula\n"
            "    return a - b\n"
        )
        await write_tool.execute(path=self.test_file, content=initial_code)

        target = "    # TODO: fix formula\n    return a - b"
        replacement = "    return a + b"

        edit_res = await edit_tool.execute(
            path=self.test_file,
            target_content=target,
            replacement_content=replacement,
        )
        self.assertIn("Successfully replaced 1 occurrence", edit_res)
        self.assertIn(".bak", edit_res)

        updated_text = await read_tool.execute(path=self.test_file)
        expected = "def calculate(a, b):\n    return a + b\n"
        self.assertEqual(updated_text, expected)

        # Verify backup holds initial code
        bak_text = await read_tool.execute(path=f"{self.test_file}.bak")
        self.assertEqual(bak_text, initial_code)

    async def test_replace_file_content_target_not_found(self):
        """Test replacement returns error when target_content is missing."""
        write_tool = WriteFileTool()
        edit_tool = ReplaceFileContentTool()

        await write_tool.execute(path=self.test_file, content="print('hello')")
        res = await edit_tool.execute(
            path=self.test_file,
            target_content="print('world')",
            replacement_content="print('bye')",
        )
        self.assertIn("Error: target_content not found", res)

    async def test_replace_file_content_multiple_matches(self):
        """Test multiple match behavior with allow_multiple flag."""
        write_tool = WriteFileTool()
        read_tool = ReadFileTool()
        edit_tool = ReplaceFileContentTool()

        content = "item = 1\nitem = 1\nitem = 1\n"
        await write_tool.execute(path=self.test_file, content=content)

        # Default allow_multiple=False should block ambiguous edits
        res_blocked = await edit_tool.execute(
            path=self.test_file,
            target_content="item = 1",
            replacement_content="item = 2",
            allow_multiple=False,
        )
        self.assertIn("Error: target_content matched 3 occurrences", res_blocked)

        # allow_multiple=True should substitute all
        res_allowed = await edit_tool.execute(
            path=self.test_file,
            target_content="item = 1",
            replacement_content="item = 2",
            allow_multiple=True,
        )
        self.assertIn("Successfully replaced 3 occurrence(s)", res_allowed)
        updated = await read_tool.execute(path=self.test_file)
        self.assertEqual(updated, "item = 2\nitem = 2\nitem = 2\n")

    async def test_rollback_file_success_and_failure(self):
        """Test restoring from .bak snapshot and error when no backup exists."""
        write_tool = WriteFileTool()
        read_tool = ReadFileTool()
        edit_tool = ReplaceFileContentTool()
        rollback_tool = RollbackFileTool()

        original_code = "version = 1.0"
        await write_tool.execute(path=self.test_file, content=original_code)

        # Edit file, creating .bak
        await edit_tool.execute(
            path=self.test_file,
            target_content="version = 1.0",
            replacement_content="version = 2.0",
        )
        self.assertEqual(await read_tool.execute(path=self.test_file), "version = 2.0")

        # Rollback
        rollback_res = await rollback_tool.execute(path=self.test_file)
        self.assertIn("Successfully rolled back", rollback_res)
        self.assertEqual(await read_tool.execute(path=self.test_file), original_code)

        # Rollback on non-existent backup
        no_bak_file = "no_bak_file.txt"
        await write_tool.execute(path=no_bak_file, content="new")
        # remove bak if created
        bak_file = WORKSPACE / f"{no_bak_file}.bak"
        if bak_file.exists():
            bak_file.unlink()
        err_res = await rollback_tool.execute(path=no_bak_file)
        self.assertIn("Error: No backup snapshot", err_res)
        # clean up
        (WORKSPACE / no_bak_file).unlink()

    async def test_tool_registry_aliases(self):
        """Test edit_file and rollback_file alias resolution in registry."""
        edit_tool = get_tool("edit_file")
        self.assertIsInstance(edit_tool, ReplaceFileContentTool)

        rollback_tool = get_tool("restore_file_backup")
        self.assertIsInstance(rollback_tool, RollbackFileTool)

    async def test_read_nonexistent_file(self):
        """Test reading a file that does not exist returns clean error string."""
        read_tool = ReadFileTool()
        result = await read_tool.execute(path="nonexistent_xyz.txt")
        self.assertIn("Error:", result)
        self.assertIn("does not exist", result)

    async def test_list_dir(self):
        """Test list_dir tool."""
        write_tool = WriteFileTool()
        list_tool = ListDirTool()

        await write_tool.execute(path="sub_folder/item.txt", content="test")
        res = await list_tool.execute(path="sub_folder")
        self.assertIn("[FILE] item.txt", res)


class TestShellTool(unittest.IsolatedAsyncioTestCase):
    """Test shell tool security guardrails and execution."""

    async def asyncSetUp(self):
        self.shell_tool = ShellTool()

    async def test_allowed_command_execution(self):
        """Test executing allowed commands."""
        result = await self.shell_tool.execute("echo 'Maidere agent'")
        self.assertEqual(result.strip(), "Maidere agent")

    async def test_python_execution(self):
        """Test running python code inside workspace."""
        result = await self.shell_tool.execute("python3 -c \"print(21 * 2)\"")
        self.assertEqual(result.strip(), "42")

    async def test_disallowed_command_blocked(self):
        """Test blocked binary commands are rejected immediately."""
        blocked_commands = [
            "curl http://example.com",
            "wget http://example.com",
            "bash -c 'echo injected'",
            "sudo apt install curl",
            "chmod 777 file",
        ]
        for cmd in blocked_commands:
            res = await self.shell_tool.execute(cmd)
            self.assertIn("Error: Command binary", res)
            self.assertIn("not permitted", res)


    async def test_unclosed_quotes_shlex_handling(self):
        """Test safe exception handling when shlex receives unclosed quotes."""
        bad_command = 'echo "unclosed quote'
        result = await self.shell_tool.execute(bad_command)
        self.assertIn("Error parsing command syntax", result)

    async def test_empty_command(self):
        """Test empty command handling."""
        result = await self.shell_tool.execute("   ")
        self.assertIn("Error: Empty command provided.", result)

    async def test_git_allowed_subcommands(self):
        """Test permitted git inspection commands run or execute."""
        # git status inside workspace
        result = await self.shell_tool.execute("git status")
        self.assertNotIn("Error: Command binary", result)
        self.assertNotIn("blocked by security policy", result)

    async def test_git_blocked_subcommands(self):
        """Test mutating git commands are blocked immediately."""
        blocked_git = [
            "git push origin main",
            "git commit -m 'test'",
            "git checkout feature",
            "git reset --hard HEAD~1",
            "git clean -fd",
            "git rebase main",
            "git merge other",
        ]
        for cmd in blocked_git:
            res = await self.shell_tool.execute(cmd)
            self.assertTrue(
                "blocked by security policy" in res or "Destructive action blocked" in res,
                msg=f"Failed to block mutating git command: {cmd}, got: {res}",
            )

    async def test_git_flags_and_empty_subcommand(self):
        """Test directory traversal flags and missing subcommands are blocked."""
        res_bare = await self.shell_tool.execute("git")
        self.assertIn("requires a permitted inspection subcommand", res_bare)

        res_escape = await self.shell_tool.execute("git -C /tmp status")
        self.assertIn("prevent workspace escaping", res_escape)

        res_worktree = await self.shell_tool.execute("git --work-tree=/etc status")
        self.assertIn("prevent workspace escaping", res_worktree)


class TestToolRegistry(unittest.IsolatedAsyncioTestCase):
    """Test tool registry and Ollama schema generation."""

    def test_ollama_schemas_format(self):
        """Test that schemas follow OpenAI-standard format for Ollama."""
        schemas = get_tools_ollama_schemas()
        self.assertGreaterEqual(len(schemas), 4)

        for s in schemas:
            self.assertIn("type", s)
            self.assertEqual(s["type"], "function")
            self.assertIn("function", s)
            fn = s["function"]
            self.assertIn("name", fn)
            self.assertIn("description", fn)
            self.assertIn("parameters", fn)
            self.assertEqual(fn["parameters"]["type"], "object")

    async def test_execute_tool_dispatcher(self):
        """Test execute_tool dispatcher handles valid and invalid tools."""
        res = await execute_tool("read_file", {"path": "dummy_nonexistent.txt"})
        self.assertIn("Error:", res)

        unknown_res = await execute_tool("unknown_tool_xyz", {})
        self.assertIn("Error: Tool 'unknown_tool_xyz' is not recognized", unknown_res)


if __name__ == "__main__":
    unittest.main()
