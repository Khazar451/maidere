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
    WriteFileTool,
    secure_path,
    WORKSPACE,
)
from tools.shell import ShellTool, ALLOWED_COMMANDS
from tools.registry import get_all_tools, get_tools_ollama_schemas, execute_tool


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
