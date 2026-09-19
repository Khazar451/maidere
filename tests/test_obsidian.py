"""Tests for Obsidian Vault integration, tools, API endpoints, and skills."""

import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import AsyncMock, patch

from fastapi.testclient import TestClient

from api.app import app
from core.config import get_obsidian_vault_info, settings
from core.skills import load_all_skills, match_skills
from tools.obsidian import ObsidianTool, format_frontmatter


class TestObsidianTool(unittest.IsolatedAsyncioTestCase):
    """Test ObsidianTool read, write, list, search, and open actions."""

    async def asyncSetUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.vault_path = Path(self.temp_dir.name) / "Test Vault"
        self.vault_path.mkdir(parents=True, exist_ok=True)
        self.tool = ObsidianTool()

    async def asyncTearDown(self):
        self.temp_dir.cleanup()

    @patch("tools.obsidian.resolve_vault_path")
    async def test_write_and_read_note_with_frontmatter(self, mock_resolve):
        mock_resolve.return_value = (self.vault_path, "Test Vault")

        # 1. Write note
        content = "# Hugging Face Architecture\n\nComprehensive details on Transformers and Safetensors."
        res = await self.tool.execute(
            action="write_note",
            title="Hugging Face Architecture",
            content=content,
            tags=["ai", "research", "huggingface"],
        )
        self.assertIn("Successfully saved note to Obsidian Vault", res)
        self.assertIn("obsidian://open?vault=Test%20Vault&file=Hugging%20Face%20Architecture.md", res)

        target_file = self.vault_path / "Hugging Face Architecture.md"
        self.assertTrue(target_file.exists())
        file_text = target_file.read_text(encoding="utf-8")
        self.assertIn("title: \"Hugging Face Architecture\"", file_text)
        self.assertIn("tags:", file_text)
        self.assertIn("  - ai", file_text)
        self.assertIn("created_by: Maidere", file_text)

        # 2. Read note
        read_res = await self.tool.execute(
            action="read_note",
            title="Hugging Face Architecture",
        )
        self.assertIn("=== OBSIDIAN NOTE: Hugging Face Architecture.md ===", read_res)
        self.assertIn("Comprehensive details on Transformers", read_res)

    @patch("tools.obsidian.resolve_vault_path")
    async def test_write_note_in_subfolder(self, mock_resolve):
        mock_resolve.return_value = (self.vault_path, "Test Vault")

        res = await self.tool.execute(
            action="write_note",
            title="DeepSeek V3",
            content="> [!NOTE] Model Specs\nMoE architecture with 671B total params.",
            folder="Research/AI",
            tags=["llm", "moe"],
        )
        self.assertIn("Successfully saved note", res)

        sub_file = self.vault_path / "Research" / "AI" / "DeepSeek V3.md"
        self.assertTrue(sub_file.exists())
        text = sub_file.read_text(encoding="utf-8")
        self.assertIn("> [!NOTE] Model Specs", text)

    @patch("tools.obsidian.resolve_vault_path")
    async def test_list_and_search_notes(self, mock_resolve):
        mock_resolve.return_value = (self.vault_path, "Test Vault")

        # Create two test notes
        (self.vault_path / "Note A.md").write_text("# Note A\nFocus on PyTorch optimization.", encoding="utf-8")
        sub_dir = self.vault_path / "Projects"
        sub_dir.mkdir(parents=True, exist_ok=True)
        (sub_dir / "Note B.md").write_text("# Note B\nFocus on CUDA kernels.", encoding="utf-8")

        # 1. List notes
        list_res = await self.tool.execute(action="list_notes")
        self.assertIn("Note A.md", list_res)
        self.assertIn("Projects/Note B.md", list_res)

        # 2. Search notes
        search_res = await self.tool.execute(action="search_notes", query="PyTorch")
        self.assertIn("Note A.md", search_res)
        self.assertIn("PyTorch optimization", search_res)

    @patch("tools.obsidian.resolve_vault_path")
    @patch("subprocess.Popen")
    async def test_open_note_action(self, mock_popen, mock_resolve):
        mock_resolve.return_value = (self.vault_path, "Test Vault")
        (self.vault_path / "Maidere.md").write_text("# Maidere", encoding="utf-8")

        open_res = await self.tool.execute(action="open_note", title="Maidere")
        self.assertIn("Triggered desktop Obsidian application", open_res)
        self.assertIn("obsidian://open?vault=Test%20Vault&file=Maidere.md", open_res)
        mock_popen.assert_called_once()


class TestObsidianSkillsAndAPI(unittest.TestCase):
    """Test Obsidian skill discovery and REST endpoints."""

    def test_obsidian_skill_registered_and_matched(self):
        skills = load_all_skills()
        self.assertIn("obsidian", skills)
        obs_skill = skills["obsidian"]
        self.assertIn("vault", obs_skill.triggers)
        self.assertIn("Obsidian Callouts", obs_skill.instructions)

        matched = match_skills("Please write a new obsidian note for this research.")
        matched_names = [s.name for s in matched]
        self.assertIn("obsidian", matched_names)

    def test_obsidian_api_endpoints(self):
        with TestClient(app) as client:
            res = client.get("/obsidian/vault")
            self.assertEqual(res.status_code, 200)
            data = res.json()
            self.assertIn("vault_path", data)
            self.assertIn("vault_name", data)

            with patch("tools.obsidian.ObsidianTool.execute", new_callable=AsyncMock) as mock_exec:
                mock_exec.return_value = "Opened note in Obsidian"
                open_res = client.post("/obsidian/open", json={"file": "Welcome.md"})
                self.assertEqual(open_res.status_code, 200)
                self.assertEqual(open_res.json()["status"], "ok")


if __name__ == "__main__":
    unittest.main()
