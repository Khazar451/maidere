import asyncio
from datetime import datetime, timezone
import json
from pathlib import Path
import tempfile
import time
import unittest
from unittest.mock import AsyncMock, patch

from core.memory_tiers import (
    MAX_INDEX_ENTRIES,
    extract_session_facts_from_messages,
    format_auto_memory_context,
    is_fact_duplicate_in_topic,
    load_instruction_memory,
    reconcile_memory_index,
    rollback_instruction_memory,
    run_session_extraction,
    save_auto_memory_fact,
    update_instruction_memory,
)
from tools.memory_tool import ManageMemoryTool
from tools.registry import get_tool


class TestMemoryTiers(unittest.IsolatedAsyncioTestCase):
    """Test suite verifying 3-Tier memory architecture, 2-step invariant, and extraction."""

    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.workspace = Path(self.temp_dir.name)
        self.maidere_dir = self.workspace / ".maidere"
        self.maidere_dir.mkdir(parents=True, exist_ok=True)
        self.memory_dir = self.maidere_dir / "memory"
        self.memory_dir.mkdir(parents=True, exist_ok=True)

    def tearDown(self):
        self.temp_dir.cleanup()

    def test_tier1_rules_lifecycle_and_backup(self):
        """Verify Tier 1 RULES.md initialization, update with .bak backup, and rollback."""
        with patch("core.memory_tiers.settings.agent_workspace", self.temp_dir.name):
            # 1. Load default
            rules = load_instruction_memory()
            self.assertIn("Maidere Instruction Memory & Operational Rules", rules)
            rules_path = self.maidere_dir / "RULES.md"
            self.assertTrue(rules_path.exists())

            # 2. Update with backup
            new_rules = "# Custom Rules\n- Rule 1: Always write tests.\n"
            updated = update_instruction_memory(new_rules)
            self.assertTrue(updated)
            self.assertIn("Rule 1: Always write tests", load_instruction_memory())

            backup_path = self.maidere_dir / "RULES.md.bak"
            self.assertTrue(backup_path.exists())
            self.assertIn("Maidere Instruction Memory", backup_path.read_text(encoding="utf-8"))

            # 3. Rollback
            rolled = rollback_instruction_memory()
            self.assertTrue(rolled)
            self.assertIn("Maidere Instruction Memory", load_instruction_memory())

    def test_tier2_two_step_save_invariant(self):
        """Verify Step 1 writes the topic markdown file and Step 2 updates index.json."""
        with patch("core.memory_tiers.settings.agent_workspace", self.temp_dir.name):
            res = save_auto_memory_fact(
                topic="user_preferences",
                fact="User prefers dark mode UI and Geist Mono font.",
                category="user",
            )
            self.assertTrue(res.get("success"))

            # Step 1 Check: Topic file exists on disk
            topic_file = self.memory_dir / "user_preferences.md"
            self.assertTrue(topic_file.exists())
            content = topic_file.read_text(encoding="utf-8")
            self.assertIn("User prefers dark mode UI and Geist Mono font", content)

            # Step 2 Check: index.json contains pointer
            index_file = self.memory_dir / "index.json"
            self.assertTrue(index_file.exists())
            idx = json.loads(index_file.read_text(encoding="utf-8"))
            entries = idx.get("entries", [])
            self.assertEqual(len(entries), 1)
            self.assertEqual(entries[0]["topic"], "user_preferences")
            self.assertEqual(entries[0]["file"], "user_preferences.md")

    def test_tier2_startup_reconciliation(self):
        """Verify startup reconciliation heals orphaned topic files and prunes dangling pointers."""
        with patch("core.memory_tiers.settings.agent_workspace", self.temp_dir.name):
            # 1. Create an orphaned topic file on disk directly
            orphaned = self.memory_dir / "arch_decisions.md"
            orphaned.write_text("# Architecture Decisions\n- Used LangGraph for cyclic agent flow.\n", encoding="utf-8")

            # 2. Create index with a dangling entry for a non-existent file
            index_file = self.memory_dir / "index.json"
            index_file.write_text(
                json.dumps({
                    "version": "1.0",
                    "entries": [
                        {
                            "id": "ghost1",
                            "topic": "deleted_topic",
                            "file": "deleted_topic.md",
                            "summary": "Old deleted fact",
                        }
                    ]
                }),
                encoding="utf-8",
            )

            # 3. Run reconciliation
            healed = reconcile_memory_index()
            entries = healed.get("entries", [])

            # Dangling entry pruned
            self.assertFalse(any(e["file"] == "deleted_topic.md" for e in entries))

            # Orphaned file healed and indexed
            arch_entry = next((e for e in entries if e["file"] == "arch_decisions.md"), None)
            self.assertIsNotNone(arch_entry)
            self.assertEqual(arch_entry["topic"], "arch_decisions")

    def test_tier2_lru_cap_enforcement(self):
        """Verify index caps entries at MAX_INDEX_ENTRIES (40) via LRU eviction."""
        with patch("core.memory_tiers.settings.agent_workspace", self.temp_dir.name):
            # Insert 45 distinct facts across different topics
            for i in range(45):
                save_auto_memory_fact(
                    topic=f"topic_{i:02d}",
                    fact=f"Fact number {i}",
                )

            index_file = self.memory_dir / "index.json"
            idx = json.loads(index_file.read_text(encoding="utf-8"))
            entries = idx.get("entries", [])

            self.assertLessEqual(len(entries), MAX_INDEX_ENTRIES)
            self.assertEqual(len(entries), 40)
            # Oldest topics (topic_00, topic_01) should have been evicted
            indexed_topics = [e["topic"] for e in entries]
            self.assertNotIn("topic_00", indexed_topics)
            self.assertIn("topic_44", indexed_topics)

    async def test_tier3_session_extraction_and_deduplication(self):
        """Verify extraction parses user preferences and deduplicates redundant facts."""
        with patch("core.memory_tiers.settings.agent_workspace", self.temp_dir.name):
            # First turn message
            messages_turn1 = [
                {"role": "user", "content": "I prefer using Python 3.12 and uv for fast builds."},
                {"role": "assistant", "content": "Understood! I will use Python 3.12 and uv."},
            ]

            extracted1 = extract_session_facts_from_messages(messages_turn1)
            self.assertEqual(len(extracted1), 1)
            self.assertEqual(extracted1[0]["topic"], "user_preferences")
            self.assertIn("Python 3.12 and uv", extracted1[0]["fact"])

            # Run extraction pipeline
            promoted = await run_session_extraction("t1", messages_turn1)
            self.assertEqual(promoted, 1)

            # Check deduplication on repeated turn
            messages_turn2 = [
                {"role": "user", "content": "I prefer using Python 3.12 and uv for fast builds."},
            ]
            promoted2 = await run_session_extraction("t2", messages_turn2)
            self.assertEqual(promoted2, 0)  # Should not duplicate

    async def test_manage_memory_tool(self):
        """Verify ManageMemoryTool actions: save_fact, read_topic, list_topics, view_rules."""
        tool = ManageMemoryTool()
        self.assertIsNotNone(get_tool("manage_memory"))
        self.assertEqual(get_tool("memory"), get_tool("manage_memory"))

        with patch("core.memory_tiers.settings.agent_workspace", self.temp_dir.name):
            # 1. save_fact
            save_out = await tool.execute(
                action="save_fact",
                topic="stack",
                fact="FastAPI is our REST API framework.",
            )
            self.assertIn("[MEMORY SAVED]", save_out)

            # 2. read_topic
            read_out = await tool.execute(action="read_topic", topic="stack")
            self.assertIn("[MEMORY TOPIC: STACK]", read_out)
            self.assertIn("FastAPI is our REST API framework", read_out)

            # 3. list_topics
            list_out = await tool.execute(action="list_topics")
            self.assertIn("[INDEXED AUTO-MEMORY TOPICS]", list_out)
            self.assertIn("stack", list_out)

            # 4. view_rules
            rules_out = await tool.execute(action="view_rules")
            self.assertIn("[TIER 1 INSTRUCTION RULES (RULES.MD)]", rules_out)


if __name__ == "__main__":
    unittest.main()
