"""Tests for vector memory storage, fastembed CPU embeddings, and recall."""

import asyncio
import tempfile
import unittest
from unittest.mock import AsyncMock, patch

from langchain_core.messages import AIMessage, HumanMessage

from core.db import get_db, init_tables
from core.embeddings import embed_one
from core.memory import (
    format_memories,
    init_memory_table,
    recall,
    serialize_vector,
    store_memory,
)


class TestVectorMemory(unittest.IsolatedAsyncioTestCase):
    """Test sqlite-vec vector memory operations."""

    async def asyncSetUp(self):
        self.temp_db = tempfile.NamedTemporaryFile(suffix=".db", delete=False)
        self.temp_db_path = self.temp_db.name
        self.temp_db.close()

        # Initialize tables including vector memory
        db = await get_db(self.temp_db_path)
        await init_tables(db)
        await db.close()

    async def asyncTearDown(self):
        import os
        if os.path.exists(self.temp_db_path):
            os.remove(self.temp_db_path)

    def test_vector_serialization(self):
        """Test vector serialization to float struct bytes."""
        vec = [0.1, 0.2, 0.3]
        serialized = serialize_vector(vec)
        self.assertIsInstance(serialized, bytes)
        self.assertEqual(len(serialized), 3 * 4)  # 3 floats, 4 bytes each

    async def test_store_and_recall_memory(self):
        """Test storing memories and semantic recall by relevance."""
        db = await get_db(self.temp_db_path)
        try:
            # Store several distinct facts
            await store_memory(db, "User favorite color is deep sapphire blue.", session_id="s1")
            await store_memory(db, "The project is written in Python 3.12 and LangGraph.", session_id="s1")
            await store_memory(db, "User loves eating Italian woodfired pizza.", session_id="s1")

            # Recall query about preferences
            recalled_color = await recall(db, query="What is the user's preferred color?", top_k=1)
            self.assertEqual(len(recalled_color), 1)
            self.assertIn("sapphire blue", recalled_color[0]["content"])

            # Recall query about coding
            recalled_tech = await recall(db, query="Which programming language does the project use?", top_k=1)
            self.assertEqual(len(recalled_tech), 1)
            self.assertIn("Python 3.12", recalled_tech[0]["content"])

            # Verify format_memories
            formatted = format_memories(recalled_color)
            self.assertIn("sapphire blue", formatted)
            self.assertTrue(formatted.startswith("- "))
        finally:
            await db.close()

    async def test_empty_recall_handling(self):
        """Test recall with empty memories or empty query."""
        db = await get_db(self.temp_db_path)
        try:
            results = await recall(db, query="", top_k=5)
            self.assertEqual(results, [])

            formatted = format_memories([])
            self.assertEqual(formatted, "No relevant memories found.")
        finally:
            await db.close()


if __name__ == "__main__":
    unittest.main()
