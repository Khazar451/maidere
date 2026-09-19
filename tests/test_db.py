"""Tests for database initialization, WAL mode, and concurrent write safety."""

import asyncio
import tempfile
import unittest

from core.db import get_db, init_tables
from tools.registry import log_audit_record


class TestDatabaseAndConcurrency(unittest.IsolatedAsyncioTestCase):
    """Test database setup, pragmas, and concurrent writes."""

    async def asyncSetUp(self):
        self.temp_db = tempfile.NamedTemporaryFile(suffix=".db", delete=False)
        self.temp_db_path = self.temp_db.name
        self.temp_db.close()

    async def asyncTearDown(self):
        import os
        if os.path.exists(self.temp_db_path):
            os.remove(self.temp_db_path)

    async def test_wal_mode_and_pragmas(self):
        """Verify journal_mode is WAL and busy_timeout is set."""
        db = await get_db(self.temp_db_path)
        try:
            cursor = await db.execute("PRAGMA journal_mode;")
            row = await cursor.fetchone()
            self.assertEqual(row[0].lower(), "wal")

            cursor = await db.execute("PRAGMA busy_timeout;")
            row = await cursor.fetchone()
            self.assertEqual(row[0], 5000)
        finally:
            await db.close()

    async def test_concurrent_write_stress(self):
        """Test 10 concurrent async workers writing simultaneously in WAL mode."""
        db = await get_db(self.temp_db_path)
        await init_tables(db)
        await db.close()

        async def worker_write(worker_id: int):
            await log_audit_record(
                thread_id=f"thread-{worker_id}",
                tool_name="shell",
                tool_args={"command": f"echo worker_{worker_id}"},
                tool_result=f"worker_{worker_id}",
                duration_ms=10,
                success=True,
                db_path=self.temp_db_path,
            )

        # Launch 10 concurrent writes
        tasks = [worker_write(i) for i in range(10)]
        await asyncio.gather(*tasks)

        # Verify all 10 records were written successfully
        db = await get_db(self.temp_db_path)
        try:
            cursor = await db.execute("SELECT COUNT(*) FROM audit_log;")
            row = await cursor.fetchone()
            self.assertEqual(row[0], 10)
        finally:
            await db.close()


if __name__ == "__main__":
    unittest.main()
