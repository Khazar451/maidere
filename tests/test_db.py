"""Tests for database initialization, WAL mode, and concurrent write safety."""

import asyncio
import tempfile
import unittest

from maidere.core.db import get_db, init_tables
from maidere.tools.registry import log_audit_record


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

    async def test_checkpoint_truncates_wal(self):
        """TRUNCATE checkpoint folds the WAL into the database and empties the -wal file."""
        import os

        from maidere.core.db import checkpoint_wal, close_shared_connections, get_db_context

        wal_path = self.temp_db_path + "-wal"
        try:
            # The outer borrow keeps the pooled connection (and its WAL) alive.
            async with get_db_context(self.temp_db_path) as db:
                await init_tables(db)
                for i in range(50):
                    await db.execute(
                        "INSERT INTO audit_log (tool_name, tool_result) VALUES (?, ?)", ["shell", "x" * 2000 + str(i)]
                    )
                await db.commit()
                self.assertGreater(os.path.getsize(wal_path), 0)

                stats = await checkpoint_wal(self.temp_db_path, "TRUNCATE")

                self.assertEqual(stats["busy"], 0)
                self.assertEqual(os.path.getsize(wal_path), 0)
        finally:
            await close_shared_connections()

    async def test_checkpoint_rejects_unknown_mode(self):
        """Mode is interpolated into a PRAGMA, so only known modes are accepted."""
        from maidere.core.db import checkpoint_wal

        with self.assertRaises(ValueError):
            await checkpoint_wal(self.temp_db_path, "TRUNCATE); DROP TABLE users;--")

    async def test_passive_checkpoint_does_not_block_active_reader(self):
        """A PASSIVE checkpoint completes while another connection holds a read transaction."""
        from maidere.core.db import checkpoint_wal, close_shared_connections

        db = await get_db(self.temp_db_path)
        await init_tables(db)
        await db.execute("INSERT INTO audit_log (tool_name) VALUES ('shell')")
        await db.commit()
        await db.execute("BEGIN")
        await db.execute_fetchall("SELECT COUNT(*) FROM audit_log")
        try:
            stats = await asyncio.wait_for(checkpoint_wal(self.temp_db_path, "PASSIVE"), timeout=5)
            self.assertIn("checkpointed", stats)
        finally:
            await db.rollback()
            await db.close()
            await close_shared_connections()

    async def test_get_db_closes_connection_on_setup_failure(self):
        """If extension loading or PRAGMA setup fails, get_db closes the connection."""
        from unittest.mock import AsyncMock, patch

        fake_conn = AsyncMock()
        fake_conn.execute = AsyncMock(side_effect=[None, RuntimeError("pragma failed")])

        with (
            patch("maidere.core.db.aiosqlite.connect", AsyncMock(return_value=fake_conn)),
            self.assertRaises(RuntimeError),
        ):
            await get_db(self.temp_db_path)

        fake_conn.close.assert_awaited_once()


if __name__ == "__main__":
    unittest.main()
