---
title: "[Performance]: Automated SQLite WAL Checkpointing & Database Compaction"
labels: ["performance", "database", "maintenance"]
---

### Problem Statement
Maidere operates SQLite in WAL (`Write-Ahead Logging`) mode with vector extensions (`sqlite-vec`). Over prolonged usage with multi-turn conversations, deliberation loops, and vector memory insertions, the `-wal` file can accumulate gigabytes of write buffers without releasing space back to disk.

Without proactive checkpointing, query latency increases and disk usage remains elevated until the application process terminates cleanly.

### Proposed Solution
1. **Periodic Background WAL Checkpointing**:
   In `core/db.py`, introduce a safe asynchronous checkpoint function:
   ```python
   async def checkpoint_wal(db_path: str = "db/maidere.db", mode: str = "TRUNCATE") -> dict[str, int]:
       """Checkpoint write-ahead log and truncate WAL file back to 0 bytes."""
       async with get_db_context(db_path) as db:
           cursor = await db.execute(f"PRAGMA wal_checkpoint({mode});")
           row = await cursor.fetchone()
           return {"busy": row[0], "log": row[1], "checkpointed": row[2]}
   ```

2. **Scheduled Maintenance Hook**:
   - Register an automatic daily or hourly job in `tools/scheduler.py` or FastAPI lifespan.
   - Run `PRAGMA wal_checkpoint(TRUNCATE);` and `PRAGMA optimize;` during application shutdown to guarantee zero dangling lock files.

### Acceptance Criteria
- [ ] SQLite `-wal` and `-shm` files are compacted during routine maintenance and on shutdown.
- [ ] Concurrency remains unaffected (`PASSIVE` / `TRUNCATE` handled safely without blocking active queries).
- [ ] Automated tests in `tests/test_db.py` verify checkpoint execution and WAL truncation.
