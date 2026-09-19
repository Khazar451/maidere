"""Persistent task scheduler for Maidere using APScheduler and SQLite SQLAlchemy store.

Supports:
- Cron expressions (e.g. '0 17 * * 5' for Every Friday at 5pm)
- Interval schedules (e.g. 'hours=24', 'minutes=30')
- One-time date schedules
- Persistent storage across restarts in db/maidere.db
"""

import asyncio
from datetime import datetime
from typing import Any

from apscheduler.jobstores.sqlalchemy import SQLAlchemyJobStore
from apscheduler.schedulers.asyncio import AsyncIOScheduler
from apscheduler.triggers.cron import CronTrigger
from apscheduler.triggers.date import DateTrigger
from apscheduler.triggers.interval import IntervalTrigger
import structlog

from core.config import settings

logger = structlog.get_logger()

_scheduler: AsyncIOScheduler | None = None


async def execute_scheduled_task(task_name: str, prompt: str) -> None:
    """Callback function invoked when a scheduled task triggers."""
    await logger.ainfo("scheduled_task_triggered", task_name=task_name, prompt=prompt)
    try:
        from core.db import get_db
        from core.memory import store_memory

        db = await get_db(settings.db_path)
        try:
            await store_memory(
                db,
                content=f"[Scheduled Task: {task_name}] Prompt: {prompt} triggered at {datetime.now().isoformat()}",
                session_id="scheduled-jobs",
            )
        finally:
            await db.close()
    except Exception as e:
        await logger.aerror("scheduled_task_memory_failed", task_name=task_name, error=str(e))


def get_scheduler(db_path: str | None = None) -> AsyncIOScheduler:
    """Retrieve or initialize the global AsyncIOScheduler instance."""
    global _scheduler
    if _scheduler is None:
        path = db_path or settings.db_path
        jobstores = {
            "default": SQLAlchemyJobStore(url=f"sqlite:///{path}")
        }
        _scheduler = AsyncIOScheduler(jobstores=jobstores, timezone="UTC")
    return _scheduler


def start_scheduler(db_path: str | None = None) -> AsyncIOScheduler:
    """Start the APScheduler if not already running."""
    scheduler = get_scheduler(db_path)
    if not scheduler.running:
        scheduler.start()
        logger.info("scheduler_started", db_path=db_path or settings.db_path)
    return scheduler


def shutdown_scheduler() -> None:
    """Shutdown scheduler on application exit."""
    global _scheduler
    if _scheduler is not None and _scheduler.running:
        _scheduler.shutdown(wait=False)
        logger.info("scheduler_shutdown")
        _scheduler = None


def parse_trigger(schedule_type: str, schedule_value: str) -> Any:
    """Parse trigger string into an APScheduler trigger object."""
    stype = schedule_type.strip().lower()
    sval = schedule_value.strip()

    if stype == "cron":
        return CronTrigger.from_crontab(sval)
    elif stype == "interval":
        parts = {}
        for token in sval.split(","):
            if "=" in token:
                k, v = token.split("=", 1)
                parts[k.strip().lower()] = int(v.strip())
        if not parts:
            raise ValueError(f"Invalid interval format '{sval}'. Expected format: 'minutes=30' or 'hours=24'.")
        return IntervalTrigger(**parts)
    elif stype == "date":
        run_date = datetime.fromisoformat(sval)
        return DateTrigger(run_date=run_date)
    else:
        raise ValueError(f"Unsupported schedule type '{schedule_type}'. Choose 'cron', 'interval', or 'date'.")


def add_scheduled_job(
    task_name: str,
    schedule_type: str,
    schedule_value: str,
    prompt: str,
    db_path: str | None = None,
) -> dict[str, Any]:
    """Add a new persistent job to APScheduler."""
    scheduler = start_scheduler(db_path)
    trigger = parse_trigger(schedule_type, schedule_value)

    job = scheduler.add_job(
        execute_scheduled_task,
        trigger=trigger,
        args=[task_name, prompt],
        id=task_name,
        name=task_name,
        replace_existing=True,
    )

    next_run = job.next_run_time.isoformat() if job.next_run_time else "N/A"
    return {
        "id": job.id,
        "name": job.name,
        "next_run_time": next_run,
        "schedule_type": schedule_type,
        "schedule_value": schedule_value,
    }


def list_scheduled_jobs(db_path: str | None = None) -> list[dict[str, Any]]:
    """List all scheduled jobs in the persistent job store."""
    scheduler = start_scheduler(db_path)
    jobs = scheduler.get_jobs()
    results = []
    for job in jobs:
        next_run = job.next_run_time.isoformat() if job.next_run_time else "N/A"
        results.append(
            {
                "id": job.id,
                "name": job.name,
                "next_run_time": next_run,
            }
        )
    return results


def delete_scheduled_job(job_id: str, db_path: str | None = None) -> bool:
    """Remove a job from the scheduler by ID."""
    scheduler = start_scheduler(db_path)
    try:
        scheduler.remove_job(job_id)
        return True
    except Exception:
        return False
