"""Scheduler tool for Maidere allowing the LLM to manage cron/interval tasks."""

import asyncio
from typing import Any

from core.scheduler import (
    add_scheduled_job,
    delete_scheduled_job,
    list_scheduled_jobs,
)
from tools.base import BaseTool


class SchedulerTool(BaseTool):
    """Tool to create, list, and delete scheduled tasks."""

    name: str = "scheduler"
    description: str = (
        "Create, list, or delete persistent background scheduled tasks (cron, interval, or date). "
        "Example cron: '0 17 * * 5' for every Friday at 5pm."
    )
    parameters: dict[str, Any] = {
        "type": "object",
        "properties": {
            "action": {
                "type": "string",
                "enum": ["create", "list", "delete"],
                "description": "Action to perform: 'create', 'list', or 'delete'.",
            },
            "task_name": {
                "type": "string",
                "description": "Unique identifier for the scheduled task (e.g. 'weekly_goal_review').",
            },
            "schedule_type": {
                "type": "string",
                "enum": ["cron", "interval", "date"],
                "description": "Type of schedule: 'cron' (crontab expression), 'interval' (e.g. 'minutes=30'), or 'date' (ISO string).",
                "default": "cron",
            },
            "schedule_value": {
                "type": "string",
                "description": "The schedule value (e.g. '0 17 * * 5' for Friday at 5pm, or 'hours=24').",
            },
            "prompt": {
                "type": "string",
                "description": "The prompt or goal for the agent to execute when triggered.",
            },
        },
        "required": ["action"],
    }

    async def execute(
        self,
        action: str,
        task_name: str | None = None,
        schedule_type: str = "cron",
        schedule_value: str | None = None,
        prompt: str = "",
        **kwargs: Any,
    ) -> str:
        act = action.strip().lower()

        if act == "create":
            if not task_name:
                return "Error: 'task_name' is required to create a scheduled task."
            if not schedule_value:
                return "Error: 'schedule_value' is required (e.g. '0 17 * * 5')."
            if not prompt:
                return "Error: 'prompt' is required to define what the task will do."

            try:
                job_info = await asyncio.to_thread(
                    add_scheduled_job,
                    task_name=task_name,
                    schedule_type=schedule_type,
                    schedule_value=schedule_value,
                    prompt=prompt,
                )
                return (
                    f"Successfully scheduled task '{job_info['name']}'. "
                    f"Type: {job_info['schedule_type']}, Schedule: '{job_info['schedule_value']}', "
                    f"Next run: {job_info['next_run_time']} UTC."
                )
            except Exception as e:
                return f"Error creating scheduled task: {str(e)}"

        elif act == "list":
            try:
                jobs = await asyncio.to_thread(list_scheduled_jobs)
                if not jobs:
                    return "No scheduled tasks currently active."
                formatted = [f"- {j['name']} (ID: {j['id']}, Next Run: {j['next_run_time']} UTC)" for j in jobs]
                return "Active scheduled tasks:\n" + "\n".join(formatted)
            except Exception as e:
                return f"Error listing scheduled tasks: {str(e)}"

        elif act == "delete":
            if not task_name:
                return "Error: 'task_name' is required to delete a task."
            try:
                success = await asyncio.to_thread(delete_scheduled_job, task_name)
                if success:
                    return f"Successfully deleted scheduled task '{task_name}'."
                return f"Error: Task '{task_name}' not found."
            except Exception as e:
                return f"Error deleting scheduled task: {str(e)}"

        else:
            return f"Error: Unknown action '{action}'. Use 'create', 'list', or 'delete'."
