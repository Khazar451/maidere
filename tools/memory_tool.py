"""ManageMemoryTool for Maidere.

Enables the agent to explicitly manage persistent Tier 2 Auto-Memory topics
and view Tier 1 Instruction Memory (RULES.md).
"""

from typing import Any

import structlog

from core.memory_tiers import (
    list_auto_memory_topics,
    load_instruction_memory,
    read_auto_memory_topic,
    save_auto_memory_fact,
)
from tools.base import BaseTool

logger = structlog.get_logger()


class ManageMemoryTool(BaseTool):
    """Tool allowing the agent to explicitly persist, inspect, and organize structured memory."""

    name: str = "manage_memory"
    description: str = (
        "Manage persistent structured project and user knowledge. "
        "Allows saving key facts to Tier 2 auto-memory topic files, reading topic files, "
        "listing existing memory topics, and viewing Tier 1 instruction rules (RULES.md). "
        "Actions: "
        "'save_fact' (requires 'topic' and 'fact'), "
        "'read_topic' (requires 'topic'), "
        "'list_topics' (shows indexed topics), "
        "'view_rules' (reads RULES.md)."
    )
    parameters: dict[str, Any] = {
        "type": "object",
        "properties": {
            "action": {
                "type": "string",
                "enum": ["save_fact", "read_topic", "list_topics", "view_rules"],
                "description": "Action to perform: 'save_fact', 'read_topic', 'list_topics', or 'view_rules'.",
            },
            "topic": {
                "type": "string",
                "description": "Topic name (e.g. 'user_preferences', 'tech_stack', 'project_decisions'). Required for save_fact and read_topic.",
            },
            "fact": {
                "type": "string",
                "description": "Concrete fact or decision to persist. Required for save_fact.",
            },
            "category": {
                "type": "string",
                "enum": ["user", "project", "reference"],
                "description": "Optional category tag for save_fact. Defaults to 'project'.",
            },
        },
        "required": ["action"],
    }

    async def execute(
        self,
        action: str = "list_topics",
        topic: str = "",
        fact: str = "",
        category: str = "project",
        **kwargs: Any,
    ) -> str:
        """Execute memory management action."""
        clean_action = action.lower().strip() if action else "list_topics"

        if clean_action == "save_fact":
            if not topic.strip() or not fact.strip():
                return "Error: Both 'topic' and 'fact' are required for action 'save_fact'."
            res = save_auto_memory_fact(topic=topic, fact=fact, category=category)
            if res.get("success"):
                return f"[MEMORY SAVED] Successfully saved fact to topic '{res.get('topic')}': {fact}"
            return f"Error saving memory fact: {res.get('error', 'unknown error')}"

        elif clean_action == "read_topic":
            if not topic.strip():
                return "Error: 'topic' parameter is required for action 'read_topic'."
            content = read_auto_memory_topic(topic=topic)
            return f"[MEMORY TOPIC: {topic.upper()}]\n{content}"

        elif clean_action == "list_topics":
            topics = list_auto_memory_topics()
            if not topics:
                return "No persistent memory topics currently indexed."
            lines = ["[INDEXED AUTO-MEMORY TOPICS]"]
            for t in topics:
                lines.append(f"- **{t.get('topic')}** ({t.get('category')}): {t.get('summary')} (file: {t.get('file')})")
            return "\n".join(lines)

        elif clean_action == "view_rules":
            rules = load_instruction_memory()
            return f"[TIER 1 INSTRUCTION RULES (RULES.MD)]\n{rules}"

        else:
            return f"Error: Unknown action '{action}'. Valid actions: save_fact, read_topic, list_topics, view_rules."
