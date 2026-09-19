"""Tools for discovering and loading specialized skills dynamically."""

from typing import Any

from core.skills import get_skills_overview, load_all_skills
from tools.base import BaseTool


async def list_available_skills() -> str:
    """List all installed skills and their capabilities."""
    return f"Available Skills:\n{get_skills_overview()}"


async def load_skill(skill_name: str) -> str:
    """Load the full instructions and guidelines of a specialized skill."""
    if not skill_name or not skill_name.strip():
        return "Error: No skill_name provided."

    skills = load_all_skills()
    target_name = skill_name.strip()
    skill = skills.get(target_name)

    if not skill:
        # Check case-insensitive match
        for s in skills.values():
            if s.name.lower() == target_name.lower():
                skill = s
                break

    if not skill:
        available = ", ".join(f"'{k}'" for k in skills.keys())
        return f"Error: Skill '{skill_name}' not found. Available skills: {available or 'None'}"

    return (
        f"=== LOADED SKILL: {skill.name} ===\n"
        f"{skill.instructions}\n"
        f"================================"
    )


class ListSkillsTool(BaseTool):
    """Tool to list all available specialized skills."""

    name: str = "list_available_skills"
    description: str = "List all available specialized skills and domain packages."
    parameters: dict[str, Any] = {
        "type": "object",
        "properties": {},
    }

    async def execute(self, **kwargs: Any) -> str:
        return await list_available_skills()


class LoadSkillTool(BaseTool):
    """Tool to load a specific skill into working context."""

    name: str = "load_skill"
    description: str = (
        "Load step-by-step instructions and constraints for a specialized skill package."
    )
    parameters: dict[str, Any] = {
        "type": "object",
        "properties": {
            "skill_name": {
                "type": "string",
                "description": "The exact name of the skill (e.g., 'git-workflow')",
            }
        },
        "required": ["skill_name"],
    }

    async def execute(self, skill_name: str, **kwargs: Any) -> str:
        return await load_skill(skill_name=skill_name)
