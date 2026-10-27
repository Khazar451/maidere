"""Skill management system for Maidere.

Discovers, parses, indexes, and activates specialized skill packages from:
1. maidere/skills/*/SKILL.md
2. workspace/.agent/skills/*/SKILL.md (and .agent/skills/*/SKILL.md)
"""

import re
from pathlib import Path
from typing import Any

import structlog
import yaml
from pydantic import BaseModel

logger = structlog.get_logger()


def get_skill_directories() -> list[Path]:
    """Bundled skills, plus any the user added under the workspace or runtime root."""
    from maidere.paths import runtime_root, skills_dir, workspace_dir

    dirs: list[Path] = []
    bundled = skills_dir()
    if bundled.is_dir():
        dirs.append(bundled)

    workspace_skills = workspace_dir() / ".agent" / "skills"
    if workspace_skills.is_dir():
        dirs.append(workspace_skills)

    root_agent_skills = runtime_root() / ".agent" / "skills"
    if root_agent_skills.is_dir() and root_agent_skills not in dirs:
        dirs.append(root_agent_skills)

    return dirs


class Skill(BaseModel):
    """Specialized skill package with YAML frontmatter metadata and instructions."""

    name: str
    description: str
    triggers: list[str] = []
    instructions: str
    path: Path


def parse_skill_file(skill_file: Path) -> Skill | None:
    """Parse YAML frontmatter and markdown body from a SKILL.md file."""
    try:
        content = skill_file.read_text(encoding="utf-8")
        # Match YAML frontmatter between --- blocks
        match = re.match(r"^---\s*\n(.*?)\n---\s*\n(.*)$", content, re.DOTALL)
        if not match:
            return None

        frontmatter_raw, instructions = match.groups()
        meta: dict[str, Any] = {}
        try:
            parsed_yaml = yaml.safe_load(frontmatter_raw)
            if isinstance(parsed_yaml, dict):
                meta = parsed_yaml
        except Exception:
            # Fallback line-by-line parser
            for line in frontmatter_raw.strip().split("\n"):
                stripped = line.strip()
                if stripped.startswith("- ") and isinstance(meta.get("triggers"), list):
                    meta["triggers"].append(stripped[2:].strip())
                elif ":" in line:
                    key, val = line.split(":", 1)
                    key = key.strip()
                    val = val.strip()
                    if key == "triggers":
                        meta[key] = [t.strip() for t in val.split(",") if t.strip()] if val else []
                    else:
                        meta[key] = val

        name = str(meta.get("name", skill_file.parent.name)).strip()
        description = str(meta.get("description", "")).strip()
        triggers_raw = meta.get("triggers", [])
        if isinstance(triggers_raw, list):
            triggers = [str(t).strip().lower() for t in triggers_raw if t]
        elif isinstance(triggers_raw, str):
            triggers = [t.strip().lower() for t in triggers_raw.split(",") if t.strip()]
        else:
            triggers = []

        return Skill(
            name=name,
            description=description,
            triggers=triggers,
            instructions=instructions.strip(),
            path=skill_file.parent,
        )
    except Exception as e:
        logger.warning("skill_parse_failed", path=str(skill_file), error=str(e))
        return None


def load_all_skills() -> dict[str, Skill]:
    """Scan and index all SKILL.md packages across configured directories."""
    skills: dict[str, Skill] = {}
    for base_dir in get_skill_directories():
        if not base_dir.exists():
            continue
        for skill_md in base_dir.glob("*/SKILL.md"):
            skill = parse_skill_file(skill_md)
            if skill:
                skills[skill.name] = skill
    return skills


def get_skills_overview() -> str:
    """Generate a lightweight summary of available skills for the LLM."""
    skills = load_all_skills()
    if not skills:
        return "No specialized skills currently installed."

    return "\n".join(f"- `{name}`: {skill.description}" for name, skill in skills.items())


def _phrase_in_query(candidate: str, query_lower: str, words: set[str]) -> bool:
    """Check whether a single-word, hyphenated, or multi-word trigger appears in the query."""
    cand = candidate.strip().lower()
    if not cand:
        return False
    if re.fullmatch(r"\w+", cand):
        return cand in words
    if re.search(rf"(?<!\w){re.escape(cand)}(?!\w)", query_lower):
        return True
    spaced = cand.replace("-", " ")
    if spaced != cand:
        return bool(re.search(rf"(?<!\w){re.escape(spaced)}(?!\w)", query_lower))
    return False


def match_skills(query: str) -> list[Skill]:
    """Match trigger keywords from skills against words and phrases in the user query."""
    if not query or not query.strip():
        return []

    query_lower = query.lower()
    words = set(re.findall(r"\w+", query_lower))
    all_skills = load_all_skills()

    return [
        skill
        for skill in all_skills.values()
        if _phrase_in_query(skill.name, query_lower, words)
        or any(_phrase_in_query(trigger, query_lower, words) for trigger in skill.triggers)
    ]
