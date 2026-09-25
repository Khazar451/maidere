"""Three-Tier Memory Architecture for Maidere.

Implements Keli Wen's harness memory patterns:
- Tier 1: Instruction Memory (workspace/.maidere/RULES.md) with .bak versioning.
- Tier 2: Auto-Memory (workspace/.maidere/memory/) topic files + capped index (max 40)
          enforcing the Two-Step Save Invariant (write topic file -> update index)
          and startup reconciliation.
- Tier 3: Session Extraction pipeline with similarity deduplication replacing raw chat dumping.
"""

from datetime import datetime, timezone
import json
from pathlib import Path
import re
import shutil
import time
from typing import Any
import uuid

import structlog

from core.config import settings

logger = structlog.get_logger()

MAX_INDEX_ENTRIES = 40

DEFAULT_RULES = """# Maidere Instruction Memory & Operational Rules

## 1. Project Standards
- Local Hardware: Tailored for consumer workstation execution (RTX 5060 8GB VRAM, 16GB RAM).
- Zero Cloud Leakage: Respect local privacy boundaries; never transmit secrets, API keys, or raw tokens.
- Precision: Provide concrete data points, verified metrics, code blocks, and source citations.
- Filesystem: Confine autonomous file operations strictly to the workspace/ boundary.
- Non-Destructive: Never delete files recursively or drop database tables without explicit user approval.
"""


def get_maidere_dir() -> Path:
    """Return workspace/.maidere path, ensuring it exists."""
    base = Path(settings.agent_workspace).resolve()
    maidere_dir = base / ".maidere"
    maidere_dir.mkdir(parents=True, exist_ok=True)
    return maidere_dir


# ==============================================================================
# Tier 1: Instruction Memory (RULES.md)
# ==============================================================================

def get_rules_path() -> Path:
    """Return path to workspace/.maidere/RULES.md."""
    return get_maidere_dir() / "RULES.md"


def load_instruction_memory() -> str:
    """Load curated instruction rules from RULES.md. Initializes with default if missing."""
    rules_file = get_rules_path()
    if not rules_file.exists():
        try:
            rules_file.write_text(DEFAULT_RULES, encoding="utf-8")
        except Exception as e:
            logger.warn("failed_initializing_default_rules", error=str(e))
            return DEFAULT_RULES

    try:
        return rules_file.read_text(encoding="utf-8").strip()
    except Exception as e:
        logger.warn("failed_reading_rules_file", error=str(e))
        return DEFAULT_RULES


def update_instruction_memory(new_rules: str) -> bool:
    """Update RULES.md with automatic .bak backup for rollback protection."""
    rules_file = get_rules_path()
    backup_file = rules_file.with_suffix(".md.bak")

    try:
        if rules_file.exists():
            shutil.copy2(rules_file, backup_file)

        rules_file.write_text(new_rules.strip() + "\n", encoding="utf-8")
        logger.info("instruction_memory_updated", backup_created=backup_file.exists())
        return True
    except Exception as e:
        logger.error("failed_updating_rules", error=str(e))
        # Attempt restore if backup exists and target was corrupted
        if backup_file.exists() and not rules_file.exists():
            shutil.copy2(backup_file, rules_file)
        return False


def rollback_instruction_memory() -> bool:
    """Roll back RULES.md to RULES.md.bak if backup exists."""
    rules_file = get_rules_path()
    backup_file = rules_file.with_suffix(".md.bak")

    if not backup_file.exists():
        return False

    try:
        shutil.copy2(backup_file, rules_file)
        logger.info("instruction_memory_rolled_back")
        return True
    except Exception as e:
        logger.error("failed_rollback_rules", error=str(e))
        return False


# ==============================================================================
# Tier 2: Auto-Memory (Topic Files + Capped Index + 2-Step Save Invariant)
# ==============================================================================

def get_auto_memory_dir() -> Path:
    """Return workspace/.maidere/memory/ directory, ensuring it exists."""
    mem_dir = get_maidere_dir() / "memory"
    mem_dir.mkdir(parents=True, exist_ok=True)
    return mem_dir


def get_index_path() -> Path:
    """Return path to workspace/.maidere/memory/index.json."""
    return get_auto_memory_dir() / "index.json"


def read_memory_index() -> dict[str, Any]:
    """Read the memory index, initializing if missing."""
    idx_path = get_index_path()
    if not idx_path.exists():
        initial = {"version": "1.0", "updated_at": datetime.now(timezone.utc).isoformat(), "entries": []}
        try:
            idx_path.write_text(json.dumps(initial, indent=2), encoding="utf-8")
        except Exception:
            pass
        return initial

    try:
        data = json.loads(idx_path.read_text(encoding="utf-8"))
        if not isinstance(data, dict) or "entries" not in data:
            return {"version": "1.0", "updated_at": datetime.now(timezone.utc).isoformat(), "entries": []}
        return data
    except Exception as e:
        logger.warn("corrupted_memory_index", error=str(e))
        return reconcile_memory_index()


def reconcile_memory_index() -> dict[str, Any]:
    """Startup reconciliation: heal index.json from actual topic files on disk.

    Resolves orphaned topic files or dangling index pointers.
    """
    mem_dir = get_auto_memory_dir()
    idx_path = get_index_path()

    existing_files = {f.name: f for f in mem_dir.glob("*.md") if f.is_file()}

    # Read current index safely
    entries: list[dict[str, Any]] = []
    if idx_path.exists():
        try:
            raw = json.loads(idx_path.read_text(encoding="utf-8"))
            if isinstance(raw, dict) and "entries" in raw:
                # Filter out dangling entries where topic file no longer exists
                for entry in raw["entries"]:
                    filename = entry.get("file")
                    if filename and filename in existing_files:
                        entries.append(entry)
        except Exception:
            entries = []

    indexed_files = {e.get("file") for e in entries if e.get("file")}

    # Add any unindexed topic files found on disk
    for fname, fpath in existing_files.items():
        if fname not in indexed_files:
            topic_name = fname[:-3] if fname.endswith(".md") else fname
            try:
                first_lines = [l.strip() for l in fpath.read_text(encoding="utf-8").splitlines() if l.strip()]
                summary = first_lines[0].lstrip("#- ")[:100] if first_lines else "Reconciled topic file"
            except Exception:
                summary = "Reconciled topic file"

            entries.append({
                "id": uuid.uuid4().hex[:8],
                "topic": topic_name,
                "file": fname,
                "summary": summary,
                "category": "project",
                "created_at": datetime.now(timezone.utc).isoformat(),
                "last_accessed": time.time(),
            })

    healed_index = {
        "version": "1.0",
        "updated_at": datetime.now(timezone.utc).isoformat(),
        "entries": entries,
    }

    try:
        idx_path.write_text(json.dumps(healed_index, indent=2), encoding="utf-8")
        logger.info("memory_index_reconciled", entry_count=len(entries))
    except Exception as e:
        logger.warn("failed_writing_reconciled_index", error=str(e))

    return healed_index


def save_auto_memory_fact(
    topic: str,
    fact: str,
    category: str = "project",
) -> dict[str, Any]:
    """Execute the Two-Step Save Invariant:
    Step 1: Write/append fact to topic markdown file.
    Step 2: Update or create pointer in index.json (enforcing max 40 LRU cap).
    """
    clean_topic = re.sub(r"[^\w\-]", "_", topic.lower().strip())
    clean_fact = fact.strip()
    if not clean_topic or not clean_fact:
        return {"success": False, "error": "Empty topic or fact"}

    mem_dir = get_auto_memory_dir()
    topic_file = mem_dir / f"{clean_topic}.md"

    # Step 1: Write to topic file
    timestamp_str = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
    fact_entry = f"- [{timestamp_str}] {clean_fact}\n"

    try:
        if not topic_file.exists():
            topic_file.write_text(f"# Memory Topic: {clean_topic}\n\n{fact_entry}", encoding="utf-8")
        else:
            with open(topic_file, "a", encoding="utf-8") as f:
                f.write(fact_entry)
    except Exception as e:
        logger.error("step1_topic_write_failed", topic=clean_topic, error=str(e))
        return {"success": False, "error": f"Failed writing topic file: {str(e)}"}

    # Step 2: Update index.json (Two-step save invariant)
    try:
        index_data = read_memory_index()
        entries = index_data.get("entries", [])

        # Check if topic already in index
        existing = next((e for e in entries if e.get("topic") == clean_topic), None)
        now_ts = time.time()

        if existing:
            existing["summary"] = clean_fact[:120]
            existing["last_accessed"] = now_ts
            existing["updated_at"] = datetime.now(timezone.utc).isoformat()
        else:
            new_entry = {
                "id": uuid.uuid4().hex[:8],
                "topic": clean_topic,
                "file": f"{clean_topic}.md",
                "summary": clean_fact[:120],
                "category": category,
                "created_at": datetime.now(timezone.utc).isoformat(),
                "last_accessed": now_ts,
            }
            entries.append(new_entry)

        # Enforce 40-entry LRU cap with consolidation
        if len(entries) > MAX_INDEX_ENTRIES:
            # Sort by last_accessed ascending (oldest first)
            entries.sort(key=lambda x: x.get("last_accessed", 0))
            excess = len(entries) - MAX_INDEX_ENTRIES
            evicted = entries[:excess]
            entries = entries[excess:]
            logger.info("memory_index_cap_enforced", evicted_topics=[e.get("topic") for e in evicted])

        index_data["entries"] = entries
        index_data["updated_at"] = datetime.now(timezone.utc).isoformat()

        get_index_path().write_text(json.dumps(index_data, indent=2), encoding="utf-8")
        return {"success": True, "topic": clean_topic, "fact": clean_fact}
    except Exception as e:
        logger.error("step2_index_update_failed", topic=clean_topic, error=str(e))
        # File is on disk, reconcile can heal it later
        return {"success": True, "topic": clean_topic, "warning": f"Topic saved, index update delayed: {str(e)}"}


def read_auto_memory_topic(topic: str) -> str:
    """Read the full content of a topic markdown file and update its last_accessed timestamp."""
    clean_topic = re.sub(r"[^\w\-]", "_", topic.lower().strip())
    topic_file = get_auto_memory_dir() / f"{clean_topic}.md"

    if not topic_file.exists():
        return f"No memory topic found for '{topic}'."

    try:
        content = topic_file.read_text(encoding="utf-8")
        # Update LRU timestamp
        idx = read_memory_index()
        for e in idx.get("entries", []):
            if e.get("topic") == clean_topic:
                e["last_accessed"] = time.time()
                try:
                    get_index_path().write_text(json.dumps(idx, indent=2), encoding="utf-8")
                except Exception:
                    pass
                break
        return content
    except Exception as e:
        return f"Error reading topic '{topic}': {str(e)}"


def list_auto_memory_topics() -> list[dict[str, Any]]:
    """Return all currently indexed memory topics."""
    idx = read_memory_index()
    return idx.get("entries", [])


def format_auto_memory_context() -> str:
    """Format the active Tier 2 memory index into a compact context block."""
    idx = read_memory_index()
    entries = idx.get("entries", [])
    if not entries:
        return ""

    lines = ["### Persistent Auto-Memory Index:"]
    for e in entries[:15]:  # include top 15 most recently used
        lines.append(f"- **{e.get('topic')}** ({e.get('category', 'project')}): {e.get('summary')}")
    return "\n".join(lines)


# ==============================================================================
# Tier 3: Session Extraction Pipeline & Deduplication
# ==============================================================================

def jaccard_similarity(text1: str, text2: str) -> float:
    """Compute token-level Jaccard similarity for deduplication."""
    words1 = set(re.findall(r"\w+", text1.lower()))
    words2 = set(re.findall(r"\w+", text2.lower()))
    if not words1 or not words2:
        return 0.0
    intersection = len(words1 & words2)
    union = len(words1 | words2)
    return intersection / union if union > 0 else 0.0


def is_fact_duplicate_in_topic(topic: str, new_fact: str, threshold: float = 0.55) -> bool:
    """Check if the given fact is semantically redundant with existing bullet points in topic."""
    content = read_auto_memory_topic(topic)
    if "No memory topic found" in content or "Error reading" in content:
        return False

    for line in content.splitlines():
        if line.startswith("- "):
            # Strip timestamp if present: - [2026-09-25 ...] fact
            clean_existing = re.sub(r"^-\s*\[.*?\]\s*", "", line).strip()
            if jaccard_similarity(clean_existing, new_fact) >= threshold:
                return True
    return False


def extract_session_facts_from_messages(messages: list[dict[str, Any]]) -> list[dict[str, str]]:
    """Extract durable facts (user preferences, project conventions, architectural decisions)
    from a finished conversation session.
    """
    extracted: list[dict[str, str]] = []

    # Heuristic pattern matching for explicit user preference declarations
    pref_patterns = [
        (r"(?:i prefer|my preference is|i like to use|always use)\s+(.+?)(?:[;\n]|\.\s|$)", "user_preferences"),
        (r"(?:i am using|my machine is|hardware is|gpu is)\s+(.+?)(?:[;\n]|\.\s|$)", "hardware_environment"),
        (r"(?:decided to use|we chose|architecture uses|pattern is)\s+(.+?)(?:[;\n]|\.\s|$)", "project_decisions"),
        (r"(?:remember that|note that|keep in mind that)\s+(.+?)(?:[;\n]|\.\s|$)", "reference_facts"),
    ]

    for msg in messages:
        role = msg.get("role") if isinstance(msg, dict) else getattr(msg, "type", "")
        content = msg.get("content", "") if isinstance(msg, dict) else getattr(msg, "content", "")
        if role not in ("user", "human") or not isinstance(content, str):
            continue

        for pattern, topic in pref_patterns:
            match = re.search(pattern, content, re.IGNORECASE)
            if match:
                captured = match.group(0).strip()
                if len(captured) >= 15:
                    extracted.append({
                        "topic": topic,
                        "fact": captured,
                        "category": "user" if "preference" in topic else "project",
                    })

    return extracted


async def run_session_extraction(thread_id: str, messages: list[Any]) -> int:
    """Run Tier 3 session extraction and promote deduplicated facts to Tier 2 auto-memory.

    Returns count of newly promoted facts.
    """
    facts = extract_session_facts_from_messages(messages)
    promoted_count = 0

    for item in facts:
        topic = item["topic"]
        fact = item["fact"]
        category = item["category"]

        # Deduplication check
        if not is_fact_duplicate_in_topic(topic, fact):
            res = save_auto_memory_fact(topic, fact, category=category)
            if res.get("success"):
                promoted_count += 1
                logger.info("tier3_fact_promoted", topic=topic, fact=fact[:60])

    return promoted_count
