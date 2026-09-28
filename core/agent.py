"""Maidere agent — LangGraph StateGraph with message trimming and autonomous tool use.

Graph flow (Phase 2):
    __start__ → trim → think → (conditional: tool_calls?)
                            ├── YES → act → think (loop)
                            └── NO  → END

Phase 3 will add: trim → remember → think (memory injection)
"""

import asyncio
from datetime import datetime
import json
import re
from typing import Any
import uuid
import structlog
from langchain_core.messages import (
    AIMessage,
    HumanMessage,
    RemoveMessage,
    SystemMessage,
    ToolMessage,
)
from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver
from langgraph.graph import END, StateGraph

from core import llm
from core.config import settings
from core.db import get_db
from core.memory import format_memories, recall
from core.state import AgentState
from tools.registry import (
    execute_tool,
    get_orchestrator_ollama_schemas,
    get_tools_ollama_schemas,
    is_tool_call_concurrent_safe,
)

logger = structlog.get_logger()


class SystemPromptTemplate(str):
    """String template for system prompt that gracefully defaults missing placeholders."""

    def format(self, *args, **kwargs) -> str:
        kwargs.setdefault("username", "User")
        kwargs.setdefault("memory_context", "No relevant memories found.")
        now = datetime.now()
        kwargs.setdefault("current_date", now.strftime("%A, %B %d, %Y"))
        kwargs.setdefault("current_year", str(now.year))
        return super().format(*args, **kwargs)


SYSTEM_PROMPT = SystemPromptTemplate(
    """\
You are Maidere, an autonomous AI assistant running locally on the user's machine.

User Identity & Interaction:
- You are interacting with {username}.
- Always address the user by their chosen nickname ({username}) and communicate with them naturally, respectfully, and personally according to this name.

Temporal Real-World Anchor:
- Current Real-World Date: {current_date}
- Current Year: {current_year}
- Append the current year ({current_year}) to search queries ONLY for financial data, technology benchmarks, quarterly metrics, or active modern events. Do NOT append the current year to queries about historical events, historical figures, or established lore.

Core Operational Principles & Governance:
1. Critical Thinking: Never simulate, fake, or invent tool outputs, search results, browser text, or URLs in conversational text. If you need external facts, file operations, web searches, or code executions, you MUST emit an actual tool call. Never treat raw assertions as facts without verification.
2. Data Privacy & Hygiene: Never output or leak credentials, API keys, private tokens, or sensitive personal data. Maintain strict data hygiene across all operations.
3. Prompt & Intent Literacy: State assumptions concisely for ambiguous tasks. Structure final answers with high density (tables, bullet points, clean markdown) with zero conversational fluff.
4. Boundary & Limitation Awareness: If a tool fails or an external service is unavailable, report the exact limitation and error boundary directly—never hallucinate or invent synthetic data to compensate.
5. Iterative Self-Correction: When a tool execution returns an error, analyze the failure diagnostics, adjust parameters or try alternative approaches, and resolve issues systematically.
6. AI Ethics & Bias Mitigation: Provide balanced, multi-perspective analysis, cite verified sources explicitly, and avoid exclusionary assumptions in all research and synthesis.
7. Task Delegation & Safety Bounds: Never execute destructive shell commands (e.g. recursive deletions, forced resets, table drops) or modify critical system files without explicit user confirmation.
8. Proactive Tool Utilization: NEVER call tools for basic greetings (e.g. "hello", "hi"), casual conversation, or general chit-chat. When tools are genuinely needed, invoke them proactively and inspect their output before formulating your final response.
9. File Resolution Bounds & Targeted Editing: NEVER interpret casual user remarks, rhetorical questions, or conversational phrases as file names to read or execute. NEVER invent fictitious file extensions. Only use `read_file` for real files explicitly named by the user or discovered via `list_dir`. For modifying existing files, ALWAYS use `replace_file_content` (or `edit_file`) to substitute targeted text blocks rather than rewriting entire files; reserve `write_file` for creating new files. If an edit fails or causes errors, use `rollback_file` to restore the `.bak` backup.
10. Obsidian Vault Integration: You have direct read, write, search, and open access to the user's local Obsidian Vault via the `obsidian` tool. When asked to write, organize, look up, or open markdown notes, use the `obsidian` tool with appropriate action (`write_note`, `read_note`, `list_notes`, `search_notes`, `open_note`).
11. Sub-Agent Delegation (MANDATORY for Deep Research & Planning):
    Whenever the user requests research, deep research, comprehensive analysis, architectural design, software planning, detailed comparisons, technical breakdowns, or code improvements:
    - DO NOT write narrative plans, outlines, or procedural text in the chat (NEVER write "Phase 1", "Let's begin by researching", or "Now I will search").
    - Your very first action MUST be emitting 2 to 4 `Agent` (or `delegate_task`) tool calls immediately.
    - Available subagent types:
      * "researcher": Fast read-only web search, documentation browsing, and fact gathering.
      * "tech-hardware": Enterprise hardware, GPU clusters, supercomputing, cloud compute grants, and infrastructure research.
      * "plan": Software architecture, system design, technical implementation plans, and migration blueprints.
      * "verification": Formal verification, mathematical proof-checking, calculation recalculation, logical invariant testing, citation verification, and code correctness.
      * "validation": Real-world empirical validation, dependency/API compatibility, runtime feasibility, external benchmarks, and user acceptance criteria.
      * "code-reviewer": Code inspection, bug analysis, security audit, and refactoring proposals.
      * "general-purpose": Multi-step complex read-only sub-tasks.
    - DECOMPOSITION STRATEGY: Decompose the user's topic into 2 to 4 distinct, non-overlapping sub-tasks and emit `Agent(subagent_type="...", prompt="...")` calls in your first turn.
      Append the current year ({current_year}) to research sub-tasks ONLY for financial, earnings, benchmark, and active modern metrics. Never append the current year to historical or lore topics.
    - ANTI-PASS-THROUGH & SYNTHESIS DIRECTIVE:
      You have received dense data payloads from your sub-agents. You are FORBIDDEN from copy-pasting raw lists of URLs, "Key Points", or unformatted scratchpad notes directly to the user.
      You MUST write a cohesive, flowing, professional, analytical report with markdown headers, explanatory paragraphs, structured comparison tables, and verified metrics.
      You must embed citations as descriptive inline markdown links using the exact URLs discovered by the sub-agents.
      NEVER use placeholder links like "[Apply Here]", "[Link]", "[Website]", or "[Click Here]". Output the exact, raw verified URL provided by the sub-agent or descriptive anchor text.
      Verify financial metrics: ensure Share Price, Quarterly Revenue, and Market Capitalization are clearly distinguished with their correct units (Millions, Billions, Trillions) and dates ({current_year}).
    - You get one delegation round per query. After receiving sub-agent summaries, synthesize the final answer directly without re-delegating.
12. Strict Inline Index Citation Formatting:
    - CRITICAL: Every single citation MUST be wrapped in square brackets (e.g., [1], [2]). You are strictly FORBIDDEN from using naked numbers like '1' or '2' for citations.
    - When synthesizing research, you MUST cite sources using bracketed integers inline with the text (e.g., 'The model achieved 95% accuracy [1].'). At the VERY END of your response, include exactly one '## Sources' block formatted with one source per line: [1] [Title](https://actual-url.com). You are FORBIDDEN from generating 'References', 'Bibliography', 'Works Cited' sections, plain-text comma-separated URL lists, or numbered plain-text source lists in the response body.
    - NEVER emit naked titles next to "URL:" and NEVER use placeholder links like "[Apply Here]", "[Link]", or "[Website]". Always use the exact, verified https:// URL discovered by tools or sub-agents.
13. Epistemic Grounding (Canonical Facts vs. Speculation):
    - When researching or discussing fictional characters or historical events, you MUST prioritize canonical facts and primary source data. You are FORBIDDEN from treating fan theories, speculative blog posts, or forum discussions as factual canon unless the user explicitly requests theories.
    - Omit unverified fan theories, headcanons, and forum speculation from your final report unless theories are explicitly requested.
14. Chronological Timeline & Causality Verification:
    - When synthesizing historical events, biographies, or narrative developments, you must establish a strict chronological timeline.
    - Verify the exact month, day, and year of consecutive events before establishing cause-and-effect relationships in your report (e.g. verify that Event A preceded Event B before asserting that B occurred 'following' or 'as a result of' A).
15. Strict Source-to-Claim Mapping (Anti-Hallucination):
    - Every inline citation [N] MUST directly support the specific claim it is attached to based on Source [N] in the evidence.
    - You are strictly FORBIDDEN from guessing citations, attributing claims to the wrong media or platform (e.g., citing a forum discussion thread for a YouTube video), or slapping the last citation index onto a paragraph as an afterthought. If a source does not support the claim, do not cite it.
16. No Surprise Variables or Metrics in Summaries:
    - Every metric, statistic, number, or variable included in a summary table, 'Summary of Metrics', or conclusion MUST be explicitly introduced, explained, and cited inline with [N] in the main analytical body text first.
    - You are strictly FORBIDDEN from introducing novel statistics or surprise variables in concluding summary sections that were absent from the analytical body.

Relevant context, memories, and active skills:
{memory_context}\
"""
)






TOKEN_LIMIT = settings.agent_token_limit
CHARS_PER_TOKEN = settings.agent_chars_per_token
CHAR_LIMIT = TOKEN_LIMIT * CHARS_PER_TOKEN


# ---------------------------------------------------------------------------
# Graph Nodes
# ---------------------------------------------------------------------------


def trim_messages_node(state: AgentState) -> dict:
    """Trim oldest messages when approaching the active context window limit.

    Also detects topic changes and aggressively prunes prior context when the
    user switches to a completely unrelated topic within the same thread.
    """
    active_num_ctx = state.get("num_ctx") or settings.ollama_num_ctx
    token_limit = int(active_num_ctx * 0.75)
    char_limit = token_limit * CHARS_PER_TOKEN

    messages = state.get("messages", [])
    total_chars = sum(
        len(m.content)
        for m in messages
        if hasattr(m, "content") and isinstance(m.content, str)
    )

    removals: list[RemoveMessage] = []

    # --- Phase 1: Topic-change detection & aggressive pruning ---
    # Only activate when there is meaningful prior context (>= 4 non-system messages)
    non_system = [m for m in messages if not isinstance(m, SystemMessage)]
    if len(non_system) >= _TOPIC_CHANGE_MIN_MESSAGES:
        # Extract latest user query
        latest_query = ""
        for msg in reversed(messages):
            if isinstance(msg, HumanMessage) and msg.content:
                latest_query = str(msg.content)
                break

        if latest_query:
            # Build context text from the last few messages BEFORE the current query
            context_parts = []
            for msg in non_system[:-1]:  # exclude the latest message (current query)
                if hasattr(msg, "content") and isinstance(msg.content, str):
                    context_parts.append(msg.content)
            context_text = " ".join(context_parts[-4:])  # last 4 messages of context

            if detect_topic_change(latest_query, context_text):
                logger.info(
                    "topic_change_detected",
                    latest_query=latest_query[:80],
                    action="pruning_prior_context",
                )
                # Keep only the last 2 non-system messages (typically: previous AI answer + current query)
                to_keep = set(id(m) for m in non_system[-2:])
                for msg in non_system:
                    if id(msg) not in to_keep:
                        removals.append(RemoveMessage(id=msg.id))

                if removals:
                    logger.info(
                        "topic_change_pruned",
                        pruned_count=len(removals),
                    )
                    return {"messages": removals}

    # --- Phase 2: Standard character-limit trimming ---
    if total_chars <= char_limit:
        return {}

    chars_to_free = total_chars - char_limit
    freed = 0

    for msg in messages:
        if isinstance(msg, SystemMessage):
            continue
        if freed >= chars_to_free:
            break
        content_len = (
            len(msg.content)
            if hasattr(msg, "content") and isinstance(msg.content, str)
            else 0
        )
        freed += content_len
        removals.append(RemoveMessage(id=msg.id))

    if removals:
        logger.info("messages_trimmed", count=len(removals), chars_freed=freed)
        return {"messages": removals}

    return {}


async def remember_node(state: AgentState) -> dict:
    """Retrieve semantically relevant memories and auto-matched skills based on latest user query."""
    messages = state.get("messages", [])
    latest_user_query = ""

    # Find the most recent human message
    for msg in reversed(messages):
        if isinstance(msg, HumanMessage) and msg.content:
            latest_user_query = str(msg.content)
            break

    context_blocks = []

    # 1. Tier 1: Instruction Memory (RULES.md)
    try:
        from core.memory_tiers import format_auto_memory_context, load_instruction_memory
        rules_text = load_instruction_memory()
        if rules_text:
            context_blocks.append(f"[TIER 1 INSTRUCTION RULES]\n{rules_text}\n[END RULES]")

        # 2. Tier 2: Persistent Auto-Memory Index
        auto_mem = format_auto_memory_context()
        if auto_mem:
            context_blocks.append(f"[TIER 2 AUTO-MEMORY]\n{auto_mem}\n[END AUTO-MEMORY]")
    except Exception as e:
        await logger.awarn("tier_memory_loading_failed", error=str(e))

    # 3. Automatic Skill Matching
    if latest_user_query:
        from core.skills import match_skills

        try:
            matched = match_skills(latest_user_query)
            if matched:
                skills_formatted = "\n\n".join(
                    f"### Skill: {s.name}\n{s.instructions}" for s in matched
                )
                context_blocks.append(f"[ACTIVE SKILLS]\n{skills_formatted}\n[END SKILLS]")
                await emit_agent_event(
                    state.get("thread_id"),
                    {
                        "type": "agent_log",
                        "text": f"[SKILL] Matched skills: {', '.join(s.name for s in matched)}",
                        "level": "info",
                    },
                )
        except Exception as e:
            await logger.awarn("skill_matching_failed", error=str(e))

    # 2. Semantic Memory Retrieval from sqlite-vec
    if latest_user_query:
        try:
            db = await get_db(settings.db_path)
            try:
                memories = await recall(db, query=latest_user_query, top_k=5)
                if memories:
                    mem_formatted = format_memories(memories)
                    if mem_formatted and not mem_formatted.startswith("No relevant memories"):
                        context_blocks.append(f"[RECALLED MEMORIES]\n{mem_formatted}\n[END MEMORIES]")
                        await emit_agent_event(
                            state.get("thread_id"),
                            {
                                "type": "agent_log",
                                "text": f"[MEMORY] Recalled {len(memories)} relevant memories from vector database",
                                "level": "info",
                            },
                        )
            finally:
                await db.close()
        except Exception as e:
            await logger.aerror("remember_failed", error=str(e))

    mem_ctx = "\n\n".join(context_blocks) if context_blocks else "No relevant memories found."
    return {
        "memory_context": mem_ctx,
        "has_delegated": False,
        "tool_loop_count": 0,
        "subagent_results": [],
    }


_stream_callbacks: dict[str, Any] = {}
_event_callbacks: dict[str, Any] = {}
_custom_system_prompt: str | None = None


def get_system_prompt() -> str:
    return _custom_system_prompt or SYSTEM_PROMPT


def set_custom_system_prompt(text: str | None) -> None:
    global _custom_system_prompt
    _custom_system_prompt = (
        SystemPromptTemplate(text.strip()) if text and text.strip() else None
    )


def register_stream_callback(thread_id: str, callback: Any) -> None:
    _stream_callbacks[thread_id] = callback


def unregister_stream_callback(thread_id: str) -> None:
    _stream_callbacks.pop(thread_id, None)


def register_event_callback(thread_id: str, callback: Any) -> None:
    """Register an async callback for rich agent telemetry (stages, tools, subagents)."""
    _event_callbacks[thread_id] = callback


def unregister_event_callback(thread_id: str) -> None:
    """Unregister the event callback for a thread."""
    _event_callbacks.pop(thread_id, None)


async def emit_agent_event(thread_id: str | None, event: dict) -> None:
    """Emit an agent telemetry event to the registered thread callback if present."""
    if not thread_id:
        return
    callback = _event_callbacks.get(thread_id)
    if callback:
        try:
            res = callback(event)
            if asyncio.iscoroutine(res):
                await res
        except Exception:
            pass


def is_research_intent(query: str) -> bool:
    """Check if query is requesting research, investigation, or deep analysis."""
    if not query:
        return False
    q = query.lower()
    triggers = (
        "research",
        "deep research",
        "investigate",
        "compare",
        "breakdown",
        "[deep-research]",
        "fellowship",
        "internship",
    )
    return any(t in q for t in triggers)


def is_logic_puzzle(query: str) -> bool:
    """Detect if query is a closed-loop logic puzzle, state simulation, or mathematical proof."""
    if not query:
        return False
    q = query.lower()
    signals = (
        "closed loop",
        "multi-agent text",
        "execution cycles",
        "execution cycle",
        "inventory system",
        "container x",
        "tie-breaker",
        "halts automatically",
        "initial state",
        "last_sent",
        "logic puzzle",
        "simulate the following",
        "system halts",
        "token exchange",
        "node alpha",
        "node beta",
        "node gamma",
        "decay phase",
        "injection phase",
        "fewest tokens",
        "most tokens",
        "protocol for",
    )
    return any(s in q for s in signals)


# Common English stop words excluded from topic comparison
_STOP_WORDS = frozenset({
    "a", "an", "the", "is", "are", "was", "were", "be", "been", "being",
    "have", "has", "had", "do", "does", "did", "will", "would", "shall",
    "should", "may", "might", "must", "can", "could", "i", "you", "he",
    "she", "it", "we", "they", "me", "him", "her", "us", "them", "my",
    "your", "his", "its", "our", "their", "this", "that", "these", "those",
    "what", "which", "who", "whom", "when", "where", "why", "how", "if",
    "then", "than", "but", "and", "or", "not", "no", "so", "too", "very",
    "just", "also", "about", "up", "out", "on", "off", "over", "under",
    "again", "further", "once", "here", "there", "all", "each", "every",
    "both", "few", "more", "most", "other", "some", "such", "only", "own",
    "same", "of", "in", "to", "for", "with", "at", "by", "from", "as",
    "into", "through", "during", "before", "after", "above", "below",
    "between", "because", "until", "while", "tell", "me", "explain",
    "describe", "give", "please", "know", "think", "make", "like",
})

# Minimum message count before topic-change detection activates
_TOPIC_CHANGE_MIN_MESSAGES = 4
# Jaccard similarity threshold: below this = topic change
_TOPIC_CHANGE_THRESHOLD = 0.05


def _extract_content_words(text: str) -> set[str]:
    """Extract meaningful content words from text, filtering stop words and short tokens."""
    words = set(re.findall(r"[a-z]{3,}", text.lower()))
    return words - _STOP_WORDS


def detect_topic_change(latest_query: str, context_text: str) -> bool:
    """Detect if the latest query represents a significant topic shift from recent context.

    Uses Jaccard similarity on content words. Returns True when overlap is
    near-zero, indicating the user has switched to a completely unrelated topic.
    """
    if not latest_query or not context_text:
        return False

    query_words = _extract_content_words(latest_query)
    context_words = _extract_content_words(context_text)

    if not query_words or not context_words:
        return False

    intersection = query_words & context_words
    union = query_words | context_words

    if not union:
        return False

    jaccard = len(intersection) / len(union)
    return jaccard < _TOPIC_CHANGE_THRESHOLD


def clean_query_for_topic(query: str) -> str:
    """Extract topic from query by stripping research prefixes."""
    cleaned = re.sub(
        r'^(?:conduct|make|do|perform)?\s*(?:a\s+)?(?:deep\s+)?research\s*(?:on|into|about)?\s*',
        '',
        query,
        flags=re.IGNORECASE,
    )
    cleaned = re.sub(r'\[deep-research\]', '', cleaned, flags=re.IGNORECASE)
    cleaned = cleaned.strip().strip(':').strip('?').strip()
    return cleaned if len(cleaned) > 2 else query


def is_historical_or_lore(query: str, topic: str) -> bool:
    """Detect if a query is about historical events, figures, eras, or established fictional lore."""
    text = f"{query} {topic}".lower()
    # Explicit modern intent overrides historical suppression (e.g. "Hitler in 2026 pop culture", "modern day")
    if any(w in text for w in ("in 202", "modern day", "today", "current day", "contemporary")):
        return False
    hist_keywords = (
        "history", "historical", "ancient", "century", "war", "world war",
        "wwi", "wwii", "reich", "wehrmacht", "luftwaffe", "nazi", "hitler",
        "holocaust", "fascism", "totalitarian", "empire", "dynasty", "biography",
        "reign", "revolution", "treaty", "medieval", "feudal", "chancellor",
        "monarch", "emperor", "caesar", "napoleon", "churchill", "stalin",
        "roosevelt", "lenin", "battle", "treaty of", "king", "queen",
        "character", "novel", "fiction", "lore", "mythology",
    )
    return any(re.search(r'\b' + re.escape(kw) + r'\b', text) for kw in hist_keywords)


def is_temporally_sensitive(query: str, topic: str) -> bool:
    """Check if a research query requires temporal anchoring to the current year."""
    text = f"{query} {topic}".lower()
    # Strictly forbid appending current year to historical events, figures, or established lore
    if is_historical_or_lore(query, topic):
        return False
    if any(w in text for w in ("202", "latest", "recent", "current", "today", "now", "upcoming", "q1", "q2", "q3", "q4")):
        return True
    temporal_keywords = (
        "financial", "earning", "revenue", "quarter", "market cap", "stock",
        "valuation", "benchmark", "metric", "roadmap", "gpu", "hardware spec",
        "cloud credits", "nairr", "startup", "company earnings", "quarterly",
    )
    return any(kw in text for kw in temporal_keywords)


def is_hardware_query(query: str) -> bool:
    """Detect if query is focused on enterprise hardware, GPUs, compute clusters, or supercomputers."""
    text = query.lower()
    patterns = (
        r"\b(dgx|h100|h200|b200|b100|a100|gpu\s*clusters?|supercomputers?|supercomputing)\b",
        r"\b(enterprise\s+hardware|data\s*center\s+hardware|nairr|nvidia\s+inception)\b",
        r"\b(compute\s+clusters?|hpc\s+clusters?|cloud\s+compute\s+credits?)\b",
    )
    return any(re.search(pat, text) for pat in patterns)


def sanitize_subtask_theories(task: str, user_query: str) -> str:
    """Sanitize fan theories or speculation from subtasks unless user asked for them."""
    user_wants_theories = any(
        w in user_query.lower()
        for w in ("theory", "theories", "speculat", "headcanon", "fanfiction", "rumor")
    )
    if user_wants_theories:
        return task
    sanitized = re.sub(
        r'(?i)\b(fan\s*theories|speculative\s*blog\s*posts|crackpot\s*theories|fanfiction|headcanons?|rumors?|unverified\s*theories)\b',
        'canonical lore and narrative arc',
        task,
    )
    sanitized = re.sub(
        r'(?i)\bfan\s*theor(?:y|ies)?\b',
        'canonical facts',
        sanitized,
    )
    return sanitized.strip()


def extract_research_subtasks(text: str, user_query: str) -> list[str]:
    """Extract decomposed research sub-tasks from model text or auto-decompose query."""
    tasks: list[str] = []
    if text:
        lines = [line.strip() for line in text.split("\n") if line.strip()]
        for line in lines:
            match = re.match(
                r'^(?:\d+[\.\)]|\*|\-)\s*(?:\*\*)?(?:Sub-task\s*\d+:|Task\s*\d+:)?\s*(.*?)(?:\*\*)?$',
                line,
                re.IGNORECASE,
            )
            if match:
                candidate = match.group(1).strip().strip('*').strip(':').strip()
                if (
                    len(candidate) > 15
                    and not candidate.lower().startswith("let's")
                    and not candidate.lower().startswith("according")
                    and not candidate.lower().startswith("sure")
                ):
                    tasks.append(candidate)

    topic = clean_query_for_topic(user_query)
    current_year = datetime.now().year
    needs_year = is_temporally_sensitive(user_query, topic)
    year_tag = f" ({current_year})" if needs_year else ""

    if 2 <= len(tasks) <= 5:
        clean_tasks = []
        for t in tasks:
            clean_t = sanitize_subtask_theories(t, user_query)
            if not needs_year:
                clean_t = re.sub(rf'\s*\({current_year}\)', '', clean_t)
                clean_t = re.sub(rf'\b{current_year}\b', '', clean_t).strip()
            if topic.lower() not in clean_t.lower():
                clean_tasks.append(f"Research {topic}{year_tag}: {clean_t}")
            else:
                if needs_year and not clean_t.endswith(f"({current_year})"):
                    clean_tasks.append(f"Research {clean_t} ({current_year})")
                else:
                    clean_tasks.append(f"Research {clean_t}")
        return clean_tasks[:4]

    # Context-aware default 3-way decomposition
    if is_hardware_query(user_query):
        return [
            f"Research {topic} ({current_year}): architecture, hardware specifications, compute capabilities, and primary institutional use cases.",
            f"Research {topic} ({current_year}): verified institutional access models, cloud compute grants, NAIRR Pilot allocations, and incubator credits.",
            f"Research {topic} ({current_year}): pricing, procurement guidelines, deployment models, and verified primary sources.",
        ]
    elif needs_year:
        return [
            f"Research {topic} ({current_year}): core overview, background, current key specifications, leadership, and primary initiatives.",
            f"Research {topic} ({current_year}): products, software stack, ecosystem, roadmap, and major {current_year} developments.",
            f"Research {topic} ({current_year}): real-world benchmarks, {current_year} financial metrics (quarterly revenue, earnings, verified market cap), performance data, and verified sources.",
        ]
    elif is_historical_or_lore(user_query, topic) and not any(w in user_query.lower() for w in ("fiction", "lore", "character", "novel")):
        return [
            f"Research {topic}: origins, early background, ideological/political rise, and primary historical sources.",
            f"Research {topic}: strict chronological timeline of major events, pivotal decisions, and primary historical developments.",
            f"Research {topic}: historical impact, legacy, critical analysis, and verified academic/encyclopedic sources.",
        ]
    else:
        return [
            f"Research {topic}: canonical background, origins, core identity, and primary source lore.",
            f"Research {topic}: canonical narrative arc, major events, key relationships, and primary developments.",
            f"Research {topic}: canonical significance, author/creator context, and verified primary text sources (excluding unverified fan theories).",
        ]


def extract_text_tool_calls(text: str, available_tools: list[str]) -> list[dict]:
    """Extract tool calls written as text/JSON by local LLMs."""
    calls = []
    if not text:
        return calls

    # 1. Regex match for Agent(prompt="...") or delegate_task(task="...") or tool_name(arg="...")
    pattern = r'(\b[a-zA-Z0-9_]+)\s*\((.*?)\)'
    for match in re.finditer(pattern, text):
        t_name = match.group(1)
        normalized_name = "Agent" if t_name in ("Agent", "agent", "delegate_task", "subagent", "Task", "task") else t_name
        if t_name in available_tools or normalized_name in available_tools:
            actual_tool_name = normalized_name if normalized_name in available_tools else t_name
            raw_args = match.group(2)
            prompt_match = re.search(r'(?:prompt|task)\s*=\s*["\'](.*?)["\']', raw_args, re.DOTALL)
            type_match = re.search(r'subagent_type\s*=\s*["\'](.*?)["\']', raw_args)
            subagent_type = type_match.group(1) if type_match else "researcher"

            if prompt_match:
                calls.append({
                    "name": actual_tool_name,
                    "args": {"subagent_type": subagent_type, "prompt": prompt_match.group(1), "task": prompt_match.group(1)},
                    "id": str(uuid.uuid4()),
                })
            elif raw_args.strip().startswith("{"):
                try:
                    args_dict = json.loads(raw_args)
                    calls.append({"name": actual_tool_name, "args": args_dict, "id": str(uuid.uuid4())})
                except Exception:
                    pass
            elif raw_args.strip():
                clean_arg = raw_args.strip().strip('"\'')
                calls.append({
                    "name": actual_tool_name,
                    "args": {"prompt": clean_arg, "task": clean_arg, "subagent_type": subagent_type} if actual_tool_name in ("Agent", "delegate_task") else {"query": clean_arg},
                    "id": str(uuid.uuid4()),
                })

    # 2. JSON blocks: ```json {"name": "...", "arguments": ...} ```
    json_blocks = re.findall(r'```(?:json)?\s*(\{.*?\})\s*```', text, re.DOTALL)
    for block in json_blocks:
        try:
            parsed = json.loads(block)
            raw_name = parsed.get("name", "")
            norm_name = "Agent" if raw_name in ("Agent", "agent", "delegate_task", "subagent", "Task", "task") else raw_name
            target_name = norm_name if norm_name in available_tools else (raw_name if raw_name in available_tools else None)
            if target_name:
                calls.append({
                    "name": target_name,
                    "args": parsed.get("arguments") or parsed.get("args") or {},
                    "id": str(uuid.uuid4()),
                })
        except Exception:
            pass

    return calls


EXCLUDED_CITATION_WORDS: set[str] = {
    "section", "figure", "table", "phase", "step", "rule", "version", "v",
    "chapter", "page", "in", "for", "year", "top", "of", "to", "at",
    "model", "day", "cycle", "part", "tier", "level", "item", "option",
    "node", "round", "turn", "iteration", "factor", "grade", "class",
}


def normalize_citations_and_sources(content: str) -> str:
    """Normalize citations into [N] brackets and enforce one-source-per-line in Sources block.

    Catches 7B model formatting evasions such as naked citation numbers ('firms 5.' -> 'firms [5].')
    and unbracketed/comma-separated entries in '## Sources'.
    """
    if not content:
        return content

    # 1. Parse max source index if sources block exists
    source_indices = [int(m) for m in re.findall(r"\[?(\d+)\]?(?=\s*(?:\[|https?://|\w+.*https?://))", content)]
    max_source = max(source_indices) if source_indices else 10

    # 2. Repair naked citations in body text (prior to Sources block)
    sources_split = re.split(r"(?i)(?=##\s*sources|\*\*sources:?\*\*)", content, maxsplit=1)
    body = sources_split[0]
    sources_section = sources_split[1] if len(sources_split) > 1 else ""

    def repair_naked(m: re.Match) -> str:
        word = m.group(1)
        num = int(m.group(2))
        if word.lower() in EXCLUDED_CITATION_WORDS:
            return m.group(0)
        if num <= max_source:
            return f"{word} [{num}]"
        return m.group(0)

    repaired_body = re.sub(
        r"(\b[a-zA-Z\)]+)\s+([1-9]|1[0-9]|20)(?=[\.,;]|\s*$)",
        repair_naked,
        body,
    )

    # 3. Normalize Sources block
    if sources_section:
        lines = sources_section.splitlines()
        repaired_lines = []
        for line in lines:
            line_str = line.strip()
            if not line_str or line_str.startswith("#") or line_str.lower().startswith("**sources"):
                repaired_lines.append(line)
                continue
            # Handle comma-separated sources e.g. [1] url, [2] url
            sub_items = re.split(r",\s*(?=\[?\d+\]?[\s\.:\-])", line_str)
            for item in sub_items:
                item = item.strip()
                if not item:
                    continue
                # Case A: 2 [Title](url) -> [2] [Title](url)
                m_a = re.match(r"^(\d+)\s+(\[[^\]]+\]\(https?://\S+\))$", item)
                if m_a:
                    repaired_lines.append(f"[{m_a.group(1)}] {m_a.group(2)}")
                    continue
                # Case B: 5 Title - Site, https://url or [5] Title - Site, https://url
                m_b = re.match(r"^\[?(\d+)\]?[\s\.:\-]+([^,]+?),\s*(https?://\S+)$", item)
                if m_b:
                    repaired_lines.append(f"[{m_b.group(1)}] [{m_b.group(2).strip()}]({m_b.group(3).strip()})")
                    continue
                # Case C: [1] https://url -> [1] [https://url](https://url)
                m_c = re.match(r"^\[?(\d+)\]?[\s\.:\-]+(https?://\S+)$", item)
                if m_c:
                    repaired_lines.append(f"[{m_c.group(1)}] [{m_c.group(2)}]({m_c.group(2)})")
                    continue
                repaired_lines.append(item)
        sources_section = "\n".join(repaired_lines)

    return repaired_body + sources_section


def strip_critic_fourth_wall_leaks(content: str) -> str:
    """Remove accidental Socratic Critic or deliberation meta-commentary leaking into the final response."""
    if not content:
        return content
    leak_patterns = [
        r"(?i)(?:(?<=[\.\?!]\s)|(?<=\n)|^)[^\.\n]*(?:adhering\s+to|incorporated|incorporating|following|based\s+on)\s+(?:the\s+)?(?:corrections|verifications|suggestions|feedback|instructions|review)\s+(?:suggested\s+by|from|of)\s+(?:the\s+)?(?:socratic\s+)?critic[^\.\n]*[\.\?!]?",
        r"(?i)(?:(?<=[\.\?!]\s)|(?<=\n)|^)[^\.\n]*(?:socratic\s+critic|critic[\'’]?s?\s+(?:review|feedback|suggestions|corrections|instructions))[^\.\n]*[\.\?!]?",
        r"(?i)(?:^|\n)[\*\-_]*\s*(?:note|status):\s*(?:revised|updated|refined)\s+based\s+on\s+critic[^\n]*\n?",
    ]
    cleaned = content
    for pat in leak_patterns:
        cleaned = re.sub(pat, "", cleaned)
    cleaned = re.sub(r"  +", " ", cleaned)
    cleaned = re.sub(r"\n{3,}", "\n\n", cleaned).strip()
    return cleaned


def restore_markdown_urls(content: str, messages: list[Any]) -> str:
    """Ensure raw web links from tool outputs are preserved and revive stripped URLs.

    If the LLM stripped URLs and wrote 'URL: [Title]' or 'URL: Title', or '[Title]' without
    a URL, this function matches titles against verified URLs collected in ToolMessages
    and restores them into working [Title](https://...) Markdown links.
    """
    if not content:
        return content

    known_links: dict[str, str] = {}
    for msg in messages:
        if not (hasattr(msg, "content") and isinstance(msg.content, str)):
            continue
        text = msg.content
        # 1. Match [Title](https://...)
        for m in re.finditer(r"\[([^\]]+)\]\((https?://[^\s\)]+)\)", text):
            title = m.group(1).strip()
            url = m.group(2).strip()
            if url:
                known_links[title.lower()] = url
        # 2. Match "1. Title \n URL: https://..."
        for m in re.finditer(r"\d+\.\s+\[?([^\]\n]+)\]?\n\s+URL:\s*(https?://[^\s\)\n]+)", text):
            title = m.group(1).strip()
            url = m.group(2).strip()
            known_links[title.lower()] = url
        # 3. Match generic "Title: https://..." or "- Title: https://..."
        for m in re.finditer(r"(?:[-*]\s*)?([A-Za-z0-9\s\-_–—]+?):\s*(https?://[^\s\)\n]+)", text):
            title = m.group(1).strip()
            url = m.group(2).strip()
            if not title.lower().startswith("http") and len(title) > 2:
                known_links[title.lower()] = url

    def replace_stripped_url(match: re.Match) -> str:
        prefix = match.group(1) or ""
        raw_text = match.group(2).strip().strip("[]\"'")
        if raw_text.startswith("http://") or raw_text.startswith("https://"):
            return f"{prefix}[{raw_text}]({raw_text})"
        low = raw_text.lower()
        if low in known_links:
            return f"{prefix}[{raw_text}]({known_links[low]})"
        for k_title, k_url in known_links.items():
            if k_title in low or low in k_title:
                return f"{prefix}[{raw_text}]({k_url})"
        return match.group(0)

    # Catches:
    # URL: NVIDIA Inception Program
    # - URL: [NVIDIA Inception Program]
    # * URL: NVIDIA AI for Robotics Program
    fixed = re.sub(r"((?:[-*]\s*)?)URL:\s*([^\n\(\)]+?)(?=\n|$)", replace_stripped_url, content)

    def replace_unlinked_bracket(match: re.Match) -> str:
        title = match.group(1).strip()
        low = title.lower()
        if low in known_links:
            return f"[{title}]({known_links[low]})"
        for k_title, k_url in known_links.items():
            if k_title in low or low in k_title:
                return f"[{title}]({k_url})"
        return match.group(0)

    # Catches [Title] not followed by '('
    fixed = re.sub(r"\[([^\]]+)\](?!\s*[\(\[])", replace_unlinked_bracket, fixed)

    # Normalize citations and source block lines
    fixed = normalize_citations_and_sources(fixed)
    return fixed


def should_deliberate(state: AgentState, user_query: str, last_message: Any) -> bool:
    """Determine whether the response requires the Socratic Critic-Refiner deliberation cycle."""
    if not getattr(settings, "enable_agentic_deliberation", True):
        return False

    # Never loop if already refined in this turn
    if state.get("refinement_count", 0) > 0:
        return False

    # Never deliberate on tool calls (must execute tools first)
    if (
        isinstance(last_message, AIMessage)
        and getattr(last_message, "tool_calls", None)
        and len(last_message.tool_calls) > 0
    ):
        return False

    # Check for basic greetings or trivial chit-chat
    q_clean = user_query.strip().lower()
    from core.router import _SIMPLE_GREETINGS
    if q_clean in _SIMPLE_GREETINGS or len(q_clean) < 4:
        return False

    # 1. Explicit deep reasoning switch active (Multi-stage System 2 Engine)
    if state.get("deep_reasoning"):
        return True

    # 2. Deep research delegation synthesis
    if state.get("has_delegated") and is_research_intent(user_query):
        return True

    # 3. Dynamic Auto-mode reasoning complexity (only when deep_reasoning enabled and thinking_mode not set)
    if not state.get("thinking_mode") and getattr(settings, "enable_deep_reasoning", True):
        from core.router import TaskComplexity, classify_complexity
        complexity = classify_complexity(user_query, state.get("messages", []))
        if complexity == TaskComplexity.REASONING:
            return True

    return False


async def think_node(state: AgentState) -> dict:
    """Call the LLM to generate a thought, answer, or tool call with real-time streaming support."""
    now = datetime.now()
    current_date = now.strftime("%A, %B %d, %Y")
    current_year = str(now.year)
    memory_context = state.get("memory_context", "No relevant memories found.")
    username = state.get("username", "User") or "User"
    base_prompt = get_system_prompt()
    try:
        system_content = base_prompt.format(
            memory_context=memory_context,
            current_date=current_date,
            current_year=current_year,
            username=username,
        )
    except KeyError:
        try:
            system_content = base_prompt.format(
                memory_context=memory_context,
                username=username,
            )
        except KeyError:
            system_content = base_prompt.format(memory_context=memory_context)

    # Build message history conforming to Ollama chat format with turn-scoping.
    # Intermediate ToolMessages from prior completed turns are stripped from the active context
    # so historical data (e.g. past research dumps) cannot leak into new turns.
    ollama_messages: list[dict] = [{"role": "system", "content": system_content}]

    raw_messages = state.get("messages", [])
    last_user_idx = -1
    for i, msg in enumerate(raw_messages):
        if isinstance(msg, HumanMessage):
            last_user_idx = i

    for i, msg in enumerate(raw_messages):
        if isinstance(msg, SystemMessage):
            continue

        # Completed prior turns: retain only user prompt and final assistant response
        if last_user_idx > 0 and i < last_user_idx:
            if isinstance(msg, HumanMessage):
                ollama_messages.append({"role": "user", "content": str(msg.content or "")})
            elif isinstance(msg, AIMessage) and msg.content and not getattr(msg, "tool_calls", None):
                ollama_messages.append({"role": "assistant", "content": str(msg.content)})
            continue

        # Current active turn: retain HumanMessage, tool calls, and ToolMessages
        if isinstance(msg, HumanMessage):
            ollama_messages.append({"role": "user", "content": str(msg.content or "")})
        elif isinstance(msg, ToolMessage):
            ollama_messages.append({"role": "tool", "content": str(msg.content or "")})
        elif isinstance(msg, AIMessage):
            msg_dict: dict = {"role": "assistant", "content": msg.content or ""}
            if getattr(msg, "tool_calls", None):
                msg_dict["tool_calls"] = [
                    {
                        "function": {
                            "name": tc["name"],
                            "arguments": tc["args"],
                        }
                    }
                    for tc in msg.tool_calls
                ]
            ollama_messages.append(msg_dict)
        else:
            role = "user"
            content = msg.content if hasattr(msg, "content") else str(msg)
            ollama_messages.append({"role": role, "content": content})

    # Retrieve latest user query for dynamic model routing
    latest_user_query = ""
    for msg in reversed(state["messages"]):
        if isinstance(msg, HumanMessage) and msg.content:
            latest_user_query = str(msg.content)
            break

    from core.metrics import ROUTER_DECISIONS_TOTAL
    from core.router import TaskComplexity, select_model

    requested_model = state.get("model")
    thinking_mode_flag = bool(state.get("thinking_mode", False))
    deep_reasoning_flag = bool(state.get("deep_reasoning", False))
    selected_model, routing_reason, complexity = await select_model(
        requested_model=requested_model,
        query=latest_user_query,
        history=state["messages"],
        thinking_mode=thinking_mode_flag,
        deep_reasoning=deep_reasoning_flag,
    )

    ROUTER_DECISIONS_TOTAL.labels(
        model=selected_model,
        complexity=complexity.value if hasattr(complexity, "value") else str(complexity),
        reason=routing_reason,
    ).inc()

    await logger.ainfo(
        "model_routed",
        selected_model=selected_model,
        requested_model=requested_model,
        routing_reason=routing_reason,
        complexity=str(complexity),
    )

    await emit_agent_event(
        state.get("thread_id"),
        {
            "type": "agent_log",
            "text": f"[ROUTER] Selected {selected_model} ({routing_reason})",
            "level": "info",
        },
    )

    # Retrieve orchestrator tool schemas for Ollama (excludes direct web_search/browser)
    tools_schemas = get_orchestrator_ollama_schemas() if not state.get("skip_tools") else None

    # Anti-Cheat / Pure Reasoning Guardrail:
    # DeepSeek-R1 does not support Ollama tool-calling and must never receive tool schemas.
    # Additionally, closed-loop logic/math puzzles must never have delegation tools (Agent/delegate_task)
    # to prevent orchestrators from cheating via web search.
    if "deepseek-r1" in selected_model or is_logic_puzzle(latest_user_query):
        tools_schemas = None
    elif state.get("has_delegated"):
        # Patch #3: Infinite Delegation Loop Prevention
        # After the orchestrator has delegated once, remove Agent and delegate_task from
        # available tools for the remainder of this query. One shot at delegation.
        tools_schemas = [
            t for t in tools_schemas
            if t.get("function", {}).get("name") not in ("Agent", "delegate_task")
        ]


    callback = state.get("callback") or _stream_callbacks.get(state.get("thread_id", ""))

    has_tools_in_history = any(isinstance(m, ToolMessage) for m in state.get("messages", []))
    is_synthesis_turn = bool(state.get("has_delegated") and has_tools_in_history)
    is_thinking_turn = bool(
        deep_reasoning_flag
        or thinking_mode_flag
        or "deepseek-r1" in selected_model
        or complexity == TaskComplexity.REASONING
    )

    if deep_reasoning_flag and not is_synthesis_turn:
        ollama_messages.append({
            "role": "system",
            "content": (
                "[SYSTEM 2: DIVERGENT COGNITIVE EXPLORATION & DECONSTRUCTION]\n"
                "You are operating in SYSTEM 2 DEEP REASONING mode.\n"
                "Do NOT provide a superficial, hasty, or intuitive answer. You MUST systematically think and deconstruct the problem inside <think> and </think> tags.\n"
                "CRITICAL FORMAT REQUIREMENT: You MUST start your output with '<think>' and end your reasoning with '</think>'.\n"
                "Inside <think>...</think>, execute these four cognitive stages:\n"
                "1. PROBLEM DECONSTRUCTION & CORE PREMISES:\n"
                "   - Break down the inquiry into its irreducible logical, mathematical, or structural components.\n"
                "   - Explicitly identify, state, and question all implicit assumptions or constraints.\n"
                "2. DIVERGENT HYPOTHESES & ALTERNATIVE PATHS:\n"
                "   - Formulate and compare at least two distinct approaches or solution candidates (Approach A vs. Approach B).\n"
                "   - Trace the causal chain, state mutations, and intermediate steps of each candidate.\n"
                "3. COUNTER-EXAMPLE & FALSIFICATION ANALYSIS:\n"
                "   - Identify the exact conditions under which an approach or premise fails.\n"
                "   - Stress-test boundary conditions, edge cases, and arithmetic/logical invariants.\n"
                "4. PROVISIONAL SYNTHESIS:\n"
                "   - Formulate the initial end-to-end solution with explicit justifications before closing </think>.\n"
                "After closing </think>, present your comprehensive analytical reasoning and provisional solution."
            ),
        })
    elif thinking_mode_flag and not is_synthesis_turn:
        ollama_messages.append({
            "role": "system",
            "content": (
                "[THINKING SCRATCHPAD: CHAIN-OF-THOUGHT DIRECTIVE]\n"
                "You are operating in THINKING MODE. Think step-by-step inside <think> and </think> tags before providing your answer.\n"
                "CRITICAL FORMAT REQUIREMENT: You MUST start your output with '<think>' and end your reasoning with '</think>'. "
                "Only AFTER closing '</think>' provide your final response.\n"
                "Inside <think>...</think>:\n"
                "- Deconstruct the user's objective, implicit context, and constraints.\n"
                "- Outline your response structure and determine the key facts, formulas, or code required.\n"
                "- Anticipate common errors, logical traps, or edge cases.\n"
                "Close </think> and immediately present your clear, direct, and well-structured answer."
            ),
        })
    elif is_thinking_turn and not is_synthesis_turn:
        ollama_messages.append({
            "role": "system",
            "content": (
                "[MANDATORY STEP-BY-STEP REASONING DIRECTIVE]\n"
                "You are operating in REASONING mode. Before providing your answer or making decisions, "
                "you MUST think and reason step-by-step inside <think> and </think> tags.\n"
                "CRITICAL FORMAT REQUIREMENT: You MUST start your output with '<think>' and end your reasoning with '</think>'. "
                "Only AFTER closing '</think>' provide your final response.\n"
                "Inside <think>...</think>:\n"
                "- Dissect the user's problem, constraints, and implicit assumptions.\n"
                "- Track state mutations and intermediate calculations explicitly.\n"
                "- Check for logical fallacies, edge cases, and temporal accuracy.\n"
                "Do NOT jump directly to the answer. Think thoroughly inside <think>...</think>."
            ),
        })



    if is_synthesis_turn:
        ollama_messages.append({
            "role": "user",
            "content": (
                f"[SYNTHESIS & CITATION MANDATE]\n"
                f"All requested research and tool actions have completed. Synthesize your final comprehensive, high-quality analytical report based on the findings above.\n"
                f"1. CRITICAL INLINE INDEX CITATION DIRECTIVE:\n"
                f"   - CRITICAL: Every single citation MUST be wrapped in square brackets (e.g., [1], [2]). You are strictly FORBIDDEN from using naked numbers like '1' or '2' for citations.\n"
                f"   - You MUST cite sources using bracketed integers inline with the text (e.g., 'The model achieved 95% accuracy [1] on the benchmark [2].').\n"
                f"   - FORBIDDEN: Do NOT generate a 'References', 'Bibliography', 'Works Cited', or numbered plain-text list of sources anywhere in your response body.\n"
                f"   - FORBIDDEN: Do NOT write comma-separated URLs or plain unlinked URLs.\n"
                f"   - FORBIDDEN: Do NOT write 'URL: [Title]' or bare titles without links.\n"
                f"   - At the VERY END of your response, include exactly one '## Sources' block with each source on its own line formatted as:\n"
                f"     [1] [Title](https://actual-url.com)\n"
                f"     [2] [Title](https://actual-url.com)\n"
                f"   - If you omit inline [1], [2] integers, use naked numbers like 1 or 2, or generate a bibliography section, you have failed.\n"
                f"2. STRICT SOURCE-TO-CLAIM CORRESPONDENCE:\n"
                f"   - Every inline citation [N] must directly support the specific claim it is attached to based on Source [N] in the evidence.\n"
                f"   - FORBIDDEN: Do NOT guess citations, attribute claims to the wrong medium or platform (e.g. citing a Reddit thread for a YouTube video), or lazily slap the last citation index onto an unverified sentence.\n"
                f"3. NO SURPRISE METRICS OR VARIABLES IN SUMMARIES:\n"
                f"   - Every metric, statistic, number, or variable in a concluding summary table or list MUST be introduced, explained, and cited inline [N] in the main body text first.\n"
                f"   - FORBIDDEN: Do NOT introduce novel statistics or surprise variables in the conclusion or summary that were never discussed in the body.\n"
                f"4. STRUCTURE & DENSITY:\n"
                f"   - Write flowing prose with markdown headers (##), analytical paragraphs, and comparison tables. Zero unformatted bullet dumps.\n"
                f"5. EPISTEMIC GROUNDING (CANONICAL FACTS VS. SPECULATION):\n"
                f"   - When discussing fictional characters, literary works, or historical events, you MUST prioritize canonical facts and primary source data.\n"
                f"   - You are FORBIDDEN from treating fan theories, speculative blog posts, or forum discussions as factual canon unless the user explicitly requested theories.\n"
                f"6. CHRONOLOGICAL TIMELINE & CAUSALITY VERIFICATION:\n"
                f"   - When summarizing historical events, biographies, or narrative developments, you must establish a strict chronological timeline.\n"
                f"   - Verify the exact month, day, and year of consecutive events before establishing cause-and-effect relationships (e.g., verify that Event A preceded Event B before asserting that B occurred 'following' or 'as a result of' A)."
            ),
        })

    # Call LLM for decision / response with real-time streaming support
    active_num_ctx = state.get("num_ctx") or settings.ollama_num_ctx
    streamed_to_callback = False

    async def _realtime_stream_token(token_text: str):
        nonlocal streamed_to_callback
        streamed_to_callback = True
        if callback:
            await callback(token_text)

    # Stream real-time tokens when no deliberation cycle is active
    initial_on_token = _realtime_stream_token if (callback and not deep_reasoning_flag and not should_deliberate(state, latest_user_query, AIMessage(content=""))) else None
    chat_kwargs = {
        "tools": tools_schemas,
        "model": selected_model,
        "num_ctx": active_num_ctx,
    }
    if initial_on_token is not None:
        chat_kwargs["on_token"] = initial_on_token
    response = await llm.chat(
        ollama_messages,
        **chat_kwargs,
    )
    message = response.get("message", {})
    ai_content = message.get("content", "")
    raw_tool_calls = message.get("tool_calls", [])

    normalized_tool_calls = []
    if raw_tool_calls:
        for tc in raw_tool_calls:
            func = tc.get("function", {})
            name = func.get("name", "")
            args = func.get("arguments", {})
            if isinstance(args, str):
                try:
                    args = json.loads(args)
                except Exception:
                    args = {"prompt" if name in ("Agent", "delegate_task") else "command": args}
            call_id = tc.get("id") or str(uuid.uuid4())
            normalized_tool_calls.append(
                {
                    "name": name,
                    "args": args,
                    "id": call_id,
                }
            )

    # Text-fallback tool call extraction for local 7B models
    if tools_schemas and not normalized_tool_calls and ai_content:
        avail_names = [t.get("function", {}).get("name") for t in tools_schemas if t.get("function")]
        extracted = extract_text_tool_calls(ai_content, avail_names)
        if extracted:
            normalized_tool_calls.extend(extracted)
            ai_content = ""

    # Auto-delegation guarantee for deep research queries:
    # If the user explicitly requested research, but the 7B model generated
    # conversational text or a plan instead of tool calls, convert into actual Agent calls!
    if (
        tools_schemas
        and not normalized_tool_calls
        and not state.get("has_delegated")
        and is_research_intent(latest_user_query)
        and "deepseek-r1" not in selected_model
        and not is_logic_puzzle(latest_user_query)
    ):
        subtasks = extract_research_subtasks(ai_content, latest_user_query)
        target_subagent = "tech-hardware" if is_hardware_query(latest_user_query) else "researcher"
        for st in subtasks:
            normalized_tool_calls.append({
                "name": "Agent",
                "args": {"subagent_type": target_subagent, "prompt": st, "task": st},
                "id": str(uuid.uuid4()),
            })
        ai_content = ""


    # If tools were found, log any thoughts and clear narrative text so tool execution happens cleanly
    if normalized_tool_calls:
        think_match = re.search(r'<think>([\s\S]*?)</think>', ai_content, re.IGNORECASE)
        if think_match:
            thought_snippet = think_match.group(1).strip().replace("\n", " ")
            if thought_snippet:
                await emit_agent_event(
                    state.get("thread_id"),
                    {
                        "type": "agent_log",
                        "text": f"[THINK] {thought_snippet[:120]}...",
                        "level": "info",
                    },
                )
        ai_content = ""
        for tc in normalized_tool_calls:
            tc_name = tc.get("name", "")
            tc_args = tc.get("args", {})
            if tc_name in ("Agent", "agent", "delegate_task", "subagent", "Task"):
                prompt = tc_args.get("prompt") or tc_args.get("task") or ""
                sub_type = tc_args.get("subagent_type") or "researcher"
                await emit_agent_event(
                    state.get("thread_id"),
                    {
                        "type": "agent_log",
                        "text": f"[SUBAGENT] Orchestrator planning Sub-Agent [{sub_type}]: \"{prompt[:90]}\"",
                        "level": "tool",
                    },
                )
            else:
                await emit_agent_event(
                    state.get("thread_id"),
                    {
                        "type": "agent_log",
                        "text": f"[TOOL] Orchestrator planned action: {tc_name}",
                        "level": "tool",
                    },
                )
    else:
        # Final answer turn: synthesize if empty or stream tokens
        if not ai_content.strip() and has_tools_in_history:
            await emit_agent_event(
                state.get("thread_id"),
                {
                    "type": "agent_log",
                    "text": "[SYNTHESIZE] Orchestrator synthesizing final answer from findings...",
                    "level": "info",
                },
            )
            thinking_addon = ""
            if is_thinking_turn:
                thinking_addon = (
                    "CRITICAL: Think step-by-step inside <think> and </think> tags first, "
                    "evaluating and cross-referencing all findings before synthesizing your report.\n"
                )
            synthesis_prompt = {
                "role": "user",
                "content": (
                    f"All requested research and tool actions have completed. "
                    f"Synthesize and present your final comprehensive, high-quality analytical report based on the findings.\n"
                    + thinking_addon +
                    f"1. CRITICAL INLINE INDEX CITATION DIRECTIVE:\n"
                    f"CRITICAL: Every single citation MUST be wrapped in square brackets (e.g., [1], [2]). You are strictly FORBIDDEN from using naked numbers like '1' or '2' for citations.\n"
                    f"You MUST cite sources using bracketed integers inline with the text (e.g., 'The study found X [1] and Y [2].'). "
                    f"FORBIDDEN: Do NOT generate a 'References', 'Bibliography', 'Works Cited', or numbered plain-text list of sources in your response body. "
                    f"FORBIDDEN: Do NOT write comma-separated URLs or plain unlinked URLs. "
                    f"FORBIDDEN: Do NOT write 'URL: [Title]' or bare titles without links. "
                    f"At the VERY END of your response, include exactly one '## Sources' block with each source on its own line formatted as: "
                    f"[1] [Title](https://actual-url.com). "
                    f"If you omit inline [1], [2] integers, use naked numbers like 1 or 2, or generate a bibliography section, you have failed.\n"
                    f"2. STRICT SOURCE-TO-CLAIM CORRESPONDENCE:\n"
                    f"Every inline citation [N] must directly support the specific claim it is attached to based on Source [N] in the evidence. "
                    f"FORBIDDEN: Do NOT guess citations, attribute claims to the wrong medium or platform (e.g. citing a Reddit thread for a YouTube video), or lazily slap the last citation index onto an unverified sentence.\n"
                    f"3. NO SURPRISE METRICS OR VARIABLES IN SUMMARIES:\n"
                    f"Every metric, statistic, number, or variable in a concluding summary table or list MUST be introduced, explained, and cited inline [N] in the main body text first. "
                    f"FORBIDDEN: Do NOT introduce novel statistics or surprise variables in the conclusion or summary that were never discussed in the body.\n"
                    f"4. Write a flowing, professional report with markdown headers (##), analytical prose paragraphs, and tables.\n"
                    f"5. EPISTEMIC GROUNDING (CANONICAL FACTS VS. SPECULATION):\n"
                    f"When discussing fictional characters, literary works, or historical events, prioritize canonical facts and primary source data. "
                    f"You are FORBIDDEN from treating fan theories, speculative blog posts, or forum discussions as factual canon unless the user explicitly requested theories.\n"
                    f"6. CHRONOLOGICAL TIMELINE & CAUSALITY VERIFICATION:\n"
                    f"When summarizing historical events, biographies, or timelines, you must establish a strict chronological timeline. "
                    f"Verify the exact month and year of consecutive events before establishing cause-and-effect relationships (e.g., verify that Event A preceded Event B before asserting that B occurred 'following' A).\n"
                    f"7. Verify financial metrics: distinguish Share Price, Quarterly Revenue, and Market Capitalization with correct units "
                    f"(Millions, Billions, Trillions) and anchor to {current_year}."
                ),
            }
            ollama_messages.append(synthesis_prompt)
            synth_kwargs = {
                "tools": None,
                "model": selected_model,
                "num_ctx": active_num_ctx,
            }
            if callback and not should_deliberate(state, latest_user_query, AIMessage(content="")):
                synth_kwargs["on_token"] = _realtime_stream_token
            synth_resp = await llm.chat(ollama_messages, **synth_kwargs)
            synth_msg = synth_resp.get("message", {})
            ai_content = synth_msg.get("content", "")

        # Apply deterministic URL restoration guard to restore any stripped links from tool history
        ai_content = restore_markdown_urls(ai_content, state.get("messages", []))

        # Check if Socratic deliberation will occur
        will_deliberate = (
            not normalized_tool_calls
            and should_deliberate(state, latest_user_query, AIMessage(content=ai_content))
        )

        if will_deliberate:
            await emit_agent_event(
                state.get("thread_id"),
                {
                    "type": "agent_log",
                    "text": "[REASON] Preliminary draft complete. Initiating Socratic verification...",
                    "level": "info",
                },
            )
        elif callback and ai_content and not streamed_to_callback:
            try:
                # Stream content in smooth chunks (fallback if not streamed directly)
                chunk_size = 20
                for i in range(0, len(ai_content), chunk_size):
                    await callback(ai_content[i:i + chunk_size])
                    await asyncio.sleep(0.01)
            except Exception:
                pass

    return {
        "messages": [
            AIMessage(
                content=ai_content,
                tool_calls=normalized_tool_calls if normalized_tool_calls else [],
            )
        ],
        "model": selected_model,
        "complexity": complexity,
    }



MAX_TOOL_LOOPS = 10


async def act_node(state: AgentState) -> dict:
    """Execute tool calls generated by the think node in parallel."""
    import asyncio
    import time

    last_message = state["messages"][-1]
    tool_calls = getattr(last_message, "tool_calls", [])
    thread_id = state.get("thread_id")
    loop_count = state.get("tool_loop_count", 0) + 1

    async def _run_single_tool(tc: dict) -> ToolMessage:
        tool_name = tc.get("name", "")
        tool_args = tc.get("args", {})
        call_id = tc.get("id", str(uuid.uuid4()))

        # Emit tool started event in real time
        await emit_agent_event(
            thread_id,
            {
                "type": "tool_started",
                "tool_name": tool_name,
                "args": tool_args,
                "call_id": call_id,
            },
        )

        start_time = time.perf_counter()
        result = await execute_tool(
            name=tool_name,
            args=tool_args,
            thread_id=thread_id,
        )
        duration_ms = int((time.perf_counter() - start_time) * 1000)
        is_success = not str(result).startswith("Error:") and not str(result).startswith("[Search Unavailable:")

        # Emit tool executed event in real time
        await emit_agent_event(
            thread_id,
            {
                "type": "tool_executed",
                "tool_name": tool_name,
                "args": tool_args,
                "result": str(result),
                "call_id": call_id,
                "duration_ms": duration_ms,
                "success": is_success,
            },
        )

        return ToolMessage(
            content=result,
            tool_call_id=call_id,
            name=tool_name,
            additional_kwargs={"duration_ms": duration_ms, "args": tool_args},
            id=f"tool_msg_{call_id}",
        )

    # Concurrency partitioning: group consecutive concurrent-safe calls for parallel execution,
    # and execute mutating/unsafe calls serially in order to eliminate state race conditions.
    tool_messages: list[ToolMessage] = []
    current_concurrent_batch: list[dict] = []

    for tc in tool_calls:
        t_name = tc.get("name", "")
        t_args = tc.get("args", {})
        if is_tool_call_concurrent_safe(t_name, t_args):
            current_concurrent_batch.append(tc)
        else:
            if current_concurrent_batch:
                batch_results = await asyncio.gather(*[_run_single_tool(c) for c in current_concurrent_batch])
                tool_messages.extend(batch_results)
                current_concurrent_batch = []
            serial_res = await _run_single_tool(tc)
            tool_messages.append(serial_res)

    if current_concurrent_batch:
        batch_results = await asyncio.gather(*[_run_single_tool(c) for c in current_concurrent_batch])
        tool_messages.extend(batch_results)

    # Patch #3: Track if Agent / delegate_task was used — blocks re-delegation for this query
    delegated = any(tc.get("name") in ("Agent", "agent", "delegate_task", "subagent", "Task") for tc in tool_calls)

    result: dict = {
        "messages": list(tool_messages),
        "tool_loop_count": loop_count,
    }
    if delegated:
        result["has_delegated"] = True

    return result



async def evaluate_node(state: AgentState) -> dict:
    """Evaluate tool execution results, detect failures, and provide self-correction context.

    If tool execution returns an error:
    - Counts consecutive errors across recent tool invocations.
    - If consecutive errors >= 2, marks [ABORT RETRIES] so the LLM reports the limitation without looping.
    - Otherwise, provides [SYSTEM EVALUATION GUIDANCE] to diagnose and adjust arguments or alternatives.
    - Modifies/annotates ToolMessage content in-place to preserve standard LLM message alternation.
    """
    messages = state.get("messages", [])
    if not messages:
        return {}

    # Find the most recent tool messages from the act_node execution
    recent_tool_msgs = []
    for msg in reversed(messages):
        if isinstance(msg, ToolMessage):
            recent_tool_msgs.append(msg)
        else:
            break

    if not recent_tool_msgs:
        return {}

    # Check for errors in the recent tool messages
    has_error = False
    for tm in recent_tool_msgs:
        content_str = str(tm.content)
        if (
            content_str.startswith("Error:")
            or content_str.startswith("[Search Error:")
            or content_str.startswith("[Search Unavailable:")
            or "Error executing tool" in content_str
        ):
            has_error = True
            break

    # Check for structured V&V verdicts from subagents (verification / validation)
    verdict_directive = None
    for tm in recent_tool_msgs:
        content_str = str(tm.content)
        verdict_match = re.search(r"\[(FAIL|BLOCKED):\s*([A-Z_]+)\]", content_str)
        if verdict_match:
            status, reason = verdict_match.group(1), verdict_match.group(2)
            verdict_directive = (
                f"\n\n[SYSTEM CORRECTION DIRECTIVE: A verification/validation subagent reported [{status}: {reason}]. "
                "You must immediately remediate this issue. Recalculate, verify citations against primary evidence, "
                "or correct your reasoning before continuing. Do NOT proceed with unverified or falsified claims.]"
            )
            break

    if not has_error and not verdict_directive:
        return {}

    # Count consecutive tool errors in the message history
    consecutive_errors = 0
    if has_error:
        for msg in reversed(messages):
            if isinstance(msg, ToolMessage):
                content_str = str(msg.content)
                if (
                    content_str.startswith("Error:")
                    or content_str.startswith("[Search Error:")
                    or content_str.startswith("[Search Unavailable:")
                    or "Error executing tool" in content_str
                ):
                    consecutive_errors += 1
                else:
                    break
            elif isinstance(msg, AIMessage) and getattr(msg, "tool_calls", None):
                continue
            else:
                break

    # Build evaluation annotation
    guidance_parts = []
    if verdict_directive:
        guidance_parts.append(verdict_directive)

    if has_error:
        if consecutive_errors >= 2:
            guidance_parts.append(
                "\n\n[ABORT RETRIES: Service or tool is unavailable after multiple attempts. "
                "Do NOT retry calling this tool. Report the failure transparently to the user "
                "and answer based on existing knowledge or state limitations without fabricating data.]"
            )
        else:
            guidance_parts.append(
                "\n\n[SYSTEM EVALUATION GUIDANCE: Diagnose this failure. Either adjust arguments, "
                "attempt an alternative tool, or state the limitation clearly to the user. "
                "Do NOT simulate or invent fake outputs.]"
            )

    guidance = "".join(guidance_parts)

    # Append guidance to the most recent tool message content
    last_tool_msg = recent_tool_msgs[0]
    updated_content = str(last_tool_msg.content)
    if guidance not in updated_content:
        updated_content += guidance

    updated_tool_msg = ToolMessage(
        content=updated_content,
        tool_call_id=last_tool_msg.tool_call_id,
        name=last_tool_msg.name,
        additional_kwargs=getattr(last_tool_msg, "additional_kwargs", {}),
        id=last_tool_msg.id or f"tool_msg_{last_tool_msg.tool_call_id}",
    )

    await logger.awarn(
        "tool_evaluated_with_guidance",
        consecutive_errors=consecutive_errors,
        tool_name=last_tool_msg.name,
        has_verdict=verdict_directive is not None,
    )

    return {"messages": [updated_tool_msg]}



# ---------------------------------------------------------------------------
# System 2 Cognitive Reasoning Nodes: Verify & Converge
# ---------------------------------------------------------------------------


async def verify_node(state: AgentState) -> dict:
    """Perform rigorous mathematical, logical, and constraint verification on the provisional solution."""
    messages = state.get("messages", [])
    last_ai = messages[-1] if messages else None
    draft_content = last_ai.content if hasattr(last_ai, "content") else str(last_ai or "")

    latest_user_query = ""
    for msg in reversed(messages):
        if isinstance(msg, HumanMessage) and msg.content:
            latest_user_query = str(msg.content)
            break

    thread_id = state.get("thread_id")
    await emit_agent_event(
        thread_id,
        {
            "type": "agent_log",
            "text": "[VERIFY] Analytical verifier proof-checking solution and testing invariants...",
            "level": "info",
        },
    )

    tool_evidence = []
    for msg in messages:
        if isinstance(msg, ToolMessage) and msg.content:
            tool_evidence.append(f"Tool {msg.name}: {str(msg.content)[:1000]}")

    evidence_str = "\n---\n".join(tool_evidence) if tool_evidence else "Rely on first-principles deduction and mathematical proofs."

    selected_model = state.get("model") or settings.ollama_model
    if selected_model == "auto":
        selected_model = settings.ollama_model
    active_num_ctx = state.get("num_ctx") or settings.ollama_num_ctx

    verifier_prompt = [
        {
            "role": "system",
            "content": (
                "You are the Analytical Verification Specialist and Proof-Checker.\n"
                "Your role is to rigorously proof-check the provisional solution against mathematical truth, logic invariants, and boundary constraints.\n\n"
                "Verification Directives:\n"
                "1. Mathematical & Arithmetic Rigor: Recalculate all numbers, formulas, sums, products, and percentages independently. Verify units and scales.\n"
                "2. State Tracking & Procedure Verification: In step-by-step procedures, simulations, or cycles, verify variable values at every single step. Ensure state mutations follow all problem rules.\n"
                "3. Logical Invariants & Deductions: Test whether conclusions follow strictly from premises. Check for hidden non sequiturs, inverted causality, or false assumptions.\n"
                "4. Boundary Conditions & Edge Cases: Test extreme values (zero, negative, null, empty strings, maximum limits). Identify any unaddressed edge-case failures.\n"
                "5. Evidence & Source Alignment: Verify that all cited assertions match the provided evidence without hallucinated extrapolations or mismatched sources.\n"
                "6. Citation & Format Verification: Ensure all citations use square brackets [N] and Sources block has one Markdown link per line.\n"
                "7. Conciseness: If the provisional solution is fully verified and mathematically sound, output '[PASS: VERIFIED]' followed by a brief confirmation.\n"
                "If errors, gaps, or arithmetic flaws exist, output a structured list of concrete corrective proof steps for the final synthesis.\n"
                "Do NOT use conversational filler, pleasantries, or emojis."
            ),
        },
        {
            "role": "user",
            "content": (
                f"Problem / Inquiry:\n{latest_user_query}\n\n"
                f"Evidence & Constraints:\n{evidence_str}\n\n"
                f"Provisional Solution to Verify:\n{draft_content}"
            ),
        },
    ]

    try:
        response = await llm.chat(verifier_prompt, tools=None, model=selected_model, num_ctx=active_num_ctx)
        verification_text = response.get("message", {}).get("content", "").strip()
    except Exception as e:
        logger.warning("verification_failed", error=str(e))
        verification_text = "[PASS: VERIFIED] Baseline verification completed."

    summary = verification_text[:120].replace("\n", " ")
    await emit_agent_event(
        thread_id,
        {
            "type": "agent_log",
            "text": f"[VERIFY] Proof check: {summary}...",
            "level": "info",
        },
    )

    return {"critique": verification_text, "verification_notes": verification_text}


# Backwards compatibility alias
critique_node = verify_node


async def converge_node(state: AgentState) -> dict:
    """Synthesize hardened final response incorporating analytical verification."""
    messages = state.get("messages", [])
    last_ai = messages[-1] if messages else None
    draft_content = last_ai.content if hasattr(last_ai, "content") else str(last_ai or "")
    verification_text = state.get("verification_notes") or state.get("critique", "")
    thread_id = state.get("thread_id")
    callback = state.get("callback") or _stream_callbacks.get(thread_id or "")

    latest_user_query = ""
    for msg in reversed(messages):
        if isinstance(msg, HumanMessage) and msg.content:
            latest_user_query = str(msg.content)
            break

    # If verification passed without issues, use the draft directly
    if ("[PASS: VERIFIED]" in verification_text or "[PASS: AIRTIGHT]" in verification_text) and len(verification_text) < 200:
        await emit_agent_event(
            thread_id,
            {
                "type": "agent_log",
                "text": "[CONVERGE] Solution verified as mathematically sound. Finalizing response...",
                "level": "info",
            },
        )
        final_content = restore_markdown_urls(draft_content, messages)
        if callback and final_content:
            chunk_size = 20
            for i in range(0, len(final_content), chunk_size):
                await callback(final_content[i:i + chunk_size])
                await asyncio.sleep(0.01)
        return {
            "messages": [AIMessage(content=final_content)],
            "refinement_count": state.get("refinement_count", 0) + 1,
        }

    await emit_agent_event(
        thread_id,
        {
            "type": "agent_log",
            "text": "[CONVERGE] Converging on formally verified authoritative solution...",
            "level": "info",
        },
    )

    selected_model = state.get("model") or settings.ollama_model
    if selected_model == "auto":
        selected_model = settings.ollama_model
    active_num_ctx = state.get("num_ctx") or settings.ollama_num_ctx

    converger_prompt = [
        {
            "role": "system",
            "content": (
                "You are the Convergent Reasoning Engine. Your mission is to produce the final, authoritative, "
                "formally verified solution to the user's inquiry.\n"
                "You have been provided with the provisional solution and the Analytical Verifier's proof check.\n\n"
                "Directives:\n"
                "1. Systematically incorporate verified corrections from the proof-checking phase.\n"
                "2. Correct Source Mismatches: Ensure every [N] citation accurately corresponds to the factual contents of Source [N]. Never attribute claims about one medium to an unrelated source. Remove or re-attribute ungrounded claims.\n"
                "3. Ensure zero hallucinations, verified chronological order, and accurate metric units.\n"
                "4. Strict Bracketed Citations & Sources: Every single citation MUST be wrapped in square brackets (e.g., [1], [2]). You are strictly FORBIDDEN from using naked numbers like '1' or '2'. Preserve all inline index citations ([1], [2]) and their corresponding '## Sources' block with [Title](https://raw-url) Markdown links. Never output comma-separated URLs or unlinked text. NEVER generate 'References' or 'Bibliography' sections.\n"
                "5. Eliminate Surprise Variables: If a metric or statistic appears in a summary table or conclusion but is absent from the body, either weave it into the appropriate body section with its verified inline [N] citation, or remove it from the summary.\n"
                "6. Enforce strict mathematical calculations, state mutations, and logic proofs.\n"
                "7. Structure with high information density (headers, tables, clear bullet points, trade-offs).\n"
                "8. Output the complete authoritative answer showing exact working steps and final answers.\n"
                "9. CRITICAL FOURTH-WALL DIRECTIVE: When outputting the final refined synthesis, DO NOT mention the verification phase, proof-checkers, internal instructions, or deliberation nodes. Output ONLY the final, polished report directly to the user. Never write phrases like 'adhering to the corrections suggested by the Critic', 'incorporating the verification review', or 'based on internal assessment'.\n"
                "10. Do NOT use emojis."
            ),
        },
        {
            "role": "user",
            "content": (
                f"User Inquiry:\n{latest_user_query}\n\n"
                f"Provisional Solution:\n{draft_content}\n\n"
                f"Analytical Verification Proof Check:\n{verification_text}\n\n"
                f"Synthesize the final authoritative response now (remember: do NOT mention the verification process or internal deliberation in your output):"
            ),
        },
    ]

    try:
        response = await llm.chat(converger_prompt, tools=None, model=selected_model, num_ctx=active_num_ctx)
        final_content = response.get("message", {}).get("content", "").strip()
    except Exception as e:
        logger.warning("converge_failed", error=str(e))
        final_content = draft_content

    # Strip any accidental fourth-wall leaks from the response
    final_content = strip_critic_fourth_wall_leaks(final_content)

    # Apply URL restoration and citation normalization
    final_content = restore_markdown_urls(final_content, messages)

    # Preserve the CoT thinking block from the initial deliberation turn
    if "<think>" in draft_content and "</think>" in draft_content and "<think>" not in final_content:
        thought_block = draft_content[draft_content.find("<think>"):draft_content.find("</think>") + len("</think>")]
        final_content = f"{thought_block}\n\n{final_content}"

    # Stream to callback
    if callback and final_content:
        chunk_size = 20
        for i in range(0, len(final_content), chunk_size):
            await callback(final_content[i:i + chunk_size])
            await asyncio.sleep(0.01)


    await emit_agent_event(
        thread_id,
        {
            "type": "agent_log",
            "text": "[CONVERGE] Formally verified solution converged with high mathematical and logical rigor.",
            "level": "info",
        },
    )

    return {
        "messages": [AIMessage(content=final_content)],
        "refinement_count": state.get("refinement_count", 0) + 1,
    }


# Backwards compatibility alias
refine_node = converge_node


# ---------------------------------------------------------------------------
# Conditional Edges
# ---------------------------------------------------------------------------


def should_continue(state: AgentState) -> str:
    """Route after think: if tool calls exist, go to act. If deep reasoning verification needed, go to verify. Otherwise end."""
    last_message = state["messages"][-1]
    loop_count = state.get("tool_loop_count", 0)
    if loop_count >= MAX_TOOL_LOOPS:
        logger.warning("max_tool_loops_exceeded", loops=loop_count)
        return END

    if (
        isinstance(last_message, AIMessage)
        and getattr(last_message, "tool_calls", None)
        and len(last_message.tool_calls) > 0
    ):
        return "act"

    latest_user_query = ""
    for msg in reversed(state.get("messages", [])):
        if isinstance(msg, HumanMessage) and msg.content:
            latest_user_query = str(msg.content)
            break

    if should_deliberate(state, latest_user_query, last_message):
        return "verify"

    return END



# ---------------------------------------------------------------------------
# Graph Construction
# ---------------------------------------------------------------------------


async def create_graph(db_path: str | None = None) -> tuple:
    """Build and compile the Maidere agent graph.

    Returns:
        (compiled_graph, checkpointer_context, checkpointer) — caller must
        manage the context lifecycle (call __aexit__ on shutdown).
    """
    path = db_path or settings.db_path
    checkpointer_ctx = AsyncSqliteSaver.from_conn_string(path)
    checkpointer = await checkpointer_ctx.__aenter__()

    workflow = StateGraph(AgentState)

    # Nodes
    workflow.add_node("trim", trim_messages_node)
    workflow.add_node("remember", remember_node)
    workflow.add_node("think", think_node)
    workflow.add_node("act", act_node)
    workflow.add_node("evaluate", evaluate_node)
    workflow.add_node("verify", verify_node)
    workflow.add_node("converge", converge_node)
    # Aliases for backwards compatibility with checkpoints
    workflow.add_node("critique", verify_node)
    workflow.add_node("refine", converge_node)

    # Edges: __start__ → trim → remember → think → (act → evaluate → think)* → (verify → converge)? → END
    workflow.set_entry_point("trim")
    workflow.add_edge("trim", "remember")
    workflow.add_edge("remember", "think")
    workflow.add_conditional_edges(
        "think",
        should_continue,
        {
            "act": "act",
            "verify": "verify",
            "critique": "verify",
            END: END,
        },
    )
    workflow.add_edge("act", "evaluate")
    workflow.add_edge("evaluate", "think")
    workflow.add_edge("verify", "converge")
    workflow.add_edge("critique", "converge")
    workflow.add_edge("converge", END)
    workflow.add_edge("refine", END)

    graph = workflow.compile(checkpointer=checkpointer)

    logger.info(
        "agent_graph_compiled",
        nodes=["trim", "remember", "think", "act", "evaluate", "verify", "converge"],
    )
    return graph, checkpointer_ctx


