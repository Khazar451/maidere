"""Sub-agent execution engine for Maidere.

Spawns focused research sub-agents, each with their own fresh context window,
to handle research sub-tasks that would overflow the main agent's 8K context.

Architecture:
- Each sub-agent runs a simple think→act loop (no LangGraph overhead)
- Read-only tool access: web_search, browser, read_file, list_dir
- Dense extraction system prompt (no vague generalizations)
- Hard turn limit with forced conclusion
- Progress callbacks for real-time WebSocket updates
- Results stored in vector memory with strict tagging

Adapted from Claude Code's sub-agent pattern for local Ollama execution.
"""

import json
from pathlib import Path
import re
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Callable, Coroutine
import uuid
import yaml

import structlog

from core import llm
from core.config import settings

logger = structlog.get_logger()

# --- Constants & Guards ---

MAX_SUBAGENT_DEPTH = 1
MAX_SUBAGENTS_PER_QUERY = 5
DEFAULT_MAX_TURNS = 6
MAX_TURNS_HARD_CAP = 10

# Strictly read-only tools — sub-agents are data-gatherers, never writers.
SUBAGENT_ALLOWED_TOOLS: frozenset[str] = frozenset({
    "web_search",
    "browser",
    "read_file",
    "list_dir",
})

# --- Progress Callback ---

_progress_callback: Callable[[dict], Coroutine] | None = None


def set_progress_callback(callback: Callable[[dict], Coroutine] | None) -> None:
    """Set async callback for real-time sub-agent progress events.

    Called from the WebSocket route handler before graph invocation.
    Cleared after invocation completes.
    """
    global _progress_callback
    _progress_callback = callback


def clear_progress_callback() -> None:
    """Clear the progress callback after query completes."""
    global _progress_callback
    _progress_callback = None


# --- Sub-Agent System Prompt ---

SUBAGENT_SYSTEM_PROMPT = """\
You are a research sub-agent. You cannot answer the research task from your own internal memory. You MUST use tools to find information.

CRITICAL DIRECTIVES — ZERO TOLERANCE:
1. Do NOT write out your plan. Do NOT explain what you are going to do. Do NOT use phrases like "Phase 1", "Executing search", "Let's begin by searching", or any conversational filler.
2. Your very first action MUST be an actual tool call (e.g., `web_search`). Wait for the tool to return data.
3. Keep calling tools (`web_search`, `browser`, `read_file`, `list_dir`) until you have gathered sufficient concrete data.
4. CONDITIONAL TEMPORAL CONSTRAINT: Append the current year (2026) to search queries ONLY for financial data, quarterly earnings, market metrics, technology benchmarks, or active modern events. Do NOT append the current year to queries about historical events, historical figures, or established lore (e.g. do NOT append 2026 to searches about WWII, 20th-century history, or fictional lore).
5. FINANCIAL & METRIC STRUCTURAL VALIDATION:
   - Verify the unit of measurement (e.g., Millions, Billions, Trillions) and distinguish between Share Price (e.g. $578), Quarterly Revenue (e.g. $39B), and Market Capitalization (e.g. $1.5 Trillion). Do NOT mix them up!
   - Verify source identity: NEVER attribute crypto tracker feeds, crypto exchange stock tickers (e.g. Kraken, Binance), or currency pairs as corporate market capitalization.
   - Always record the exact fiscal quarter or reporting period (e.g., Q2 2026, FY 2025).
6. EPISTEMIC GROUNDING (CANONICAL FACTS VS. SPECULATION):
   - When researching fictional characters or historical events, you MUST prioritize canonical facts and primary source data. You are FORBIDDEN from treating fan theories, speculative blog posts, or forum discussions as factual canon unless the user explicitly requests theories.
   - Do NOT search for or include terms like "fan theories", "critical analyses", "speculation", or forum debates (e.g., Reddit, fandom forums) in your search queries unless explicitly requested by the user. Focus searches on canonical sources, official wikis, primary texts, and author/creator statements.
7. CHRONOLOGICAL TIMELINE & CAUSALITY VERIFICATION:
   - When extracting historical events, you must establish a strict chronological timeline. Verify the exact month, day, and year of consecutive events before establishing cause-and-effect relationships (e.g., verify that Event A preceded Event B before claiming Event B occurred "following" or "as a result of" Event A).
8. SEARCH DIVERSITY & REDUNDANCY PREVENTION:
   - If you execute multiple search queries, they must be highly diverse and target completely different aspects of the topic to prevent retrieving the same source URL multiple times. Do NOT repeat queries or re-search minor rephrasings of the same topic.
   - If a source or Wikipedia article has already been retrieved, query distinct sub-topics, alternative perspectives, primary documents, or specific institutional programs.
9. STRICT MARKDOWN URL CITATIONS:
   - Every source citation MUST use strict Markdown link format with verified raw source URLs: `[Title](https://www.actual-link.com)`.
   - You are strictly FORBIDDEN from writing "URL: [Title]" or outputting page titles without their raw `https://` web address.
   - NEVER use placeholder links like "[Apply Here]" or "[Link]".
10. Once you have gathered sufficient data via tools, return a DENSE DATA PAYLOAD:
   - Extract and retain exact numbers, benchmarks, version numbers, dates, terminal commands, code snippets, pricing, and direct quotes.
   - Format with bullet points and markdown tables.
   - Zero conversational text or narrative fluff.

Research Task:
{task}\
"""


TECH_HARDWARE_SYSTEM_PROMPT = """\
You are an expert enterprise hardware and high-performance computing (HPC) research sub-agent.
Your role is to research data center infrastructure, enterprise AI accelerators, GPU clusters, supercomputers, cloud compute credits, and institutional research grants with strict domain grounding.

CRITICAL DIRECTIVES — ZERO TOLERANCE:
1. Do NOT write out your plan. Do NOT explain what you are going to do. Do NOT use phrases like "Phase 1", "Executing search", or any conversational filler.
2. Your very first action MUST be an actual tool call (`web_search`). Wait for the tool to return data.
3. Keep calling tools (`web_search`, `browser`, `read_file`, `list_dir`) until you have gathered sufficient concrete data.
4. REALITY CHECK & ENTERPRISE HARDWARE GROUNDING:
   - When researching enterprise hardware (such as NVIDIA DGX systems, H100/H200/B200 GPU clusters, data center supercomputers), financial assets, or expensive infrastructure, distinguish between physical ownership and cloud/grant access.
   - You are strictly FORBIDDEN from suggesting or searching second-hand consumer marketplaces (like Craigslist, eBay, Facebook Marketplace, or developer forums) for enterprise data center hardware costing hundreds of thousands of dollars.
   - Ground research in verified institutional access, research grants, and accredited programs:
     * National AI Research Resource (NAIRR) Pilot: The primary avenue in 2026 for academic & institutional researchers, granting exclusive allocations on up to 4 NVIDIA DGX nodes, NSF supercomputers, and DOE national laboratory clusters.
     * Cloud Compute Grants & Incubator Credits: NVIDIA Inception Program provides up to $100,000 in DGX Cloud credits (it does NOT mail physical DGX boxes to startups); AWS Activate; Google for Startups Cloud credits; Microsoft for Startups Founders Hub.
     * Institutional & Academic HPC: University supercomputing clusters, NSF ACCESS allocations, and research grant sponsorships.
5. TEMPORAL CONSTRAINT: For all search queries regarding hardware specifications, benchmarks, GPU pricing, availability, and cloud credits, ALWAYS append the current year (2026) to prevent retrieving outdated historical specs.
6. STRICT MARKDOWN URL CITATIONS:
   - Every source citation MUST use strict Markdown link format with verified raw source URLs: `[Title](https://www.actual-link.com)`.
   - You are strictly FORBIDDEN from writing "URL: [Title]" or outputting page titles without their raw `https://` web address.
   - NEVER use placeholder links like "[Apply Here]" or "[Link]".
7. Once you have gathered sufficient data via tools, return a DENSE DATA PAYLOAD:
   - Extract and retain exact numbers, benchmarks, pricing, memory bandwidth, TDP, architecture specs, grant allocation terms, and direct quotes.
   - Format with bullet points and markdown tables.
   - Zero conversational text or narrative fluff.

Hardware Research Task:
{task}\
"""


# --- Data Classes ---

@dataclass
class SubAgentDefinition:
    """Claude Code subagent configuration parsed from frontmatter markdown."""

    name: str
    description: str
    tools: list[str] = field(default_factory=lambda: ["web_search", "browser", "read_file", "list_dir"])
    disallowed_tools: list[str] = field(default_factory=list)
    model: str | None = None
    max_turns: int = DEFAULT_MAX_TURNS
    system_prompt: str = SUBAGENT_SYSTEM_PROMPT


@dataclass
class SubAgentResult:
    """Result from a sub-agent execution."""

    task: str
    summary: str
    subagent_type: str = "researcher"
    tools_used: list[dict] = field(default_factory=list)
    duration_ms: int = 0
    success: bool = True
    turns_used: int = 0
    artifact_path: str = ""
    brief: str = ""


# --- Artifact Bus & Handoff Engineering ---

def prune_artifacts(
    artifacts_dir: Path | None = None,
    max_artifacts: int = 100,
    max_age_days: int = 14,
) -> int:
    """Prune old artifacts in workspace/.maidere/artifacts/ to prevent unbounded disk growth.

    Enforces two constraints:
    1. Time-to-live (TTL): deletes files older than max_age_days.
    2. Capacity cap: if remaining files exceed max_artifacts, removes oldest by modification time.

    Returns the count of pruned files.
    """
    target_dir = artifacts_dir or (Path(settings.agent_workspace) / ".maidere" / "artifacts")
    if not target_dir.exists():
        return 0

    now = time.time()
    max_age_seconds = max_age_days * 86400
    pruned_count = 0

    try:
        files = [f for f in target_dir.glob("*.md") if f.is_file()]
        surviving_files: list[tuple[Path, float]] = []

        # 1. TTL Pruning
        for f in files:
            try:
                mtime = f.stat().st_mtime
                if now - mtime > max_age_seconds:
                    f.unlink(missing_ok=True)
                    pruned_count += 1
                else:
                    surviving_files.append((f, mtime))
            except OSError:
                continue

        # 2. Capacity Pruning (oldest mtime first)
        if len(surviving_files) > max_artifacts:
            surviving_files.sort(key=lambda x: x[1])
            to_remove = len(surviving_files) - max_artifacts
            for f, _ in surviving_files[:to_remove]:
                try:
                    f.unlink(missing_ok=True)
                    pruned_count += 1
                except OSError:
                    continue
    except Exception as e:
        logger.warn("artifact_pruning_failed", error=str(e))

    return pruned_count


def write_subagent_artifact(
    task: str,
    content: str,
    subagent_type: str,
    tools_used: list[dict],
    duration_ms: int,
    turns_used: int,
    model: str,
    artifacts_dir: Path | None = None,
) -> Path:
    """Write subagent output to a collision-proof artifact file."""
    target_dir = artifacts_dir or (Path(settings.agent_workspace) / ".maidere" / "artifacts")
    target_dir.mkdir(parents=True, exist_ok=True)

    timestamp_str = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S_%f")
    unique_suffix = uuid.uuid4().hex[:8]
    clean_type = re.sub(r"[^\w\-]", "_", subagent_type.lower())
    filename = f"{timestamp_str}_{clean_type}_{unique_suffix}.md"
    file_path = target_dir / filename

    tools_summary = ", ".join(
        f"{t.get('tool_name', 'tool')}({'OK' if t.get('success') else 'FAIL'})"
        for t in tools_used
    ) or "none"

    artifact_md = f"""# Sub-Agent Research Artifact: {task}

- **Agent Type**: {subagent_type}
- **Timestamp**: {datetime.now(timezone.utc).isoformat()}
- **Model**: {model}
- **Duration**: {duration_ms}ms
- **Turns Used**: {turns_used}
- **Tools**: {tools_summary}

---

## Detailed Payload

{content}
"""
    file_path.write_text(artifact_md, encoding="utf-8")
    return file_path


def generate_handoff_brief(
    task: str,
    full_content: str,
    artifact_path: Path,
    tools_used: list[dict],
    duration_ms: int,
    turns_used: int,
    subagent_type: str,
) -> str:
    """Generate a structured, compact executive handoff brief for the orchestrator.

    Provides 70-90% context compression compared to raw multi-turn tool outputs
    while retaining critical decision-making data points, metrics, citations,
    and a direct pointer to the persisted artifact on disk.
    """
    if not full_content or not full_content.strip():
        return f"[No content generated by {subagent_type}]"

    lines = [line.strip() for line in full_content.split("\n") if line.strip()]

    bullet_points: list[str] = []
    citations: set[str] = set()
    metrics: list[str] = []

    url_regex = re.compile(r'\[([^\]]+)\]\((https?://[^\)]+)\)')

    for line in lines:
        # Check for URLs / citations
        for match in url_regex.finditer(line):
            title, url = match.group(1), match.group(2)
            if len(citations) < 5:
                citations.add(f"[{title}]({url})")

        # Check for lines containing quantitative figures
        if any(char in line for char in ["$", "%", "ms", "GB", "TB", "GHz", "TDP", "FLOPS", "million", "billion"]):
            if len(metrics) < 4 and len(line) < 200:
                metrics.append(line.lstrip("-*• ").strip())

        # Check for bullet points
        if line.startswith(("-", "*", "•", "1.", "2.", "3.", "4.", "5.")):
            clean_bullet = line.lstrip("-*• 0123456789.").strip()
            if len(clean_bullet) > 15 and clean_bullet not in bullet_points:
                bullet_points.append(clean_bullet)

    # If no bullet points found, extract leading substantive sentences
    if not bullet_points:
        substantive = [l for l in lines if not l.startswith("#") and len(l) > 30]
        bullet_points = substantive[:4]

    selected_bullets = bullet_points[:6]
    findings_block = (
        "\n".join(f"• {b}" for b in selected_bullets)
        if selected_bullets
        else "• Research completed. See artifact for full details."
    )

    sections = [
        f"**Key Findings:**\n{findings_block}",
    ]

    if metrics:
        unique_metrics = [m for m in metrics if not any(m in b for b in selected_bullets)][:3]
        if unique_metrics:
            sections.append(f"**Verified Metrics:**\n" + "\n".join(f"• {m}" for m in unique_metrics))

    if citations:
        sections.append(f"**Verified Sources:**\n" + ", ".join(sorted(citations)))

    try:
        rel_path = artifact_path.relative_to(Path(settings.agent_workspace))
    except Exception:
        rel_path = artifact_path

    sections.append(f"📁 **Full Artifact Stored At:** `{rel_path}`")

    brief_text = "\n\n".join(sections)

    if len(brief_text) > 1400:
        brief_text = brief_text[:1390].rsplit("\n", 1)[0] + f"\n\n📁 **Full Artifact Stored At:** `{rel_path}`"

    return brief_text


def parse_subagent_file(agent_file: Path) -> SubAgentDefinition | None:
    """Parse YAML frontmatter and markdown body from an agent definition file."""
    try:
        content = agent_file.read_text(encoding="utf-8")
        match = re.match(r"^---\s*\n(.*?)\n---\s*\n(.*)$", content, re.DOTALL)
        if not match:
            return None

        frontmatter_raw, prompt_body = match.groups()
        meta: dict[str, Any] = {}
        try:
            parsed = yaml.safe_load(frontmatter_raw)
            if isinstance(parsed, dict):
                meta = parsed
        except Exception:
            for line in frontmatter_raw.strip().split("\n"):
                if ":" in line:
                    k, v = line.split(":", 1)
                    meta[k.strip()] = v.strip()

        name = str(meta.get("name", agent_file.stem)).strip().lower()
        description = str(meta.get("description", f"Specialized subagent for {name}")).strip()

        tools_val = meta.get("tools", "")
        if isinstance(tools_val, list):
            tools = [str(t).strip() for t in tools_val if str(t).strip()]
        elif isinstance(tools_val, str) and tools_val.strip():
            tools = [t.strip() for t in tools_val.split(",") if t.strip()]
        else:
            tools = list(SUBAGENT_ALLOWED_TOOLS)

        disallowed_val = meta.get("disallowedTools", meta.get("disallowed_tools", ""))
        if isinstance(disallowed_val, list):
            disallowed = [str(t).strip() for t in disallowed_val if str(t).strip()]
        elif isinstance(disallowed_val, str) and disallowed_val.strip():
            disallowed = [t.strip() for t in disallowed_val.split(",") if t.strip()]
        else:
            disallowed = []

        model_val = meta.get("model")
        max_turns_val = meta.get("maxTurns", meta.get("max_turns", DEFAULT_MAX_TURNS))
        try:
            max_turns_int = int(max_turns_val)
        except (ValueError, TypeError):
            max_turns_int = DEFAULT_MAX_TURNS

        body = prompt_body.strip() or SUBAGENT_SYSTEM_PROMPT

        return SubAgentDefinition(
            name=name,
            description=description,
            tools=tools,
            disallowed_tools=disallowed,
            model=str(model_val) if model_val else None,
            max_turns=max_turns_int,
            system_prompt=body,
        )
    except Exception as e:
        logger.warn("failed_parsing_subagent_file", path=str(agent_file), error=str(e))
        return None


def load_subagent_definitions() -> dict[str, SubAgentDefinition]:
    """Scan and load subagent definitions from project and user agent directories.

    Follows Claude Code scope priority:
    1. Project agents: .maidere/agents/ and agents/
    2. User agents: ~/.maidere/agents/
    3. Built-in agents: researcher, code-reviewer, general-purpose
    """
    definitions: dict[str, SubAgentDefinition] = {}

    builtins = [
        SubAgentDefinition(
            name="researcher",
            description="Fast, read-only research agent for web search, documentation, and technical facts.",
            tools=["web_search", "browser", "read_file", "list_dir"],
            max_turns=6,
            system_prompt=SUBAGENT_SYSTEM_PROMPT,
        ),
        SubAgentDefinition(
            name="explore",
            description="Alias for researcher. Fast, read-only codebase and web exploration agent.",
            tools=["web_search", "browser", "read_file", "list_dir"],
            max_turns=6,
            system_prompt=SUBAGENT_SYSTEM_PROMPT,
        ),
        SubAgentDefinition(
            name="plan",
            description="Software architecture and implementation planning specialist.",
            tools=["read_file", "list_dir", "web_search", "browser"],
            max_turns=6,
            system_prompt=(
                "You are an expert software architect and implementation planning sub-agent.\n"
                "Your role is to deeply analyze software projects, examine codebases, and formulate comprehensive, "
                "production-ready implementation plans, system architectures, and technical designs.\n\n"
                "CRITICAL DIRECTIVES:\n"
                "1. READ-ONLY SCOPE: You have read-only access to inspect files (read_file, list_dir), search documentation (web_search), and browse reference architecture pages (browser). You DO NOT write files or run shell commands.\n"
                "2. DISCOVERY FIRST: Your first action MUST be using tools to inspect relevant project files, directory layouts, configuration files, or documentation. Never guess codebase layout.\n"
                "3. CONCRETE & STEP-BY-STEP: Once you have gathered sufficient codebase context, formulate an actionable plan with system design, components, step-by-step implementation sequence, edge cases, and test plan.\n"
                "4. NO PLACEHOLDERS: Provide concrete file paths, class names, method signatures, and schema designs.\n\n"
                "Planning Task:\n{task}"
            ),
        ),
        SubAgentDefinition(
            name="code-reviewer",
            description="Scans code files and suggests improvements for quality, security, and performance.",
            tools=["read_file", "list_dir"],
            max_turns=4,
            system_prompt=(
                "You are a code improvement and security review specialist.\n"
                "Inspect the specified code files and provide clear, actionable feedback with code blocks.\n\n"
                "Review Task:\n{task}"
            ),
        ),
        SubAgentDefinition(
            name="general-purpose",
            description="General-purpose subagent for multi-step exploration and problem-solving.",
            tools=["web_search", "browser", "read_file", "list_dir"],
            max_turns=6,
            system_prompt=SUBAGENT_SYSTEM_PROMPT,
        ),
        SubAgentDefinition(
            name="reviewer",
            description="Adversarial code, diff, and plan reviewer. Evaluates security, regressions, and quality.",
            tools=["read_file", "list_dir"],
            max_turns=4,
            system_prompt=(
                "You are an expert adversarial code, diff, and architectural reviewer.\n"
                "Your role is to critically analyze code changes, implementation plans, and git diffs.\n"
                "You MUST look for regressions, missing tests, security risks, syntax errors, and edge cases.\n"
                "Provide clear, numbered findings categorized by severity (BLOCKER, WARNING, SUGGESTION).\n\n"
                "Review Task:\n{task}"
            ),
        ),
        SubAgentDefinition(
            name="staffer",
            description="Fast, general-purpose staffer for scoped multi-step task execution.",
            tools=["web_search", "browser", "read_file", "list_dir"],
            max_turns=6,
            system_prompt=SUBAGENT_SYSTEM_PROMPT,
        ),
        SubAgentDefinition(
            name="tech-hardware",
            description="Enterprise hardware, compute infrastructure, GPU clusters, and high-performance computing (HPC) research specialist.",
            tools=["web_search", "browser", "read_file", "list_dir"],
            max_turns=6,
            system_prompt=TECH_HARDWARE_SYSTEM_PROMPT,
        ),
    ]
    for b in builtins:
        definitions[b.name.lower()] = b

    scan_dirs = [
        Path.home() / ".maidere" / "agents",
        Path.cwd() / ".maidere" / "agents",
        Path.cwd() / "agents",
    ]

    for d in scan_dirs:
        if d.is_dir():
            for f in sorted(d.glob("*.md")):
                parsed = parse_subagent_file(f)
                if parsed:
                    definitions[parsed.name.lower()] = parsed

    return definitions


def normalize_subagent_url(url: str) -> str:
    """Normalize URL by stripping trailing slash, hash fragments, and whitespace."""
    if not url:
        return ""
    clean = url.strip().split("#")[0].rstrip("/")
    return clean.lower()


def is_search_query_redundant(new_query: str, past_queries: list[str], threshold: float = 0.60) -> bool:
    """Check if a new search query has high lexical overlap with previous queries in this session."""
    stop_words = {
        "the", "and", "for", "with", "from", "that", "this", "what", "how", "who",
        "which", "are", "was", "were", "about", "into", "key", "programs", "during",
    }
    new_words = {w for w in re.findall(r'\w+', new_query.lower()) if len(w) > 2 and w not in stop_words}
    if not new_words:
        return False
    for past in past_queries:
        past_words = {w for w in re.findall(r'\w+', past.lower()) if len(w) > 2 and w not in stop_words}
        if not past_words:
            continue
        intersection = new_words & past_words
        min_len = min(len(new_words), len(past_words))
        if min_len > 0 and (len(intersection) / min_len) >= threshold:
            return True
    return False


# --- Core Execution Engine ---

async def run_subagent(
    task: str,
    subagent_type: str = "researcher",
    max_turns: int | None = None,
    model: str | None = None,
    subagent_index: int = 1,
    total_subagents: int = 1,
    num_ctx: int | None = None,
) -> SubAgentResult:
    """Run a focused sub-agent with its own fresh context window.

    Follows Claude Code subagent specification:
    - Runs in an isolated context window
    - Receives only its dedicated system prompt and task
    - Restricted to allowed read-only tools
    - Hard turn limit with dense data extraction
    - Persists tagged memory upon completion

    Args:
        task: Specific subagent task description.
        subagent_type: Type/name of subagent (e.g. 'researcher', 'code-reviewer').
        max_turns: Max think→act cycles (clamped to MAX_TURNS_HARD_CAP).
        model: Ollama model to use (defaults to primary 7B).
        subagent_index: 1-based index for progress reporting.
        total_subagents: Total sub-agents in this delegation batch.

    Returns:
        SubAgentResult with the summary and execution metadata.
    """
    start_time = time.perf_counter()

    definitions = load_subagent_definitions()
    resolved_type = subagent_type.lower().strip() if subagent_type else "researcher"
    agent_def = definitions.get(resolved_type) or definitions.get("researcher") or next(iter(definitions.values()))

    # Resolve model
    raw_model = model or agent_def.model or settings.ollama_model
    target_model = settings.ollama_model if raw_model == "inherit" else raw_model

    # Resolve max turns
    turn_limit = max_turns or agent_def.max_turns or DEFAULT_MAX_TURNS
    effective_max_turns = max(1, min(turn_limit, MAX_TURNS_HARD_CAP))

    tools_used: list[dict] = []
    turns_used = 0
    visited_urls: set[str] = set()
    executed_search_queries: list[str] = []
    retrieved_search_urls: set[str] = set()

    # Load allowed tools (strictly read-only)
    from tools.registry import get_tool

    effective_tool_names = set(agent_def.tools)
    if agent_def.disallowed_tools:
        effective_tool_names -= set(agent_def.disallowed_tools)
    effective_tool_names &= SUBAGENT_ALLOWED_TOOLS  # enforce read-only boundary

    allowed_tools: dict[str, Any] = {}
    for tool_name in sorted(effective_tool_names):
        tool = get_tool(tool_name)
        if tool is not None:
            allowed_tools[tool_name] = tool

    tool_schemas = [tool.to_ollama_schema() for tool in allowed_tools.values()]

    # Build fresh conversation — completely independent from parent context
    if "{task}" in agent_def.system_prompt:
        system_content = agent_def.system_prompt.format(task=task)
    else:
        system_content = f"{agent_def.system_prompt}\n\nTask:\n{task}"

    messages: list[dict] = [
        {"role": "system", "content": system_content},
        {"role": "user", "content": f"Execute this task now: {task}"},
    ]

    final_content = ""

    # Emit progress: started
    if _progress_callback:
        try:
            await _progress_callback({
                "type": "subagent_started",
                "task": task,
                "subagent_type": agent_def.name,
                "index": subagent_index,
                "total": total_subagents,
            })
        except Exception:
            pass

    try:
        for turn in range(effective_max_turns):
            turns_used = turn + 1

            # Think: call LLM
            response = await llm.chat(messages, tools=tool_schemas, model=target_model, num_ctx=num_ctx)
            message = response.get("message", {})
            ai_content = message.get("content", "")
            raw_tool_calls = message.get("tool_calls", [])

            if not raw_tool_calls:
                # No tool calls — sub-agent is done
                final_content = ai_content
                break

            # Append assistant message with tool calls to conversation
            msg_dict: dict = {"role": "assistant", "content": ai_content or ""}
            msg_dict["tool_calls"] = raw_tool_calls
            messages.append(msg_dict)

            # Act: execute each tool call sequentially (read-only only)
            for tc in raw_tool_calls:
                func = tc.get("function", {})
                tool_name = func.get("name", "")
                tool_args = func.get("arguments", {})

                # Normalize string arguments to dict
                if isinstance(tool_args, str):
                    try:
                        tool_args = json.loads(tool_args)
                    except Exception:
                        tool_args = {"query": tool_args}

                # Emit progress: tool started
                if _progress_callback:
                    try:
                        await _progress_callback({
                            "type": "subagent_tool_started",
                            "tool_name": tool_name,
                            "args": tool_args,
                            "subagent_type": agent_def.name,
                            "subagent_index": subagent_index,
                            "total": total_subagents,
                        })
                    except Exception:
                        pass

                tool_start = time.perf_counter()

                # Security: strictly enforce read-only allowlist
                if tool_name not in allowed_tools:
                    result = (
                        f"Error: Tool '{tool_name}' is not available to sub-agents. "
                        f"You can only use: {', '.join(sorted(SUBAGENT_ALLOWED_TOOLS))}."
                    )
                    tool_success = False
                elif tool_name == "browser":
                    raw_url = str(tool_args.get("url") or tool_args.get("target") or "").strip()
                    norm_url = normalize_subagent_url(raw_url)
                    if norm_url and norm_url in visited_urls:
                        result = (
                            f"[Redundant URL Notice: '{raw_url}' has already been browsed in this research session. "
                            f"To maintain research diversity and prevent redundant reading, query a different source or call web_search with a new diverse angle.]"
                        )
                        tool_success = True
                    else:
                        try:
                            tool_instance = allowed_tools[tool_name]
                            result = await tool_instance.execute(**tool_args)
                            tool_success = not str(result).startswith("Error:")
                            if tool_success and norm_url:
                                visited_urls.add(norm_url)
                        except Exception as e:
                            result = f"Error: {str(e)}"
                            tool_success = False
                elif tool_name == "web_search":
                    query_str = str(tool_args.get("query") or "").strip()
                    is_redundant_q = is_search_query_redundant(query_str, executed_search_queries)
                    try:
                        tool_instance = allowed_tools[tool_name]
                        result = await tool_instance.execute(**tool_args)
                        tool_success = not str(result).startswith("Error:")
                        if tool_success:
                            found_urls = re.findall(r'https?://[^\s\)\"\']+', str(result))
                            norm_found = [normalize_subagent_url(u) for u in found_urls if u]
                            repeated_urls = [u for u in norm_found if u in retrieved_search_urls]
                            has_url_redundancy = bool(norm_found and len(repeated_urls) >= max(1, len(norm_found) // 2))

                            if is_redundant_q or has_url_redundancy:
                                result = (
                                    f"{result}\n\n"
                                    f"[Search Redundancy Notice: This query or its returned sources overlap with previous searches in this session. "
                                    f"Your next action MUST be highly diverse: target a completely different sub-topic/aspect, or proceed to synthesize your findings now. "
                                    f"Do NOT execute redundant queries.]"
                                )
                            retrieved_search_urls.update(norm_found)
                            if query_str:
                                executed_search_queries.append(query_str)
                    except Exception as e:
                        result = f"Error: {str(e)}"
                        tool_success = False
                else:
                    try:
                        tool_instance = allowed_tools[tool_name]
                        result = await tool_instance.execute(**tool_args)
                        tool_success = not str(result).startswith("Error:")
                    except Exception as e:
                        result = f"Error: {str(e)}"
                        tool_success = False

                tool_duration_ms = int((time.perf_counter() - tool_start) * 1000)

                tools_used.append({
                    "tool_name": tool_name,
                    "args": tool_args,
                    "success": tool_success,
                })

                messages.append({"role": "tool", "content": str(result)})

                # Emit progress: tool executed
                if _progress_callback:
                    try:
                        res_str = str(result)
                        preview = res_str[:120].replace("\n", " ").strip()
                        await _progress_callback({
                            "type": "subagent_tool",
                            "tool_name": tool_name,
                            "args": tool_args,
                            "subagent_type": agent_def.name,
                            "subagent_index": subagent_index,
                            "total": total_subagents,
                            "duration_ms": tool_duration_ms,
                            "success": tool_success,
                            "result_preview": preview,
                        })
                    except Exception:
                        pass

            # Context guard: if approaching limit, force conclusion
            total_chars = sum(len(m.get("content", "") or "") for m in messages)
            char_limit = settings.agent_token_limit * settings.agent_chars_per_token
            if total_chars > int(char_limit * 0.80):
                messages.append({
                    "role": "user",
                    "content": (
                        "Context limit approaching. Provide your FINAL research summary "
                        "NOW based on everything you've gathered so far. Use the dense "
                        "data payload format with specific facts and sources."
                    ),
                })
        else:
            # Exhausted all turns — force a final summary with no tools
            messages.append({
                "role": "user",
                "content": (
                    "Maximum research turns reached. Provide your FINAL research summary "
                    "NOW. Include all specific data points, metrics, and sources you found."
                ),
            })
            response = await llm.chat(messages, tools=None, model=target_model, num_ctx=num_ctx)
            final_content = response.get("message", {}).get("content", "")

        duration_ms = int((time.perf_counter() - start_time) * 1000)

        # 1. Prune old artifacts to keep directory within retention limits (100 files / 14 days)
        prune_artifacts()

        # 2. Write full output to collision-proof artifact file
        artifact_file = write_subagent_artifact(
            task=task,
            content=final_content,
            subagent_type=agent_def.name,
            tools_used=tools_used,
            duration_ms=duration_ms,
            turns_used=turns_used,
            model=target_model,
        )

        # 3. Generate compact adaptive handoff brief for orchestrator context
        brief = generate_handoff_brief(
            task=task,
            full_content=final_content,
            artifact_path=artifact_file,
            tools_used=tools_used,
            duration_ms=duration_ms,
            turns_used=turns_used,
            subagent_type=agent_def.name,
        )

        # Store result in vector memory with strict tagging (user requirement)
        await _store_subagent_memory(task, final_content, subagent_type=agent_def.name)

        if _progress_callback:
            try:
                await _progress_callback({
                    "type": "agent_log",
                    "text": f"[MEMORY] Sub-Agent [{agent_def.name}] persisted research findings to vector database",
                    "level": "info",
                })
            except Exception:
                pass

        await logger.ainfo(
            "subagent_completed",
            task=task[:100],
            subagent_type=agent_def.name,
            turns_used=turns_used,
            tools_count=len(tools_used),
            duration_ms=duration_ms,
            summary_length=len(final_content),
            artifact_path=str(artifact_file),
        )

        # Emit progress: completed
        if _progress_callback:
            try:
                await _progress_callback({
                    "type": "subagent_completed",
                    "index": subagent_index,
                    "total": total_subagents,
                    "subagent_type": agent_def.name,
                    "task": task[:100],
                    "duration_ms": duration_ms,
                    "summary_preview": final_content[:150] if final_content else "",
                    "artifact_path": str(artifact_file),
                })
            except Exception:
                pass

        return SubAgentResult(
            task=task,
            summary=final_content,
            subagent_type=agent_def.name,
            tools_used=tools_used,
            duration_ms=duration_ms,
            success=True,
            turns_used=turns_used,
            artifact_path=str(artifact_file),
            brief=brief,
        )

    except Exception as e:
        duration_ms = int((time.perf_counter() - start_time) * 1000)
        await logger.aerror("subagent_failed", task=task[:100], subagent_type=agent_def.name, error=str(e))
        return SubAgentResult(
            task=task,
            summary=f"Sub-agent [{agent_def.name}] failed: {str(e)}",
            subagent_type=agent_def.name,
            tools_used=tools_used,
            duration_ms=duration_ms,
            success=False,
            turns_used=turns_used,
            artifact_path="",
            brief=f"Sub-agent [{agent_def.name}] failed: {str(e)}",
        )


async def _store_subagent_memory(task: str, summary: str, subagent_type: str = "researcher") -> None:
    """Store sub-agent result in vector memory with strict source tagging.

    Tags prevent the LLM from treating older summaries as absolute truth —
    the tag tells it 'this is a synthesized summary from a past sub-agent'.
    """
    if not summary or not summary.strip() or len(summary.strip()) < 20:
        return

    try:
        from core.db import get_db
        from core.memory import store_memory

        tagged_content = (
            f"[source: subagent_{subagent_type}] [task: {task[:120]}] "
            f"{summary[:2000]}"
        )

        db = await get_db(settings.db_path)
        try:
            await store_memory(db, content=tagged_content, session_id="subagent")
        finally:
            await db.close()
    except Exception as e:
        await logger.awarn("subagent_memory_store_failed", error=str(e))
