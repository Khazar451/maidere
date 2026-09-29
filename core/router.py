"""Dynamic dual-LLM model router and task complexity classifier for Maidere."""

import re
import time
from enum import Enum
from typing import Any

import httpx
import structlog

from core.config import settings

logger = structlog.get_logger()


class TaskComplexity(str, Enum):
    """Task complexity levels for model routing."""

    SIMPLE = "simple"
    COMPLEX = "complex"
    REASONING = "reasoning"


# Keywords and patterns indicative of deep reasoning, mathematical proofs, and logic puzzles
_REASONING_PATTERNS = [
    r"\b(prove|proof|theorem|derive|derivation|lemma|axiom)\b",
    r"\b(solve\s+for|equation|calculus|integral|derivative|algebra)\b",
    r"\b(logic\s+puzzle|riddle|brain\s*teaser|counter\s*factual)\b",
    r"\b(step-by-step\s+(reasoning|proof|deduction|analysis))\b",
    r"\b(think\s+(deeply|through|step-by-step)|chain\s+of\s+thought)\b",
    r"\b(implications\s+of|game\s+theory|probabilistic|combinatorics)\b",
    r"\b(philosophical\s+analysis|formal\s+logic|deduce|deductive)\b",
    r"\b(tie-breaker|initial\s+state|state\s+tracking|state\s+machine)\b",
    r"\b(closed\s+loop|execution\s+cycles?|protocol\s+for\s+\d+\s+(days?|cycles?|steps?))\b",
    r"\b(token\s+exchange\s+network|inventory\s+system|container\s+[a-z]|node\s+[a-z])\b",
]

# Keywords and patterns indicative of complex multi-step reasoning, tool execution, or coding
_COMPLEX_PATTERNS = [
    r"\b(write|create|edit|modify|delete|remove|read)\s+(file|script|code|function|class|module|program)\b",
    r"\b(execute|run|test|benchmark|profile|debug|fix|compile|build)\b",
    r"\b(python|bash|sh|shell|sql|javascript|typescript|docker|git|api)\b",
    r"\b(search|browse|scrape|fetch|download|curl|http|url|website)\b",
    r"\b(schedule|cron|reminder|timer|task|job)\b",
    r"\b(data|csv|pandas|dataframe|analysis|statistics|plot|graph)\b",
    r"\b(research|investigate|obsidian|synthesize|decompose)\b",
    r"\b(step[s]?-by-step|algorithm|refactor|architecture|database)\b",
    r"```",  # code snippets in user prompt
]

_SIMPLE_GREETINGS = {
    "hello", "hi", "hey", "good morning", "good evening", "good afternoon",
    "how are you", "who are you", "what can you do", "what is your name",
    "thanks", "thank you", "bye", "goodbye", "help", "ping", "test",
}

# Cache for available Ollama models
_MODELS_CACHE: list[str] = []
_LAST_CACHE_TIME: float = 0.0
_CACHE_TTL: float = 30.0  # seconds


def classify_complexity(query: str, history: list[Any] | None = None) -> TaskComplexity:
    """Classify the complexity of a user query based on intent, triggers, and context."""
    if not query or not query.strip():
        return TaskComplexity.SIMPLE

    clean_query = query.strip().lower()

    # 1. Exact greeting matches or very short conversational statements
    if clean_query in _SIMPLE_GREETINGS:
        return TaskComplexity.SIMPLE

    # 2. Check for reasoning patterns (math proofs, logic, deep deduction)
    for pattern in _REASONING_PATTERNS:
        if re.search(pattern, clean_query):
            return TaskComplexity.REASONING

    # 3. Check for complex patterns (coding, tools, filesystem, search, analysis)
    for pattern in _COMPLEX_PATTERNS:
        if re.search(pattern, clean_query):
            return TaskComplexity.COMPLEX

    # 4. Check for previous tool calls in recent history
    if history:
        for msg in history[-4:]:
            tool_calls = getattr(msg, "tool_calls", None)
            if tool_calls and len(tool_calls) > 0:
                return TaskComplexity.COMPLEX
            if getattr(msg, "type", None) == "tool":
                return TaskComplexity.COMPLEX

    # 5. Long queries (> 400 chars) typically require deeper synthesis
    if len(query) > 400:
        return TaskComplexity.COMPLEX

    return TaskComplexity.SIMPLE


async def get_available_models(force_refresh: bool = False) -> list[str]:
    """Retrieve installed Ollama models and configured cloud AI models with caching."""
    global _MODELS_CACHE, _LAST_CACHE_TIME

    cloud_models: list[str] = []
    api_key, _, default_cloud_model = settings.get_effective_cloud_config()
    if api_key:
        for cm in (default_cloud_model, "nvidia/nemotron-3-ultra-550b-a55b", "meta/llama-3.3-70b-instruct"):
            if cm and cm not in cloud_models:
                cloud_models.append(cm)

    now = time.time()
    if not force_refresh and _MODELS_CACHE and (now - _LAST_CACHE_TIME < _CACHE_TTL):
        return cloud_models + [m for m in _MODELS_CACHE if m not in cloud_models]

    local_models = []
    try:
        async with httpx.AsyncClient(timeout=3.0) as client:
            res = await client.get(f"{settings.ollama_url}/api/tags")
            if res.status_code == 200:
                tags = res.json().get("models", [])
                models = [m["name"] for m in tags if m.get("name")]
                if models:
                    _MODELS_CACHE = models
                    _LAST_CACHE_TIME = now
                    local_models = models
    except Exception:
        pass

    if not local_models:
        thinking_model = getattr(settings, "ollama_thinking_model", "deepseek-r1:7b")
        defaults = [
            settings.ollama_model,
            settings.ollama_fast_model,
            thinking_model,
            "deepseek-r1:8b",
        ]
        for d in defaults:
            if d not in local_models:
                local_models.append(d)

    return cloud_models + [m for m in local_models if m not in cloud_models]


async def select_model(
    requested_model: str | None = None,
    query: str = "",
    history: list[Any] | None = None,
    thinking_mode: bool = False,
    deep_reasoning: bool = False,
    *args: Any,
    **kwargs: Any,
) -> tuple[str, str, TaskComplexity]:
    """Select the optimal model for an execution turn.

    Returns:
        (selected_model_name, routing_reason, complexity)
    """
    complexity = (
        TaskComplexity.REASONING
        if (thinking_mode or deep_reasoning)
        else classify_complexity(query, history)
    )

    # 1. Explicit thinking mode or deep reasoning mode override
    available = await get_available_models()
    thinking_model = getattr(settings, "ollama_thinking_model", "deepseek-r1:7b")

    if deep_reasoning:
        if requested_model and requested_model.lower() not in ("auto", "none", ""):
            if requested_model.lower() != settings.ollama_model.lower():
                return requested_model, "user_override_deep_reasoning", TaskComplexity.REASONING
        matched = None
        if any(thinking_model in m or "deepseek-r1" in m for m in available):
            matched = next((m for m in available if thinking_model in m or "deepseek-r1" in m), None)
        elif any("nemotron" in m.lower() for m in available):
            matched = next((m for m in available if "nemotron" in m.lower()), None)
        if matched:
            return matched, "deep_reasoning_model", TaskComplexity.REASONING
        primary = settings.ollama_model if (not available or settings.ollama_model in available) else available[0]
        return primary, "deep_reasoning_primary", TaskComplexity.REASONING

    if thinking_mode:
        if requested_model and requested_model.lower() not in ("auto", "none", ""):
            if requested_model.lower() != settings.ollama_model.lower():
                return requested_model, "user_override_thinking", TaskComplexity.REASONING
        matched = None
        if any(thinking_model in m or "deepseek-r1" in m for m in available):
            matched = next((m for m in available if thinking_model in m or "deepseek-r1" in m), None)
        elif any("nemotron" in m.lower() for m in available):
            matched = next((m for m in available if "nemotron" in m.lower()), None)
        if matched:
            return matched, "forced_thinking", TaskComplexity.REASONING
        # If DeepSeek or Nemotron is not available, safely use the primary 7B model
        primary = settings.ollama_model if (not available or settings.ollama_model in available) else available[0]
        await logger.awarn(
            "deepseek_not_available",
            message=f"thinking_mode requested but '{thinking_model}' is not installed in Ollama. "
                    f"Falling back to '{primary}'. Run 'ollama pull {thinking_model}' to enable native thinking.",
            thinking_model=thinking_model,
            fallback_model=primary,
        )
        return primary, "thinking_mode_primary", TaskComplexity.REASONING



    # 2. User manual override
    if requested_model and requested_model.lower() not in ("auto", "none", ""):
        return requested_model, "user_override", complexity

    # 3. If routing disabled, use default primary model
    if not settings.enable_model_routing:
        return settings.ollama_model, "routing_disabled", complexity

    # 4. Dynamic Smart Routing
    if complexity == TaskComplexity.REASONING:
        thinking_available = any(thinking_model in m or "deepseek-r1" in m for m in available)
        if thinking_available:
            matched = next((m for m in available if thinking_model in m or "deepseek-r1" in m), thinking_model)
            return matched, "auto_reasoning", complexity
        else:
            await logger.awarn(
                "deepseek_not_available",
                message=f"REASONING query detected but '{thinking_model}' is not installed in Ollama. "
                        f"Falling back to '{settings.ollama_model}'. Run 'ollama pull {thinking_model}' to enable deep reasoning.",
                thinking_model=thinking_model,
                fallback_model=settings.ollama_model,
            )
            return settings.ollama_model, "fallback_primary", complexity

    if complexity == TaskComplexity.SIMPLE:
        fast_available = any(settings.ollama_fast_model in m for m in available)
        if fast_available:
            return settings.ollama_fast_model, "auto_simple", complexity
        else:
            return settings.ollama_model, "fallback_primary", complexity

    # Complex tasks route to primary 7B model
    return settings.ollama_model, "auto_complex", complexity

