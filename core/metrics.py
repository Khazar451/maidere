"""Custom Prometheus metrics and instrumentation for Maidere."""

from prometheus_client import Counter, Gauge, Histogram

# ---------------------------------------------------------------------------
# Tool Execution Metrics
# ---------------------------------------------------------------------------

TOOL_CALLS_TOTAL = Counter(
    "maidere_tool_calls_total",
    "Total number of tool invocations executed by the agent",
    ["tool_name", "status"],  # status: "success" | "error"
)

TOOL_DURATION_SECONDS = Histogram(
    "maidere_tool_duration_seconds",
    "Execution duration of tools in seconds",
    ["tool_name"],
    buckets=(0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0),
)

# ---------------------------------------------------------------------------
# LLM Inference Metrics
# ---------------------------------------------------------------------------

LLM_REQUESTS_TOTAL = Counter(
    "maidere_llm_requests_total",
    "Total number of LLM inference calls",
    ["model", "status"],  # status: "success" | "error"
)

LLM_DURATION_SECONDS = Histogram(
    "maidere_llm_duration_seconds",
    "Duration of LLM inference requests in seconds",
    ["model"],
    buckets=(0.1, 0.25, 0.5, 1.0, 2.0, 5.0, 10.0, 20.0, 30.0, 60.0),
)

LLM_TOKENS_TOTAL = Counter(
    "maidere_llm_tokens_total",
    "Total tokens evaluated by the LLM",
    ["model", "token_type"],  # token_type: "prompt" | "completion"
)

# ---------------------------------------------------------------------------
# Vector Memory Retrieval Metrics
# ---------------------------------------------------------------------------

MEMORY_RECALLS_TOTAL = Counter(
    "maidere_memory_recalls_total",
    "Total number of vector memory retrieval queries executed",
    ["status"],  # status: "success" | "empty" | "error"
)

MEMORY_RECALL_DURATION_SECONDS = Histogram(
    "maidere_memory_recall_duration_seconds",
    "Duration of vector memory recall operations in seconds",
    buckets=(0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0),
)

# ---------------------------------------------------------------------------
# Session / Thread Gauges & Routing
# ---------------------------------------------------------------------------

ACTIVE_SESSIONS = Gauge(
    "maidere_active_sessions",
    "Number of active conversations / threads",
)

ROUTER_DECISIONS_TOTAL = Counter(
    "maidere_router_decisions_total",
    "Total number of model routing decisions made",
    ["model", "complexity", "reason"],
)

