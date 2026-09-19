# Maidere (v4.1) — Production Architecture & Technical Reference Manual

> **A 100% Self-Hosted, Zero-API-Cost Autonomous AI Agent Designed for Consumer Hardware (NVIDIA RTX 5060 8GB VRAM, 16GB System RAM on Linux Mint / Ubuntu).**

---

## 1. Executive Summary & Hardware Budget Architecture

### Core Mission
Maidere is an autonomous local AI agent engineered for sustained local execution on consumer workstations and laptops. It eliminates cloud API subscriptions, data leakage, and GPU model-swapping latency by maintaining a strictly partitioned memory budget: the primary 7B LLM stays pinned permanently in GPU VRAM, while semantic embeddings, vector index searches, and tool runtimes execute entirely on host CPU and system RAM.

### Subsystem Topology
```
┌────────────────────────────────────────────────────────────────────────────────────────┐
│                                 HOST OS (Linux Mint / Ubuntu)                          │
├─────────────────────────────────────────┬──────────────────────────────────────────────┤
│               GPU VRAM (8.0 GB)         │             SYSTEM RAM (16.0 GB)             │
│ ┌─────────────────────────────────────┐ │ ┌──────────────────────────────────────────┐ │
│ │ Ollama Daemon                       │ │ │ FastAPI Server (Uvicorn Async Worker)    │ │
│ │  - Primary: qwen2.5:7b-instruct     │ │ │  - REST Endpoints & WebSocket Handler    │ │
│ │    (Q4_K_M pinned: ~5.5 GB)         │ │ │  - Dynamic Dual-Model Router             │ │
│ │  - Fast: qwen2.5:3b (on-demand)     │ │ │  - Prometheus Instrumentator (/metrics)  │ │
│ │  - Locked KV Cache (num_ctx: 8192)  │ │ ├──────────────────────────────────────────┤ │
│ │    (~1.0 GB allocation)             │ │ │ LangGraph Agent Core                     │ │
│ │  - Compositor & CUDA Buffers (~1.5G)│ │ │  - trim -> remember -> think -> act loop │ │
│ └─────────────────────────────────────┘ │ │  - Symmetrical Filesystem Sandbox        │ │
│                                         │ │  - Safe Subprocess & Shell Allowlist     │ │
│                                         │ ├──────────────────────────────────────────┤ │
│                                         │ │ In-Process CPU & Storage Subsystems      │ │
│                                         │ │  - FastEmbed (bge-small-en-v1.5 on CPU)  │ │
│                                         │ │  - SQLite Database (WAL Mode Enabled)    │ │
│                                         │ │    * sqlite-vec (vec0 Virtual Table)     │ │
│                                         │ │    * LangGraph Checkpoints & Audit Log   │ │
│                                         │ │    * Persistent APScheduler DB Store     │ │
│                                         │ ├──────────────────────────────────────────┤ │
│                                         │ │ Docker Bridge Services                   │ │
│                                         │ │  - SearXNG Metasearch (Port 8080, 256MB) │ │
│                                         │ │  - Prometheus TSDB (Port 9090, 256MB)    │ │
│                                         │ │  - Grafana Dashboards (Port 3000, 256MB) │ │
│                                         │ └──────────────────────────────────────────┘ │
└─────────────────────────────────────────┴──────────────────────────────────────────────┘
```

### Hardware Allocation Matrix

| Subsystem / Resource | Component / Process | Allocated Size | Execution Target | Rationale & Guardrails |
| :--- | :--- | :--- | :--- | :--- |
| **GPU VRAM (8.0 GB Budget)** | Primary LLM (`qwen2.5:7b-instruct`) | ~5.5 GB | NVIDIA RTX 5060 (CUDA) | Pinned 24/7 in 4-bit quantization (Q4_K_M); zero cold-swapping. |
| | KV Cache Allocation (`num_ctx: 8192`) | ~1.0 GB | NVIDIA RTX 5060 (CUDA) | Hard limit in Ollama options; prevents VRAM fragmentation and OOM. |
| | Desktop / Display Compositor Buffer | ~0.5 GB | Linux X11 / Wayland | OS display output headroom. |
| | CUDA Driver & Runtime Headroom | ~1.0 GB | GPU Kernel Space | Dynamic CUDA allocation buffer. |
| **System RAM (16.0 GB Budget)** | Host OS & Kernel Services | ~2.5 GB | Host CPU | Base operating system footprint. |
| | Ollama Daemon & Fast Model (`qwen2.5:3b`) | ~2.0 GB | Host CPU / RAM | Ephemeral routing for high-throughput simple conversational turns. |
| | FastAPI + LangGraph Agent Core | ~0.8 GB | Python 3.12 Process | In-process agent runtime, checkpointers, and connection pools. |
| | FastEmbed CPU Engine (`bge-small-en-v1.5`) | ~0.4 GB | ONNX Runtime (CPU) | 384-dimensional dense embeddings; 0 MB GPU VRAM consumed. |
| | SQLite WAL & `sqlite-vec` Index | ~0.3 GB | In-Process Memory Cache | High-speed vector similarity and ACID checkpoints. |
| | Playwright Headless Browser (Ephemeral) | ~0.6 GB | Chromium Subprocess | Scrapes target URLs on demand; destroyed immediately after parse. |
| | SearXNG Metasearch Engine | ~0.25 GB | Docker Container | Local private metasearch engine capped at 256MB. |
| | Prometheus Monitoring TSDB | ~0.25 GB | Docker Container | Metric collection capped at 1GB disk / 7 days retention. |
| | Grafana Dashboard Server | ~0.25 GB | Docker Container | Real-time agent telemetry visualization. |
| | **Free / Available Headroom** | **~8.65 GB** | System Reserve | Buffers large file reads, code execution, and system caches. |

---

## 2. Five Core Non-Negotiable Architectural Patterns

### Pattern 1: SQLite WAL Mode & Concurrent Safe Connections
- **Problem**: Default SQLite database locking (`ROLLBACK` journal) causes `sqlite3.OperationalError: database is locked` when multiple asynchronous tasks (LangGraph checkpointer, memory embedding ingestion, tool audit logger, and API requests) write concurrently.
- **Pattern**: Force write-ahead logging (`WAL`), 5-second busy timeout handler, normal synchronization, and foreign key constraint enforcement across every SQLite connection.
- **Implementation**:
```python
# In core/db.py
async def get_db(db_path: str = "db/maidere.db") -> aiosqlite.Connection:
    db = await aiosqlite.connect(db_path)
    await db.execute("PRAGMA journal_mode = WAL;")
    await db.execute("PRAGMA busy_timeout = 5000;")
    await db.execute("PRAGMA synchronous = NORMAL;")
    await db.execute("PRAGMA foreign_keys = ON;")
    return db
```

### Pattern 2: Zero-VRAM CPU Embeddings via FastEmbed
- **Problem**: Running embedding models on the GPU forces Ollama to unload the 7B generative model or causes catastrophic CUDA Out-Of-Memory (OOM) faults on 8GB cards.
- **Pattern**: Leverage CPU-only ONNX Runtime execution via `fastembed` with `BAAI/bge-small-en-v1.5` (384 dimensions). Embedding generation takes ~5ms on modern CPUs while consuming exactly 0 MB of GPU VRAM.
- **Implementation**:
```python
# In core/embeddings.py
from fastembed import TextEmbedding

_model = TextEmbedding(model_name="BAAI/bge-small-en-v1.5")

def embed_one(text: str) -> list[float]:
    embeddings = list(_model.embed([text]))
    return embeddings[0].tolist()
```

### Pattern 3: In-Process Vector Search via `sqlite-vec`
- **Problem**: Standalone vector databases (Milvus, Qdrant, Pinecone, Chroma) add container bloat, network latency, and memory overhead.
- **Pattern**: Load the lightweight `sqlite-vec` C-extension directly into SQLite and maintain a native virtual table (`vec0`) supporting L2 distance metric vector queries.
- **Implementation**:
```python
# In core/memory.py
import struct
import sqlite_vec

def serialize_vector(vector: list[float]) -> bytes:
    return struct.pack(f"{len(vector)}f", *vector)

async def recall(db: aiosqlite.Connection, query: str, top_k: int = 5) -> list[dict]:
    vector = embed_one(query)
    rows = await db.execute_fetchall(
        """
        SELECT content, timestamp, distance
        FROM memories
        WHERE embedding MATCH ?
        ORDER BY distance
        LIMIT ?
        """,
        [serialize_vector(vector), top_k],
    )
    return [{"content": r[0], "timestamp": r[1], "distance": float(r[2])} for r in rows]
```

### Pattern 4: Playwright Zombie Process & Leak Prevention
- **Problem**: Long-running browser singletons leak memory over time, orphan zombie Chromium processes, and crash if a page hangs.
- **Pattern**: Enforce an ephemeral browser lifecycle. Every scraping task launches Chromium asynchronously, opens an isolated context, enforces a 15-second strict timeout, caps extracted HTML/text at 4,000 characters, and guarantees termination via `try...finally`.
- **Implementation**:
```python
# In tools/browser.py
from playwright.async_api import async_playwright

async def browse_url(url: str) -> str:
    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=True)
        context = await browser.new_context()
        page = await context.new_page()
        try:
            await page.goto(url, timeout=15000, wait_until="domcontentloaded")
            text = await page.inner_text("body")
            return text[:4000]
        finally:
            await context.close()
            await browser.close()
```

### Pattern 5: Context Window Guard & Token Trimming
- **Problem**: As conversation histories accumulate, exceeding 8,192 tokens causes Ollama to crash with VRAM allocation errors.
- **Pattern**: A dedicated `trim_messages` node in the LangGraph pipeline computes cumulative characters before calling LLM inference. If length exceeds 24,000 characters (~6,000 tokens), older messages are pruned from volatile graph state while preserving system messages intact. Pruned content remains preserved in SQLite and recoverable via semantic vector search.
- **Implementation**:
```python
# In core/agent.py
def trim_messages_node(state: AgentState) -> dict:
    messages = state["messages"]
    total_chars = sum(len(m.content) for m in messages if isinstance(m.content, str))
    if total_chars <= 24000:
        return {}
    
    removals = []
    chars_to_free = total_chars - 24000
    freed = 0
    for msg in messages:
        if isinstance(msg, SystemMessage):
            continue
        if freed >= chars_to_free:
            break
        freed += len(msg.content)
        removals.append(RemoveMessage(id=msg.id))
    return {"messages": removals}
```

---

## 3. Phase-by-Phase Technical Breakdown (Phases 1 to 8)

```
┌─────────┐     ┌──────────┐     ┌───────────┐     ┌──────────┐     ┌──────────┐
│ Phase 1 │ ──> │ Phase 2  │ ──> │  Phase 3  │ ──> │ Phase 4  │ ──> │ Phase 5  │
│ Engine  │     │ Sandbox  │     │  Memory   │     │ Runners  │     │ Search   │
└─────────┘     └──────────┘     └───────────┘     └──────────┘     └──────────┘
                                                                          │
┌─────────┐     ┌──────────┐     ┌───────────┐                            │
│ Phase 8 │ <── │ Phase 7  │ <── │  Phase 6  │ <──────────────────────────┘
│ Routing │     │ Telemetry│     │  Web UI   │
└─────────┘     └──────────┘     └───────────┘
```

### Phase 1: Core Engine & Lifecycle
- **FastAPI Lifespan**: Initialized directories (`db/`, `workspace/`, `logs/`, `static/`), verified WAL PRAGMA settings, compiled LangGraph `StateGraph`, and initialized connection pools.
- **Structured Logging**: Configured `structlog` to emit JSON Lines to `logs/maidere.jsonl` with ISO-8601 timestamps and execution context.
- **Direct Ollama Client**: Built `core/llm.py` with zero LiteLLM dependencies, using raw `httpx` async calls enforcing `num_ctx: 8192`.

### Phase 2: Security Sandbox & Multi-Step Agentic Loop
- **Symmetrical Path Sandbox**: Enforces that all file read/write operations resolve strictly within `workspace/` (`target.is_relative_to(WORKSPACE)`).
- **Safe Shell Executor**: Enforces `shell=False` in `subprocess.run()`, parses arguments with `shlex.split()`, wraps syntax errors, and validates command binaries against a strict allowlist (`python3`, `git`, `ls`, `grep`, `mkdir`, `pytest`, etc.).
- **Audit Logging**: Every tool invocation persists an audit record (`thread_id`, `tool_name`, `tool_args`, `tool_result`, `duration_ms`, `success`) to SQLite table `audit_log`.
- **Cyclic Agentic Loop**: Built LangGraph pipeline: `trim -> remember -> think -> (act -> think)* -> END`.

### Phase 3: Persistent Memory & Conversation Summarization
- **Vector Memory Ingestion**: Ingests every human message and assistant response into the `memories` table with 384-dimensional FastEmbed vectors.
- **Dynamic Context Injection**: `remember_node` performs semantic search on the latest user query, dynamically injecting top-K relevant memories into the prompt.
- **Automatic Summarization**: Triggers a background conversation summarization when message turns exceed 20, compressing dialogue state.

### Phase 4: Code Runner, Ephemeral Scraper & Scheduler
- **Isolated Code Runner**: Executes Python scripts directly in `workspace/` with auto-cleanup of temporary scripts.
- **Ephemeral Playwright Scraper**: Headless Chromium scraping with DOM inner-text parsing and strict timeout enforcement.
- **Persistent APScheduler**: Runs background jobs backed by SQLite `SQLAlchemyJobStore`, enabling recurring cron schedules (`interval`, `cron`, `date`).

### Phase 5: Local Web Search via SearXNG
- **Private Metasearch Integration**: Connects to local Dockerized SearXNG container on port 8080.
- **Synthesized Search & Browse Workflow**: Enables the agent to query SearXNG, extract top URLs, scrape deep technical content with Playwright, and cross-reference answers.
- **Graceful Offline Fallback**: Detects SearXNG downtime and emits `[Search Unavailable: SearXNG container is offline]`.

### Phase 6: Minimalist Developer Web UI
- **Design System**: Monochromatic slate/zinc palette (Dark `#09090b` / Light `#ffffff`), Geist/Geist Mono typography, and 1px subtle borders.
- **Thread State Persistence**: Full URL query parameter synchronization (`/?thread=<id>`) and `localStorage` persistence.
- **Real-Time Streaming**: WebSocket endpoint (`/ws/chat`) streaming progress stages, tool execution pills with execution time in ms (`[🟢 Ran tool: shell • 32ms • Success]`), and markdown parsing with copyable code blocks.

### Phase 7: Telemetry & Observability
- **Prometheus Metrics**: Exported at `GET /metrics` via `prometheus-fastapi-instrumentator`.
- **Custom Metrics**: Tracks tool execution counts and latency histograms (`maidere_tool_calls_total`, `maidere_tool_duration_seconds`), LLM inference duration and token counts (`maidere_llm_duration_seconds`, `maidere_llm_tokens_total`), and vector recall latency.
- **Grafana Provisioning**: Pre-built dashboard visualizing request rates, tool errors, token generation speed, and P95 latency.

### Phase 8: Dynamic Dual-Model Smart Routing
- **Task Complexity Classifier**: Evaluates user prompts in sub-millisecond heuristic passes to classify tasks as `SIMPLE` (greetings, chit-chat, summaries) or `COMPLEX` (multi-step tools, coding, research, filesystem writes).
- **Dual-Model Execution**: Automatically routes `SIMPLE` tasks to `qwen2.5:3b` (~65+ tok/s) and `COMPLEX` tasks to `qwen2.5:7b-instruct`.
- **Automatic Fallback**: Gracefully routes to primary model if the fast model is not pulled.

---

## 4. Extensible Capabilities & Skills System

Maidere features an Antigravity-style skill system located in `skills/*/SKILL.md` (and workspace-level `.agent/skills/`).

### Package Architecture
Each skill package contains YAML frontmatter metadata and markdown instructions:
```markdown
---
name: deep-research
description: Conducts autonomous multi-step iterative research across the web and outputs structured, interlinked notes in Obsidian Markdown format.
triggers:
  - research
  - deep research
  - investigate
  - obsidian
  - write note
  - lit review
---

# Deep Research & Obsidian Documentation Protocol
...
```

### Deep Research & Obsidian Protocol Specification
When activated, Maidere executes a multi-step research cycle:
1. **Query Decomposition**: Decomposes user goals into 2–4 targeted queries.
2. **Metasearch**: Executes `web_search` for each decomposed query.
3. **Deep Scraping**: Fetches pages via `browser` for data extraction.
4. **Synthesis & Deduplication**: Cross-references claims and resolves conflicting facts.
5. **Obsidian File Persistence**: Uses `write_file` to save formatted notes to `workspace/notes/<topic>.md`:
   - Full YAML frontmatter (`title`, `date`, `tags`, `aliases`, `sources_count`).
   - Executive summary callout `> [!SUMMARY] Core Thesis`.
   - Structural mechanics with ASCII diagrams or tables.
   - Wikilinks `[[Concept Name]]` integration.
   - Pitfalls and warnings `> [!WARNING] Critical Bottlenecks`.
   - Verified references list `## 🔗 References`.

---

## 5. Security Matrix & Sandbox Boundaries

| # | Security Control | Threat Mitigated | Technical Implementation |
| :---: | :--- | :--- | :--- |
| **1** | Directory Traversal Defense | Unauthorized system file read/write (`/etc/passwd`) | `pathlib.Path.resolve()` + `is_relative_to(WORKSPACE)` validation. |
| **2** | Symmetrical Path Resolution | Comparison failure between relative and absolute paths | Base `WORKSPACE` resolved to absolute path at startup. |
| **3** | Shell Injection Immunity | Malicious shell metacharacter execution (`rm -rf /;`) | `subprocess.run(shell=False)` with argument list. |
| **4** | Shell Binary Allowlist | Unauthorized system binary execution | Strict allowlist: `python3`, `git`, `ls`, `grep`, `mkdir`, `pytest`, etc. |
| **5** | Safe `shlex.split` Handling | Unhandled crashes on malformed LLM quotes | Enclosed in `try...except ValueError` returning clean error string. |
| **6** | Ephemeral Browser Lifecycle | Memory leaks and orphaned Chromium processes | `async with async_playwright()` with guaranteed `finally` termination. |
| **7** | Browser Timeout Caps | Infinite hanging on slow or hostile websites | 15,000ms navigation timeout cap. |
| **8** | HTML Content Caps | LLM context overflow from scraping massive pages | Strict 4,000 character maximum slice per URL. |
| **9** | Context Window Enforcer | Ollama VRAM allocation crashes and OOM | `trim_messages` node capping active history at 24,000 chars. |
| **10** | Zero GPU Model Thrashing | GPU latency spikes and VRAM fragmentation | FastEmbed embeddings run 100% on CPU ONNX Runtime. |
| **11** | SQLite Concurrency Locking | `sqlite3.OperationalError: database is locked` | `PRAGMA journal_mode = WAL` + `busy_timeout = 5000`. |
| **12** | Database Foreign Key Checks | Orphaned checkpoints and relational corruption | `PRAGMA foreign_keys = ON;`. |
| **13** | Audit Log Persistence | Unaccountable agent tool modifications | Every tool execution logged with timestamp, duration, args, and status. |
| **14** | SearXNG Container Isolation | Public search tracker tracking & network sprawl | Private local Docker container capped at 256MB RAM. |
| **15** | Prometheus Scrape Protection | Telemetry overhead loops | `/metrics` excluded from endpoint instrumentation logs. |
| **16** | TSDB Disk Retention Caps | Disk exhaustion on local development machine | `--storage.tsdb.retention.time=7d` and `--storage.tsdb.retention.size=1GB`. |
| **17** | Strict System Prompt Bounds | Over-eager tool invocation on chit-chat | Strict negative prompt boundaries against invoking tools on greetings. |
| **18** | Dual-Model Fallback Security | Agent termination when secondary model missing | Dynamic router automatically falls back to primary model. |
| **19** | Pydantic Request Validation | Malformed HTTP payloads and parameter exploits | Strict Pydantic models on all FastAPI routes. |
| **20** | Temporary File Cleanup | Disk clutter from ephemeral code runs | Automatic deletion of `.tmp/` scratch files after execution. |

---

## 6. Verification & Automated Test Suite

Maidere maintains an automated test suite with **66/66 passing tests** across 12 distinct test suites.

```bash
.venv/bin/python -m unittest discover -s tests -p "test_*.py" -v
```

### Test Suite Execution Summary

| Test Suite File | Test Count | Status | Subsystem Tested |
| :--- | :---: | :---: | :--- |
| `tests/test_router.py` | 8 | **PASS** | Task complexity classification, fast/primary routing, overrides, fallbacks, LangGraph routing |
| `tests/test_monitoring.py` | 4 | **PASS** | Prometheus `/metrics` exposition, tool duration histograms, token counters, memory metrics |
| `tests/test_skills.py` | 10 | **PASS** | Skill discovery, YAML parsing, trigger matching, `list_available_skills` & `load_skill` tools |
| `tests/test_web_ui.py` | 3 | **PASS** | Static UI serving (`GET /`), thread state persistence, dynamic `/models` switcher |
| `tests/test_search.py` | 5 | **PASS** | SearXNG query formatting, offline fallback, agent search-and-browse multi-step loop |
| `tests/test_phase4.py` | 9 | **PASS** | Code runner execution, browser cleanup, persistent APScheduler cron jobs |
| `tests/test_tools.py` | 12 | **PASS** | Filesystem sandbox boundaries, shell allowlist, safe quote handling |
| `tests/test_agent.py` | 7 | **PASS** | LangGraph agentic loop, multi-step tool calls, memory recall |
| `tests/test_memory.py` | 3 | **PASS** | `sqlite-vec` vector storage, fastembed CPU embeddings |
| `tests/test_api.py` | 3 | **PASS** | `/chat`, `/history/{thread_id}`, `/memories`, `/health` endpoints |
| `tests/test_db.py` | 2 | **PASS** | WAL mode active, 10 concurrent async writers stress test |
| **Total** | **66** | **100% PASS** | **Zero failures, Zero errors** |

---

## 7. API & Tool Reference

### Registered Agent Tools

```
┌─────────────────────────┬────────────────────────────────────────────────────────────────────────┐
│ Tool Name               │ Purpose & Parameters                                                   │
├─────────────────────────┼────────────────────────────────────────────────────────────────────────┤
│ read_file               │ Read file contents from workspace. Args: {path: str}                   │
│ write_file              │ Write text to file in workspace. Args: {path: str, content: str}       │
│ list_dir                │ List files in directory. Args: {path: str}                             │
│ shell                   │ Run allowed command. Args: {command: str}                              │
│ code_runner             │ Execute Python code in workspace. Args: {code: str}                     │
│ browser                 │ Fetch webpage text via Playwright. Args: {url: str}                    │
│ scheduler               │ Schedule tasks via APScheduler. Args: {action: str, job_id: str, ...}  │
│ web_search              │ Metasearch via SearXNG. Args: {query: str}                             │
│ list_available_skills   │ List installed specialized skills. Args: {}                            │
│ load_skill              │ Load full skill instructions. Args: {skill_name: str}                  │
└─────────────────────────┴────────────────────────────────────────────────────────────────────────┘
```

### REST API Directory

| Method | Path | Request Body / Params | Response Schema | Purpose |
| :--- | :--- | :--- | :--- | :--- |
| `POST` | `/chat` | `{"message": str, "thread_id": str?, "model": str?}` | `ChatResponse` | Primary interaction endpoint. |
| `WS` | `/ws/chat` | `{"message": str, "thread_id": str?, "model": str?}` | Streaming JSON | Real-time WebSocket streaming. |
| `GET` | `/threads` | None | `ThreadsListResponse` | List past conversation threads. |
| `DELETE`| `/threads/{id}`| `thread_id: str` | `{"status": "deleted"}` | Delete a thread and its checkpoints. |
| `GET` | `/history/{id}`| `thread_id: str` | `HistoryResponse` | Retrieve full thread conversation history. |
| `GET` | `/memories` | `query: str, top_k: int` | `MemoryRecallResponse` | Query semantic memories. |
| `GET` | `/models` | None | `{"models": list[str]}` | List available Ollama models + `auto`. |
| `GET` | `/metrics` | None | `text/plain` | Prometheus telemetry exposition. |
| `GET` | `/health` | None | `HealthResponse` | Service and Ollama status. |

---

## 8. Complete Operations & Quickstart Guide

### Prerequisites
- **OS**: Linux Mint 21+ / Ubuntu 22.04+
- **Hardware**: NVIDIA GPU with >= 8GB VRAM (e.g. RTX 5060 / 4060 / 3060), 16GB RAM.
- **Runtimes**: Python 3.12+, Docker Engine & Docker Compose, Ollama.

### Step 1: Environment Setup
```bash
# Clone the repository
git clone https://github.com/Khazar451/maidere.git
cd maidere

# Create and activate Python virtual environment
python3 -m venv .venv
source .venv/bin/activate

# Install dependencies
.venv/bin/python -m pip install -e .
```

### Step 2: Configure Environment Variables
Create a `.env` file in the project root:
```ini
OLLAMA_URL=http://localhost:11434
OLLAMA_MODEL=qwen2.5:7b-instruct
OLLAMA_FAST_MODEL=qwen2.5:3b
OLLAMA_NUM_CTX=8192
ENABLE_MODEL_ROUTING=true

DB_PATH=db/maidere.db
EMBEDDING_MODEL=BAAI/bge-small-en-v1.5
AGENT_WORKSPACE=workspace
LOG_PATH=logs/maidere.jsonl
LOG_LEVEL=INFO

SEARXNG_URL=http://localhost:8080
API_HOST=0.0.0.0
API_PORT=8000
```

### Step 3: Start Ollama & Pull Models
```bash
# Start Ollama service (if not already running)
systemctl start ollama || ollama serve &

# Pull the primary 7B model and fast 3B routing model
ollama pull qwen2.5:7b-instruct
ollama pull qwen2.5:3b
```

### Step 4: Launch Docker Infrastructure (SearXNG, Prometheus, Grafana)
```bash
docker compose up -d
```
- **SearXNG Metasearch**: `http://localhost:8080`
- **Prometheus TSDB**: `http://localhost:9090`
- **Grafana Dashboard**: `http://localhost:3000` (Login: `admin` / `admin`)

### Step 5: Start Maidere Agent Server
```bash
.venv/bin/uvicorn api.app:app --host 0.0.0.0 --port 8000 --reload
```

### Step 6: Access & Verify
- **Web UI**: Open `http://localhost:8000` in any browser.
- **Run Automated Test Suite**:
```bash
.venv/bin/python -m unittest discover -s tests -p "test_*.py" -v
```
