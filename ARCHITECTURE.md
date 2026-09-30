# Maidere (v5.0) — Technical Architecture & Reference Manual

> **An Autonomous, Privacy-First AI Agent & Cognitive Engine Designed for Local Workstations (NVIDIA RTX 5060 8GB VRAM / 16GB System RAM) with Cloud Acceleration.**

---

## 1. Executive Summary & Hardware Budget

### Core Philosophy
Maidere is an autonomous local AI agent engineered for sustained execution on personal workstations and laptops. It eliminates recurring cloud API subscriptions, data leakage, and GPU model-swapping latency by maintaining a strictly partitioned memory and execution budget:
1. **Zero-VRAM Semantic Pipeline**: The primary 7B LLM stays pinned permanently in GPU VRAM, while semantic embeddings, vector index searches, and tool runtimes execute entirely on host CPU and system RAM.
2. **Hybrid Cloud Scalability**: When tasks exceed local hardware boundaries, Maidere seamlessly bridges to high-context cloud clusters (e.g. NVIDIA Nemotron 3 Ultra 550B up to 128K context) with automatic fallback to local models upon disconnection.
3. **Native Desktop Integration**: Maidere ships with a native Linux desktop application manager providing system dock integration, automatic process lifecycle handling, and a continuous development mode with live hot-reloading (`--dev`).

### Subsystem Topology
```
┌────────────────────────────────────────────────────────────────────────────────────────┐
│                             HOST OS (Linux Mint / Ubuntu)                              │
├─────────────────────────────────────────┬──────────────────────────────────────────────┤
│           LOCAL GPU VRAM (8.0 GB)       │             SYSTEM RAM (16.0 GB)             │
│ ┌─────────────────────────────────────┐ │ ┌──────────────────────────────────────────┐ │
│ │ Ollama Daemon                       │ │ │ Native Desktop & FastAPI Server          │ │
│ │  - Primary: qwen2.5:7b-instruct     │ │ │  - Window Manager & Process Lifecycle    │ │
│ │    (Pinned Q4_K_M: ~4.7 GB)         │ │ │  - WebSocket Token & Event Stream        │ │
│ │  - Reasoning: deepseek-r1:7b        │ │ │  - Tri-Model Dynamic Intent Router       │ │
│ │    (Proof verification: ~4.7 GB)    │ │ │  - Dynamic Context Selector (8K–128K)    │ │
│ │  - Fast: qwen2.5:3b (on-demand)     │ │ │  - Prometheus Instrumentator (/metrics)  │ │
│ │  - Dynamic KV Cache (8K/16K/32K)    │ │ ├──────────────────────────────────────────┤ │
│ │  - CUDA Driver Buffers (~1.5 GB)    │ │ │ LangGraph Agent Core                     │ │
│ └─────────────────────────────────────┘ │ │  - Cyclic: trim -> remember -> think ->  │ │
├─────────────────────────────────────────┤ │    act -> evaluate -> verify -> converge │ │
│         CLOUD INFERENCE BRIDGE          │ │  - Isolated Subagent Contexts            │ │
│ ┌─────────────────────────────────────┐ │ │  - File-Based Artifact Bus (.maidere/)   │ │
│ │ NVIDIA NIM / OpenAI Cloud Endpoints │ │ ├──────────────────────────────────────────┤ │
│ │  - nvidia/nemotron-3-ultra-550b     │ │ │ In-Process CPU & Storage Subsystems      │ │
│ │  - Context: 64K Massive / 128K Extr │ │ │  - FastEmbed (bge-small-en-v1.5 on CPU)  │ │
│ │  - Thinking Mode & Reasoning Delta  │ │ │  - SQLite Database (WAL Mode Enabled)    │ │
│ └─────────────────────────────────────┘ │ │    * sqlite-vec (vec0 Virtual Table)     │ │
│                                         │ │    * LangGraph Checkpoints & Audit Log   │ │
│                                         │ │    * Persistent APScheduler Job Store    │ │
│                                         │ ├──────────────────────────────────────────┤ │
│                                         │ │ Docker Bridge Services                   │ │
│                                         │ │  - SearXNG Metasearch (Port 8080)        │ │
│                                         │ │  - Prometheus TSDB (Port 9090)           │ │
│                                         │ │  - Grafana Telemetry (Port 3000)         │ │
│                                         │ └──────────────────────────────────────────┘ │
└─────────────────────────────────────────┴──────────────────────────────────────────────┘
```

### Hardware Allocation Matrix

| Subsystem / Resource | Component / Process | Allocated Size | Execution Target | Rationale & Guardrails |
| :--- | :--- | :--- | :--- | :--- |
| **GPU VRAM (8.0 GB Budget)** | Primary LLM (`qwen2.5:7b-instruct`) | ~4.7 GB | NVIDIA RTX 5060 (CUDA) | Pinned in 4-bit quantization (Q4_K_M); zero cold-swapping delay. |
| | KV Cache (`num_ctx: 16384`) | ~0.92 GB | NVIDIA RTX 5060 (CUDA) | Sweet spot allocation; configurable between 8K (~0.46 GB) and 32K (~1.84 GB). |
| | Desktop / Display Compositor Buffer | ~0.5 GB | Linux X11 / Wayland | OS display output headroom. |
| | CUDA Driver & Runtime Headroom | ~1.5 GB | GPU Kernel Space | Dynamic CUDA allocation buffer preventing out-of-memory faults. |
| **System RAM (16.0 GB Budget)** | Host OS & Kernel Services | ~2.5 GB | Host CPU | Base operating system footprint. |
| | Ollama Daemon & Fast Model (`3B`) | ~2.0 GB | Host CPU / RAM | Ephemeral routing for high-throughput conversational queries. |
| | FastAPI + LangGraph Agent Core | ~0.8 GB | Python 3.12 Process | In-process agent runtime, checkpointers, and connection pools. |
| | FastEmbed CPU Engine (`bge-small`) | ~0.4 GB | ONNX Runtime (CPU) | 384-dimensional dense embeddings; 0 MB GPU VRAM consumed. |
| | SQLite WAL & `sqlite-vec` Index | ~0.3 GB | In-Process Memory Cache | High-speed vector similarity and ACID checkpoints. |
| | Playwright Headless Browser (Ephemeral) | ~0.6 GB | Chromium Subprocess | Scrapes target URLs on demand; destroyed immediately after parse. |
| | SearXNG Metasearch Engine | ~0.25 GB | Docker Container | Local private metasearch engine capped at 256MB. |
| | Prometheus Monitoring TSDB | ~0.25 GB | Docker Container | Metric collection capped at 1GB disk / 7 days retention. |
| | Grafana Dashboard Server | ~0.25 GB | Docker Container | Real-time agent telemetry visualization. |
| | **Free / Available Headroom** | **~8.65 GB** | System Reserve | Buffers large file reads, code execution, and OS caches. |

---

## 2. Core Architectural Invariants

### 1. SQLite WAL Concurrency & Safe Connection Pools
- **Problem**: Default SQLite database locking (`ROLLBACK` journal) causes `sqlite3.OperationalError: database is locked` when multiple asynchronous tasks (LangGraph checkpointer, memory embedding ingestion, tool audit logger, and API requests) write concurrently.
- **Pattern**: Force write-ahead logging (`WAL`), 5-second busy timeout handler, normal synchronization, and foreign key constraint enforcement across every SQLite connection.
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

### 2. Zero-VRAM CPU Embeddings via FastEmbed
- **Problem**: Running embedding models on the GPU forces Ollama to unload the 7B generative model or causes catastrophic CUDA Out-Of-Memory (OOM) faults on 8GB cards.
- **Pattern**: Leverage CPU-only ONNX Runtime execution via `fastembed` with `BAAI/bge-small-en-v1.5` (384 dimensions). Embedding generation takes ~5ms on modern CPUs while consuming exactly 0 MB of GPU VRAM.
```python
# In core/embeddings.py
from fastembed import TextEmbedding

_model = TextEmbedding(model_name="BAAI/bge-small-en-v1.5")

def embed_one(text: str) -> list[float]:
    embeddings = list(_model.embed([text]))
    return embeddings[0].tolist()
```

### 3. In-Process Vector Search via `sqlite-vec`
- **Problem**: Standalone vector databases (Milvus, Qdrant, Pinecone, Chroma) add container bloat, network latency, and memory overhead.
- **Pattern**: Load the lightweight `sqlite-vec` C-extension directly into SQLite and maintain a native virtual table (`vec0`) supporting L2 distance metric vector queries.
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

### 4. Dynamic Context Window & Token Trimming
- **Problem**: As conversation turns accumulate, exceeding model context boundaries causes Ollama or cloud providers to crash or truncate abruptly.
- **Pattern**: A dedicated `trim_messages` node dynamically calculates the active token budget based on the selected context tier (8K, 16K, 32K, 64K, 128K). At 75% utilization, older conversational turns are pruned from volatile graph state while preserving system prompts and instructions intact. Pruned content remains preserved in SQLite and recoverable via semantic vector search.

### 5. Multi-Stage System 2 Deliberation Pipeline
- **Problem**: Direct, single-pass LLM generation frequently suffers from confirmation bias, arithmetic errors, and hallucinated API boundaries.
- **Pattern**: When operating in `Deep Reason` mode, Maidere splits synthesis into three distinct, non-leaking cognitive stages:
  1. **Divergent Cognitive Exploration**: Premise deconstruction, divergent hypothesis generation (Approach A vs. Approach B), and counter-example falsification inside `<think>...</think>` tags.
  2. **Analytical Verification Specialist**: Independent recalculation of math, logical invariants, boundary conditions, and citation integrity.
  3. **Convergent Synthesis**: Formulates the final polished synthesis while deterministically stripping internal deliberation prompts and fourth-wall reviewer leaks (`strip_critic_fourth_wall_leaks`).

### 6. Ephemeral Playwright Lifecycle
- **Problem**: Long-running browser instances leak memory over time, orphan zombie Chromium processes, and crash if a page hangs.
- **Pattern**: Enforce an ephemeral browser lifecycle. Every scraping task launches Chromium asynchronously, opens an isolated context, enforces a 15-second strict timeout, caps extracted HTML/text at 4,000 characters, and guarantees termination via `try...finally`.

---

## 3. Subagent Fleet & File-Based Artifact Bus

Maidere orchestrates a team of specialized subagents running in isolated ephemeral execution contexts:

```mermaid
graph TD
    User([User Request]) --> Router[Tri-Model Router]
    Router --> Orchestrator[Orchestrator Agent]
    
    subgraph Fleet["Specialized Ephemeral Subagent Fleet"]
        Researcher["researcher: Metasearch & Web Scraping"]
        PlanAgent["plan: Software Architecture & Migration"]
        Verifier["verification: Math Proofs & Citation Check"]
        Validator["validation: 2026 Feasibility & Dependency Check"]
        Reviewer["code-reviewer: Security & Bug Audit"]
        Staffer["agy_staffer: External Antigravity / Gemini 3.8"]
    end

    Orchestrator -->|delegate_task| Fleet
    Fleet -->|Microsecond Collision-Safe Disk Writes| Bus[".maidere/artifacts/*.md"]
    Bus -->|Structured Handoff Briefs<br/>(300-1400 chars)| Orchestrator
    Orchestrator -->|Synthesize Authoritative Answer| Response([Final Response])
```

### Artifact Bus Mechanics
- **Disk Persistence**: Subagents write full, dense findings directly to disk (`workspace/.maidere/artifacts/`) using microsecond timestamps and UUID suffixes to prevent race collisions.
- **Automated Retention**: An automated maintenance routine enforces a 14-day TTL and a 100-file cap, purging stale artifacts automatically.
- **Adaptive Handoff Briefs**: Instead of polluting the parent orchestrator's context with thousands of tokens of raw web text, dispatches return a structured 300–1,400 character brief containing:
  - Executive summary
  - Key verified facts & numerical metrics
  - Exact source URLs for citation recovery
  - File path to the full artifact on disk
- **Context Savings**: Reduces parent turn context consumption by $\ge 70\%$.

---

## 4. 3-Tier Self-Healing Memory Engine

```
┌────────────────────────────────────────────────────────────────────────┐
│                        3-TIER MEMORY ENGINE                            │
├────────────────────────────────────────────────────────────────────────┤
│ Tier 1: Instruction Memory (.maidere/RULES.md)                         │
│  - System-wide behavioral overrides and persistent project directives  │
│  - Single-step rollback safety: writes automatically snapshot .bak    │
├────────────────────────────────────────────────────────────────────────┤
│ Tier 2: Structured Auto-Memory (.maidere/memory/*.md + index.json)     │
│  - Topic-specific knowledge files with structured markdown headers    │
│  - Two-Step Save Invariant: write topic file -> update index metadata  │
│  - LRU Cache Eviction capped at 40 indexed entries                     │
│  - Startup Reconciliation: re-indexes uncataloged disk files on boot   │
├────────────────────────────────────────────────────────────────────────┤
│ Tier 3: Session Fact Extraction (Turn Completion Hook)                 │
│  - Auto-extracts user preferences, stack choices, and tech constraints │
│  - Jaccard Novelty Filter: evaluates word overlap against active memory│
│  - Auto-promotion: novel facts (overlap < 0.6) promoted to Tier 2     │
└────────────────────────────────────────────────────────────────────────┘
```

---

## 5. Security Architecture & Sandbox Controls

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
| **9** | Context Window Enforcer | Ollama VRAM allocation crashes and OOM | `trim_messages` node dynamically scaling with selected context tier. |
| **10** | Zero GPU Model Thrashing | GPU latency spikes and VRAM fragmentation | FastEmbed embeddings run 100% on CPU ONNX Runtime. |
| **11** | SQLite Concurrency Locking | `sqlite3.OperationalError: database is locked` | `PRAGMA journal_mode = WAL` + `busy_timeout = 5000`. |
| **12** | Database Foreign Key Checks | Orphaned checkpoints and relational corruption | `PRAGMA foreign_keys = ON;`. |
| **13** | Audit Log Persistence | Unaccountable agent tool modifications | Every tool execution logged with timestamp, duration, args, and status. |
| **14** | SearXNG Container Isolation | Public search tracker tracking & network sprawl | Private local Docker container capped at 256MB RAM. |
| **15** | Prometheus Scrape Protection | Telemetry overhead loops | `/metrics` excluded from endpoint instrumentation logs. |
| **16** | TSDB Disk Retention Caps | Disk exhaustion on local development machine | `--storage.tsdb.retention.time=7d` and `--storage.tsdb.retention.size=1GB`. |
| **17** | Strict System Prompt Bounds | Over-eager tool invocation on chit-chat | Strict negative prompt boundaries against invoking tools on greetings. |
| **18** | Dual-Model Fallback Security | Agent termination when secondary model missing | Dynamic router automatically falls back to primary model. |
| **19** | Epistemic Grounding | SEO/Theory Scrape Collapse and fan fiction | Prioritizes primary sources; omits unverified speculation unless requested. |
| **20** | Temporary File Cleanup | Disk clutter from ephemeral code runs | Automatic deletion of `.tmp/` scratch files after execution. |

---

## 6. Automated Test Suite Matrix

Maidere maintains an automated test suite comprising **245 unit and integration tests across 23 test modules**:

```bash
HF_HUB_OFFLINE=1 .venv/bin/python -m unittest discover -s tests -p "test_*.py" -v
```

```text
Ran 245 tests in 4.808s — OK (100% Passed, 0 Failures, 0 Errors)
```

| Test Module | Tests | Status | Target Subsystem & Validated Invariants |
| :--- | :---: | :---: | :--- |
| `tests/test_agent.py` | 8 | **PASS** | LangGraph cyclic execution, message trimming, tool loops, state transitions |
| `tests/test_agy_bridge.py` | 7 | **PASS** | External Antigravity CLI bridge, git status check, rate-limit fallback |
| `tests/test_api.py` | 5 | **PASS** | REST endpoints (`/chat`, `/threads`, `/history`, `/memories`, `/health`) |
| `tests/test_cloud_llm.py` | 12 | **PASS** | NVIDIA NIM client, Nemotron thinking stream, context tier overrides |
| `tests/test_db.py` | 3 | **PASS** | SQLite WAL mode, foreign key validation, concurrent async writer stress test |
| `tests/test_desktop_launcher.py` | 8 | **PASS** | Desktop window launcher, port prober, signal handling, XDG launcher setup |
| `tests/test_ethics_and_eval.py` | 14 | **PASS** | Epistemic grounding, reality checks, chronological causality verification |
| `tests/test_github.py` | 9 | **PASS** | GitHub tool integration, issue/PR management, branch validation |
| `tests/test_llm.py` | 5 | **PASS** | Direct Ollama client, raw HTTP streaming, connection timeout handling |
| `tests/test_memory.py` | 4 | **PASS** | `sqlite-vec` vector similarity search, FastEmbed CPU embedding generation |
| `tests/test_memory_tiers.py` | 16 | **PASS** | Tier 1 RULES.md backups, Tier 2 LRU auto-memory, Tier 3 Jaccard novelty filter |
| `tests/test_monitoring.py` | 5 | **PASS** | Prometheus `/metrics` exposition, tool duration histograms, token counters |
| `tests/test_obsidian.py` | 9 | **PASS** | Obsidian vault path resolution, frontmatter formatting, note search |
| `tests/test_phase4.py` | 9 | **PASS** | Isolated Python code runner, Playwright browser cleanup, APScheduler crons |
| `tests/test_router.py` | 18 | **PASS** | Tri-model intent classifier, reasoning detection, model fallback logic |
| `tests/test_search.py` | 7 | **PASS** | SearXNG query formatting, offline fallback, agent search-and-browse loop |
| `tests/test_skills.py` | 11 | **PASS** | Skill discovery, YAML parsing, trigger matching, dynamic skill loader |
| `tests/test_subagent.py` | 32 | **PASS** | Subagent isolation, microsecond-safe artifact bus, handoff brief generation |
| `tests/test_subagent_artifacts.py` | 10 | **PASS** | Artifact TTL pruning, 100-file cap enforcer, context savings verification |
| `tests/test_synthesis_hardening.py` | 8 | **PASS** | Citation bracket normalization `[N]`, fourth-wall reviewer leak stripping |
| `tests/test_tools.py` | 14 | **PASS** | Filesystem path sandbox, shell allowlist, safe quote handling |
| `tests/test_v01_enhancements.py` | 11 | **PASS** | Real-time token streaming, Markdown code block copy, thinking toggle UI |
| `tests/test_web_ui.py` | 10 | **PASS** | Static UI serving, WebSocket chat protocol, theme settings, thread persistence |
| **Total** | **245** | **100% PASS** | **Zero failures, Zero errors** |

---

## 7. Configuration Reference (.env)

| Variable | Default Value | Description |
| :--- | :--- | :--- |
| `OLLAMA_URL` | `http://localhost:11434` | Endpoint for the local Ollama daemon. |
| `OLLAMA_MODEL` | `qwen2.5:7b-instruct` | Primary pinned local model for complex tasks. |
| `OLLAMA_FAST_MODEL` | `qwen2.5:3b` | Fast local model for simple conversational turns. |
| `OLLAMA_THINKING_MODEL`| `deepseek-r1:7b` | Local reasoning model for proofs and logic puzzles. |
| `OLLAMA_NUM_CTX` | `16384` | Default context window size in tokens. |
| `ENABLE_MODEL_ROUTING` | `true` | Enables automatic task complexity classifier. |
| `CLOUD_API_BASE` | `https://integrate.api.nvidia.com/v1` | Base URL for OpenAI-compatible / NVIDIA NIM endpoints. |
| `CLOUD_API_KEY` | `""` | API key for cloud model inference. |
| `CLOUD_MODEL` | `nvidia/nemotron-3-ultra-550b-a55b` | Cloud model identifier. |
| `CLOUD_NUM_CTX` | `65536` | Context window size for cloud inference (64K or 128K). |
| `DB_PATH` | `db/maidere.db` | Local SQLite database file path. |
| `EMBEDDING_MODEL` | `BAAI/bge-small-en-v1.5` | FastEmbed ONNX model for CPU embeddings. |
| `AGENT_WORKSPACE` | `workspace` | Root sandbox directory for file read/write operations. |
| `LOG_PATH` | `logs/maidere.jsonl` | Structured JSON log file destination. |
| `LOG_LEVEL` | `INFO` | Logging verbosity (`DEBUG`, `INFO`, `WARNING`, `ERROR`). |
| `SEARXNG_URL` | `http://localhost:8080` | Endpoint for local SearXNG Docker container. |
| `OBSIDIAN_VAULT_PATH` | `""` | Optional explicit path to user Obsidian vault. |
| `API_HOST` | `0.0.0.0` | Bind host for FastAPI Uvicorn server. |
| `API_PORT` | `8000` | Port for FastAPI Uvicorn server. |
