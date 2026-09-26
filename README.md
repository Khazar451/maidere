# [Maidere] Local Autonomous AI Agent

> **100% Free, Self-Hosted Autonomous AI Agent for Consumer Hardware (NVIDIA RTX 5060 8GB VRAM / 16GB System RAM on Linux).**

Maidere is an autonomous local AI agent engineered for privacy, speed, and sustained local execution. It runs fully offline with zero subscription costs, zero cloud dependencies, and zero GPU model-swapping delays.

---

## Quickstart in 60 Seconds

### 1. Prerequisites
- **OS**: Linux Mint 21+ / Ubuntu 22.04+
- **GPU**: NVIDIA GPU with >= 8GB VRAM (e.g. RTX 5060 / 4060 / 3060)
- **RAM**: 16GB System RAM
- **Software**: Python 3.12, Docker & Docker Compose, Ollama

### 2. Setup & Installation
```bash
# Clone the repository
git clone https://github.com/Khazar451/maidere.git
cd maidere

# Create virtual environment and install dependencies
python3 -m venv .venv
source .venv/bin/activate
.venv/bin/python -m pip install -e .

# Start Ollama & pull models
ollama pull qwen2.5:7b-instruct
ollama pull qwen2.5:3b
ollama pull deepseek-r1:7b     # DeepSeek-R1-Distill-Qwen-7B (~4.7 GB)
# Optional Llama-distill alternative:
# ollama pull deepseek-r1:8b   # DeepSeek-R1-Distill-Llama-8B (~4.9 GB)

# Start background services (SearXNG search, Prometheus, Grafana)
docker compose up -d

# Start Maidere Agent Server
.venv/bin/uvicorn api.app:app --host 0.0.0.0 --port 8000
```

Open **`http://localhost:8000`** in your browser to access the Web UI.

---

## Dynamic Context Window Switcher

Maidere features an interactive real-time context window selector in the UI top navigation bar (`CTX:` pill):

| Context Tier | Tokens (`num_ctx`) | KV Cache VRAM | Total VRAM (7B Q4_K_M) | Use Case |
| :--- | :--- | :--- | :--- | :--- |
| **8K - Medium** | `8192` | ~0.46 GB | ~5.16 GB | Maximum speed; lightweight queries |
| **16K - High** *(Default)* | `16384` | ~0.92 GB | ~5.62 GB | **Recommended sweet spot**; 2x context with 2.4 GB headroom |
| **32K - Ultra** | `32768` | ~1.84 GB | ~6.54 GB | Deep multi-agent research and large codebases |

Context window size persists across browser sessions via `localStorage` and scales LangGraph's dynamic turn pruning threshold (75% context utilization) automatically.

---

## Auto vs. Thinking Mode (Deep Reasoning)

Maidere includes dedicated **Auto** and **Thinking** controls (top header toggle group and input deck `[THINK] Deep Think` pill):

| Mode | Trigger / Button | Behavior & Model Routing |
| :--- | :--- | :--- |
| **Auto** *(Default)* | `Auto` button | **Dynamic Smart Routing**: Automatically classifies incoming query complexity (`SIMPLE` -> `qwen2.5:3b`, `COMPLEX` -> `qwen2.5:7b-instruct`, `REASONING` -> `deepseek-r1:7b` / `8b`). |
| **Thinking** | `Thinking` / `[THINK]` pill | **Enforced Chain-of-Thought**: Enforces step-by-step reasoning inside `<think>...</think>` tags before answering or executing tools, routes directly to DeepSeek-R1 distill models, streams live thinking pulses, and renders collapsible thought blocks in the UI. |

- **Agent Shell Visibility**: Thoughts prior to tool calls are extracted and logged live as `[THINK]` entries in the Agent Shell console.
- **Interactive Accordion**: `<think>...</think>` blocks are rendered as interactive `<details class="thought-box">` accordions with real-time pulsing animations during token streaming.

---

## Subsystem Architecture

```
+----------------------------------------------------------------------------------------+
|                                 HOST OS (Linux Mint / Ubuntu)                          |
+-----------------------------------------+----------------------------------------------+
|               GPU VRAM (8.0 GB)         |             SYSTEM RAM (16.0 GB)             |
| +-------------------------------------+ | +------------------------------------------+ |
| | Ollama Daemon                       | | | FastAPI Server (Uvicorn Async Worker)    | |
| |  - Primary: qwen2.5:7b-instruct     | | |  - REST Endpoints & WebSocket Handler    | |
| |    (Q4_K_M pinned: ~4.7 GB)         | | |  - Tri-Model Router (Fast, Think, Primary)| |
| |  - Thinking: deepseek-r1:7b / 8b    | | |  - Dynamic Context Switcher (8K/16K/32K) | |
| |    (Reasoning & Proofs: ~4.7-4.9 GB)| | |  - Prometheus Instrumentator (/metrics)  | |
| |  - Fast: qwen2.5:3b (on-demand)     | | +------------------------------------------+ |
| |  - Dynamic KV Cache (num_ctx:       | | | LangGraph Agent Core                     | |
| |    8192 / 16384 / 32768) (~0.5-1.8G)| | |  - trim -> remember -> think -> act loop | |
| |  - CUDA & Headroom Buffers (~1.5 GB)| | |  - Claude Code Subagents (research, plan)| |
| +-------------------------------------+ | |  - Symmetrical Filesystem Sandbox        | |
|                                         | |  - Safe Subprocess & Shell Allowlist     | |
|                                         | +------------------------------------------+ |
|                                         | | In-Process CPU & Storage Subsystems      | |
|                                         | |  - FastEmbed (bge-small-en-v1.5 on CPU)  | |
|                                         | |  - SQLite Database (WAL Mode Enabled)    | |
|                                         | |    * sqlite-vec (vec0 Virtual Table)     | |
|                                         | |    * LangGraph Checkpoints & Audit Log   | |
|                                         | |    * Persistent APScheduler DB Store     | |
|                                         | +------------------------------------------+ |
|                                         | | Docker Bridge Services                   | |
|                                         | |  - SearXNG Metasearch (Port 8080, 256MB) | |
|                                         | |  - Prometheus TSDB (Port 9090, 256MB)    | |
|                                         | |  - Grafana Dashboards (Port 3000, 256MB) | |
|                                         | +------------------------------------------+ |
+-----------------------------------------+----------------------------------------------+
```

---

## Core Architectural Decisions

1. **Zero GPU Model Thrashing**: Semantic embeddings run 100% on CPU via `fastembed` (`BAAI/bge-small-en-v1.5`), keeping LLMs pinned in VRAM without thrashing.
2. **Tri-Model Dynamic Routing**: Analyzes task complexity and intent:
   - `SIMPLE` tasks (greetings, simple queries) route to `qwen2.5:3b` (~65+ tok/s).
   - `REASONING` tasks (mathematical proofs, logic puzzles, algorithm derivations, step-by-step thinking) route to `deepseek-r1:7b` / `deepseek-r1:8b`.
   - `COMPLEX` multi-step tool and research tasks route to `qwen2.5:7b-instruct`.
3. **Interactive Reasoning Accordion**: Native real-time parsing of DeepSeek-R1's `<think>...</think>` tokens into an expandable, collapsible thought block with live streaming pulse indicator.
4. **In-Process Vector Storage**: Single SQLite database file (`db/maidere.db`) with native `sqlite-vec` extension (`vec0` virtual table), eliminating external vector DB overhead.
5. **SQLite WAL Concurrency**: Forces `PRAGMA journal_mode = WAL` and `busy_timeout = 5000` for crash resilience and non-blocking asynchronous multi-client access.
6. **Claude Code Subagents**: Subagents (`researcher`, `plan`, `verification`, `validation`, `code-reviewer`) execute in isolated ephemeral contexts with strict read-only tool boundaries and structured JSON reporting.
7. **Turn-Scoping & State Isolation**: Intermediate tool payloads from prior turns are discarded before calling the LLM to prevent cross-turn context pollution.
8. **Domain Grounding & Reality Check**: Distinguishes physical hardware ownership from institutional grant allocations (NAIRR Pilot, DGX Cloud credits) for high-end enterprise computing queries.
9. **Deterministic URL Citation Guard**: Automatically validates and revives markdown links `[Title](https://...)` from tool outputs, preventing URL stripping or placeholder dead links.
10. **Symmetrical Sandbox**: Resolves absolute file paths and strictly verifies that operations stay inside `workspace/`.
11. **Safe Subprocess Shell**: Enforces `shell=False` inside `subprocess.run()`, parses arguments with `shlex.split()`, and checks binaries against an allowlist.
12. **Ephemeral Browser Lifespan**: Playwright launches headless Chromium per scraping request with strict 15s timeouts and auto-termination to prevent zombie processes.
13. **Built-in Telemetry**: Exposes Prometheus metrics on `/metrics` with pre-configured Grafana dashboards for latency, tokens/s, and tool error rates.
14. **Epistemic Grounding (Canonical Lore vs. Speculation)**: Strictly separates canonical ground truth from speculative forum debates, blog essays, and fan theories. Prevents SEO/Theory Scrape Collapse by prioritizing primary source data and official wikis while omitting unverified internet theories unless the user explicitly requests them.
15. **Conditional Temporal Grounding & Chronological Causality**: Distinguishes active modern metrics from historical events. Appends the current year ({current_year}) strictly for financial metrics, earnings, and tech benchmarks, while strictly forbidding temporal distortion on historical figures, eras, or lore. Enforces strict chronological timeline verification before asserting cause-and-effect relationships (e.g., verifying Event A preceded Event B before stating B occurred 'following' A). Prevents search redundancy and Wikipedia loops via session-scoped URL deduplication and lexical query similarity checks.

---

## Registered Agent Tools

| Tool | Purpose | Args |
| :--- | :--- | :--- |
| `read_file` | Read file contents from workspace | `path: str` |
| `write_file` | Write text to file in workspace | `path: str, content: str` |
| `list_dir` | List files and directories in workspace | `path: str` |
| `shell` | Run allowlisted shell command | `command: str` |
| `code_runner` | Execute Python script inside workspace | `code: str` |
| `browser` | Scrape webpage DOM text via Playwright | `url: str` |
| `delegate_task` | Spawn autonomous subagent (`researcher`, `plan`, `verification`, `validation`, `code-reviewer`) | `prompt: str, subagent_type: str, max_turns: int` |
| `obsidian` | Search, read, write, and create notes in Obsidian vault | `action: str, note_name: str, content: str` |
| `scheduler` | Schedule recurring or one-shot cron jobs via APScheduler | `action: str, job_id: str, ...` |
| `web_search` | Private local metasearch via SearXNG | `query: str` |
| `list_available_skills` | List installed specialized skills | `{}` |
| `load_skill` | Load instructions for a specific skill | `skill_name: str` |

---

## Automated Test Suite

```bash
# Run full automated test suite (148 tests across 16 test suites)
.venv/bin/python -m unittest discover -s tests -p "test_*.py" -v
```

```text
Ran 148 tests in 3.550s — OK (100% Passed, 0 Failures, 0 Errors)
```

---

## Comprehensive Documentation

For the exhaustive technical specification, hardware budget allocation matrix, security controls, and skill specifications, see **[ARCHITECTURE.md](ARCHITECTURE.md)**.

---

## License

MIT License — free for personal, educational, and commercial use.

