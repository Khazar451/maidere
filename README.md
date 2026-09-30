# [Maidere] Local Autonomous AI Agent

> **100% Free, Self-Hosted Autonomous AI Agent for Consumer Hardware (NVIDIA RTX 5060 8GB VRAM / 16GB System RAM on Linux).**

Maidere is an autonomous local AI agent engineered for privacy, speed, and sustained local execution. It runs fully offline with zero subscription costs, zero cloud dependencies, and zero GPU model-swapping delays.

---

## Quickstart in 60 Seconds

### 1. Prerequisites
- **OS**: Linux Mint 21+ / Ubuntu 22.04+
- **GPU**: NVIDIA GPU with >= 8GB VRAM (e.g. RTX 5060 / 4060 / 3060)
- **RAM**: 16GB System RAM
- **Software**: Python 3.12 (`>=3.12, <3.14`), Docker & Docker Compose, Ollama, [uv](https://docs.astral.sh/uv/) (recommended)

### 2. Setup & Installation
```bash
# Clone the repository
git clone https://github.com/Khazar451/maidere.git
cd maidere

# Option A: Install with uv (fastest & recommended)
uv sync

# Option B: Standard Python venv
python3 -m venv .venv
source .venv/bin/activate
pip install -e .

# Start Ollama & pull models
ollama pull qwen2.5:7b-instruct
ollama pull qwen2.5:3b
ollama pull deepseek-r1:7b     # DeepSeek-R1-Distill-Qwen-7B (~4.7 GB)
# Optional Llama-distill alternative:
# ollama pull deepseek-r1:8b   # DeepSeek-R1-Distill-Llama-8B (~4.9 GB)

# Start background services (SearXNG search, Prometheus, Grafana)
docker compose up -d

# Option A: Launch Native Desktop Application (Default)
python3 maidere.py
# or in continuous development mode with live hot-reload:
python3 maidere.py --dev

# Option B: Install into Linux Start Menu / Dock
python3 maidere.py install-desktop

# Option C: Start backend server only (headless)
uv run uvicorn api.app:app --host 0.0.0.0 --port 8000
```

Open **`http://localhost:8000`** in your browser if running in server-only mode.

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

## Cognitive Modes: Auto, Thinking, and Deep Reasoning

Maidere provides three distinct operational paradigms selectable via the top navigation toggle group (`Auto`, `Thinking`, `Deep Reason`):

| Mode | Selector | Architecture & Operational Flow |
| :--- | :--- | :--- |
| **Auto** *(Default)* | `Auto` | **Dynamic Smart Routing**: Classifies task complexity (`SIMPLE` &rarr; `qwen2.5:3b`, `COMPLEX` &rarr; `qwen2.5:7b-instruct`, `REASONING` &rarr; `deepseek-r1:7b` / `8b`) without deliberation overhead. |
| **Thinking** | `Thinking` | **Fast Single-Pass Chain-of-Thought**: Generates step-by-step reasoning inside `<think>...</think>` scratchpad tags with real-time UI token streaming and collapsible accordion rendering. Bypasses multi-stage review loops to minimize latency. |
| **Deep Reason** | `Deep Reason` | **Multi-Stage System 2 Deliberation Pipeline**: Executes an exhaustive 3-stage cognitive framework:<br>1. *Divergent Cognitive Exploration*: Premise deconstruction, divergent hypothesis generation, and falsification analysis.<br>2. *Analytical Verification Specialist*: Independent recalculation of math, logical invariants, boundary conditions, and citation integrity.<br>3. *Convergent Synthesis*: Rigorous, verified authoritative solution generation. |

- **Agent Shell Visibility**: Chain-of-thought scratchpads and intermediate verification verdicts are logged live as `[THINK]` and `[VERIFY]` entries in the Agent Shell console.
- **Interactive Accordion**: `<think>...</think>` blocks render as interactive `<details class="thought-box">` accordions with pulsing token streaming animations.

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
| |    (Q4_K_M pinned: ~4.7 GB)         | | |  - Tri-Model Router (Fast, Think, Reason)| |
| |  - Thinking: deepseek-r1:7b / 8b    | | |  - Dynamic Context Switcher (8K/16K/32K) | |
| |    (Reasoning & Proofs: ~4.7-4.9 GB)| | |  - Prometheus Instrumentator (/metrics)  | |
| |  - Fast: qwen2.5:3b (on-demand)     | | +------------------------------------------+ |
| |  - Dynamic KV Cache (num_ctx:       | | | LangGraph Agent Core                     | |
| |    8192 / 16384 / 32768) (~0.5-1.8G)| | |  - trim -> remember -> think -> act ->   | |
| |  - CUDA & Headroom Buffers (~1.5 GB)| | |    evaluate -> verify -> converge loop   | |
| +-------------------------------------+ | |  - Dual V&V Subagents (verify/validate)  | |
|                                         | |  - File Artifact Bus (.maidere/artifacts)| |
|                                         | |  - External AGY Staffer (Gemini 3.8 Flash)| |
|                                         | +------------------------------------------+ |
|                                         | | In-Process Storage & Memory Subsystems   | |
|                                         | |  - 3-Tier Memory Engine:                 | |
|                                         | |    * Tier 1: RULES.md Instructions (.bak)| |
|                                         | |    * Tier 2: Auto-Memory (*.md + index)  | |
|                                         | |    * Tier 3: Turn Extraction (Jaccard)   | |
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
6. **Dual-Stage V&V Subagent Framework**: Subagents (`researcher`, `plan`, `verification`, `validation`, `reviewer`, `staffer`) execute in isolated ephemeral contexts with strict read-only tool boundaries:
   - **`verification` (alias `verifier`)**: Mathematical proof recalculation, invariant tracking, static contract checking, and citation bracket checks (`[PASS: VERIFIED]`).
   - **`validation` (alias `validator`)**: Real-world 2026 dependency and API feasibility checks, environment constraints, and user acceptance criteria (`[PASS: VALIDATED]`).
7. **File-Based Artifact Bus & Adaptive Handoff Briefs**: Subagents write dense findings directly to disk (`workspace/.maidere/artifacts/`) with microsecond and UUID collision-resistance and automated 14-day TTL / 100-file cap pruning. Dispatches return structured 300–1,400 character handoff briefs, achieving $\ge 70\%$ orchestrator context savings.
8. **External AGY Staffer Bridge (`agy_staffer`)**: Seamlessly connects to Google Antigravity CLI (Gemini 3.8 Flash) for high-context tasks across 5 personas (`researcher`, `reviewer`, `implementer`, `ask`, `staffer`) with pre-dispatch git dirty-state checks and automatic local subagent fallback on rate limits (`429` / `RESOURCE_EXHAUSTED`).
9. **3-Tier Memory Engine**:
   - **Tier 1 (Instruction Memory)**: Manages `RULES.md` in `.maidere/` with automatic `.bak` backups before write operations for single-step rollback.
   - **Tier 2 (Auto-Memory)**: Manages topic markdown files (`.maidere/memory/*.md`) and `index.json`, enforcing the Two-Step Save Invariant, LRU eviction capped at 40 indexed entries, and startup index reconciliation.
   - **Tier 3 (Session Fact Extraction)**: Extracts user preferences and project decisions at turn completion, evaluates semantic novelty using Jaccard word-overlap scoring, and auto-promotes facts to Tier 2 topics.
10. **Local 7B Synthesis Hardening & Anti-Hallucination**: Enforces strict `[N]` bracket wrapping for citations, deterministically normalizes unbracketed citation integers while protecting legitimate numbers (`step 2`, `table 1`), normalizes plain URL lists into markdown links, and strips fourth-wall reviewer leakage.
11. **Turn-Scoping & State Isolation**: Intermediate tool payloads from prior turns are discarded before calling the LLM to prevent cross-turn context pollution.
12. **Domain Grounding & Reality Check**: Distinguishes physical hardware ownership from institutional grant allocations (NAIRR Pilot, DGX Cloud credits) for high-end enterprise computing queries.
13. **Deterministic URL Citation Guard**: Automatically validates and revives markdown links `[Title](https://...)` from tool outputs, preventing URL stripping or placeholder dead links.
14. **Symmetrical Sandbox**: Resolves absolute file paths and strictly verifies that operations stay inside `workspace/`.
15. **Safe Subprocess Shell**: Enforces `shell=False` inside `subprocess.run()`, parses arguments with `shlex.split()`, and checks binaries against an allowlist.
16. **Ephemeral Browser Lifespan**: Playwright launches headless Chromium per scraping request with strict 15s timeouts and auto-termination to prevent zombie processes.
17. **Built-in Telemetry**: Exposes Prometheus metrics on `/metrics` with pre-configured Grafana dashboards for latency, tokens/s, and tool error rates.
18. **Epistemic Grounding (Canonical Lore vs. Speculation)**: Strictly separates canonical ground truth from speculative forum debates, blog essays, and fan theories. Prevents SEO/Theory Scrape Collapse by prioritizing primary source data and official wikis while omitting unverified internet theories unless the user explicitly requests them.
19. **Conditional Temporal Grounding & Chronological Causality**: Distinguishes active modern metrics from historical events. Appends the current year strictly for financial metrics, earnings, and tech benchmarks, while strictly forbidding temporal distortion on historical figures, eras, or lore. Enforces strict chronological timeline verification before asserting cause-and-effect relationships. Prevents search redundancy and Wikipedia loops via session-scoped URL deduplication and lexical query similarity checks.

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
| `delegate_task` | Spawn autonomous subagent (`researcher`, `plan`, `verification`, `validation`, `reviewer`, `staffer`) | `prompt: str, subagent_type: str, max_turns: int` |
| `manage_memory` | 3-Tier memory manager (read/save topic facts, view rules) | `action: str, topic: str, content: str` |
| `agy_staffer` | External Gemini 3.8 Flash delegate via Antigravity CLI (`researcher`, `reviewer`, `implementer`, `ask`, `staffer`) | `prompt: str, persona: str` |
| `obsidian` | Search, read, write, and create notes in Obsidian vault | `action: str, note_name: str, content: str` |
| `scheduler` | Schedule recurring or one-shot cron jobs via APScheduler | `action: str, job_id: str, ...` |
| `web_search` | Private local metasearch via SearXNG | `query: str` |
| `list_available_skills` | List installed specialized skills | `{}` |
| `load_skill` | Load instructions for a specific skill | `skill_name: str` |

---

## Automated Test Suite

```bash
# Run full automated test suite (209 tests across 21 test suites)
uv run --no-sync python -m unittest discover -s tests -p "test_*.py" -v
# or with active venv:
.venv/bin/python -m unittest discover -s tests -p "test_*.py" -v
```

```text
Ran 209 tests in 3.945s — OK (100% Passed, 0 Failures, 0 Errors)
```

---

## Comprehensive Documentation

For the exhaustive technical specification, hardware budget allocation matrix, security controls, and skill specifications, see **[ARCHITECTURE.md](ARCHITECTURE.md)**.

---

## License

MIT License — free for personal, educational, and commercial use.

