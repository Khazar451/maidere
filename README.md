<div align="center">

# Maidere

**Autonomous, Privacy-First AI Agent & Cognitive Engine for Local Workstations & Cloud Clusters**

[![Python 3.12](https://img.shields.io/badge/Python-3.12-3776AB?style=flat-square&logo=python&logoColor=white)](https://www.python.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.115+-009688?style=flat-square&logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com/)
[![LangGraph](https://img.shields.io/badge/LangGraph-StateGraph-FF6F00?style=flat-square)](https://langchain-ai.github.io/langgraph/)
[![Ollama](https://img.shields.io/badge/Ollama-Local_Inference-000000?style=flat-square&logo=ollama&logoColor=white)](https://ollama.com/)
[![NVIDIA NIM](https://img.shields.io/badge/NVIDIA-NIM_%2F_Nemotron-76B900?style=flat-square&logo=nvidia&logoColor=white)](https://build.nvidia.com/)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg?style=flat-square)](LICENSE)
[![CI](https://github.com/Khazar451/maidere/actions/workflows/ci.yml/badge.svg)](https://github.com/Khazar451/maidere/actions/workflows/ci.yml)

*Engineered for zero subscription costs, zero GPU model-thrashing, and enterprise-grade reasoning on consumer hardware (NVIDIA RTX 5060 8GB / 16GB RAM) with instant hybrid scaling to massive cloud models (Nemotron-3-Ultra 550B up to 128K context).*

[Quickstart](#quickstart-in-60-seconds) • [Architecture](#subsystem-architecture) • [Cognitive Modes](#cognitive-engine--operational-modes) • [Multimodal Vision](#multimodal-vision--antigravity-context-deck) • [Hybrid LLM & Cloud](#hybrid-inference--cloud-scaling) • [Memory Engine](#3-tier-memory-engine) • [Tools](#registered-agent-tools) • [Documentation](ARCHITECTURE.md)

</div>

---

## Highlights & Key Capabilities

- **Antigravity-Grade Multimodal Vision & Input Deck**: Paste screenshots (`Ctrl+V`), drag-and-drop UI mockups or documents, and open the upward-popping **Add Context** (`+`) menu for media, `@ file mentions`, `/ actions`, and browser links. Automatically routes visual queries to `qwen2.5-vl:7b` with high-speed in-memory PDF/document text extraction.
- **Zero-VRAM CPU Embeddings**: High-performance semantic vector embeddings executed entirely on CPU via `fastembed` (`BAAI/bge-small-en-v1.5`), preserving 100% of GPU VRAM for LLM generation.
- **Native Desktop & Continuous Development App**: Runs as a lightweight native desktop application (`maidere` / `python -m maidere`) with OS dock integration, intelligent server lifecycle management, and a live hot-reloading development mode (`--dev`).
- **Dynamic 5-Tier Context Selector**: Real-time context window switcher scalable from **8K (Local Fast)** up to **128K (Cloud Extreme)** with automatic token trimming and context-aware KV cache management.
- **System 2 Deliberation Pipeline**: Multi-stage cognitive deconstruction, divergent exploration, formal proof verification, and convergent synthesis with real-time `<think>...</think>` streaming.
- **Hybrid Local + Cloud Execution**: Auto-routes between local Ollama instances (`qwen2.5:7b-instruct`, `qwen2.5:3b`, `deepseek-r1:7b`) and cloud-accelerated endpoints (NVIDIA Nemotron 3 Ultra 550B, OpenAI-compatible APIs).
- **In-Process SQLite Vector Database**: ACID-compliant SQLite WAL mode with `sqlite-vec` extension (`vec0` virtual table), eliminating external vector database overhead.
- **3-Tier Self-Healing Memory**: Persistent hierarchical memory combining rule enforcement (`RULES.md`), structured topic auto-memory with LRU eviction (`memory/*.md`), and turn-level fact extraction with Jaccard novelty filtering.
- **Specialized Subagent Fleet**: Ephemeral subagents (`researcher`, `plan`, `verification`, `validation`, `reviewer`, `staffer`) communicating via a microsecond-stamped file artifact bus.
- **Local Metasearch & Namespace-Isolated Tools**: Private metasearch via local SearXNG, ephemeral Playwright web scraper, `shell`/`code_runner` subprocesses confined with [bubblewrap](https://github.com/containers/bubblewrap) (read-only root, workspace-only writes, no network, scrubbed environment), and bidirectional Obsidian vault synchronization.

---

## Quickstart in 60 Seconds

### 1. System Requirements
- **Workstation OS**: Linux Mint 21+, Ubuntu 22.04+, or modern Linux distribution.
- **Local GPU**: NVIDIA GPU with $\ge$ 8GB VRAM (e.g. RTX 5060, 4060, 3060).
- **System Memory**: 16 GB RAM.
- **Prerequisites**: Python 3.12 (`>=3.12, <3.14`), Docker & Docker Compose, Ollama, [uv](https://docs.astral.sh/uv/) (recommended), `bubblewrap` (`sudo apt install bubblewrap`) for tool isolation.

### 2. Setup & Installation
```bash
# Clone the repository
git clone https://github.com/Khazar451/maidere.git
cd maidere

# Option A: Install using uv (recommended for sub-second installs)
uv sync

# Option B: Standard Python virtual environment
python3 -m venv .venv
source .venv/bin/activate
pip install -e .

# Start local Ollama & pull recommended models
ollama pull qwen2.5:7b-instruct
ollama pull qwen2.5:3b
ollama pull deepseek-r1:7b
ollama pull qwen2.5-vl:7b # Multimodal vision for screenshots & documents

# Configure secrets for the Docker sidecars (compose refuses to start without them)
cp .env.example .env
sed -i "s/^SEARXNG_SECRET=.*/SEARXNG_SECRET=$(openssl rand -hex 32)/" .env
sed -i "s/^GRAFANA_ADMIN_PASSWORD=.*/GRAFANA_ADMIN_PASSWORD=$(openssl rand -hex 16)/" .env

# Launch background infrastructure (SearXNG metasearch, Prometheus, Grafana)
docker compose up -d
```

### 3. Launching Maidere

Maidere can be launched in three modes:

```bash
# 1. Native Desktop Application (Default)
maidere
# or, from a checkout before the console script is on PATH:
python -m maidere

# 2. Continuous Development Mode (Live hot-reloading on file change)
maidere --dev

# 3. Register Desktop App in OS Start Menu & App Launcher
maidere install-desktop

# 4. Headless Server Mode (Browser UI at http://127.0.0.1:8000)
.venv/bin/uvicorn maidere.api.app:app --host 127.0.0.1 --port 8000
```

`uv sync` installs Maidere as a package, so the `maidere` command and `python -m maidere` work from any directory. Relative paths such as `db/maidere.db` and `workspace/` stay inside this repository when you run from a checkout, and move to `~/.local/share/maidere` for a plain `pip install`. Set `MAIDERE_RUNTIME_ROOT` to choose the base explicitly. Logs are written as JSON to `logs/maidere.jsonl` (rotated at 5 MB), not only printed.

### 4. Access Control

The API binds to **loopback only** (`127.0.0.1`) and every request requires a bearer token. The agent can run shell commands and read your workspace, so this is not optional.

- On first start Maidere generates a random token and stores it in `~/.config/maidere/token` (mode `0600`). The desktop launcher and `launch.sh` pass it to the UI automatically via the URL fragment, which the browser never sends over the network.
- Opening `http://127.0.0.1:8000` by hand shows a prompt; paste the token from that file once and it is remembered in the browser.
- Scripts and other clients send `Authorization: Bearer <token>` (or `X-Maidere-Token: <token>`). The WebSocket at `/ws/chat` expects `{"type": "auth", "token": "<token>"}` as its first frame and replies `{"type": "auth_ok"}`.
- To pin a token yourself set `API_TOKEN=` in `.env`. To expose Maidere beyond your machine, put a TLS-terminating reverse proxy in front of it, keep `API_HOST=127.0.0.1`, and add the proxy's origin to `ALLOWED_ORIGINS=`.
- `/health`, `/` and `/static/*` are public. `/metrics` accepts unauthenticated scrapes from loopback only (the bundled Prometheus runs on the host network), or from anywhere with a token.

```bash
TOKEN=$(cat ~/.config/maidere/token)
curl -H "Authorization: Bearer $TOKEN" http://127.0.0.1:8000/threads
```

### 5. Tool Sandbox

The `shell` and `code_runner` tools run every subprocess through `tools/sandbox.py`. `GET /health` reports the active mode as `sandbox_mode`.

- **`bwrap`** (default when available): a fresh bubblewrap sandbox per command with a read-only `/usr`, the workspace as the *only* writable host path, tmpfs `/tmp` and `HOME`, new user/PID/IPC/UTS namespaces, **no network** (`TOOL_SANDBOX_ALLOW_NETWORK=true` to opt in), a cleared environment (the server's API keys are never inherited), `prlimit` memory/file-size caps, and a 30 s timeout. The project virtualenv is mounted read-only so `code_runner` keeps its packages. Your home directory, `/etc/passwd`, and the Maidere repository itself do not exist inside the sandbox.
- **`restricted`** (automatic fallback, logged as a warning): no namespaces. Path arguments must stay inside the workspace, interpreters refuse inline code (`python -c`, `node -e`), `find -exec` is blocked, and the environment is still scrubbed. This confines arguments, not what a script does once it runs; install `bubblewrap` rather than relying on it.
- **`off`**: cwd only. Development use.

Set `TOOL_SANDBOX_MODE=bwrap` to fail closed: tools refuse to run at all if bubblewrap is unavailable. On Ubuntu 24.04+ unprivileged user namespaces are restricted by AppArmor, but the stock `bubblewrap` package ships a profile that permits them.

---

## Subsystem Architecture

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
│ │    (Proof verification: ~4.7 GB)    │ │ │  - Prometheus Instrumentator (/metrics)  │ │
│ │  - Fast: qwen2.5:3b (on-demand)     │ │ ├──────────────────────────────────────────┤ │
│ │  - Dynamic KV Cache (8K/16K/32K)    │ │ │ LangGraph Agent Core                     │ │
│ │  - CUDA Driver Buffers (~1.5 GB)    │ │ │  - Cyclic: trim -> remember -> think ->  │ │
│ └─────────────────────────────────────┘ │ │    act -> evaluate -> verify -> converge │ │
├─────────────────────────────────────────┤ │  - Isolated Subagent Contexts            │ │
│         CLOUD INFERENCE BRIDGE          │ │  - File-Based Artifact Bus (.maidere/)   │ │
│ ┌─────────────────────────────────────┐ │ ├──────────────────────────────────────────┤ │
│ │ NVIDIA NIM / OpenAI Cloud Endpoints │ │ │ In-Process CPU & Storage Subsystems      │ │
│ │  - nvidia/nemotron-3-ultra-550b     │ │ │  - FastEmbed (bge-small-en-v1.5 on CPU)  │ │
│ │  - Context: 64K Massive / 128K Extr │ │ │  - SQLite Database (WAL Mode Enabled)    │ │
│ │  - Thinking Mode & Reasoning Delta  │ │ │    * sqlite-vec (vec0 Virtual Table)     │ │
│ └─────────────────────────────────────┘ │ │    * LangGraph Checkpoints & Audit Log   │ │
│                                         │ │    * Persistent APScheduler Job Store    │ │
│                                         │ ├──────────────────────────────────────────┤ │
│                                         │ │ Docker Bridge Services                   │ │
│                                         │ │  - SearXNG Metasearch (Port 8080)        │ │
│                                         │ │  - Prometheus TSDB (Port 9090)           │ │
│                                         │ │  - Grafana Telemetry (Port 3000)         │ │
│                                         │ └──────────────────────────────────────────┘ │
└─────────────────────────────────────────┴──────────────────────────────────────────────┘
```

---

## Cognitive Engine & Operational Modes

Maidere provides three cognitive paradigms selectable from the top navigation bar or bottom input deck:

| Mode | Selector | Execution Pipeline & Behavior |
| :--- | :--- | :--- |
| **Auto** *(Default)* | `Auto` | **Dynamic Complexity & Vision Routing**: Classifies task complexity (`SIMPLE` &rarr; `qwen2.5:3b`, `COMPLEX` &rarr; `qwen2.5:7b-instruct`, `REASONING` &rarr; `deepseek-r1:7b`, `VISION` &rarr; `qwen2.5-vl:7b`) with sub-millisecond intent evaluation. |
| **Thinking** | `Thinking` | **Single-Pass Chain-of-Thought**: Generates step-by-step reasoning inside `<think>...</think>` scratchpad tags with real-time token streaming and collapsible UI accordions. |
| **Deep Reason** | `Deep Reason` | **Multi-Stage System 2 Deliberation**: Exhaustive 3-phase cognitive framework:<br>1. *Divergent Exploration*: Problem deconstruction, alternative hypotheses, falsification stress-testing.<br>2. *Analytical Verification*: Independent recalculation of math, state mutations, and citation integrity.<br>3. *Convergent Synthesis*: Formal synthesis with fourth-wall reviewer leakage prevention. |

---

## Multimodal Vision & Antigravity Context Deck

Maidere features an **Antigravity-grade context ingestion deck** designed for rapid multi-modal interaction:

- **Upward-Opening Add Context Menu (`+`)**:
  - `🖼️ Media`: Multi-file picker for screenshots (`.png`, `.jpg`, `.webp`), PDFs, documents, and code files.
  - `@ Mentions`: Instant `@` token injection for referencing workspace files, notes, or memories.
  - `☑ Actions`: Triggers slash workflows (`/plan`, `/review`, `[deep-research]`).
  - `🌐 Browser`: Quick web URL scraping and browsing query injection.
- **Attachment Preview Tray & Lightbox**:
  - Rendered above the input textarea with real-time file previews.
  - Image thumbnails feature hover-zoom and full-screen lightbox modal on click.
  - Document chips feature color-coded format badges (`PDF`, `CODE`, `TEXT`), formatted file sizes, and single-click removal buttons.
- **Clipboard & Drag-and-Drop Ingestion**:
  - Direct `Ctrl+V` screenshot pasting anywhere on the interface.
  - Drag-and-drop onto the input deck or chat stream with active dropzone glow highlighting.
- **Fast Attachment Processing Engine**:
  - High-speed PDF text extraction via `/usr/bin/pdftotext` with pure-Python FlateDecode zlib stream decompression fallback.
  - Automatic code and markdown formatting injection into turn prompts.
- **Automatic Multimodal Routing**:
  - Image payloads automatically elevate complexity to `TaskComplexity.VISION` and route to `qwen2.5-vl:7b` (or cloud OpenAI vision endpoints).

---

## Dynamic Context Window Selector

Maidere allows users to tune the context window size on the fly via the top bar `CTX:` selector:

| Context Tier | Tokens (`num_ctx`) | Target Engine | KV Cache Impact | Best For |
| :--- | :--- | :--- | :--- | :--- |
| **8K - Medium** | `8192` | Local Ollama | ~0.46 GB | High-speed chit-chat & lightweight scripts |
| **16K - High** *(Default)* | `16384` | Local Ollama | ~0.92 GB | **Balanced standard**; 2.4 GB VRAM headroom |
| **32K - Ultra** | `32768` | Local Ollama | ~1.84 GB | Deep multi-turn conversations & complex code |
| **64K - Cloud Massive** | `65536` | NVIDIA Nemotron | Offloaded | Enterprise refactoring & massive documentation |
| **128K - Cloud Extreme** | `131072` | NVIDIA Nemotron | Offloaded | Full repository analysis & extensive research |

*Selecting a cloud model automatically promotes the context window to 64K, and switches back safely when returning to local models.*

---

## Hybrid Inference & Cloud Scaling

Configure cloud acceleration in `.env` to unlock models like **NVIDIA Nemotron 3 Ultra 550B**:

```ini
# NVIDIA NIM / OpenAI-Compatible Cloud Endpoint
CLOUD_API_BASE=https://integrate.api.nvidia.com/v1
CLOUD_API_KEY=nvapi-your-key-here
CLOUD_MODEL=nvidia/nemotron-3-ultra-550b-a55b
CLOUD_NUM_CTX=65536
```

- **Native Thinking Protocol**: Automatically sends `extra_body={"chat_template_kwargs": {"enable_thinking": True}}` and streams `delta.reasoning_content` in real time.
- **Fail-Safe Fallback**: If the cloud API is unreachable or rate-limited, Maidere falls back gracefully to local Ollama models.

---

## 3-Tier Memory Engine

```mermaid
graph TD
    subgraph T1["Tier 1: Instruction Memory"]
        R["RULES.md"] --> BAK[".RULES.md.bak"]
    end

    subgraph T2["Tier 2: Structured Auto-Memory"]
        IDX["index.json (LRU Cache, Max 40)"] <--> TOPICS["memory/*.md"]
    end

    subgraph T3["Tier 3: Session Fact Extraction"]
        INSPECT["Turn Completion"] --> NOVELTY{"Jaccard Overlap < 0.6?"}
        NOVELTY -->|Yes| PROMO["Promote to Tier 2 Topic"]
        NOVELTY -->|No| DEDUP["Deduplicate & Discard"]
    end

    T1 --> INJECT["Turn Prompt Context"]
    T2 --> INJECT
    VEC["sqlite-vec Vector Recall"] --> INJECT
```

1. **Tier 1 (Instruction Memory)**: Manages `.maidere/RULES.md` with automatic `.bak` backups created prior to file mutation for one-step rollback.
2. **Tier 2 (Structured Auto-Memory)**: Topic markdown files (`.maidere/memory/*.md`) indexed via `index.json`, enforcing the Two-Step Save Invariant, LRU eviction capped at 40 indexed entries, and startup reconciliation.
3. **Tier 3 (Session Fact Extraction)**: Extracts user preferences, architectural rules, and project decisions upon turn completion, verifies novelty with Jaccard word-overlap scoring, and auto-promotes unique facts to Tier 2.

---

## Registered Agent Tools

| Tool | Purpose | Signature |
| :--- | :--- | :--- |
| `read_file` | Read workspace file contents with path confinement | `path: str` |
| `write_file` | Atomic file write with automatic directory creation | `path: str, content: str` |
| `list_dir` | List files and directories within the sandbox | `path: str` |
| `shell` | Run allowlisted command inside a bubblewrap sandbox (`shell=False`, no network, workspace-only writes) | `command: str` |
| `code_runner` | Execute a Python script inside the same sandbox (`python -I`, venv packages available) | `code: str` |
| `browser` | Scrape webpage DOM text via ephemeral Playwright | `url: str` |
| `delegate_task` | Spawn isolated subagents (`researcher`, `plan`, `verification`, `validation`, `reviewer`, `staffer`) | `prompt: str, subagent_type: str, max_turns: int` |
| `manage_memory`| 3-Tier memory manager (read/save topic facts, view rules) | `action: str, topic: str, content: str` |
| `agy_staffer` | External Gemini 3.8 Flash delegate via Antigravity CLI | `prompt: str, persona: str` |
| `obsidian` | Bi-directional Obsidian note search, read, write, and open | `action: str, note_name: str, content: str` |
| `scheduler` | Schedule persistent cron jobs via SQLite APScheduler | `action: str, job_id: str, ...` |
| `web_search` | Private local metasearch via SearXNG | `query: str` |
| `list_available_skills` | Discover installed specialized skills | `{}` |
| `load_skill` | Load executable instructions for a skill | `skill_name: str` |

---

## Automated Verification Suite

Maidere maintains an exhaustive automated test suite with **258 tests across 24 modules**:

```bash
# Execute full test suite
HF_HUB_OFFLINE=1 .venv/bin/python -m unittest discover -s tests -p "test_*.py" -v
```

```text
Ran 258 tests in 5.322s — OK (100% Passed, 0 Failures, 0 Errors)
```

| Test Module | Coverage Area | Status |
| :--- | :--- | :---: |
| `test_agent.py` | LangGraph cyclic execution, message trimming, tool loops | **PASS** |
| `test_multimodal.py` | Multimodal vision routing, PDF stream extraction, base64 normalization, cloud formats | **PASS** |
| `test_cloud_llm.py` | NVIDIA NIM & OpenAI cloud client, Nemotron thinking stream | **PASS** |
| `test_desktop_launcher.py`| Desktop lifecycle, port probing, signal handling, XDG launcher | **PASS** |
| `test_memory_tiers.py` | Tier 1 RULES.md backups, Tier 2 LRU auto-memory, Tier 3 Jaccard filter | **PASS** |
| `test_subagent.py` | Ephemeral subagent isolation, handoff briefs, tool restrictions | **PASS** |
| `test_router.py` | Tri-model intent classifier, reasoning detection, overrides | **PASS** |
| `test_synthesis_hardening.py` | Inline citation bracket normalization, fourth-wall leak filtering | **PASS** |
| `test_tools.py` | Path traversal prevention, shell allowlist, safe argument parsing | **PASS** |
| `test_obsidian.py` | Obsidian vault resolution, frontmatter formatting, note search | **PASS** |
| `test_monitoring.py` | Prometheus metrics exposition, latency histograms, token counters | **PASS** |
| `test_phase4.py` | Python code runner, Playwright browser cleanup, APScheduler crons | **PASS** |
| `test_web_ui.py` | Static serving, WebSocket protocol, theme settings, thread persistence | **PASS** |

---

## Documentation

- **[ARCHITECTURE.md](ARCHITECTURE.md)**: Exhaustive technical specification, hardware budget allocation matrix, security controls, and design patterns.
- **[CONTRIBUTING.md](CONTRIBUTING.md)**: Guidelines for extending tools, adding skills, and running benchmarks.

---

## Running in Docker

Sidecars only (SearXNG, Prometheus, Grafana):

```bash
docker compose up -d
```

Maidere and Ollama as well, with the API published only on `127.0.0.1:8000`:

```bash
docker compose --profile app up -d --build
docker compose --profile app exec maidere cat /data/api-token
```

Open `http://127.0.0.1:8000/#token=` plus that value. On a machine with an NVIDIA GPU, pass the GPU through to Ollama:

```bash
docker compose -f docker-compose.yml -f docker-compose.gpu.yml --profile app up -d
```

Inside the container the process listens on `0.0.0.0` so Docker can publish the port. The host binding stays on loopback. Tool commands use the restricted sandbox there, because bubblewrap user namespaces are not available in a default container.

`/live` answers when the process is up. `/ready` also checks that the database accepts a write, the data volume has free space, and Ollama is reachable. SearXNG is reported and does not by itself mark the service unready.

## License

MIT License — Free for personal, academic, and commercial use.
