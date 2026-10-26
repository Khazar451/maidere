# Contributing to Maidere

This document outlines the architecture, coding standards, development workflows, and testing procedures for contributing to Maidere. As this is a private repository, these guidelines apply to all contributors and team members.

---

## 1. Core Principles

Maidere is an autonomous, self-hosted AI agent architecture built for local execution without external API dependencies.

1. **Local-First & Zero Cloud Leakage**: All computation, inference, embeddings, and persistence must run locally. Never introduce dependencies that silently ship user queries, prompt history, or vector embeddings to third-party endpoints.
2. **Deterministic Fallbacks**: Local models (such as 7B and 3B parameter models) occasionally make formatting errors. Every extraction, routing, tool parsing, and citation mechanism must have deterministic fallback guards.
3. **Strict Data Hygiene**: Credentials, session tokens, absolute user directory paths, and databases must never be committed to git or exposed in application logs.
4. **Emoji-Free Codebase**: Do not use emojis in application code, logs, user interface elements, system prompt templates, or documentation.

---

## 2. Development Environment Setup

### Prerequisites
- Python 3.12 or higher
- `uv` package manager (recommended) or standard Python `venv`
- `ollama` with the following models pulled locally:
  - `qwen2.5:7b-instruct` (Primary instruction model)
  - `qwen2.5:3b` (Optional; used only by the complexity router for API requests that send `"model": "auto"`)
  - `deepseek-r1:7b` (Optional, required for Deep Reasoning mode)
- `docker` and `docker compose` (for auxiliary services: SearXNG, Prometheus, Grafana)

### Installation
1. Clone the repository:
   ```bash
   git clone git@github.com:Khazar451/maidere.git
   cd maidere
   ```

2. Create and activate a virtual environment:
   ```bash
   uv venv
   source .venv/bin/activate
   ```

3. Install project dependencies in editable mode with development packages:
   ```bash
   uv pip install -e ".[dev]"
   ```

4. Initialize environment configuration:
   ```bash
   cp .env.example .env
   ```
   Inspect [.env](.env) to ensure default hostnames and ports match your local environment. Note that `.env` is ignored by git.

---

## 3. Architecture Overview

Contributors should familiarize themselves with the modular layout before making changes:

- **[`core/`](src/maidere/core/)**: Agent runtime engine
  - [`agent.py`](src/maidere/core/agent.py): LangGraph StateGraph definition, node pipelines (`trim`, `remember`, `think`, `act`, `evaluate`, `critique`, `refine`), anti-cheat guardrails, and citation formatting.
  - [`state.py`](src/maidere/core/state.py): LangGraph typed state dictionary schema.
  - [`router.py`](src/maidere/core/router.py): Dynamic model selector and task complexity classifier (SIMPLE, COMPLEX, REASONING).
  - [`llm.py`](src/maidere/core/llm.py): Asynchronous Ollama HTTP client and token streaming handler.
  - [`memory.py`](src/maidere/core/memory.py): Vector similarity store and recall backed by `sqlite-vec`.
  - [`embeddings.py`](src/maidere/core/embeddings.py): Local CPU embeddings via FastEmbed with deterministic offline fallback.
  - [`subagent.py`](src/maidere/core/subagent.py): Sub-agent orchestration, execution loops, and lifecycle tracking.
  - [`audit.py`](src/maidere/core/audit.py): Secret sanitization and audit log recording.

- **[`api/`](src/maidere/api/)**: Web backend
  - [`app.py`](src/maidere/api/app.py): FastAPI application initialization, lifecycle hooks, and CORS setup.
  - [`routes.py`](src/maidere/api/routes.py): REST endpoints and real-time streaming WebSocket (`/ws/chat`).
  - [`schemas.py`](src/maidere/api/schemas.py): Pydantic request and response models.

- **[`tools/`](src/maidere/tools/)**: Agent capabilities
  - [`filesystem.py`](src/maidere/tools/filesystem.py): Sandboxed workspace read and write tools.
  - [`shell.py`](src/maidere/tools/shell.py): Safe command execution tool.
  - [`web_search.py`](src/maidere/tools/web_search.py): Local SearXNG meta-search integration.
  - [`browser.py`](src/maidere/tools/browser.py): Headless page content scraper and text extractor.
  - [`obsidian.py`](src/maidere/tools/obsidian.py): Local Obsidian Vault integration and note manager.
  - [`delegate.py`](src/maidere/tools/delegate.py): Multi-agent delegation tool for spawning specialized sub-agents.
  - [`registry.py`](src/maidere/tools/registry.py): Central tool registry and Ollama JSON function schema generation.

- **[`agents/`](src/maidere/agents/)**: Markdown system prompt templates for sub-agent roles (`researcher.md`, `critic.md`, `plan.md`, `code-reviewer.md`, `tech-hardware.md`, `verification.md`, `validation.md`).

- **[`skills/`](src/maidere/skills/)**: Procedural step-by-step skill markdown guidelines loaded dynamically into memory.

- **[`static/`](src/maidere/static/)**: Browser interface (`index.html`, `style.css`, `app.js`).

---

## 4. Coding Standards & Conventions

### Asynchronous Design
- All database operations, external process executions, and HTTP requests must be non-blocking using `async` and `await`.
- Use `asyncio.create_task` for background operations, and ensure cancellation exceptions (`asyncio.CancelledError`) are caught and handled cleanly.

### Typing & Modern Python
- Use Python 3.12+ syntax throughout (e.g., `list[str]`, `dict[str, Any]`, `X | None`).
- Annotate all function parameters and return types.
- Avoid raw type casting or `Any` where a specific type or `TypedDict` can be expressed.

### Logging
- Use `structlog` for all application logging:
  ```python
  import structlog
  logger = structlog.get_logger()
  
  await logger.ainfo("operation_completed", item_count=len(items))
  ```
- Never use `print()` or the standard unconfigured `logging` library.
- Provide structured key-value arguments rather than string interpolations in log calls.

### Citation Standards
- When the agent synthesizes research, sources must be formatted as inline bracketed integers (e.g., `[1]`, `[2]`).
- A dedicated `## Sources` block at the end of the text must map each integer to a markdown link: `[1] [Title](https://actual-url.com)`.
- Never generate free-form "References" or "Bibliography" sections in prompt templates or agent outputs.

### Style and Emojis
- Do not include emojis in code, log messages, UI elements, or documentation.

---

## 5. Security & Data Hygiene

1. **Ignored Files**: Never force-add files ignored by [.gitignore](.gitignore). This includes:
   - Environment files (`.env`, `.env.*`)
   - SQLite databases (`db/*.db`, `*.sqlite3`)
   - Log files (`logs/`, `*.log`)
   - Test cache (`.pytest_cache/`)
   - Workspace directories (`workspace/`)
2. **Secret Sanitization**:
   - Any text stored into memories or audit tables must pass through `sanitize_secrets()` in [`core/audit.py`](src/maidere/core/audit.py).
   - Test strings in unit tests must use standard mock placeholders (e.g., `AKIAIOSFODNN7EXAMPLE`) and never real keys.
3. **Workspace Isolation**:
   - File read and write operations initiated by the agent must remain bounded to `settings.agent_workspace`.
4. **API Authentication**:
   - All routes are protected by the bearer-token middleware in [`api/auth.py`](src/maidere/api/auth.py). Do not add routes to the public allowlist (`PUBLIC_EXACT_PATHS` / `PUBLIC_PREFIXES`) unless they expose nothing and are needed before the UI has a token (e.g. `/health`).
   - New frontend calls must go through `apiFetch()` in `static/app.js`, never raw `fetch()`, so the `Authorization` header is attached.
   - Tests that build the app must pin a token and send it: use `patch_api_token()` and `AUTH_HEADERS` from `tests/authhelpers.py` (see `tests/test_api.py`). WebSocket tests send `WS_AUTH_FRAME` first. Never let a test read or write `~/.config/maidere/token`.

---

## 6. Testing Requirements

### Running Tests
The suite is `unittest`, collected by pytest so coverage is measured. It runs offline: Hugging Face Hub stays disabled, and the one end-to-end turn talks to an in-process Ollama stub (`tests/test_ollama_roundtrip.py`), not a real model server.

```bash
uv sync --group dev
HF_HUB_OFFLINE=1 .venv/bin/ruff check src tests
HF_HUB_OFFLINE=1 .venv/bin/ruff format --check src tests
HF_HUB_OFFLINE=1 .venv/bin/mypy
HF_HUB_OFFLINE=1 .venv/bin/pytest
```

`mypy` type-checks the entire `src/maidere` package with `disallow_untyped_defs` and `no_implicit_optional`; do not silence new type errors without a specific justification.

### Writing New Tests
- Place tests in the [`tests/`](tests/) directory following the `test_<module>.py` naming convention.
- Use `unittest.mock.AsyncMock` or `patch` to mock `maidere.core.llm.chat` and `maidere.core.llm.chat_stream`.
- Do not rely on an active Ollama instance running during tests. A local stub is fine.
- When creating asynchronous test cases, use standard `unittest.IsolatedAsyncioTestCase` or `asyncio.run(coroutine)` within test methods.
- Ensure 100% of tests pass before opening a pull request.

---

## 7. Git Workflow

### Branching Strategy
- `main`: Production-ready, stable codebase.
- Feature branches: `feature/<feature-name>`
- Bugfix branches: `fix/<bug-description>`
- Refactor branches: `refactor/<target-module>`

### Commit Messages
Use clear, imperative commit messages describing the rationale:
```
feat: add topic change pruning to trim_messages_node
fix: prevent router fallback from bypassing thinking mode
test: add unit tests for index citation extraction
docs: update setup instructions in README
```

### Pre-Push Verification Checklist
Before pushing any branch or committing changes, run through this checklist:
- [ ] Test suite passes with exit code 0 (`HF_HUB_OFFLINE=1 .venv/bin/python -m unittest discover -s tests -p "test_*.py"`).
- [ ] `git status` shows no untracked database files, credentials, or logs.
- [ ] No live API keys, private tokens, or personal paths are introduced.
- [ ] No emojis were added to any files.
