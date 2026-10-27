# Changelog

## Unreleased

### Security
- The API binds to loopback and requires a bearer token. Uploads can no longer leave the workspace.
- `shell` and `code_runner` run inside bubblewrap when it is available, with home-directory exposure blocked when the workspace is inside `$HOME` and symlink/root-boundary escapes rejected.
- Dependency audit fixes for `urllib3` and `langgraph-sdk`.

### Added
- Installable package (`maidere`, `python -m maidere`) that runs from any working directory.
- Schema migrations, a shared database connection, and per-chat subagent progress events.
- `/live` and `/ready` checks, a container image, and a tagged release workflow.
- `/scheduler/tasks` endpoint for inspecting active scheduled jobs.
- Hourly SQLite WAL checkpoints (`DB_CHECKPOINT_INTERVAL_SECONDS`) and a compacting checkpoint on shutdown.
- `EXTRA_ALLOWED_COMMANDS` lets you permit more binaries in the shell tool (for example `uv,pytest,cargo,go`). Shells, `sudo` and network clients stay blocked.
- Web search falls back to Tavily or Serper (when a key is set) and then DuckDuckGo if SearXNG is offline. Set `SEARCH_FALLBACK_ENABLED=false` to keep search local.
- An empty model reply is retried (`EMPTY_RESPONSE_RETRIES`) and then reported with a clear message instead of a blank answer.
- Terminal tabs render ANSI colours and keep the last 1,000 lines.
- Uploaded images stay available for follow-up questions (`IMAGE_MEMORY_TURNS`, default 2; `IMAGE_MEMORY_MAX_IMAGES`, default 4). They are re-attached only when the question refers back to them, and stored images are downscaled to fit 1920x1080.

### Fixed
- Restored the `https://api.github.com` base URL in the GitHub tool after package rename.
- Sub-agent spawn limits (`AgentTool`) now reset per turn and isolate counters across concurrent chat threads.
- Skill matching now handles hyphenated skill names, multi-word trigger phrases, and YAML list triggers.
- User and project sub-agent definitions (`~/.maidere/agents` and `.maidere/agents`) now override built-in personas as documented, and custom system prompts safely escape literal braces.
- `restore_markdown_urls` no longer rewrites short numeric citations or non-title brackets into search links.
- Cloud endpoint resolution now respects custom `NVIDIA_BASE_URL` and `OPENAI_BASE_URL` values.
- Authenticated thread export (`exportThread`) now sends the bearer token and triggers a download instead of navigating to an unauthenticated tab, and WebSocket `final_response` events extract `session_id` from top-level fields.
- Timed-out subprocesses in `pdftotext` and `agy` are reaped cleanly, and invalid base64 text attachments return a clear placeholder.
- Shared SQLite initialization closes half-opened connections on startup failure.
- A turn that had already delegated no longer raises `TypeError` when tools are disabled.
- Images sent in Auto mode now reach a vision model. The default text model was being treated as the user's explicit choice, so the text-only model got the image and Ollama rejected it.
- The vision model is named `qwen2.5vl` (Ollama has no `qwen2.5-vl`). The default setting and the README pull command now use the real name.
- Tool schemas are left out for models that cannot take tools, such as vision models, which previously failed with a 400. Image turns on those models are text-only.
- A direct `web_search` or `browser` call from the orchestrator model is now handed to a researcher sub-agent. Before, Ollama discarded it and the user got an empty reply.
- Thread ages in the sidebar are correct outside UTC.
- The desktop launcher exits non-zero when the backend fails to start. Sponsored DuckDuckGo results are dropped from the search fallback.

### Changed
- Auto mode is described as what it does: the configured model, switching to the vision model for images. The complexity router (`OLLAMA_FAST_MODEL`, `ENABLE_MODEL_ROUTING`) runs only for API requests that send `"model": "auto"`. The model dropdown label and docs were corrected to match.
- Replaced silent `try/except/pass` blocks with structured debug logging or explicit `contextlib.suppress`, and enabled `RUF013`, `SIM`, `PIE`, `PERF`, `C4`, and `S110` lint rules across the codebase.
- Logs are JSON lines in `logs/maidere.jsonl`.
- The GitHub tool no longer assumes a personal repository. Set `GITHUB_DEFAULT_REPO` to choose one.

## 0.1.0

- First published source of the local Maidere agent.
