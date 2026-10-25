# Changelog

## Unreleased

### Security
- The API binds to loopback and requires a bearer token. Uploads can no longer leave the workspace.
- `shell` and `code_runner` run inside bubblewrap when it is available.
- Dependency audit fixes for `urllib3` and `langgraph-sdk`.

### Added
- Installable package (`maidere`, `python -m maidere`) that runs from any working directory.
- Schema migrations, a shared database connection, and per-chat subagent progress events.
- `/live` and `/ready` checks, a container image, and a tagged release workflow.
- Hourly SQLite WAL checkpoints (`DB_CHECKPOINT_INTERVAL_SECONDS`) and a compacting checkpoint on shutdown.
- `EXTRA_ALLOWED_COMMANDS` lets you permit more binaries in the shell tool (for example `uv,pytest,cargo,go`). Shells, `sudo` and network clients stay blocked.
- Web search falls back to Tavily or Serper (when a key is set) and then DuckDuckGo if SearXNG is offline. Set `SEARCH_FALLBACK_ENABLED=false` to keep search local.
- An empty model reply is retried (`EMPTY_RESPONSE_RETRIES`) and then reported with a clear message instead of a blank answer.
- Terminal tabs render ANSI colours and keep the last 1,000 lines.
- Uploaded images stay available for follow-up questions (`IMAGE_MEMORY_TURNS`, default 2; `IMAGE_MEMORY_MAX_IMAGES`, default 4). They are re-attached only when the question refers back to them, and stored images are downscaled to fit 1920x1080.

### Fixed
- A direct `web_search` or `browser` call from the orchestrator model is now handed to a researcher sub-agent. Before, Ollama discarded it and the user got an empty reply.
- Thread ages in the sidebar are correct outside UTC.
- The desktop launcher exits non-zero when the backend fails to start. Sponsored DuckDuckGo results are dropped from the search fallback.

### Changed
- Logs are JSON lines in `logs/maidere.jsonl`.
- The GitHub tool no longer assumes a personal repository. Set `GITHUB_DEFAULT_REPO` to choose one.

## 0.1.0

- First published source of the local Maidere agent.
