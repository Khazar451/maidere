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

### Changed
- Logs are JSON lines in `logs/maidere.jsonl`.
- The GitHub tool no longer assumes a personal repository. Set `GITHUB_DEFAULT_REPO` to choose one.

## 0.1.0

- First published source of the local Maidere agent.
