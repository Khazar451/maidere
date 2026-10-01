---
title: "[Feature]: Configurable Developer Shell Allowlist (uv, pytest, cargo, go)"
labels: ["enhancement", "security", "tools"]
---

### Problem Statement
`tools/shell.py` currently enforces a strict, hardcoded set of permitted binaries:

```python
ALLOWED_COMMANDS: set[str] = {
    "python3", "python", "node", "ls", "cat", "grep",
    "mkdir", "echo", "touch", "find", "head", "tail",
    "pwd", "cp", "mv", "rm", "wc", "git",
}
```

When users ask Maidere to run project test suites (`pytest`, `unittest`), modern Python package managers (`uv run`, `poetry run`), Rust tools (`cargo test`, `cargo clippy`), or Go commands (`go test`), execution is blocked by the allowlist:

```text
Error: Command 'uv' is not allowed. Permitted commands: ...
```

### Proposed Solution
1. **Configurable Settings Parameter**:
   Add an optional configuration key in `core/config.py`:
   ```python
   extra_allowed_commands: list[str] = ["uv", "pytest", "cargo", "go"]
   ```
   Configurable via `.env`:
   ```ini
   MAIDERE_EXTRA_ALLOWED_COMMANDS=uv,pytest,cargo,go,ruff
   ```

2. **Safe Resolution**:
   - Merge `ALLOWED_COMMANDS` with `settings.extra_allowed_commands`.
   - Before executing, verify that the binary resolves via `shutil.which(cmd)` within system paths.
   - Retain all security constraints: `shell=False`, argument list execution, timeout enforcement (30s), and working directory confinement inside `workspace/`.

### Acceptance Criteria
- [ ] Users can safely execute configured development tools (`uv`, `pytest`, `cargo`) within the workspace.
- [ ] Unknown or unauthorized system binaries remain strictly rejected.
- [ ] Unit tests in `tests/test_tools.py` verify custom allowlist configuration and boundary enforcement.
