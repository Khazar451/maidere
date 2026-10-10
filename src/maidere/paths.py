"""Filesystem locations that do not depend on the process working directory.

Relative settings (``db/maidere.db``, ``workspace``, ``logs/maidere.jsonl``) are
anchored at the repository when running from a checkout, and at the XDG data
directory when running from an installed wheel. ``MAIDERE_RUNTIME_ROOT``
overrides both. Absolute paths are used as given.
"""

from __future__ import annotations

import os
from pathlib import Path

_PKG = Path(__file__).resolve().parent


def package_dir() -> Path:
    """Directory containing this package (the checkout or site-packages)."""
    return _PKG


def repo_root() -> Path | None:
    """Repository root for a source checkout or an editable install."""
    candidate = _PKG.parent.parent  # <repo>/src/maidere -> <repo>
    if (candidate / "pyproject.toml").is_file() and (candidate / "src" / "maidere").is_dir():
        return candidate
    return None


def data_dir() -> Path:
    """XDG data directory used by an installed copy (db, logs, workspace)."""
    override = os.environ.get("MAIDERE_DATA_DIR", "").strip()
    if override:
        path = Path(override).expanduser()
    else:
        base = os.environ.get("XDG_DATA_HOME", "").strip()
        root = Path(base).expanduser() if base else Path.home() / ".local" / "share"
        path = root / "maidere"
    path.mkdir(parents=True, exist_ok=True)
    return path


def runtime_root() -> Path:
    """Base directory for relative configuration paths."""
    override = os.environ.get("MAIDERE_RUNTIME_ROOT", "").strip()
    if override:
        path = Path(override).expanduser().resolve()
        path.mkdir(parents=True, exist_ok=True)
        return path
    return repo_root() or data_dir()


def resolve_config_path(value: str) -> Path:
    """Resolve a settings path without consulting the working directory."""
    path = Path(value).expanduser()
    if path.is_absolute():
        return path
    return (runtime_root() / path).resolve()


def resolve_db_path(value: str) -> str:
    """Absolute database path, or ``:memory:`` unchanged."""
    if value == ":memory:":
        return value
    resolved = resolve_config_path(value)
    resolved.parent.mkdir(parents=True, exist_ok=True)
    return str(resolved)


def workspace_dir() -> Path:
    """Agent workspace, created if needed. Reads settings at call time."""
    from maidere.core.config import settings

    path = resolve_config_path(settings.agent_workspace)
    path.mkdir(parents=True, exist_ok=True)
    return path


def static_dir() -> Path:
    """Bundled web UI."""
    bundled = package_dir() / "static"
    if bundled.is_dir():
        return bundled
    root = repo_root()
    if root is not None and (root / "static").is_dir():
        return root / "static"
    return bundled


def skills_dir() -> Path:
    """Bundled skill packages (SKILL.md)."""
    bundled = package_dir() / "skills"
    if bundled.is_dir():
        return bundled
    root = repo_root()
    if root is not None and (root / "skills").is_dir():
        return root / "skills"
    return bundled


def agents_dir() -> Path:
    """Bundled subagent prompt files."""
    bundled = package_dir() / "agents"
    if bundled.is_dir():
        return bundled
    root = repo_root()
    if root is not None and (root / "agents").is_dir():
        return root / "agents"
    return bundled
