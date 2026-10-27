"""API access token management for Maidere.

The API is protected by a single bearer token. Resolution order:

1. ``settings.api_token`` (``API_TOKEN`` env var / ``.env``) if non-empty.
2. The contents of ``settings.api_token_file`` (default ``~/.config/maidere/token``).
3. If the file does not exist, a new random token is generated and written
   there with mode 0600.

This module deliberately has no FastAPI dependency so the desktop launcher
can resolve the same token before the server process starts.
"""

from __future__ import annotations

import contextlib
import os
import secrets
import stat
from pathlib import Path

import structlog

from maidere.core.config import settings

logger = structlog.get_logger()

_TOKEN_BYTES = 32
_MIN_TOKEN_LENGTH = 16


def token_file_path() -> Path:
    """Absolute path of the persisted token file."""
    return Path(settings.api_token_file).expanduser()


def _read_token_file(path: Path) -> str | None:
    try:
        value = path.read_text(encoding="utf-8").strip()
    except FileNotFoundError:
        return None
    except OSError as exc:
        logger.warning("api_token_file_unreadable", path=str(path), error=str(exc))
        return None
    return value or None


def _write_token_file(path: Path, token: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_name(f".{path.name}.tmp")
    fd = os.open(tmp_path, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            handle.write(token + "\n")
    except Exception:
        tmp_path.unlink(missing_ok=True)
        raise
    os.replace(tmp_path, path)
    with contextlib.suppress(OSError):
        path.chmod(stat.S_IRUSR | stat.S_IWUSR)


def generate_token() -> str:
    """Generate a new URL-safe random token."""
    return secrets.token_urlsafe(_TOKEN_BYTES)


def load_or_create_token() -> str:
    """Resolve the active API token, creating and persisting one if needed."""
    configured = (settings.api_token or "").strip()
    if configured:
        return configured

    path = token_file_path()
    existing = _read_token_file(path)
    if existing:
        return existing

    token = generate_token()
    _write_token_file(path, token)
    logger.info("api_token_created", path=str(path))
    return token


def get_api_token() -> str:
    """Return the active API token (alias kept short for call sites)."""
    return load_or_create_token()


def verify_token(candidate: str | None) -> bool:
    """Constant-time comparison of a presented token against the active token."""
    if not candidate:
        return False
    expected = get_api_token()
    return secrets.compare_digest(candidate.strip().encode("utf-8"), expected.encode("utf-8"))


def token_strength_warning() -> str | None:
    """Return a human-readable warning if the configured token is weak, else None."""
    token = get_api_token()
    if len(token) < _MIN_TOKEN_LENGTH:
        return (
            f"API_TOKEN is only {len(token)} characters; use at least {_MIN_TOKEN_LENGTH} "
            "(or leave it unset to auto-generate a strong one)."
        )
    return None


def rotate_token() -> str:
    """Generate a new token, persist it, and return it.

    Only meaningful when the token comes from the file (not from ``API_TOKEN``).
    """
    token = generate_token()
    _write_token_file(token_file_path(), token)
    logger.info("api_token_rotated", path=str(token_file_path()))
    return token
