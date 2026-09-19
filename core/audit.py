"""Audit logging and secret sanitization for Maidere data hygiene."""

import json
import re
from typing import Any

import structlog

from core.db import get_db

logger = structlog.get_logger()

# ---------------------------------------------------------------------------
# Regex patterns for sensitive credentials, API keys, tokens, and private keys
# ---------------------------------------------------------------------------
SECRET_PATTERNS = [
    # OpenAI & Standard API Keys
    (r"\bsk-[a-zA-Z0-9_\-]{20,}\b", "[REDACTED_SECRET]"),
    # Anthropic API Keys
    (r"\bsk-ant-[a-zA-Z0-9_\-]{20,}\b", "[REDACTED_SECRET]"),
    # Hugging Face Tokens
    (r"\bhf_[a-zA-Z0-9]{20,}\b", "[REDACTED_SECRET]"),
    # GitHub Tokens (Personal Access, OAuth, User-to-Server)
    (r"\b(ghp|gho|ghu|ghs|ghr)_[a-zA-Z0-9]{30,}\b", "[REDACTED_SECRET]"),
    (r"\bgithub_pat_[a-zA-Z0-9_]{30,}\b", "[REDACTED_SECRET]"),
    # AWS Access Key IDs
    (r"\bAKIA[0-9A-Z]{16}\b", "[REDACTED_SECRET]"),
    # Google AIza Keys
    (r"\bAIza[0-9A-Za-z\-_]{35}\b", "[REDACTED_SECRET]"),
    # Slack Tokens
    (r"\bxox[baprs]-[0-9a-zA-Z]{10,}-[0-9a-zA-Z]{10,}-[a-zA-Z0-9]{20,}\b", "[REDACTED_SECRET]"),
    # Bearer Tokens
    (r"(?i)\bBearer\s+[a-zA-Z0-9_\-\.]{16,}\b", "Bearer [REDACTED_SECRET]"),
    # Basic Auth Base64 Strings
    (r"(?i)\bBasic\s+[a-zA-Z0-9+/=]{16,}\b", "Basic [REDACTED_SECRET]"),
    # RSA / EC / OpenSSH Private Keys
    (r"-----BEGIN [A-Z ]*PRIVATE KEY-----[\s\S]*?-----END [A-Z ]*PRIVATE KEY-----", "[REDACTED_PRIVATE_KEY]"),
    # Key/Password Assignments in text, JSON, or env formats (e.g. "password": "...", api_key="...")
    (
        r"(?i)([\"']?(?:api[_-]?key|secret[_-]?key|auth[_-]?token|access[_-]?token|password|passwd|pwd|client[_-]?secret)[\"']?)\s*([:=])\s*([\"']?)([^\"'\s,;}{]{4,})([\"']?)",
        r"\1\2 \3[REDACTED_SECRET]\5",
    ),

]


def sanitize_secrets(text: str) -> str:
    """Sanitize sensitive credentials, API keys, tokens, and passwords from text."""
    if not text or not isinstance(text, str):
        return text

    sanitized = text
    for pattern, replacement in SECRET_PATTERNS:
        sanitized = re.sub(pattern, replacement, sanitized)

    return sanitized


async def record_audit(
    thread_id: str | None,
    tool_name: str,
    tool_args: dict[str, Any],
    tool_result: str,
    duration_ms: int,
    success: bool,
    db_path: str | None = None,
) -> None:
    """Sanitize and write tool execution entry to the audit_log database table."""
    try:
        from core.config import settings

        path = db_path or settings.db_path
        db = await get_db(path)
        try:
            # Sanitize arguments and results before persisting
            sanitized_args_str = sanitize_secrets(json.dumps(tool_args))
            sanitized_result = sanitize_secrets(tool_result)[:2000]

            await db.execute(
                """
                INSERT INTO audit_log (thread_id, tool_name, tool_args, tool_result, duration_ms, success)
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                (
                    thread_id,
                    tool_name,
                    sanitized_args_str,
                    sanitized_result,
                    duration_ms,
                    1 if success else 0,
                ),
            )
            await db.commit()
        finally:
            await db.close()
    except Exception as e:
        await logger.aerror("audit_log_failed", tool=tool_name, error=str(e))


# Alias for backward compatibility
log_audit_record = record_audit
