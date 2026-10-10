"""Shared helpers for authenticating test clients against the Maidere API.

Tests patch ``settings.api_token`` to a known value so no token file is ever
read from or written to the developer's home directory.
"""

from unittest.mock import patch

from maidere.core.config import settings

TEST_TOKEN = "maidere-test-token-0123456789abcdef"
AUTH_HEADERS = {"Authorization": f"Bearer {TEST_TOKEN}"}
WS_AUTH_FRAME = {"type": "auth", "token": TEST_TOKEN}


def patch_api_token(token: str = TEST_TOKEN):
    """Return a started patcher that pins ``settings.api_token``; caller stops it."""
    patcher = patch.object(settings, "api_token", token)
    patcher.start()
    return patcher
