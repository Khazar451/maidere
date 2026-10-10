"""Authentication and origin enforcement for the Maidere API.

Design
------
* HTTP: every route requires ``Authorization: Bearer <token>`` (or the
  ``X-Maidere-Token`` header) except a small public allowlist (the UI shell,
  static assets, and ``/health`` so the desktop launcher can poll readiness).
* ``/metrics``: accepted from loopback clients without a token (Prometheus
  runs on the host network) or from anyone presenting a valid token.
* WebSocket: the ``Origin`` header, when present, must be in the allowed
  origin list; otherwise the handshake is rejected with 403 before ``accept``.
  Token verification for WebSockets happens in the route via the first frame
  (see ``api.routes.websocket_chat``), because browsers cannot set headers on
  WebSocket connections.

The middleware is pure ASGI (no ``BaseHTTPMiddleware``) so it does not buffer
streaming responses or interfere with WebSocket upgrades.
"""

from __future__ import annotations

import json
from collections.abc import Awaitable, Callable
from typing import Any

import structlog
from fastapi import HTTPException, Request, status

from core.config import settings
from core.tokens import verify_token

logger = structlog.get_logger()

Scope = dict[str, Any]
Receive = Callable[[], Awaitable[dict[str, Any]]]
Send = Callable[[dict[str, Any]], Awaitable[None]]

PUBLIC_EXACT_PATHS: frozenset[str] = frozenset({"/", "/health", "/favicon.ico", "/manifest.json"})
PUBLIC_PREFIXES: tuple[str, ...] = ("/static/",)
METRICS_PATH = "/metrics"

LOOPBACK_HOSTS: frozenset[str] = frozenset({"127.0.0.1", "::1", "::ffff:127.0.0.1"})

AUTH_HEADER = b"authorization"
TOKEN_HEADER = b"x-maidere-token"


def _header(scope: Scope, name: bytes) -> str | None:
    for key, value in scope.get("headers") or []:
        if key == name:
            return value.decode("latin-1")
    return None


def extract_token_from_headers(scope: Scope) -> str | None:
    """Pull a bearer token from Authorization or X-Maidere-Token headers."""
    auth = _header(scope, AUTH_HEADER)
    if auth:
        scheme, _, credentials = auth.partition(" ")
        if scheme.lower() == "bearer" and credentials.strip():
            return credentials.strip()
    alt = _header(scope, TOKEN_HEADER)
    if alt and alt.strip():
        return alt.strip()
    return None


def is_public_path(path: str) -> bool:
    return path in PUBLIC_EXACT_PATHS or path.startswith(PUBLIC_PREFIXES)


def is_loopback_client(scope: Scope) -> bool:
    client = scope.get("client")
    if not client:
        return False
    host = str(client[0])
    return host in LOOPBACK_HOSTS or host.startswith("127.")


def is_origin_allowed(origin: str | None) -> bool:
    """Non-browser clients send no Origin; browsers must match the allowlist."""
    if origin is None:
        return True
    return origin.rstrip("/") in settings.get_allowed_origins()


async def _send_json(send: Send, status_code: int, payload: dict[str, Any]) -> None:
    body = json.dumps(payload).encode("utf-8")
    await send(
        {
            "type": "http.response.start",
            "status": status_code,
            "headers": [
                (b"content-type", b"application/json"),
                (b"content-length", str(len(body)).encode("ascii")),
                (b"www-authenticate", b"Bearer"),
            ],
        }
    )
    await send({"type": "http.response.body", "body": body})


class AuthMiddleware:
    """ASGI middleware enforcing bearer-token auth and WebSocket origin checks."""

    def __init__(self, app: Callable[[Scope, Receive, Send], Awaitable[None]]) -> None:
        self.app = app

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        scope_type = scope.get("type")
        if scope_type == "http":
            await self._handle_http(scope, receive, send)
        elif scope_type == "websocket":
            await self._handle_websocket(scope, receive, send)
        else:
            await self.app(scope, receive, send)

    async def _handle_http(self, scope: Scope, receive: Receive, send: Send) -> None:
        if not settings.api_auth_enabled:
            await self.app(scope, receive, send)
            return

        path: str = scope.get("path", "")
        if is_public_path(path):
            await self.app(scope, receive, send)
            return

        token = extract_token_from_headers(scope)
        if token and verify_token(token):
            await self.app(scope, receive, send)
            return

        if path == METRICS_PATH and settings.metrics_allow_loopback and is_loopback_client(scope):
            await self.app(scope, receive, send)
            return

        logger.warning(
            "auth_rejected",
            path=path,
            client=(scope.get("client") or ("?",))[0],
            reason="missing_token" if not token else "invalid_token",
        )
        await _send_json(
            send,
            status.HTTP_401_UNAUTHORIZED,
            {"detail": "Unauthorized: a valid API token is required."},
        )

    async def _handle_websocket(self, scope: Scope, receive: Receive, send: Send) -> None:
        origin = _header(scope, b"origin")
        if not is_origin_allowed(origin):
            logger.warning("ws_origin_rejected", origin=origin, path=scope.get("path"))
            # Closing before accept makes the server respond 403 to the handshake.
            await send({"type": "websocket.close", "code": 4403})
            return
        await self.app(scope, receive, send)


# ---------------------------------------------------------------------------
# Helpers usable inside route handlers
# ---------------------------------------------------------------------------

WS_AUTH_TIMEOUT_SECONDS = 5.0
WS_CLOSE_UNAUTHORIZED = 4401


def websocket_auth_required() -> bool:
    return settings.api_auth_enabled


def websocket_frame_is_authenticated(frame: Any) -> bool:
    """Validate the first WebSocket frame: ``{"type": "auth", "token": "..."}``."""
    if not isinstance(frame, dict):
        return False
    if frame.get("type") != "auth":
        return False
    return verify_token(frame.get("token"))


def require_token(request: Request) -> None:
    """FastAPI dependency for routes that want an explicit per-route check.

    The middleware already protects everything; this exists for routes mounted
    outside the main app (or tests) that want the same semantics.
    """
    if not settings.api_auth_enabled:
        return
    token = extract_token_from_headers(request.scope)
    if not (token and verify_token(token)):
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Unauthorized")
