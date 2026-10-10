"""Tests for API authentication, origin enforcement, upload hardening, and token storage."""

import os
import stat
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import httpx
from fastapi.testclient import TestClient
from httpx import ASGITransport
from starlette.websockets import WebSocketDisconnect

from api.app import create_app
from api.routes import sanitize_upload_filename
from core import tokens
from core.audit import sanitize_secrets
from core.config import settings
from tests.authhelpers import AUTH_HEADERS, TEST_TOKEN, WS_AUTH_FRAME, patch_api_token


class TestHTTPAuthentication(unittest.TestCase):
    """Bearer-token enforcement on HTTP routes."""

    def setUp(self):
        self.token_patcher = patch_api_token()
        self.app = create_app()
        self.client = TestClient(self.app)

    def tearDown(self):
        self.token_patcher.stop()

    def test_protected_routes_reject_missing_token(self):
        for method, path in [
            ("get", "/threads"),
            ("get", "/memories"),
            ("get", "/settings/system-prompt"),
            ("get", "/skills"),
            ("get", "/user/profile"),
            ("delete", "/threads/some-thread"),
            ("delete", "/memories/1"),
        ]:
            with self.subTest(method=method, path=path):
                response = getattr(self.client, method)(path)
                self.assertEqual(response.status_code, 401)
                self.assertEqual(response.headers.get("www-authenticate"), "Bearer")
                self.assertIn("token", response.json()["detail"].lower())

        response = self.client.post("/chat", json={"message": "hi there"})
        self.assertEqual(response.status_code, 401)
        response = self.client.put("/settings/system-prompt", json={"system_prompt": "x"})
        self.assertEqual(response.status_code, 401)

    def test_protected_routes_reject_wrong_token(self):
        response = self.client.get("/threads", headers={"Authorization": "Bearer definitely-not-the-token"})
        self.assertEqual(response.status_code, 401)
        response = self.client.get("/threads", headers={"Authorization": f"Basic {TEST_TOKEN}"})
        self.assertEqual(response.status_code, 401)

    def test_protected_routes_accept_bearer_header(self):
        response = self.client.get("/threads", headers=AUTH_HEADERS)
        self.assertEqual(response.status_code, 200)

    def test_protected_routes_accept_alt_header(self):
        response = self.client.get("/threads", headers={"X-Maidere-Token": TEST_TOKEN})
        self.assertEqual(response.status_code, 200)

    def test_public_paths_need_no_token(self):
        self.assertEqual(self.client.get("/").status_code, 200)
        self.assertEqual(self.client.get("/static/app.js").status_code, 200)
        with patch("core.llm.is_available", return_value=True):
            self.assertEqual(self.client.get("/health").status_code, 200)

    def test_auth_can_be_disabled_explicitly(self):
        with patch.object(settings, "api_auth_enabled", False):
            app = create_app()
            client = TestClient(app)
            self.assertEqual(client.get("/threads").status_code, 200)

    def test_cors_only_allows_configured_origins(self):
        allowed = f"http://127.0.0.1:{settings.api_port}"
        response = self.client.get("/threads", headers={**AUTH_HEADERS, "Origin": allowed})
        self.assertEqual(response.headers.get("access-control-allow-origin"), allowed)

        response = self.client.get("/threads", headers={**AUTH_HEADERS, "Origin": "http://evil.example"})
        self.assertIsNone(response.headers.get("access-control-allow-origin"))

        # Preflight from a disallowed origin is refused before reaching auth.
        response = self.client.options(
            "/threads",
            headers={
                "Origin": "http://evil.example",
                "Access-Control-Request-Method": "GET",
                "Access-Control-Request-Headers": "authorization",
            },
        )
        self.assertNotEqual(response.status_code, 200)

    def test_extra_allowed_origins_setting(self):
        with patch.object(settings, "allowed_origins", "https://maidere.example.com, http://10.0.0.5:8000/"):
            origins = settings.get_allowed_origins()
        self.assertIn("https://maidere.example.com", origins)
        self.assertIn("http://10.0.0.5:8000", origins)
        self.assertIn(f"http://localhost:{settings.api_port}", origins)


class TestMetricsAccess(unittest.IsolatedAsyncioTestCase):
    """/metrics is reachable from loopback without a token, otherwise needs one."""

    async def asyncSetUp(self):
        self.token_patcher = patch_api_token()
        self.app = create_app()

    async def asyncTearDown(self):
        self.token_patcher.stop()

    async def _get_metrics(self, client_addr, headers=None):
        transport = ASGITransport(app=self.app, client=client_addr)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            return await client.get("/metrics", headers=headers or {})

    async def test_loopback_scrape_without_token(self):
        response = await self._get_metrics(("127.0.0.1", 40000))
        self.assertEqual(response.status_code, 200)
        self.assertIn("http_requests_total", response.text)

    async def test_remote_scrape_requires_token(self):
        response = await self._get_metrics(("203.0.113.5", 40000))
        self.assertEqual(response.status_code, 401)
        response = await self._get_metrics(("203.0.113.5", 40000), headers=AUTH_HEADERS)
        self.assertEqual(response.status_code, 200)

    async def test_loopback_exemption_can_be_disabled(self):
        with patch.object(settings, "metrics_allow_loopback", False):
            response = await self._get_metrics(("127.0.0.1", 40000))
        self.assertEqual(response.status_code, 401)


class TestWebSocketAuthentication(unittest.TestCase):
    """First-frame token auth and Origin enforcement on /ws/chat."""

    def setUp(self):
        self.token_patcher = patch_api_token()
        self.app = create_app()
        self.client = TestClient(self.app)

    def tearDown(self):
        self.token_patcher.stop()

    def test_valid_auth_frame_is_acknowledged(self):
        with self.client.websocket_connect("/ws/chat") as ws:
            ws.send_json(WS_AUTH_FRAME)
            self.assertEqual(ws.receive_json(), {"type": "auth_ok"})

    def test_invalid_token_closes_socket_4401(self):
        with self.client.websocket_connect("/ws/chat") as ws:
            ws.send_json({"type": "auth", "token": "nope"})
            with self.assertRaises(WebSocketDisconnect) as ctx:
                ws.receive_json()
            self.assertEqual(ctx.exception.code, 4401)

    def test_message_before_auth_closes_socket_4401(self):
        with self.client.websocket_connect("/ws/chat") as ws:
            ws.send_json({"message": "hello", "thread_id": "t1"})
            with self.assertRaises(WebSocketDisconnect) as ctx:
                ws.receive_json()
            self.assertEqual(ctx.exception.code, 4401)

    def test_disallowed_origin_rejected_at_handshake(self):
        with self.assertRaises(WebSocketDisconnect) as ctx:
            with self.client.websocket_connect("/ws/chat", headers={"Origin": "http://evil.example"}):
                pass
        self.assertEqual(ctx.exception.code, 4403)

    def test_allowed_origin_accepted(self):
        origin = f"http://localhost:{settings.api_port}"
        with self.client.websocket_connect("/ws/chat", headers={"Origin": origin}) as ws:
            ws.send_json(WS_AUTH_FRAME)
            self.assertEqual(ws.receive_json(), {"type": "auth_ok"})

    def test_auth_frame_acknowledged_when_auth_disabled(self):
        with patch.object(settings, "api_auth_enabled", False):
            client = TestClient(create_app())
            with client.websocket_connect("/ws/chat") as ws:
                ws.send_json({"type": "auth", "token": ""})
                self.assertEqual(ws.receive_json(), {"type": "auth_ok"})


class TestUploadHardening(unittest.TestCase):
    """Uploads cannot escape workspace/uploads and respect the size cap."""

    def setUp(self):
        self.token_patcher = patch_api_token()
        self.tmp = tempfile.TemporaryDirectory()
        self.workspace = Path(self.tmp.name) / "workspace"
        self.workspace.mkdir()
        self.ws_patcher = patch.object(settings, "agent_workspace", str(self.workspace))
        self.ws_patcher.start()
        self.client = TestClient(create_app(), headers=AUTH_HEADERS)

    def tearDown(self):
        self.ws_patcher.stop()
        self.token_patcher.stop()
        self.tmp.cleanup()

    def test_sanitize_upload_filename(self):
        cases = {
            "report.pdf": "report.pdf",
            "../../etc/passwd": "passwd",
            "/etc/passwd": "passwd",
            "..\\..\\windows\\system.ini": "system.ini",
            ".hidden": "hidden",
            "weird name (1).txt": "weird_name_1_.txt",
            "..": "",
            "": "",
        }
        for raw, expected in cases.items():
            with self.subTest(raw=raw):
                result = sanitize_upload_filename(raw)
                if expected:
                    self.assertEqual(result, expected)
                else:
                    self.assertTrue(result.startswith("upload_") and result.endswith(".bin"))
                self.assertNotIn("/", result)
                self.assertNotIn("..", result)

    def test_traversal_filename_stays_inside_uploads(self):
        response = self.client.post(
            "/upload",
            files={"file": ("../../escaped.txt", b"owned", "text/plain")},
        )
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertEqual(data["filename"], "escaped.txt")
        self.assertTrue((self.workspace / "uploads" / "escaped.txt").exists())
        self.assertFalse((Path(self.tmp.name) / "escaped.txt").exists())
        self.assertFalse((self.workspace.parent / "escaped.txt").exists())

    def test_upload_size_limit_enforced(self):
        with patch.object(settings, "max_upload_bytes", 1024):
            response = self.client.post(
                "/upload",
                files={"file": ("big.bin", b"x" * 4096, "application/octet-stream")},
            )
        self.assertEqual(response.status_code, 413)
        leftovers = list((self.workspace / "uploads").glob("*"))
        self.assertEqual(leftovers, [], f"partial files left behind: {leftovers}")

    def test_upload_requires_token(self):
        client = TestClient(create_app())
        response = client.post("/upload", files={"file": ("a.txt", b"a", "text/plain")})
        self.assertEqual(response.status_code, 401)


class TestTokenStorage(unittest.TestCase):
    """Token resolution order and on-disk persistence."""

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.token_path = Path(self.tmp.name) / "nested" / "token"
        self.patchers = [
            patch.object(settings, "api_token", ""),
            patch.object(settings, "api_token_file", str(self.token_path)),
        ]
        for p in self.patchers:
            p.start()

    def tearDown(self):
        for p in self.patchers:
            p.stop()
        self.tmp.cleanup()

    def test_env_token_takes_precedence(self):
        with patch.object(settings, "api_token", "from-env-token-0123456789"):
            self.assertEqual(tokens.load_or_create_token(), "from-env-token-0123456789")
        self.assertFalse(self.token_path.exists())

    def test_token_file_created_with_restrictive_mode(self):
        first = tokens.load_or_create_token()
        self.assertTrue(self.token_path.exists())
        mode = stat.S_IMODE(os.stat(self.token_path).st_mode)
        self.assertEqual(mode, 0o600)
        self.assertGreaterEqual(len(first), 32)
        # Stable across calls, and verify() agrees.
        self.assertEqual(tokens.load_or_create_token(), first)
        self.assertTrue(tokens.verify_token(first))
        self.assertTrue(tokens.verify_token(f"  {first}\n"))
        self.assertFalse(tokens.verify_token(first[:-1]))
        self.assertFalse(tokens.verify_token(""))
        self.assertFalse(tokens.verify_token(None))

    def test_existing_token_file_is_reused(self):
        self.token_path.parent.mkdir(parents=True)
        self.token_path.write_text("pre-existing-token-abcdefghijkl\n", encoding="utf-8")
        self.assertEqual(tokens.load_or_create_token(), "pre-existing-token-abcdefghijkl")

    def test_rotate_replaces_token(self):
        first = tokens.load_or_create_token()
        second = tokens.rotate_token()
        self.assertNotEqual(first, second)
        self.assertEqual(tokens.load_or_create_token(), second)
        self.assertFalse(tokens.verify_token(first))

    def test_weak_token_warning(self):
        with patch.object(settings, "api_token", "short"):
            self.assertIsNotNone(tokens.token_strength_warning())
        with patch.object(settings, "api_token", "a-perfectly-long-enough-token"):
            self.assertIsNone(tokens.token_strength_warning())


class TestSecretSanitizerNvidia(unittest.TestCase):
    def test_nvapi_key_redacted(self):
        text = "key is nvapi-AbCdEfGhIjKlMnOpQrStUvWxYz0123456789-_xyz and more"
        sanitized = sanitize_secrets(text)
        self.assertNotIn("nvapi-AbCd", sanitized)
        self.assertIn("[REDACTED_SECRET]", sanitized)


if __name__ == "__main__":
    unittest.main()
