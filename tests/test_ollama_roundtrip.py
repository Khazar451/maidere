"""One real graph turn against a local stub of Ollama's HTTP API.

The agent, the WebSocket, and the HTTP client are the production code.
Only the model server is fake, so a regression in request shape or streaming
assembly fails here instead of only in a fully mocked unit test.
"""

import json
import os
import tempfile
import threading
import unittest
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from unittest.mock import patch

from fastapi.testclient import TestClient

from maidere.api.app import create_app
from maidere.api.routes import set_graph
from maidere.core.agent import create_graph
from maidere.core.config import settings
from maidere.core.db import get_db, init_tables
from tests.authhelpers import TEST_TOKEN, WS_AUTH_FRAME, patch_api_token


class _OllamaStub(BaseHTTPRequestHandler):
    def do_GET(self) -> None:
        body = json.dumps({"models": [{"name": "qwen2.5:7b-instruct"}]}).encode()
        self._send(200, body, "application/json")

    def do_POST(self) -> None:
        length = int(self.headers.get("Content-Length", "0"))
        raw = self.rfile.read(length) if length else b"{}"
        request = json.loads(raw.decode() or "{}")
        if request.get("stream"):
            self.send_response(200)
            self.send_header("Content-Type", "application/x-ndjson")
            self.end_headers()
            for piece in ("stub-", "ok"):
                line = {"message": {"role": "assistant", "content": piece}, "done": False}
                self.wfile.write((json.dumps(line) + "\n").encode())
            done = {
                "message": {"role": "assistant", "content": ""},
                "done": True,
                "prompt_eval_count": 3,
                "eval_count": 2,
            }
            self.wfile.write((json.dumps(done) + "\n").encode())
            return
        body = json.dumps(
            {"message": {"role": "assistant", "content": "stub-ok", "tool_calls": []}, "done": True}
        ).encode()
        self._send(200, body, "application/json")

    def _send(self, status: int, body: bytes, content_type: str) -> None:
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, format: str, *args: object) -> None:
        return


class TestOllamaRoundTrip(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self) -> None:
        self.server = ThreadingHTTPServer(("127.0.0.1", 0), _OllamaStub)
        self.port = self.server.server_address[1]
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        self.thread.start()

        with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as temp_db:
            self.temp_db_path = temp_db.name

        self.patches = [
            patch_api_token(),
            patch.dict(os.environ, {"no_proxy": "*", "NO_PROXY": "*"}),
            patch.object(settings, "db_path", self.temp_db_path),
            patch.object(settings, "ollama_url", f"http://127.0.0.1:{self.port}"),
            patch.object(settings, "ollama_model", "qwen2.5:7b-instruct"),
            patch("maidere.core.embeddings.embed_one", side_effect=lambda _t: [0.05] * 384),
            patch("maidere.core.memory.embed_one", side_effect=lambda _t: [0.05] * 384),
        ]
        for item in self.patches:
            item.start()

        db = await get_db(self.temp_db_path)
        await init_tables(db)
        await db.close()
        self.graph, self.checkpointer = await create_graph(self.temp_db_path)
        set_graph(self.graph)
        self.client = TestClient(create_app())

    async def asyncTearDown(self) -> None:
        self.client.close()
        await self.checkpointer.__aexit__(None, None, None)
        self.server.shutdown()
        self.server.server_close()
        self.thread.join(timeout=5)
        for item in reversed(self.patches):
            item.stop()
        if os.path.exists(self.temp_db_path):
            os.remove(self.temp_db_path)

    def test_websocket_turn_streams_stub_tokens(self) -> None:
        with self.client.websocket_connect("/ws/chat") as ws:
            ws.send_json(WS_AUTH_FRAME)
            self.assertEqual(ws.receive_json(), {"type": "auth_ok"})
            ws.send_json(
                {
                    "message": "Reply with a short greeting.",
                    "thread_id": "roundtrip-1",
                    "model": "qwen2.5:7b-instruct",
                    "thinking_mode": False,
                    "deep_reasoning": False,
                    "username": "User",
                }
            )
            tokens: list[str] = []
            final = None
            for _ in range(40):
                message = ws.receive_json()
                if message.get("type") == "token":
                    tokens.append(message.get("text", ""))
                elif message.get("type") == "response":
                    final = message
                    break
                elif message.get("type") == "error":
                    self.fail(message.get("message"))
            self.assertIsNotNone(final)
            assert final is not None
            self.assertIn("stub-ok", final.get("content", ""))
            self.assertIn("stub-ok", "".join(tokens))

    def test_rest_chat_uses_the_same_stub(self) -> None:
        response = self.client.post(
            "/chat",
            headers={"Authorization": f"Bearer {TEST_TOKEN}"},
            json={
                "message": "Reply with a short greeting.",
                "thread_id": "roundtrip-rest",
                "model": "qwen2.5:7b-instruct",
            },
        )
        self.assertEqual(response.status_code, 200, response.text)
        self.assertIn("stub-ok", response.json()["response"])
