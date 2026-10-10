"""Liveness and readiness HTTP checks."""

import tempfile
import unittest
from unittest.mock import AsyncMock, patch

from fastapi.testclient import TestClient

from maidere.api.app import create_app
from maidere.core.config import settings
from maidere.core.readiness import readiness_status
from tests.authhelpers import patch_api_token


class TestReadinessStatus(unittest.TestCase):
    def test_searxng_is_optional(self):
        ready, status = readiness_status({"database": True, "disk": True, "ollama": True, "searxng": False})
        self.assertTrue(ready)
        self.assertEqual(status, "degraded")

    def test_missing_model_server_is_not_ready(self):
        ready, status = readiness_status({"database": True, "disk": True, "ollama": False, "searxng": True})
        self.assertFalse(ready)
        self.assertEqual(status, "unavailable")


class TestLiveAndReadyEndpoints(unittest.TestCase):
    def setUp(self):
        self.token = patch_api_token()
        self.tmp = tempfile.TemporaryDirectory()
        self.db = patch.object(settings, "db_path", f"{self.tmp.name}/maidere.db")
        self.db.start()
        self.client = TestClient(create_app())

    def tearDown(self):
        self.db.stop()
        self.token.stop()
        self.tmp.cleanup()

    def test_live_is_public(self):
        response = self.client.get("/live")
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json(), {"status": "live"})

    def test_ready_when_dependencies_pass(self):
        with (
            patch("maidere.core.llm.is_available", new_callable=AsyncMock, return_value=True),
            patch("maidere.core.readiness.searxng_reachable", new_callable=AsyncMock, return_value=True),
        ):
            response = self.client.get("/ready")
        self.assertEqual(response.status_code, 200, response.text)
        body = response.json()
        self.assertTrue(body["ready"])
        self.assertEqual(body["status"], "ok")
        self.assertTrue(body["checks"]["database"])
        self.assertTrue(body["checks"]["disk"])

    def test_ready_is_degraded_without_searxng(self):
        with (
            patch("maidere.core.llm.is_available", new_callable=AsyncMock, return_value=True),
            patch("maidere.core.readiness.searxng_reachable", new_callable=AsyncMock, return_value=False),
        ):
            response = self.client.get("/ready")
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["status"], "degraded")
        self.assertTrue(response.json()["ready"])

    def test_ready_503_when_disk_is_low(self):
        with (
            patch("maidere.core.llm.is_available", new_callable=AsyncMock, return_value=True),
            patch("maidere.core.readiness.searxng_reachable", new_callable=AsyncMock, return_value=True),
            patch("maidere.core.readiness.disk_has_space", return_value=False),
        ):
            response = self.client.get("/ready")
        self.assertEqual(response.status_code, 503)
        self.assertFalse(response.json()["detail"]["ready"])
        self.assertFalse(response.json()["detail"]["checks"]["disk"])
