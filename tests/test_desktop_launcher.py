"""Tests for Maidere Desktop Application Launcher and Lifecycle Manager."""

from __future__ import annotations

import tempfile
import unittest
import urllib.error
from pathlib import Path
from unittest.mock import MagicMock, patch

from maidere.core.config import settings
from maidere.desktop.launcher import (
    DEFAULT_HOST,
    DEFAULT_PORT,
    build_ui_url,
    find_browser_engine,
    install_desktop_entry,
    is_server_healthy,
    resolve_api_token,
    start_server_process,
    wait_for_server,
)


class TestDesktopLauncher(unittest.TestCase):
    """Unit tests for desktop launcher functions."""

    def test_build_ui_url_puts_token_in_fragment(self):
        """The token must travel in the URL fragment, never the query string."""
        url = build_ui_url("127.0.0.1", 8000, "abc123")
        self.assertEqual(url, "http://127.0.0.1:8000/#token=abc123")
        self.assertNotIn("?", url)
        self.assertEqual(build_ui_url("127.0.0.1", 8000, None), "http://127.0.0.1:8000/")

    def test_resolve_api_token_uses_server_resolution(self):
        """Launcher resolves the same token the server will accept."""
        with patch.object(settings, "api_token", "launcher-token-0123456789"):
            self.assertEqual(resolve_api_token(), "launcher-token-0123456789")
        with patch.object(settings, "api_auth_enabled", False):
            self.assertIsNone(resolve_api_token())

    @patch("urllib.request.urlopen")
    def test_is_server_healthy_success(self, mock_urlopen):
        """Verify is_server_healthy returns True on HTTP 200 response."""
        mock_resp = MagicMock()
        mock_resp.status = 200
        mock_urlopen.return_value.__enter__.return_value = mock_resp

        self.assertTrue(is_server_healthy("127.0.0.1", 8000))
        mock_urlopen.assert_called_once()

    @patch("urllib.request.urlopen")
    def test_is_server_healthy_connection_refused(self, mock_urlopen):
        """Verify is_server_healthy returns False when server is down."""
        mock_urlopen.side_effect = urllib.error.URLError("Connection refused")
        self.assertFalse(is_server_healthy("127.0.0.1", 8000))

    @patch("maidere.desktop.launcher.is_server_healthy")
    def test_wait_for_server_immediate_ready(self, mock_healthy):
        """Verify wait_for_server returns True when server responds."""
        mock_healthy.return_value = True
        self.assertTrue(wait_for_server("127.0.0.1", 8000, max_wait_sec=1.0))

    @patch("maidere.desktop.launcher.is_server_healthy")
    def test_wait_for_server_timeout(self, mock_healthy):
        """Verify wait_for_server returns False when server fails to respond."""
        mock_healthy.return_value = False
        self.assertFalse(wait_for_server("127.0.0.1", 8000, max_wait_sec=0.5))

    @patch("subprocess.Popen")
    def test_start_server_process_dev_mode(self, mock_popen):
        """Verify dev_mode passes --reload and watch directories to uvicorn."""
        mock_popen.return_value = MagicMock()
        start_server_process(DEFAULT_HOST, DEFAULT_PORT, dev_mode=True)

        mock_popen.assert_called_once()
        cmd = mock_popen.call_args[0][0]
        self.assertIn("--reload", cmd)
        self.assertIn("--reload-dir", cmd)
        self.assertIn("maidere.api.app:app", cmd)

        env = mock_popen.call_args[1]["env"]
        self.assertEqual(env.get("MAIDERE_DEV"), "1")

    @patch("subprocess.Popen")
    def test_start_server_process_standard_mode(self, mock_popen):
        """Verify standard production mode omits --reload flags."""
        mock_popen.return_value = MagicMock()
        start_server_process(DEFAULT_HOST, DEFAULT_PORT, dev_mode=False)

        mock_popen.assert_called_once()
        cmd = mock_popen.call_args[0][0]
        self.assertNotIn("--reload", cmd)
        self.assertIn("maidere.api.app:app", cmd)

    def test_find_browser_engine_detects_or_safely_falls_back(self):
        """Verify browser detection returns valid tuple."""
        engine, args = find_browser_engine()
        if engine:
            self.assertIsInstance(engine, str)
            self.assertIsInstance(args, list)
            self.assertTrue(len(args) > 0)
        else:
            self.assertIsNone(engine)
            self.assertEqual(args, [])

    def test_install_desktop_entry_xdg_format(self):
        """Verify install_desktop_entry generates a valid Desktop Entry."""
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_home = Path(temp_dir)
            with patch("pathlib.Path.home", return_value=temp_home):
                success = install_desktop_entry()
                self.assertTrue(success)

                desktop_file = temp_home / ".local" / "share" / "applications" / "maidere.desktop"
                self.assertTrue(desktop_file.exists())

                content = desktop_file.read_text(encoding="utf-8")
                self.assertIn("[Desktop Entry]", content)
                self.assertIn("Name=Maidere", content)
                self.assertIn("Exec=", content)
                self.assertIn("Icon=", content)
                self.assertIn("Categories=Development;Utility;", content)


if __name__ == "__main__":
    unittest.main()
