"""Tests for browser/window selection and the desktop launcher lifecycle."""

from __future__ import annotations

import signal
import sys
import tempfile
import unittest
from pathlib import Path
from types import ModuleType
from unittest.mock import MagicMock, patch

from maidere.desktop import launcher


class TestFindBrowserEngine(unittest.TestCase):
    def _find(self, available: dict[str, str]):
        with (
            tempfile.TemporaryDirectory() as home,
            patch.object(Path, "home", return_value=Path(home)),
            patch("shutil.which", side_effect=lambda name: available.get(name)),
        ):
            return launcher.find_browser_engine(), Path(home)

    def test_prefers_chromium_family_with_isolated_profile(self):
        (name, cmd), home = self._find({"chromium": "/usr/bin/chromium", "firefox": "/usr/bin/firefox"})
        self.assertEqual(name, "chromium")
        self.assertEqual(cmd[0], "/usr/bin/chromium")
        self.assertIn(f"--user-data-dir={home / '.config' / 'maidere' / 'browser_profile'}", cmd)
        self.assertIn("--class=Maidere", cmd)

    def test_chrome_stable_wins_over_other_chromium_builds(self):
        (name, _), _ = self._find({"google-chrome-stable": "/opt/chrome", "chromium": "/usr/bin/chromium"})
        self.assertEqual(name, "google-chrome-stable")

    def test_falls_back_to_firefox(self):
        (name, cmd), _ = self._find({"firefox": "/usr/bin/firefox"})
        self.assertEqual(name, "firefox")
        self.assertEqual(cmd[0], "/usr/bin/firefox")

    def test_none_when_no_browser_installed(self):
        self.assertEqual(self._find({})[0], (None, []))


class TestLaunchWindow(unittest.TestCase):
    URL = "http://127.0.0.1:8000/#token=t"

    def test_uses_pywebview_when_installed(self):
        fake = ModuleType("webview")
        fake.create_window = MagicMock()  # type: ignore[attr-defined]
        fake.start = MagicMock()  # type: ignore[attr-defined]
        with patch.dict(sys.modules, {"webview": fake}):
            result = launcher.launch_window(self.URL, title="T", dev_mode=True)
        self.assertIsNone(result)
        self.assertEqual(fake.create_window.call_args.kwargs["url"], self.URL)
        fake.start.assert_called_once_with(debug=True)

    def _without_pywebview(self):
        return patch.dict(sys.modules, {"webview": None})  # makes `import webview` raise ImportError

    def test_chromium_gets_app_mode_url(self):
        with (
            self._without_pywebview(),
            patch.object(
                launcher, "find_browser_engine", return_value=("chromium", ["/usr/bin/chromium", "--no-first-run"])
            ),
            patch("subprocess.Popen") as popen,
        ):
            result = launcher.launch_window(self.URL)
        self.assertIs(result, popen.return_value)
        self.assertEqual(popen.call_args.args[0], ["/usr/bin/chromium", "--no-first-run", f"--app={self.URL}"])

    def test_firefox_gets_plain_url(self):
        with (
            self._without_pywebview(),
            patch.object(
                launcher, "find_browser_engine", return_value=("firefox", ["/usr/bin/firefox", "--new-window"])
            ),
            patch("subprocess.Popen") as popen,
        ):
            launcher.launch_window(self.URL)
        self.assertEqual(popen.call_args.args[0], ["/usr/bin/firefox", "--new-window", self.URL])

    def test_falls_back_to_default_browser(self):
        with (
            self._without_pywebview(),
            patch.object(launcher, "find_browser_engine", return_value=(None, [])),
            patch("webbrowser.open") as opener,
        ):
            self.assertIsNone(launcher.launch_window(self.URL))
        opener.assert_called_once_with(self.URL)


class TestRunDesktop(unittest.TestCase):
    """Server and window lifecycle, with every process, signal and sleep faked."""

    def setUp(self):
        self.handlers: dict[int, object] = {}
        patches = [
            patch("signal.signal", side_effect=lambda sig, fn: self.handlers.__setitem__(sig, fn)),
            patch("time.sleep"),
            patch.object(launcher, "resolve_api_token", return_value="tok"),
        ]
        for p in patches:
            p.start()
            self.addCleanup(p.stop)

    @staticmethod
    def _proc(poll=None):
        proc = MagicMock()
        proc.poll.return_value = poll
        return proc

    def test_registers_signal_handlers(self):
        with (
            patch.object(launcher, "is_server_healthy", return_value=True),
            patch.object(launcher, "launch_window", return_value=None),
            patch.object(launcher, "start_server_process"),
            patch("time.sleep", side_effect=KeyboardInterrupt),
        ):
            launcher.run_desktop()
        self.assertIn(signal.SIGINT, self.handlers)
        self.assertIn(signal.SIGTERM, self.handlers)

    def test_attaches_to_running_server_and_leaves_it_alone(self):
        window = self._proc()
        with (
            patch.object(launcher, "is_server_healthy", return_value=True),
            patch.object(launcher, "start_server_process") as start,
            patch.object(launcher, "launch_window", return_value=window) as launch,
        ):
            self.assertEqual(launcher.run_desktop(host="127.0.0.1", port=8123), 0)
        start.assert_not_called()
        self.assertEqual(launch.call_args.args[0], "http://127.0.0.1:8123/#token=tok")
        window.wait.assert_called_once()
        window.terminate.assert_called_once()  # window still alive at shutdown -> closed

    def test_starts_backend_and_stops_it_with_sigint(self):
        server, window = self._proc(), self._proc(poll=0)
        with (
            patch.object(launcher, "is_server_healthy", return_value=False),
            patch.object(launcher, "start_server_process", return_value=server) as start,
            patch.object(launcher, "wait_for_server", return_value=True),
            patch.object(launcher, "launch_window", return_value=window),
        ):
            self.assertEqual(launcher.run_desktop(dev_mode=True), 0)
        start.assert_called_once_with("127.0.0.1", 8000, dev_mode=True)
        server.send_signal.assert_called_once_with(signal.SIGINT)
        server.wait.assert_called_once_with(timeout=5.0)
        window.terminate.assert_not_called()  # already closed by the user

    def test_backend_that_will_not_stop_is_killed(self):
        server = self._proc()
        server.wait.side_effect = TimeoutError
        with (
            patch.object(launcher, "is_server_healthy", return_value=False),
            patch.object(launcher, "start_server_process", return_value=server),
            patch.object(launcher, "wait_for_server", return_value=True),
            patch.object(launcher, "launch_window", return_value=self._proc(poll=0)),
        ):
            launcher.run_desktop()
        server.kill.assert_called_once()

    def test_failed_startup_returns_nonzero_and_stops_the_child(self):
        server = self._proc()
        with (
            patch.object(launcher, "is_server_healthy", return_value=False),
            patch.object(launcher, "start_server_process", return_value=server),
            patch.object(launcher, "wait_for_server", return_value=False),
            patch.object(launcher, "launch_window") as launch,
        ):
            self.assertEqual(launcher.run_desktop(), 1)
        launch.assert_not_called()
        server.send_signal.assert_called_once_with(signal.SIGINT)

    def test_server_only_runs_until_backend_exits(self):
        server = MagicMock()
        server.poll.side_effect = [None, 0, 0]
        with (
            patch.object(launcher, "is_server_healthy", return_value=False),
            patch.object(launcher, "start_server_process", return_value=server),
            patch.object(launcher, "wait_for_server", return_value=True),
            patch.object(launcher, "launch_window") as launch,
        ):
            self.assertEqual(launcher.run_desktop(server_only=True), 0)
        launch.assert_not_called()

    def test_server_only_stops_on_ctrl_c(self):
        server = self._proc()
        with (
            patch.object(launcher, "is_server_healthy", return_value=False),
            patch.object(launcher, "start_server_process", return_value=server),
            patch.object(launcher, "wait_for_server", return_value=True),
            patch("time.sleep", side_effect=KeyboardInterrupt),
        ):
            self.assertEqual(launcher.run_desktop(server_only=True), 0)
        server.send_signal.assert_called_once_with(signal.SIGINT)

    def test_waits_on_backend_when_window_runs_in_process(self):
        server = self._proc()
        with (
            patch.object(launcher, "is_server_healthy", return_value=False),
            patch.object(launcher, "start_server_process", return_value=server),
            patch.object(launcher, "wait_for_server", return_value=True),
            patch.object(launcher, "launch_window", return_value=None),
        ):
            self.assertEqual(launcher.run_desktop(), 0)
        server.wait.assert_any_call()

    def test_sigint_handler_shuts_down_and_exits_zero(self):
        server = self._proc()
        with (
            patch.object(launcher, "is_server_healthy", return_value=False),
            patch.object(launcher, "start_server_process", return_value=server),
            patch.object(launcher, "wait_for_server", return_value=True),
            patch.object(launcher, "launch_window", return_value=None),
        ):
            server.wait.side_effect = lambda *a, **k: (
                (_ for _ in ()).throw(KeyboardInterrupt) if not a and not k else None
            )
            launcher.run_desktop()
            handler = self.handlers[signal.SIGINT]
            with self.assertRaises(SystemExit) as ctx:
                handler(signal.SIGINT, None)  # type: ignore[operator]
        self.assertEqual(ctx.exception.code, 0)
        server.send_signal.assert_called_with(signal.SIGINT)


if __name__ == "__main__":
    unittest.main()
