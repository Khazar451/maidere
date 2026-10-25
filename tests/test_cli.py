"""Tests for the ``maidere`` command-line entry point and ``python -m maidere``."""

import runpy
import unittest
from unittest.mock import patch

from maidere.cli import main


def run_cli(*argv: str) -> int:
    with patch("sys.argv", ["maidere", *argv]):
        return main()


class TestCli(unittest.TestCase):
    def test_defaults_launch_desktop_on_loopback(self):
        with patch("maidere.cli.run_desktop", return_value=0) as run:
            self.assertEqual(run_cli(), 0)
        run.assert_called_once_with(host="127.0.0.1", port=8000, dev_mode=False, server_only=False)

    def test_flags_are_passed_through(self):
        with patch("maidere.cli.run_desktop", return_value=0) as run:
            run_cli("run", "--dev", "--server-only", "--host", "0.0.0.0", "--port", "9100")
        run.assert_called_once_with(host="0.0.0.0", port=9100, dev_mode=True, server_only=True)

    def test_exit_code_of_the_launcher_is_returned(self):
        with patch("maidere.cli.run_desktop", return_value=1):
            self.assertEqual(run_cli(), 1)

    def test_install_desktop_maps_result_to_exit_code(self):
        with (
            patch("maidere.cli.install_desktop_entry", return_value=True) as install,
            patch("maidere.cli.run_desktop") as run,
        ):
            self.assertEqual(run_cli("install-desktop"), 0)
            run.assert_not_called()
        install.assert_called_once_with()
        with patch("maidere.cli.install_desktop_entry", return_value=False):
            self.assertEqual(run_cli("install-desktop"), 1)

    def test_unknown_command_is_rejected(self):
        with self.assertRaises(SystemExit) as ctx:
            run_cli("explode")
        self.assertEqual(ctx.exception.code, 2)

    def test_non_integer_port_is_rejected(self):
        with self.assertRaises(SystemExit) as ctx:
            run_cli("--port", "eighty")
        self.assertEqual(ctx.exception.code, 2)

    def test_help_exits_cleanly(self):
        with self.assertRaises(SystemExit) as ctx, patch("sys.stdout"):
            run_cli("--help")
        self.assertEqual(ctx.exception.code, 0)

    def test_python_dash_m_runs_main_and_exits_with_its_code(self):
        with patch("maidere.cli.main", return_value=3), self.assertRaises(SystemExit) as ctx:
            runpy.run_module("maidere", run_name="__main__")
        self.assertEqual(ctx.exception.code, 3)


if __name__ == "__main__":
    unittest.main()
