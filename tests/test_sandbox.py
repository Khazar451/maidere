"""Tests for tool subprocess isolation (tools/sandbox.py) and its use by shell/code_runner.

Two layers:
- Policy/unit tests that run everywhere (command construction, restricted-mode
  validation, environment scrubbing, mode resolution).
- Escape probes that run only when bubblewrap works on the host. These are the
  exact commands that used to succeed against the old "sandbox" (reading
  /etc/hostname, listing $HOME, reaching the network, writing outside the
  workspace).
"""

import os
import shutil
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from maidere.core.config import settings
from maidere.tools import sandbox
from maidere.tools.code_runner import CodeRunnerTool
from maidere.tools.sandbox import (
    SandboxError,
    SandboxMode,
    build_bwrap_command,
    build_environment,
    resolve_mode,
    run_sandboxed,
    validate_restricted,
)
from maidere.tools.shell import ShellTool

BWRAP_WORKS = sandbox.bwrap_available()
needs_bwrap = unittest.skipUnless(BWRAP_WORKS, "bubblewrap not usable on this host")


class TestModeResolution(unittest.TestCase):
    def tearDown(self):
        sandbox.reset_probe_cache()

    def test_explicit_modes(self):
        with patch.object(settings, "tool_sandbox_mode", "off"):
            self.assertIs(resolve_mode(), SandboxMode.OFF)
        with patch.object(settings, "tool_sandbox_mode", "restricted"):
            self.assertIs(resolve_mode(), SandboxMode.RESTRICTED)

    def test_auto_falls_back_to_restricted_without_bwrap(self):
        with patch.object(settings, "tool_sandbox_mode", "auto"), \
             patch.object(sandbox, "bwrap_available", return_value=False):
            self.assertIs(resolve_mode(), SandboxMode.RESTRICTED)

    def test_forced_bwrap_fails_closed_without_bwrap(self):
        with patch.object(settings, "tool_sandbox_mode", "bwrap"), \
             patch.object(sandbox, "bwrap_available", return_value=False):
            with self.assertRaises(SandboxError):
                resolve_mode()
            info = sandbox.describe()
            self.assertEqual(info["active"], "unavailable")
            self.assertIn("bubblewrap", info["error"])

    def test_describe_reports_active_mode(self):
        with patch.object(settings, "tool_sandbox_mode", "off"):
            info = sandbox.describe()
        self.assertEqual(info["active"], "off")
        self.assertEqual(info["configured"], "off")


class TestEnvironmentScrubbing(unittest.TestCase):
    def test_server_env_is_not_inherited(self):
        with patch.dict(os.environ, {"NVIDIA_API_KEY": "nvapi-secret", "CLOUD_API_KEY": "x", "TERM": "xterm"}):
            env = build_environment(home="/tmp/h", tmp="/tmp")
        self.assertNotIn("NVIDIA_API_KEY", env)
        self.assertNotIn("CLOUD_API_KEY", env)
        self.assertEqual(env["HOME"], "/tmp/h")
        self.assertEqual(env["TERM"], "xterm")
        self.assertEqual(env["MAIDERE_SANDBOX"], "1")
        self.assertTrue(env["PATH"].startswith("/usr/local/sbin"))


class TestBwrapCommandConstruction(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.workspace = Path(self.tmp.name) / "ws"
        self.workspace.mkdir()

    def tearDown(self):
        self.tmp.cleanup()

    def _cmd(self, argv, **kw):
        with patch.object(sandbox, "_bwrap_path", "/usr/bin/bwrap"):
            return build_bwrap_command(argv, self.workspace, **kw)

    def test_core_isolation_flags(self):
        cmd = self._cmd(["/usr/bin/ls", "-la"])
        joined = " ".join(cmd)
        self.assertEqual(cmd[0], "/usr/bin/bwrap")
        self.assertIn("--ro-bind /usr /usr", joined)
        self.assertIn(f"--bind {self.workspace} {self.workspace}", joined)
        self.assertIn(f"--chdir {self.workspace}", joined)
        self.assertIn("--unshare-all", cmd)
        self.assertNotIn("--share-net", cmd)
        self.assertIn("--die-with-parent", cmd)
        self.assertIn("--new-session", cmd)
        self.assertIn("--clearenv", cmd)
        self.assertIn("--tmpfs /tmp", joined)
        self.assertIn("--proc /proc", joined)
        self.assertIn("--dev /dev", joined)
        self.assertEqual(cmd[-2:], ["/usr/bin/ls", "-la"])
        # The host's home, /etc/passwd, and the server's own files are never bound.
        self.assertNotIn("/etc/passwd", joined)
        self.assertNotIn("--bind /home", joined)
        self.assertNotIn("--ro-bind /home " , joined)

    def test_network_opt_in(self):
        cmd = self._cmd(["/usr/bin/ls"], allow_network=True)
        self.assertIn("--share-net", cmd)
        self.assertIn("/etc/resolv.conf", cmd)

    def test_workspace_is_the_only_rw_bind(self):
        cmd = self._cmd(["/usr/bin/ls"])
        rw_binds = [cmd[i + 1] for i, tok in enumerate(cmd) if tok == "--bind"]
        self.assertEqual(rw_binds, [str(self.workspace)])

    def test_env_is_explicit_and_secret_free(self):
        with patch.dict(os.environ, {"NVIDIA_API_KEY": "nvapi-secret"}):
            cmd = self._cmd(["/usr/bin/ls"])
        setenv_keys = [cmd[i + 1] for i, tok in enumerate(cmd) if tok == "--setenv"]
        self.assertIn("PATH", setenv_keys)
        self.assertIn("HOME", setenv_keys)
        self.assertNotIn("NVIDIA_API_KEY", setenv_keys)

    def test_executable_outside_usr_gets_its_install_root(self):
        fake_root = Path(self.tmp.name) / "opt" / "mynode"
        (fake_root / "bin").mkdir(parents=True)
        exe = fake_root / "bin" / "node"
        exe.write_text("#!/bin/sh\n")
        cmd = self._cmd([str(exe), "-v"])
        self.assertIn(f"--ro-bind {fake_root.resolve()} {fake_root.resolve()}", " ".join(cmd))


class TestRestrictedValidation(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.workspace = Path(self.tmp.name) / "ws"
        (self.workspace / "notes").mkdir(parents=True)
        (self.workspace / "notes" / "a.txt").write_text("hi")

    def tearDown(self):
        self.tmp.cleanup()

    def _bad(self, argv, needle):
        msg = validate_restricted(argv, self.workspace)
        self.assertIsNotNone(msg, f"expected rejection for {argv}")
        self.assertIn(needle, msg)

    def _ok(self, argv):
        self.assertIsNone(validate_restricted(argv, self.workspace), f"expected acceptance for {argv}")

    def test_blocks_absolute_and_parent_paths(self):
        self._bad(["cat", "/etc/hostname"], "outside the workspace")
        self._bad(["cat", "../secrets.env"], "outside the workspace")
        self._bad(["ls", "~/.ssh"], "outside the workspace")
        self._bad(["cp", "/home/user/.ssh/id_rsa", "."], "outside the workspace")
        self._bad(["find", "/", "-name", "etc"], "outside the workspace")
        self._bad(["git", "--work-tree=/etc", "status"], "outside the workspace")
        self._bad(["grep", "-r", "x", "--include=../../etc/passwd"], "outside the workspace")

    def test_blocks_inline_code_and_find_exec(self):
        self._bad(["python3", "-c", "print(1)"], "inline code")
        self._bad(["python", "-c", "print(1)"], "inline code")
        self._bad(["node", "-e", "1"], "inline code")
        self._bad(["node", "--eval=1"], "inline code")
        self._bad(["find", ".", "-exec", "rm", "{}", ";"], "not permitted")

    def test_allows_workspace_relative_and_absolute_inside(self):
        self._ok(["ls", "-la"])
        self._ok(["cat", "notes/a.txt"])
        self._ok(["cat", str(self.workspace / "notes" / "a.txt")])
        self._ok(["python3", "notes/script.py"])
        self._ok(["python3", "-m", "json.tool", "notes/a.txt"])
        self._ok(["git", "status", "--porcelain"])
        self._ok(["echo", "the ratio is 3/4 and ../ is just text?"])

    def test_dotdot_that_stays_inside_is_fine(self):
        self._ok(["cat", "notes/../notes/a.txt"])


class TestRunSandboxedModes(unittest.TestCase):
    """Behaviour that must hold in every mode."""

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.workspace = Path(self.tmp.name) / "ws"
        self.workspace.mkdir()

    def tearDown(self):
        self.tmp.cleanup()

    def test_off_mode_runs_with_scrubbed_env(self):
        with patch.dict(os.environ, {"MAIDERE_SECRET_PROBE": "leaked"}):
            script = self.workspace / "p.py"
            script.write_text("import os; print(os.environ.get('MAIDERE_SECRET_PROBE', 'unset'), os.getcwd())")
            res = run_sandboxed(["python3", str(script)], workspace=self.workspace, mode=SandboxMode.OFF)
        self.assertEqual(res.returncode, 0, res.stderr)
        self.assertTrue(res.stdout.startswith("unset "))
        self.assertIn(str(self.workspace.resolve()), res.stdout)

    def test_restricted_mode_refuses_before_running(self):
        with self.assertRaises(SandboxError) as ctx:
            run_sandboxed(["cat", "/etc/hostname"], workspace=self.workspace, mode=SandboxMode.RESTRICTED)
        self.assertIn("outside the workspace", str(ctx.exception))

    def test_timeout_is_reported(self):
        script = self.workspace / "sleep.py"
        script.write_text("import time; time.sleep(10)")
        res = run_sandboxed(["python3", str(script)], workspace=self.workspace, timeout=1)
        self.assertTrue(res.timed_out)
        self.assertIn("timed out", res.stderr)

    def test_output_is_capped(self):
        script = self.workspace / "big.py"
        script.write_text("print('x' * 50000)")
        with patch.object(settings, "tool_sandbox_max_output_chars", 2000):
            res = run_sandboxed(["python3", str(script)], workspace=self.workspace)
        self.assertTrue(res.truncated)
        self.assertLessEqual(len(res.stdout), 2000)
        self.assertIn("[output truncated]", res.format())


@needs_bwrap
class TestBwrapEscapeProbes(unittest.IsolatedAsyncioTestCase):
    """The commands that escaped the previous sandbox must now fail."""

    async def asyncSetUp(self):
        self.mode_patcher = patch.object(settings, "tool_sandbox_mode", "bwrap")
        self.mode_patcher.start()
        self.shell = ShellTool()
        self.runner = CodeRunnerTool()

    async def asyncTearDown(self):
        self.mode_patcher.stop()

    async def test_cannot_read_host_etc(self):
        out = await self.shell.execute(command="cat /etc/hostname")
        self.assertIn("No such file or directory", out)
        self.assertNotIn(os.uname().nodename, out)

    async def test_cannot_see_host_home(self):
        out = await self.runner.execute(
            code="import os; print(sorted(os.listdir(os.path.expanduser('~'))))"
        )
        self.assertEqual(out.strip(), "[]", out)

    async def test_find_root_shows_only_sandbox(self):
        out = await self.shell.execute(command="find / -maxdepth 1")
        self.assertIn("/usr", out)
        self.assertNotIn("/root", out)
        self.assertNotIn("/var", out)
        self.assertNotIn("/boot", out)

    async def test_no_network(self):
        out = await self.runner.execute(
            code=(
                "import socket\n"
                "socket.setdefaulttimeout(2)\n"
                "try:\n"
                "    socket.create_connection(('1.1.1.1', 53)); print('CONNECTED')\n"
                "except OSError as e:\n"
                "    print('BLOCKED', e.errno)\n"
            )
        )
        self.assertIn("BLOCKED", out)
        self.assertNotIn("CONNECTED", out)

    async def test_cannot_write_outside_workspace(self):
        out = await self.runner.execute(
            code=(
                "import os\n"
                "for p in ['/usr/pwned', '/tmp/../etc/pwned', os.path.expanduser('~/.bashrc_pwned')]:\n"
                "    try:\n"
                "        open(p, 'w').write('x'); print('WROTE', p)\n"
                "    except OSError as e:\n"
                "        print('DENIED', p, type(e).__name__)\n"
            )
        )
        self.assertNotIn("WROTE /usr", out)
        self.assertNotIn("WROTE /tmp/../etc", out)
        self.assertIn("DENIED /usr/pwned", out)
        # ~/.bashrc_pwned lands in the tmpfs HOME inside the sandbox, never on the host.
        self.assertFalse(Path.home().joinpath(".bashrc_pwned").exists())

    async def test_workspace_write_and_venv_packages_work(self):
        from maidere.tools.shell import WORKSPACE

        marker = WORKSPACE / ".tmp" / "sandbox_marker.txt"
        try:
            out = await self.runner.execute(
                code=(
                    "import json, pathlib\n"
                    "pathlib.Path('.tmp/sandbox_marker.txt').write_text('ok')\n"
                    "import numpy\n"
                    "print('numpy', numpy.__version__.split('.')[0])\n"
                )
            )
            self.assertIn("numpy", out)
            self.assertTrue(marker.exists())
        finally:
            marker.unlink(missing_ok=True)

    async def test_server_secrets_not_visible(self):
        with patch.dict(os.environ, {"NVIDIA_API_KEY": "nvapi-should-not-leak"}):
            out = await self.runner.execute(
                code="import os; print(os.environ.get('NVIDIA_API_KEY', 'unset'), os.environ.get('HOME'))"
            )
        self.assertTrue(out.startswith("unset "), out)

    async def test_health_reports_bwrap(self):
        self.assertEqual(sandbox.describe()["active"], "bwrap")


class TestShellToolWithRestrictedMode(unittest.IsolatedAsyncioTestCase):
    """Shell tool surfaces restricted-mode refusals as readable errors."""

    async def asyncSetUp(self):
        self.mode_patcher = patch.object(settings, "tool_sandbox_mode", "restricted")
        self.mode_patcher.start()
        self.shell = ShellTool()

    async def asyncTearDown(self):
        self.mode_patcher.stop()

    async def test_host_path_refused(self):
        out = await self.shell.execute(command="cat /etc/hostname")
        self.assertTrue(out.startswith("Error:"))
        self.assertIn("outside the workspace", out)

    async def test_inline_python_refused(self):
        out = await self.shell.execute(command='python3 -c "print(1)"')
        self.assertTrue(out.startswith("Error:"))
        self.assertIn("inline code", out)

    async def test_plain_command_still_works(self):
        out = await self.shell.execute(command="echo restricted-ok")
        self.assertEqual(out.strip(), "restricted-ok")

    @unittest.skipUnless(shutil.which("python3"), "python3 missing")
    async def test_code_runner_works_in_restricted_mode(self):
        out = await CodeRunnerTool().execute(code="print(6 * 7)")
        self.assertEqual(out.strip(), "42")


if __name__ == "__main__":
    unittest.main()
