"""Subprocess isolation for agent tools (``shell`` and ``code_runner``).

Three modes, selected by ``settings.tool_sandbox_mode``:

``bwrap`` (preferred)
    Each command runs inside a bubblewrap sandbox: read-only ``/usr`` (and the
    merged ``/bin``, ``/lib`` symlinks), the workspace bind-mounted read-write
    at its real path, tmpfs ``/tmp`` and ``HOME``, fresh user/PID/IPC/UTS/
    cgroup namespaces, **no network** unless ``tool_sandbox_allow_network`` is
    set, ``--die-with-parent`` and ``--new-session``. The environment is
    cleared (the server's env may contain API keys) and resource limits are
    applied through ``prlimit``. The project virtualenv is mounted read-only so
    ``code_runner`` keeps its packages.

``restricted`` (fallback when bubblewrap is missing or blocked)
    No namespaces. Arguments are validated so that path-like arguments stay
    inside the workspace, interpreters cannot take inline code (``-c``/``-e``),
    and ``find`` cannot ``-exec``. The environment is still scrubbed and
    rlimits still apply. This confines *arguments*, not the code a script can
    run once started; it is a best effort, and the startup log says so.

``off``
    Direct execution with cwd=workspace and a scrubbed env. Development only.

``auto`` resolves to ``bwrap`` if a one-time probe succeeds, else ``restricted``.
"""

from __future__ import annotations

import os
import re
import shutil
import subprocess
import sys
import threading
from dataclasses import dataclass
from enum import Enum
from pathlib import Path

import structlog

from maidere.core.config import settings

logger = structlog.get_logger()


class SandboxMode(str, Enum):
    BWRAP = "bwrap"
    RESTRICTED = "restricted"
    OFF = "off"


class SandboxError(Exception):
    """Raised when a command cannot be run under the active policy."""


@dataclass
class SandboxResult:
    stdout: str
    stderr: str
    returncode: int
    timed_out: bool
    mode: SandboxMode
    truncated: bool = False

    def format(self, success_message: str = "Command executed successfully (no output).") -> str:
        """Render in the format the agent tools have always returned."""
        parts: list[str] = []
        if self.stdout:
            parts.append(self.stdout)
        if self.stderr:
            parts.append(f"[stderr]\n{self.stderr}")
        if self.truncated:
            parts.append("[output truncated]")
        if self.returncode != 0 and not self.timed_out:
            parts.append(f"[Process exited with code {self.returncode}]")
        return "\n".join(parts) if parts else success_message


# ---------------------------------------------------------------------------
# Host introspection (cached)
# ---------------------------------------------------------------------------

_probe_lock = threading.Lock()
_bwrap_path: str | None = None
_bwrap_ok: bool | None = None
_prlimit_path: str | None | bool = False  # False = not probed yet
_warned_fallback = False


def _merged_usr_links() -> list[tuple[str, str]]:
    """Return (target, path) for /bin, /sbin, /lib, /lib64 when they are symlinks into /usr."""
    links = []
    for name in ("bin", "sbin", "lib", "lib64", "lib32", "libx32"):
        p = Path("/") / name
        if p.is_symlink():
            links.append((os.readlink(p), f"/{name}"))
    return links


def _probe_bwrap() -> tuple[str | None, bool]:
    """Locate bubblewrap and check that it can create a sandbox on this host."""
    path = shutil.which("bwrap")
    if not path:
        return None, False
    cmd = [path, "--ro-bind", "/usr", "/usr"]
    for target, link in _merged_usr_links():
        cmd += ["--symlink", target, link]
    for name in ("bin", "lib", "lib64"):
        p = Path("/") / name
        if p.exists() and not p.is_symlink():
            cmd += ["--ro-bind", str(p), str(p)]
    cmd += ["--proc", "/proc", "--dev", "/dev", "--unshare-all", "--die-with-parent", "--", "/bin/true"]
    try:
        proc = subprocess.run(cmd, capture_output=True, timeout=10, check=False)
        ok = proc.returncode == 0
        if not ok:
            logger.warning("bwrap_probe_failed", returncode=proc.returncode, stderr=proc.stderr.decode(errors="replace")[:300])
        return path, ok
    except Exception as exc:  # pragma: no cover - defensive
        logger.warning("bwrap_probe_error", error=str(exc))
        return path, False


def bwrap_available() -> bool:
    global _bwrap_path, _bwrap_ok
    with _probe_lock:
        if _bwrap_ok is None:
            _bwrap_path, _bwrap_ok = _probe_bwrap()
        return bool(_bwrap_ok)


def _prlimit() -> str | None:
    global _prlimit_path
    if _prlimit_path is False:
        _prlimit_path = shutil.which("prlimit")
    return _prlimit_path  # type: ignore[return-value]


def reset_probe_cache() -> None:
    """Testing hook: forget cached host probes."""
    global _bwrap_path, _bwrap_ok, _prlimit_path, _warned_fallback
    with _probe_lock:
        _bwrap_path = None
        _bwrap_ok = None
        _prlimit_path = False
        _warned_fallback = False


def resolve_mode() -> SandboxMode:
    """Map the configured mode to the mode that will actually be used."""
    global _warned_fallback
    configured = (settings.tool_sandbox_mode or "auto").strip().lower()
    if configured == "off":
        return SandboxMode.OFF
    if configured == "restricted":
        return SandboxMode.RESTRICTED
    if configured == "bwrap":
        if not bwrap_available():
            raise SandboxError(
                "TOOL_SANDBOX_MODE=bwrap but bubblewrap is unavailable or cannot create "
                "user namespaces on this host. Install `bubblewrap` or set "
                "TOOL_SANDBOX_MODE=restricted."
            )
        return SandboxMode.BWRAP
    # auto
    if bwrap_available():
        return SandboxMode.BWRAP
    if not _warned_fallback:
        _warned_fallback = True
        logger.warning(
            "tool_sandbox_fallback",
            mode="restricted",
            detail="bubblewrap not usable; tool subprocesses are confined by argument checks only.",
        )
    return SandboxMode.RESTRICTED


def describe() -> dict[str, object]:
    """Summary for /health and startup logs."""
    try:
        mode = resolve_mode().value
        error = None
    except SandboxError as exc:
        mode = "unavailable"
        error = str(exc)
    return {
        "configured": settings.tool_sandbox_mode,
        "active": mode,
        "bwrap": bwrap_available(),
        "network": bool(settings.tool_sandbox_allow_network),
        "error": error,
    }


# ---------------------------------------------------------------------------
# Environment and limits (all modes)
# ---------------------------------------------------------------------------

_SAFE_ENV_PASSTHROUGH = ("TERM",)


def build_environment(home: str, tmp: str) -> dict[str, str]:
    """Minimal environment: never inherit the server's env (it may hold API keys)."""
    env = {
        "PATH": "/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin",
        "HOME": home,
        "TMPDIR": tmp,
        "LANG": "C.UTF-8",
        "LC_ALL": "C.UTF-8",
        "PYTHONDONTWRITEBYTECODE": "1",
        "PYTHONUNBUFFERED": "1",
        "PYTHONIOENCODING": "utf-8",
        "NO_COLOR": "1",
        "GIT_TERMINAL_PROMPT": "0",
        "MAIDERE_SANDBOX": "1",
    }
    for key in _SAFE_ENV_PASSTHROUGH:
        if key in os.environ:
            env[key] = os.environ[key]
    return env


def _rlimit_prefix() -> list[str]:
    prl = _prlimit()
    if not prl:
        return []
    args = [prl]
    mem_mb = int(settings.tool_sandbox_memory_mb)
    if mem_mb > 0:  # 0 disables the address-space cap (V8/node needs a large VA reservation)
        args.append(f"--as={max(64, mem_mb) * 1024 * 1024}")
    fsize = max(1, int(settings.tool_sandbox_max_file_mb)) * 1024 * 1024
    args += [f"--fsize={fsize}", "--core=0", "--"]
    return args


# ---------------------------------------------------------------------------
# bwrap command construction
# ---------------------------------------------------------------------------

_SANDBOX_HOME = "/tmp/home"


def _extra_ro_paths() -> list[Path]:
    paths = []
    for raw in (settings.tool_sandbox_extra_ro_paths or "").split(","):
        raw = raw.strip()
        if raw:
            p = Path(raw).expanduser()
            if p.exists():
                paths.append(p.resolve())
    return paths


def _is_under(path: Path, roots: list[Path]) -> bool:
    return any(path == r or r in path.parents for r in roots)


def build_bwrap_command(
    argv: list[str],
    workspace: Path,
    *,
    allow_network: bool = False,
    extra_ro: list[Path] | None = None,
) -> list[str]:
    """Construct the full bubblewrap argv for ``argv`` confined to ``workspace``."""
    if not _bwrap_path:
        bwrap_available()
    bwrap = _bwrap_path or "bwrap"
    workspace = workspace.resolve()

    cmd: list[str] = [bwrap, "--ro-bind", "/usr", "/usr"]
    for target, link in _merged_usr_links():
        cmd += ["--symlink", target, link]
    for name in ("bin", "sbin", "lib", "lib64"):
        p = Path("/") / name
        if p.exists() and not p.is_symlink():
            cmd += ["--ro-bind", str(p), str(p)]

    # Small, non-sensitive pieces of /etc that interpreters and git expect.
    for etc in (
        "/etc/alternatives",
        "/etc/ld.so.cache",
        "/etc/ld.so.conf",
        "/etc/ld.so.conf.d",
        "/etc/localtime",
        "/etc/timezone",
        "/etc/python3",
        "/etc/python3.12",
        "/etc/python3.13",
        # Ubuntu's apport sitecustomize hook reads these on uncaught exceptions;
        # exposing them read-only avoids noisy warnings in tracebacks.
        "/etc/apt",
        "/etc/default/apport",
    ):
        cmd += ["--ro-bind-try", etc, etc]
    if allow_network:
        for etc in ("/etc/resolv.conf", "/etc/hosts", "/etc/ssl", "/etc/ca-certificates", "/etc/nsswitch.conf"):
            cmd += ["--ro-bind-try", etc, etc]
    # Deliberately no /etc/passwd, /etc/shadow, /home, or the server's own
    # repository: nothing outside /usr and the workspace is visible.
    cmd += ["--proc", "/proc", "--dev", "/dev", "--tmpfs", "/tmp", "--dir", _SANDBOX_HOME]

    # Project virtualenv (read-only) so code_runner keeps its packages.
    bound_roots: list[Path] = [Path("/usr")]
    venv = Path(sys.prefix).resolve()
    if venv != Path(sys.base_prefix).resolve() and venv.exists() and not _is_under(venv, [workspace]):
        cmd += ["--ro-bind", str(venv), str(venv)]
        bound_roots.append(venv)
    base = Path(sys.base_prefix).resolve()
    if not _is_under(base, bound_roots) and base.exists():
        cmd += ["--ro-bind", str(base), str(base)]
        bound_roots.append(base)

    for extra in extra_ro or []:
        if not _is_under(extra, bound_roots):
            cmd += ["--ro-bind", str(extra), str(extra)]
            bound_roots.append(extra)

    # The executable itself may live outside /usr (e.g. a user-local node).
    # Mount its install root read-only and exec the real path (symlinks such
    # as ~/.local/bin/node do not exist inside the sandbox).
    argv = list(argv)
    exe = Path(argv[0])
    if exe.is_absolute():
        real = exe.resolve()
        if real.exists() and not _is_under(real, bound_roots + [workspace]):
            install_root = real.parent.parent if real.parent.name == "bin" else real.parent
            cmd += ["--ro-bind", str(install_root), str(install_root)]
            bound_roots.append(install_root)
            argv[0] = str(real)

    # Workspace: the only writable host path.
    cmd += ["--bind", str(workspace), str(workspace), "--chdir", str(workspace)]

    # Namespaces and process hygiene.
    cmd += ["--unshare-all"]
    if allow_network:
        cmd += ["--share-net"]
    cmd += ["--die-with-parent", "--new-session", "--hostname", "maidere-sandbox", "--cap-drop", "ALL"]
    # Make the sandbox's private root read-only; /tmp (tmpfs) and the workspace
    # bind are separate mounts and stay writable.
    cmd += ["--remount-ro", "/"]

    # Environment.
    cmd += ["--clearenv"]
    for key, value in build_environment(_SANDBOX_HOME, "/tmp").items():
        cmd += ["--setenv", key, value]

    cmd += ["--"] + list(argv)
    return cmd


# ---------------------------------------------------------------------------
# Restricted-mode argument validation
# ---------------------------------------------------------------------------

_INLINE_CODE_FLAGS: dict[str, set[str]] = {
    "python": {"-c"},
    "node": {"-e", "--eval", "-p", "--print", "-r", "--require", "--import"},
}
_INTERPRETER_CANON = re.compile(r"^(?P<name>python|node)(js)?[0-9.]*$")


def canonical_binary_name(argv0: str) -> str:
    """``/usr/bin/python3.12`` -> ``python``; ``nodejs`` -> ``node``; otherwise the basename."""
    name = Path(argv0).name
    m = _INTERPRETER_CANON.match(name)
    return m.group("name") if m else name
_FIND_BLOCKED = {"-exec", "-execdir", "-ok", "-okdir", "-fprint", "-fprint0", "-fprintf", "-fls"}
_PATH_FLAG_PREFIXES = ("--work-tree=", "--git-dir=", "--output=", "-o=")


def _looks_like_path(arg: str) -> bool:
    return arg.startswith(("/", "~")) or ".." in arg.split("/") or arg.startswith("./..")


def _path_inside_workspace(arg: str, workspace: Path) -> bool:
    if arg.startswith("~"):
        return False
    candidate = Path(arg)
    target = candidate if candidate.is_absolute() else workspace / candidate
    try:
        resolved = target.resolve()
    except (OSError, RuntimeError):
        return False
    return resolved == workspace or workspace in resolved.parents


def validate_restricted(argv: list[str], workspace: Path) -> str | None:
    """Return an error message if ``argv`` is not acceptable in restricted mode."""
    if not argv:
        return "Error: No command arguments parsed."
    workspace = workspace.resolve()
    binary = canonical_binary_name(argv[0])

    blocked_flags = _INLINE_CODE_FLAGS.get(binary, set())
    for arg in argv[1:]:
        if arg in blocked_flags or any(arg.startswith(f + "=") for f in blocked_flags):
            return (
                f"Error: '{binary} {arg}' (inline code) is not permitted in restricted sandbox mode. "
                "Write the code to a file in the workspace and run that file, or install bubblewrap "
                "to enable full isolation."
            )
    if binary == "find":
        for arg in argv[1:]:
            if arg in _FIND_BLOCKED:
                return f"Error: 'find {arg}' is not permitted in restricted sandbox mode."

    for arg in argv[1:]:
        value = arg
        for prefix in _PATH_FLAG_PREFIXES:
            if arg.startswith(prefix):
                value = arg[len(prefix):]
                break
        else:
            if "=" in arg and arg.startswith("-"):
                value = arg.split("=", 1)[1]
        if _looks_like_path(value) and not _path_inside_workspace(value, workspace):
            return (
                f"Error: Path argument '{value}' is outside the workspace sandbox. "
                "Only paths inside the workspace are permitted."
            )
    return None


# ---------------------------------------------------------------------------
# Execution
# ---------------------------------------------------------------------------


def _truncate(text: str, limit: int) -> tuple[str, bool]:
    if len(text) <= limit:
        return text, False
    return text[:limit], True


def run_sandboxed(
    argv: list[str],
    *,
    workspace: Path,
    timeout: float | None = None,
    allow_network: bool | None = None,
    mode: SandboxMode | None = None,
) -> SandboxResult:
    """Run ``argv`` under the active sandbox policy. Blocking; call via ``asyncio.to_thread``.

    Raises ``SandboxError`` when the policy refuses to run the command.
    """
    if not argv:
        raise SandboxError("Error: No command arguments parsed.")
    workspace = workspace.resolve()
    workspace.mkdir(parents=True, exist_ok=True)
    timeout = float(timeout if timeout is not None else settings.tool_sandbox_timeout_seconds)
    allow_network = bool(settings.tool_sandbox_allow_network if allow_network is None else allow_network)
    active = mode or resolve_mode()
    out_limit = max(1000, int(settings.tool_sandbox_max_output_chars))

    if active is SandboxMode.BWRAP:
        full_cmd = _rlimit_prefix() + build_bwrap_command(
            argv, workspace, allow_network=allow_network, extra_ro=_extra_ro_paths()
        )
        env = None  # bwrap --clearenv/--setenv governs the inner env
        cwd = str(workspace)
    else:
        if active is SandboxMode.RESTRICTED:
            problem = validate_restricted(argv, workspace)
            if problem:
                raise SandboxError(problem)
        tmp_dir = workspace / ".tmp"
        tmp_dir.mkdir(parents=True, exist_ok=True)
        full_cmd = _rlimit_prefix() + list(argv)
        env = build_environment(home=str(tmp_dir), tmp=str(tmp_dir))
        cwd = str(workspace)

    try:
        proc = subprocess.run(
            full_cmd,
            cwd=cwd,
            env=env,
            shell=False,
            capture_output=True,
            text=True,
            errors="replace",
            timeout=timeout,
            stdin=subprocess.DEVNULL,
            check=False,
        )
    except subprocess.TimeoutExpired as exc:
        out = (exc.stdout or b"")
        err = (exc.stderr or b"")
        out_s = out.decode(errors="replace") if isinstance(out, bytes) else str(out)
        err_s = err.decode(errors="replace") if isinstance(err, bytes) else str(err)
        return SandboxResult(
            stdout=out_s.strip()[:out_limit],
            stderr=(err_s.strip() + f"\nError: Command execution timed out after {timeout:g} seconds.").strip(),
            returncode=-1,
            timed_out=True,
            mode=active,
        )

    stdout, t1 = _truncate(proc.stdout.strip(), out_limit)
    stderr, t2 = _truncate(proc.stderr.strip(), out_limit)
    return SandboxResult(
        stdout=stdout,
        stderr=stderr,
        returncode=proc.returncode,
        timed_out=False,
        mode=active,
        truncated=t1 or t2,
    )
