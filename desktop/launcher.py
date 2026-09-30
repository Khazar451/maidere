"""Maidere Desktop Application Launcher & Lifecycle Manager.

Manages backend server startup, health check polling, native standalone window
spawning, and clean process teardown. Supports both standard desktop execution
and live-reloading continuous development (--dev).
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import time
from typing import Any
import urllib.error
import urllib.request
import webbrowser

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_HOST = "127.0.0.1"
DEFAULT_PORT = 8000


def is_server_healthy(host: str = DEFAULT_HOST, port: int = DEFAULT_PORT, timeout: float = 1.0) -> bool:
    """Check if Maidere server is actively responding on /health."""
    url = f"http://{host}:{port}/health"
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "Maidere-Desktop-Launcher"})
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return resp.status == 200
    except (urllib.error.URLError, TimeoutError, ConnectionRefusedError, OSError):
        return False


def wait_for_server(host: str = DEFAULT_HOST, port: int = DEFAULT_PORT, max_wait_sec: float = 15.0) -> bool:
    """Poll /health until server is ready or timeout expires."""
    start_time = time.time()
    while time.time() - start_time < max_wait_sec:
        if is_server_healthy(host, port):
            return True
        time.sleep(0.25)
    return False


def start_server_process(host: str = DEFAULT_HOST, port: int = DEFAULT_PORT, dev_mode: bool = False) -> subprocess.Popen:
    """Launch Uvicorn backend server as a managed child subprocess."""
    python_bin = sys.executable
    cmd = [
        python_bin,
        "-m",
        "uvicorn",
        "api.app:app",
        "--host",
        host,
        "--port",
        str(port),
    ]

    if dev_mode:
        cmd.extend([
            "--reload",
            "--reload-dir",
            str(PROJECT_ROOT / "core"),
            "--reload-dir",
            str(PROJECT_ROOT / "api"),
            "--reload-dir",
            str(PROJECT_ROOT / "tools"),
            "--reload-dir",
            str(PROJECT_ROOT / "static"),
        ])

    env = os.environ.copy()
    env["PYTHONUNBUFFERED"] = "1"
    if dev_mode:
        env["MAIDERE_DEV"] = "1"

    proc = subprocess.Popen(
        cmd,
        cwd=str(PROJECT_ROOT),
        env=env,
        stdout=sys.stdout,
        stderr=sys.stderr,
    )
    return proc


def find_browser_engine() -> tuple[str | None, list[str]]:
    """Detect available standalone browser window engines on Linux / OS.

    Returns (engine_name, command_arguments_prefix).
    """
    # 1. Chromium-based browsers support dedicated --app window mode
    chromium_candidates = [
        "google-chrome-stable",
        "google-chrome",
        "brave-browser",
        "chromium",
        "chromium-browser",
        "microsoft-edge",
    ]
    for candidate in chromium_candidates:
        bin_path = shutil.which(candidate)
        if bin_path:
            profile_dir = Path.home() / ".config" / "maidere" / "browser_profile"
            profile_dir.mkdir(parents=True, exist_ok=True)
            return candidate, [
                bin_path,
                f"--user-data-dir={profile_dir}",
                "--no-first-run",
                "--no-default-browser-check",
                "--class=Maidere",
                "--window-size=1440,900",
            ]

    # 2. Firefox standalone window
    firefox_bin = shutil.which("firefox")
    if firefox_bin:
        return "firefox", [firefox_bin, "--new-window", "--class", "Maidere"]

    return None, []


def launch_window(url: str, title: str = "Maidere", dev_mode: bool = False) -> subprocess.Popen | None:
    """Open the Maidere application in a native standalone window."""
    # Priority 1: pywebview if installed
    try:
        import webview  # type: ignore

        print(f"[DESKTOP] Launching via pywebview window engine (dev_mode={dev_mode})...")
        window = webview.create_window(
            title=title,
            url=url,
            width=1440,
            height=900,
            min_size=(960, 640),
            background_color="#0d0b18",
        )
        webview.start(debug=dev_mode)
        return None
    except ImportError:
        pass

    # Priority 2: Standalone browser app mode
    engine_name, cmd_prefix = find_browser_engine()
    if engine_name and cmd_prefix:
        print(f"[DESKTOP] Launching standalone window using {engine_name}...")
        if "--app" not in " ".join(cmd_prefix) and engine_name != "firefox":
            cmd = cmd_prefix + [f"--app={url}"]
        elif engine_name == "firefox":
            cmd = cmd_prefix + [url]
        else:
            cmd = cmd_prefix + [f"--app={url}"]

        return subprocess.Popen(
            cmd,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )

    # Priority 3: Default system browser
    print("[DESKTOP] Falling back to default system browser...")
    webbrowser.open(url)
    return None


def install_desktop_entry() -> bool:
    """Install Linux XDG .desktop launcher file and application icon."""
    apps_dir = Path.home() / ".local" / "share" / "applications"
    icons_dir = Path.home() / ".local" / "share" / "icons" / "hicolor" / "scalable" / "apps"
    apps_dir.mkdir(parents=True, exist_ok=True)
    icons_dir.mkdir(parents=True, exist_ok=True)

    src_icon = PROJECT_ROOT / "static" / "icon.svg"
    target_icon = icons_dir / "maidere.svg"
    if src_icon.exists():
        shutil.copy2(src_icon, target_icon)
        print(f"[INSTALL] Copied desktop icon to {target_icon}")

    python_bin = sys.executable
    main_script = PROJECT_ROOT / "maidere.py"

    desktop_content = f"""[Desktop Entry]
Version=1.0
Type=Application
Name=Maidere
GenericName=Autonomous AI Agent
Comment=100% Free, self-hosted autonomous AI agent running locally
Exec={python_bin} {main_script}
Icon={target_icon}
Terminal=false
Categories=Development;Utility;ArtificialIntelligence;
StartupWMClass=Maidere
Keywords=AI;Agent;Assistant;LLM;Ollama;
"""
    desktop_file = apps_dir / "maidere.desktop"
    desktop_file.write_text(desktop_content, encoding="utf-8")
    desktop_file.chmod(0o755)
    print(f"[INSTALL] Created desktop launcher at {desktop_file}")

    # Update system desktop database if utility exists
    if shutil.which("update-desktop-database"):
        subprocess.run(["update-desktop-database", str(apps_dir)], check=False)
        print("[INSTALL] Updated Linux desktop database.")

    print("\n[SUCCESS] Maidere is now installed in your Linux application menu!")
    print("Search 'Maidere' in your Start Menu or pin it to your desktop/dock.")
    return True


def run_desktop(
    host: str = DEFAULT_HOST,
    port: int = DEFAULT_PORT,
    dev_mode: bool = False,
    server_only: bool = False,
) -> int:
    """Main desktop application entrypoint managing server and window lifecycle."""
    server_proc: subprocess.Popen | None = None
    window_proc: subprocess.Popen | None = None
    started_our_server = False

    url = f"http://{host}:{port}"

    def _cleanup(signum: Any = None, frame: Any = None):
        nonlocal server_proc, window_proc
        print("\n[DESKTOP] Shutting down Maidere desktop session...")
        if window_proc and window_proc.poll() is None:
            try:
                window_proc.terminate()
            except Exception:
                pass

        if started_our_server and server_proc and server_proc.poll() is None:
            print("[DESKTOP] Stopping backend server...")
            try:
                server_proc.send_signal(signal.SIGINT)
                server_proc.wait(timeout=5.0)
            except Exception:
                server_proc.kill()
        sys.exit(0)

    signal.signal(signal.SIGINT, _cleanup)
    signal.signal(signal.SIGTERM, _cleanup)

    # 1. Check or start server
    if is_server_healthy(host, port):
        print(f"[DESKTOP] Detected active Maidere server running on {url}")
    else:
        print(f"[DESKTOP] Starting Maidere backend server on {url} (dev_mode={dev_mode})...")
        server_proc = start_server_process(host, port, dev_mode=dev_mode)
        started_our_server = True

        if not wait_for_server(host, port, max_wait_sec=15.0):
            print(f"[ERROR] Maidere server failed to become healthy on {url} within 15 seconds.")
            _cleanup()
            return 1
        print(f"[DESKTOP] Backend server ready at {url}")

    if server_only:
        print("[DESKTOP] Running in server-only headless mode. Press Ctrl+C to stop.")
        try:
            while True:
                time.sleep(1.0)
                if server_proc and server_proc.poll() is not None:
                    break
        except KeyboardInterrupt:
            pass
        finally:
            _cleanup()
        return 0

    # 2. Launch native application window
    window_proc = launch_window(url, title="Maidere — Local Autonomous AI Agent", dev_mode=dev_mode)

    # 3. Keep launcher alive while window or server is running
    try:
        if window_proc:
            window_proc.wait()
            print("[DESKTOP] Application window closed.")
        else:
            # If pywebview or browser took control or ran synchronously
            if server_proc:
                server_proc.wait()
            else:
                print("[DESKTOP] App running in browser. Press Ctrl+C to exit launcher.")
                while True:
                    time.sleep(1.0)
    except KeyboardInterrupt:
        pass
    finally:
        _cleanup()

    return 0
