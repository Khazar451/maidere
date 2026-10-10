"""Command-line entry point. Installed as the ``maidere`` console script."""

from __future__ import annotations

import argparse

from maidere.desktop.launcher import install_desktop_entry, run_desktop


def main() -> int:
    """Launch the desktop app, a headless server, or install the OS launcher."""
    parser = argparse.ArgumentParser(
        prog="maidere",
        description="Maidere: Autonomous Local AI Agent Desktop Application",
    )
    parser.add_argument(
        "--dev",
        action="store_true",
        help="Enable live hot-reloading development mode",
    )
    parser.add_argument(
        "--server-only",
        action="store_true",
        help="Run the backend in headless mode without opening a desktop window",
    )
    parser.add_argument(
        "--host",
        default="127.0.0.1",
        help="Server binding host (default: 127.0.0.1)",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=8000,
        help="Server binding port (default: 8000)",
    )
    parser.add_argument(
        "command",
        nargs="?",
        choices=["run", "install-desktop"],
        default="run",
        help="Action to perform: 'run' (default) or 'install-desktop'",
    )

    args = parser.parse_args()

    if args.command == "install-desktop":
        return 0 if install_desktop_entry() else 1

    return run_desktop(
        host=args.host,
        port=args.port,
        dev_mode=args.dev,
        server_only=args.server_only,
    )
