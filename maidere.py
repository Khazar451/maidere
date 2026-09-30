#!/usr/bin/env python3
"""Maidere — 100% Free, Self-Hosted Autonomous AI Agent.

CLI Entrypoint for running the desktop application, server, or managing OS integration.

Usage:
  maidere                     # Launch native desktop application
  maidere --dev               # Launch in live hot-reload development mode
  maidere --server-only       # Start backend server only (headless)
  maidere install-desktop     # Install Linux desktop icon & Start Menu entry
"""

from __future__ import annotations

import argparse
import sys
from desktop.launcher import install_desktop_entry, run_desktop


def main() -> int:
    """Main CLI entrypoint for Maidere."""
    parser = argparse.ArgumentParser(
        prog="maidere",
        description="Maidere: Autonomous Local AI Agent Desktop Application",
    )
    parser.add_argument(
        "--dev",
        action="store_true",
        help="Enable live hot-reloading development mode (watches core/, api/, tools/, static/)",
    )
    parser.add_argument(
        "--server-only",
        action="store_true",
        help="Run backend server in headless mode without opening a desktop window",
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
        help="Action to perform: 'run' (default) or 'install-desktop' to create Linux launcher",
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


if __name__ == "__main__":
    sys.exit(main())
