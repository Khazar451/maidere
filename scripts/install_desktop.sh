#!/usr/bin/env bash
# ==============================================================================
# Maidere Desktop Application Installer for Linux Mint / Ubuntu
# ==============================================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

echo "=================================================="
echo " Installing Maidere Desktop Application..."
echo " Project root: $PROJECT_ROOT"
echo "=================================================="

# Check if uv or python is available
if [ -d "$PROJECT_ROOT/.venv" ]; then
    PYTHON_BIN="$PROJECT_ROOT/.venv/bin/python"
elif command -v uv >/dev/null 2>&1; then
    PYTHON_BIN="$(uv run which python)"
elif command -v python3 >/dev/null 2>&1; then
    PYTHON_BIN="$(which python3)"
else
    echo "Error: Python binary could not be found." >&2
    exit 1
fi

echo "Using Python binary: $PYTHON_BIN"

# Run installation through Python launcher
cd "$PROJECT_ROOT"
"$PYTHON_BIN" maidere.py install-desktop

echo ""
echo "Installation complete!"
