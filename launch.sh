#!/usr/bin/env bash
# ==============================================================================
# Maidere Standalone CLI Launcher
# ==============================================================================
set -e

# 1. Dynamically resolve the absolute repository root directory (follows symlinks)
SOURCE="${BASH_SOURCE[0]}"
while [ -h "$SOURCE" ]; do
  DIR="$(cd -P "$(dirname "$SOURCE")" >/dev/null 2>&1 && pwd)"
  SOURCE="$(readlink "$SOURCE")"
  [[ $SOURCE != /* ]] && SOURCE="$DIR/$SOURCE"
done
REPO_ROOT="$(cd -P "$(dirname "$SOURCE")" >/dev/null 2>&1 && pwd)"

# Change working directory to Maidere repository root.
# src/ is on the path so the package imports even before an editable install.
cd "$REPO_ROOT"
export PYTHONPATH="$REPO_ROOT/src${PYTHONPATH:+:$PYTHONPATH}"

echo "================================================================="
echo "   Starting Maidere Autonomous AI Agent (v4.1)"
echo "   Working Directory: $REPO_ROOT"
echo "================================================================="

# 2. Check and start Docker auxiliary services (SearXNG, Prometheus, Grafana)
if [ -f "$REPO_ROOT/docker-compose.yml" ]; then
  if command -v docker >/dev/null 2>&1 && docker compose version >/dev/null 2>&1; then
    echo "   Starting Docker services (SearXNG, Prometheus, Grafana)..."
    docker compose up -d >/dev/null 2>&1 || true
  elif command -v podman-compose >/dev/null 2>&1; then
    podman-compose up -d >/dev/null 2>&1 || true
  fi
fi

# 3. Check Ollama connectivity
if ! curl -s -f "http://localhost:11434/api/tags" >/dev/null 2>&1; then
  echo ""
  echo "    [WARNING] Ollama is not responding on http://localhost:11434"
  echo "      Make sure to start Ollama with: 'ollama serve' or 'systemctl start ollama'"
  echo ""
else
  echo "   Ollama LLM service detected and active."
fi

# 4. Resolve runner and the API access token (created on first run, mode 0600)
HOST="${API_HOST:-127.0.0.1}"
PORT="${API_PORT:-8000}"

if [ -f "$REPO_ROOT/.venv/bin/python" ]; then
  PY="$REPO_ROOT/.venv/bin/python"
elif command -v uv >/dev/null 2>&1; then
  PY="uv run python"
else
  PY="python3"
fi

TOKEN="$($PY -c 'from maidere.core.tokens import load_or_create_token; print(load_or_create_token())' 2>/dev/null || true)"
UI_URL="http://${HOST}:${PORT}/"
if [ -n "$TOKEN" ]; then
  # The token travels in the URL fragment: browsers never send it to the server.
  OPEN_URL="${UI_URL}#token=${TOKEN}"
else
  OPEN_URL="$UI_URL"
fi

# 5. Trigger Web UI launch in default browser non-blockingly (~1.5s delay)
(
  sleep 1.5
  if command -v xdg-open >/dev/null 2>&1 && [ -n "$DISPLAY$WAYLAND_DISPLAY" ]; then
    xdg-open "$OPEN_URL" >/dev/null 2>&1 || true
  elif command -v open >/dev/null 2>&1; then
    open "$OPEN_URL" >/dev/null 2>&1 || true
  fi
) &

echo "   Web UI accessible at: $UI_URL"
echo "   API token file:       ~/.config/maidere/token"
echo "   Press Ctrl + C to stop Maidere"
echo "================================================================="
echo ""

# 6. Exec uvicorn bound to loopback for clean signal handling
exec $PY -m uvicorn maidere.api.app:app --host "$HOST" --port "$PORT" --reload
