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

# Change working directory to Maidere repository root
cd "$REPO_ROOT"

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

# 4. Trigger Web UI launch in default browser non-blockingly (~1.5s delay)
(
  sleep 1.5
  if command -v xdg-open >/dev/null 2>&1 && [ -n "$DISPLAY$WAYLAND_DISPLAY" ]; then
    xdg-open "http://localhost:8000" >/dev/null 2>&1 || true
  elif command -v open >/dev/null 2>&1; then
    open "http://localhost:8000" >/dev/null 2>&1 || true
  fi
) &

echo "   Web UI accessible at: http://localhost:8000"
echo "   Press Ctrl + C to stop Maidere"
echo "================================================================="
echo ""

# 5. Determine Python / Uvicorn runner and exec process for clean signal handling
if command -v uv >/dev/null 2>&1; then
  exec uv run uvicorn api.app:app --host 0.0.0.0 --port 8000 --reload
elif [ -f "$REPO_ROOT/.venv/bin/uvicorn" ]; then
  exec "$REPO_ROOT/.venv/bin/uvicorn" api.app:app --host 0.0.0.0 --port 8000 --reload
elif [ -f "$REPO_ROOT/.venv/bin/python" ]; then
  exec "$REPO_ROOT/.venv/bin/python" -m uvicorn api.app:app --host 0.0.0.0 --port 8000 --reload
else
  exec uvicorn api.app:app --host 0.0.0.0 --port 8000 --reload
fi
