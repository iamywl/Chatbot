#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

PYTHON_BIN="${PYTHON_BIN:-/opt/anaconda3/bin/python3}"
if [ ! -x "$PYTHON_BIN" ]; then
  PYTHON_BIN="$(command -v python3)"
fi

HOST="${HOST:-127.0.0.1}"
PORT="${PORT:-8000}"

# Idempotent restart: stop any existing server bound to this app/port
if pgrep -f "uvicorn backend.main:app" >/dev/null 2>&1; then
  echo "Stopping existing uvicorn backend.main:app..."
  pkill -TERM -f "uvicorn backend.main:app" || true
  for _ in $(seq 1 20); do
    pgrep -f "uvicorn backend.main:app" >/dev/null 2>&1 || break
    sleep 0.25
  done
  pgrep -f "uvicorn backend.main:app" >/dev/null 2>&1 && pkill -KILL -f "uvicorn backend.main:app" || true
fi

echo "Starting Private MLX Chatbot on http://${HOST}:${PORT}"
exec "$PYTHON_BIN" -m uvicorn backend.main:app --host "$HOST" --port "$PORT"
