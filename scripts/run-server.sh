#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
set -a
source "$ROOT/config/model.env"
set +a

PYTHON="${PYTHON:-$ROOT/.venv-mlx/bin/python}"
if [[ ! -x "$PYTHON" ]]; then
  echo "Missing Python environment: $PYTHON" >&2
  echo "Run: ./scripts/setup.sh" >&2
  exit 1
fi

cd "$ROOT"
exec "$PYTHON" -m uvicorn \
  local_openai_mlx_provider.server:app \
  --host "$HOST" \
  --port "$PORT"
