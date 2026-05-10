#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
source "$ROOT/config/model.env"

BASE_URL="${BASE_URL:-http://127.0.0.1:${PORT}}"
EMBEDDING_BASE_URL="${EMBEDDING_BASE_URL:-http://127.0.0.1:${EMBEDDING_PORT}}"

echo "# chat"
curl -fsS "${BASE_URL}/health"
echo
curl -fsS "${BASE_URL}/v1/models"
echo

echo "# embedding"
curl -fsS "${EMBEDDING_BASE_URL}/health"
echo
curl -fsS "${EMBEDDING_BASE_URL}/v1/models"
echo
