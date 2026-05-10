#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
source "$ROOT/config/model.env"

echo "Local:"
echo "  Chat:      http://127.0.0.1:${PORT}/v1"
echo "  Embedding: http://127.0.0.1:${EMBEDDING_PORT}/v1"

if command -v tailscale >/dev/null 2>&1; then
  TS_IP="$(tailscale ip -4 2>/dev/null | head -n 1 || true)"
  if [[ -n "$TS_IP" ]]; then
    echo "Tailscale:"
    echo "  Chat:      http://${TS_IP}:${PORT}/v1"
    echo "  Embedding: http://${TS_IP}:${EMBEDDING_PORT}/v1"
  fi
fi

echo "Model:"
echo "  Chat:      ${MODEL_HF}"
echo "  Embedding: ${EMBEDDING_MODEL_HF}"
