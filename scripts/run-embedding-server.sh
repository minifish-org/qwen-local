#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
source "$ROOT/config/model.env"

BIN="$ROOT/llama.cpp/build/bin/llama-server"
if [[ ! -x "$BIN" ]]; then
  echo "llama-server is not built yet: $BIN" >&2
  exit 1
fi

exec "$BIN" \
  -hf "$EMBEDDING_MODEL_HF" \
  -c "$EMBEDDING_CTX_SIZE" \
  -ngl "$GPU_LAYERS" \
  -fa on \
  --embedding \
  --pooling last \
  -b "$EMBEDDING_BATCH_SIZE" \
  -ub "$EMBEDDING_UBATCH_SIZE" \
  --host "$HOST" \
  --port "$EMBEDDING_PORT"
