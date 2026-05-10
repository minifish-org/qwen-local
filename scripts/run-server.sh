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
  -hf "$MODEL_HF" \
  -c "$CTX_SIZE" \
  -ngl "$GPU_LAYERS" \
  -fa on \
  --reasoning "$REASONING" \
  -b "$BATCH_SIZE" \
  -ub "$UBATCH_SIZE" \
  --host "$HOST" \
  --port "$PORT"
