#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
source "$ROOT/config/model.env"

BIN="$ROOT/llama.cpp/build/bin/llama-bench"
if [[ ! -x "$BIN" ]]; then
  echo "llama-bench is not built yet: $BIN" >&2
  exit 1
fi

exec "$BIN" \
  -hf "$MODEL_HF" \
  -p 16384 \
  -n "$BENCH_GEN_TOKENS" \
  -r 1 \
  -ngl "$GPU_LAYERS" \
  -fa 1 \
  -b "$BATCH_SIZE" \
  -ub "$UBATCH_SIZE" \
  --progress
