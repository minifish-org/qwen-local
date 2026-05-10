#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
source "$ROOT/config/model.env"

BIN="$ROOT/llama.cpp/build/bin/llama-bench"
OUT="$ROOT/notes/benchmarks.md"

if [[ ! -x "$BIN" ]]; then
  echo "llama-bench is not built yet: $BIN" >&2
  exit 1
fi

mkdir -p "$ROOT/notes"

{
  echo "# Benchmarks"
  echo
  echo "Generated: $(date '+%Y-%m-%d %H:%M:%S %Z')"
  echo
  echo "Model: \`$MODEL_HF\`"
  echo "Prompt tokens: \`$BENCH_MATRIX_PROMPT_TOKENS\`"
  echo "Generation tokens: \`$BENCH_MATRIX_GEN_TOKENS\`"
  echo "Repetitions: \`$BENCH_MATRIX_REPETITIONS\`"
  echo
  echo "## Matrix"
  echo
} > "$OUT"

run_case() {
  local name="$1"
  shift

  echo "Running $name..."
  {
    echo "### $name"
    echo
    echo '```text'
    "$BIN" \
      -hf "$MODEL_HF" \
      -p "$BENCH_MATRIX_PROMPT_TOKENS" \
      -n "$BENCH_MATRIX_GEN_TOKENS" \
      -r "$BENCH_MATRIX_REPETITIONS" \
      -ngl "$GPU_LAYERS" \
      -fa 1 \
      "$@"
    echo '```'
    echo
  } >> "$OUT"
}

run_case "b512-ub256-f16kv"  -b 512  -ub 256 -ctk f16  -ctv f16
run_case "b1024-ub256-f16kv" -b 1024 -ub 256 -ctk f16  -ctv f16
run_case "b1024-ub512-f16kv" -b 1024 -ub 512 -ctk f16  -ctv f16
run_case "b512-ub256-q8kv"   -b 512  -ub 256 -ctk q8_0 -ctv q8_0
run_case "b1024-ub256-q8kv"  -b 1024 -ub 256 -ctk q8_0 -ctv q8_0

echo "Wrote $OUT"
