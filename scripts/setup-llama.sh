#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
LLAMA_DIR="$ROOT/llama.cpp"
LLAMA_REPO="https://github.com/ggml-org/llama.cpp.git"
LLAMA_COMMIT="1e5ad35d560b90a8ac447d149c8f8447ae1fcaa0"

if [[ ! -d "$LLAMA_DIR/.git" ]]; then
  git clone "$LLAMA_REPO" "$LLAMA_DIR"
fi

if [[ "$(git -C "$LLAMA_DIR" rev-parse HEAD 2>/dev/null || true)" != "$LLAMA_COMMIT" ]]; then
  git -C "$LLAMA_DIR" fetch origin
fi

git -C "$LLAMA_DIR" checkout --detach "$LLAMA_COMMIT"

cmake -S "$LLAMA_DIR" -B "$LLAMA_DIR/build" \
  -DGGML_METAL=ON \
  -DCMAKE_BUILD_TYPE=Release

cmake --build "$LLAMA_DIR/build" --config Release -j "$(sysctl -n hw.ncpu)"
