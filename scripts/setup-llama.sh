#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
LLAMA_DIR="$ROOT/llama.cpp"
LLAMA_REPO="https://github.com/ggml-org/llama.cpp.git"
LLAMA_COMMIT="1e5ad35"

if [[ ! -d "$LLAMA_DIR/.git" ]]; then
  git clone "$LLAMA_REPO" "$LLAMA_DIR"
fi

git -C "$LLAMA_DIR" fetch --depth 1 origin "$LLAMA_COMMIT"
git -C "$LLAMA_DIR" checkout --detach "$LLAMA_COMMIT"

cmake -S "$LLAMA_DIR" -B "$LLAMA_DIR/build" \
  -DGGML_METAL=ON \
  -DCMAKE_BUILD_TYPE=Release

cmake --build "$LLAMA_DIR/build" --config Release -j "$(sysctl -n hw.ncpu)"
