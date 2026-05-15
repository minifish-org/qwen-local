#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python3.12}"

cd "$ROOT"
"$PYTHON_BIN" -m venv .venv-mlx
. .venv-mlx/bin/activate
python -m pip install --upgrade pip
python -m pip install -r local_openai_mlx_provider/requirements.txt
