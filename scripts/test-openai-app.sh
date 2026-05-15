#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
source "$ROOT/config/model.env"

PYTHON="${PYTHON:-$ROOT/.venv-mlx/bin/python}"
BASE_URL="${BASE_URL:-http://127.0.0.1:${PORT}/v1}"

if [[ ! -x "$PYTHON" ]]; then
  echo "Missing Python environment: $PYTHON" >&2
  echo "Run: ./scripts/setup.sh" >&2
  exit 1
fi

"$PYTHON" - <<'PY'
import os
from openai import OpenAI

base_url = os.environ.get("BASE_URL", "http://127.0.0.1:8000/v1")
chat_model = os.environ.get("API_LLM_MODEL", "local-llm")
embedding_model = os.environ.get("API_EMBEDDING_MODEL", "local-embedding")

client = OpenAI(base_url=base_url, api_key="local")

models = [model.id for model in client.models.list().data]
print("models:", models)
assert chat_model in models
assert embedding_model in models

chat = client.chat.completions.create(
    model=chat_model,
    messages=[{"role": "user", "content": "hello"}],
    max_tokens=32,
)
print("chat:", chat.choices[0].message.content[:120])

stream = client.chat.completions.create(
    model=chat_model,
    messages=[{"role": "user", "content": "用一句话解释 KV cache。"}],
    max_tokens=64,
    stream=True,
)
stream_text = "".join(chunk.choices[0].delta.content or "" for chunk in stream)
print("stream:", stream_text[:120])
assert stream_text

embed = client.embeddings.create(
    model=embedding_model,
    input=["OceanBase capacity sizing"],
)
print("embedding dimensions:", len(embed.data[0].embedding))
assert len(embed.data[0].embedding) == 1024
PY
