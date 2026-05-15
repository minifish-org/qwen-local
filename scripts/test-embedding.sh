#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
source "$ROOT/config/model.env"

BASE_URL="${BASE_URL:-http://127.0.0.1:${PORT}}"

curl -fsS "${BASE_URL}/v1/embeddings" \
  -H "Content-Type: application/json" \
  -d "{
    \"model\":\"${API_EMBEDDING_MODEL}\",
    \"input\":\"hello local embeddings\"
  }" | python3 -c '
import json, sys
d=json.load(sys.stdin)
item=d["data"][0]
vec=item["embedding"]
print("embedding dimensions:", len(vec))
print("first values:", [round(x, 6) for x in vec[:5]])
'
