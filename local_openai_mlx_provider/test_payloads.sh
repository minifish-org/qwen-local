#!/usr/bin/env bash
set -euo pipefail

BASE_URL="${BASE_URL:-http://127.0.0.1:8000}"

curl -fsS "$BASE_URL/v1/models"
echo

curl -fsS "$BASE_URL/v1/chat/completions" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "local-llm",
    "messages": [{"role": "user", "content": "hello"}],
    "max_tokens": 32,
    "stream": false
  }'
echo

curl -fsS "$BASE_URL/v1/chat/completions" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "local-llm",
    "messages": [{"role": "user", "content": "hello"}],
    "max_tokens": 32,
    "stream": true
  }'
echo

curl -fsS "$BASE_URL/v1/embeddings" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "local-embedding",
    "input": ["OceanBase capacity sizing"]
  }'
echo
