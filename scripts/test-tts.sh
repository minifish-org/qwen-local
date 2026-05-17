#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
source "$ROOT/config/model.env"

BASE_URL="${BASE_URL:-http://127.0.0.1:${PORT}}"
OUT="${OUT:-$ROOT/speech.wav}"

curl -fsS "${BASE_URL}/v1/audio/speech" \
  -H "Authorization: Bearer local" \
  -H "Content-Type: application/json" \
  -d "{
    \"model\": \"${API_TTS_MODEL}\",
    \"input\": \"Hello, this is a local Kokoro text to speech test.\",
    \"voice\": \"default\",
    \"response_format\": \"wav\",
    \"speed\": 1.0
  }" \
  --output "$OUT"

echo "Wrote $OUT"
