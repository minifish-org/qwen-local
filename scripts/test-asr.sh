#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
source "$ROOT/config/model.env"

BASE_URL="${BASE_URL:-http://127.0.0.1:${PORT}}"
AUDIO="${AUDIO:-$ROOT/speech.wav}"

if [[ ! -s "$AUDIO" ]]; then
  echo "Missing audio file: $AUDIO" >&2
  echo "Generate one first with: ./scripts/test-tts.sh" >&2
  exit 1
fi

curl -fsS "${BASE_URL}/v1/audio/transcriptions" \
  -H "Authorization: Bearer local" \
  -F "model=${API_ASR_MODEL}" \
  -F "file=@${AUDIO}" \
  -F "response_format=json"
echo
