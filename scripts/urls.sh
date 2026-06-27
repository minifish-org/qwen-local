#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
source "$ROOT/config/model.env"

echo "Local:"
echo "  OpenAI API: http://127.0.0.1:${PORT}/v1"

if command -v tailscale >/dev/null 2>&1; then
  TS_IP="$(tailscale ip -4 2>/dev/null | head -n 1 || true)"
  if [[ -n "$TS_IP" ]]; then
    echo "Tailscale:"
    echo "  OpenAI API: http://${TS_IP}:${PORT}/v1"
  fi
fi

echo "Models:"
echo "  Chat:      ${API_LLM_MODEL} (${LLM_MODEL})"
echo "  Embedding: ${API_EMBEDDING_MODEL} (${EMBEDDING_MODEL})"
echo "  TTS:       ${API_TTS_MODEL} (${TTS_BACKEND}, ${TTS_MODEL}, voice=${TTS_DEFAULT_VOICE}, format=${TTS_RESPONSE_FORMAT})"
echo "  TTS VD:    ${API_TTS_VOICE_DESIGN_MODEL} (${TTS_BACKEND}, ${TTS_VOICE_DESIGN_MODEL}, instruct=required, format=${TTS_RESPONSE_FORMAT})"
echo "  ASR:       ${API_ASR_MODEL} (${ASR_BACKEND}, ${ASR_MODEL})"
