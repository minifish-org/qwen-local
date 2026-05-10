#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
source "$ROOT/config/model.env"

LLAMA_BASE_URL="${LLAMA_BASE_URL:-http://127.0.0.1:${PORT}}"
OLLAMA_BASE_URL="${OLLAMA_BASE_URL:-http://127.0.0.1:11434}"
OLLAMA_MODEL="${OLLAMA_MODEL:-qwen3-4b-q4-local}"
PROMPT="${PROMPT:-用三段话解释 KV cache 为什么会影响长上下文推理性能。}"
RUNS="${RUNS:-3}"
MAX_TOKENS="${MAX_TOKENS:-256}"

json_escape() {
  python3 -c 'import json,sys; print(json.dumps(sys.stdin.read()))'
}

PROMPT_JSON="$(printf '%s' "$PROMPT" | json_escape)"

echo "# llama.cpp OpenAI-compatible"
for i in $(seq 1 "$RUNS"); do
  curl -s "${LLAMA_BASE_URL}/v1/chat/completions" \
    -H "Content-Type: application/json" \
    -d "{
      \"model\":\"${MODEL_HF}\",
      \"messages\":[{\"role\":\"user\",\"content\":${PROMPT_JSON}}],
      \"temperature\":0,
      \"max_tokens\":${MAX_TOKENS}
    }" | python3 -c '
import json, sys
d=json.load(sys.stdin)
t=d.get("timings", {})
u=d.get("usage", {})
print("run prompt={} output={} prompt_tps={:.2f} output_tps={:.2f}".format(
    u.get("prompt_tokens"),
    u.get("completion_tokens"),
    t.get("prompt_per_second", 0),
    t.get("predicted_per_second", 0),
))
'
done

echo
echo "# Ollama"
for i in $(seq 1 "$RUNS"); do
  curl -s "${OLLAMA_BASE_URL}/api/generate" \
    -H "Content-Type: application/json" \
    -d "{
      \"model\":\"${OLLAMA_MODEL}\",
      \"prompt\":${PROMPT_JSON},
      \"stream\":false,
      \"options\":{\"temperature\":0,\"num_predict\":${MAX_TOKENS},\"num_ctx\":${CTX_SIZE}}
    }" | python3 -c '
import json, sys
d=json.load(sys.stdin)
prompt_n=d.get("prompt_eval_count") or 0
prompt_s=(d.get("prompt_eval_duration") or 0)/1e9
eval_n=d.get("eval_count") or 0
eval_s=(d.get("eval_duration") or 0)/1e9
ptps=prompt_n/prompt_s if prompt_s else 0
otps=eval_n/eval_s if eval_s else 0
print(f"run prompt={prompt_n} output={eval_n} prompt_tps={ptps:.2f} output_tps={otps:.2f}")
'
done
