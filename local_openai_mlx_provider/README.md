# local-openai-mlx-provider

Small local OpenAI-compatible API server backed by MLX models on Apple Silicon.

It exposes only:

- `GET /v1/models`
- `POST /v1/chat/completions`
- `POST /v1/embeddings`

It does not implement `/v1/responses`, `/v1/rerank`, RAG, vector storage, tool
calling, agents, ASR, TTS, OCR, image generation, or VLM workflows.

## Setup

```sh
cd /Users/yusp/work/qwen-local
python3.12 -m venv .venv-mlx
source .venv-mlx/bin/activate
pip install -r local_openai_mlx_provider/requirements.txt
```

The default model names are configurable. The checked-in defaults are:

```text
LLM_MODEL=mlx-community/Qwen3.5-4B-MLX-4bit
EMBEDDING_MODEL=mlx-community/Qwen3-Embedding-0.6B-4bit-DWQ
HOST=127.0.0.1
PORT=11435
MAX_CONTEXT_TOKENS=8192
```

## Run

```sh
LLM_MODEL=mlx-community/Qwen3.5-4B-MLX-4bit \
EMBEDDING_MODEL=mlx-community/Qwen3-Embedding-0.6B-4bit-DWQ \
HOST=127.0.0.1 \
PORT=11435 \
MAX_CONTEXT_TOKENS=8192 \
python -m local_openai_mlx_provider.server
```

Or run it directly with uvicorn:

```sh
uvicorn local_openai_mlx_provider.server:app --host 127.0.0.1 --port 11435
```

The first request for each capability loads its model lazily. The first run may
download model files; inference itself runs locally after the models are cached.

## OpenAI Client

```python
from openai import OpenAI

client = OpenAI(
    base_url="http://127.0.0.1:11435/v1",
    api_key="local",
)

chat_resp = client.chat.completions.create(
    model="local-llm",
    messages=[{"role": "user", "content": "hello"}],
)
print(chat_resp.choices[0].message.content)

embed_resp = client.embeddings.create(
    model="local-embedding",
    input=["OceanBase capacity sizing"],
)
print(len(embed_resp.data[0].embedding))
```

## Curl

```sh
curl http://127.0.0.1:11435/v1/models
```

```sh
curl http://127.0.0.1:11435/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "local-llm",
    "messages": [
      {"role": "system", "content": "You are a helpful local assistant."},
      {"role": "user", "content": "Explain OceanBase in one sentence."}
    ],
    "temperature": 0.7,
    "max_tokens": 512,
    "stream": false
  }'
```

```sh
curl http://127.0.0.1:11435/v1/embeddings \
  -H "Content-Type: application/json" \
  -d '{
    "model": "local-embedding",
    "input": [
      "OceanBase capacity sizing",
      "OB 集群容量评估"
    ]
  }'
```
