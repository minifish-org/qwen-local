# qwen-local

OpenAI-compatible local Qwen service for a 16 GB Apple Silicon Mac.

This project is a thin FastAPI adapter around local MLX models. It runs chat
and embeddings in one process, on one port, with no external API calls during
inference after models are cached.

## Defaults

- Runtime: MLX / mlx-lm
- Chat model: `mlx-community/Qwen3.5-4B-MLX-4bit`
- Embedding model: `mlx-community/Qwen3-Embedding-0.6B-4bit-DWQ`
- API URL: `http://127.0.0.1:8000/v1`
- Tailscale URL: `http://<mac-tailscale-ip>:8000/v1`
- Chat API model: `local-llm`
- Embedding API model: `local-embedding`
- Context: 8192 tokens
- Default temperature: 0
- Streaming chat: supported
- Thinking: disabled in the Qwen chat template

## Setup

```sh
./scripts/setup.sh
```

The first chat or embedding request may download model files from Hugging Face.
After that, inference runs locally from the model cache.

## Run

```sh
./scripts/run-server.sh
```

Check URLs and model names:

```sh
./scripts/urls.sh
```

Check that the server is responding:

```sh
./scripts/health.sh
```

Test embeddings:

```sh
./scripts/test-embedding.sh
```

## OpenAI Client

```python
from openai import OpenAI

client = OpenAI(
    base_url="http://127.0.0.1:8000/v1",
    api_key="local",
)

chat_resp = client.chat.completions.create(
    model="local-llm",
    messages=[{"role": "user", "content": "用一句话解释 KV cache。"}],
    max_tokens=128,
    stream=True,
)

for chunk in chat_resp:
    text = chunk.choices[0].delta.content or ""
    print(text, end="")

embed_resp = client.embeddings.create(
    model="local-embedding",
    input=["OceanBase capacity sizing"],
)
print(len(embed_resp.data[0].embedding))
```

## Launchd

Install a user-level LaunchAgent so the server starts when you log in:

```sh
./scripts/install-launchd.sh
```

Remove it:

```sh
./scripts/uninstall-launchd.sh
```

Logs are written under `logs/`.
