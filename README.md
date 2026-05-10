# qwen-local

OpenAI-compatible local Qwen3-4B service for a 16 GB Apple Silicon Mac.

This project is a small wrapper around `llama.cpp`, not a faster replacement
for Ollama. Its value is a simple, transparent local OpenAI-compatible endpoint
for fixed local chat and embedding GGUF models that are easy to use over
Tailscale.

Baseline:

- Runtime: llama.cpp
- Chat model: `Qwen/Qwen3-4B-GGUF:Q4_K_M`
- Embedding model: `Qwen/Qwen3-Embedding-0.6B-GGUF:Q8_0`
- Context: 16384 tokens
- Embedding context: 8192 tokens
- Backend: Metal
- Flash attention: on
- Reasoning: off by default
- Server slots: 1
- Prompt cache RAM cap: 2048 MiB

## Setup

```sh
./scripts/setup-llama.sh
```

The first server or benchmark run downloads the GGUF through llama.cpp's
Hugging Face loader.

## Run

```sh
./scripts/run-server.sh
```

Local server URL on this Mac:

```text
Chat:      http://127.0.0.1:8000/v1
Embedding: http://127.0.0.1:8001/v1
```

From another device on the same Tailscale network, use this Mac's Tailscale IP:

```text
Chat:      http://<mac-tailscale-ip>:8000/v1
Embedding: http://<mac-tailscale-ip>:8001/v1
```

On this Mac, the current Tailscale IPv4 address can usually be checked with:

```sh
tailscale ip -4
```

Model IDs:

```text
Chat:      Qwen/Qwen3-4B-GGUF:Q4_K_M
Embedding: Qwen/Qwen3-Embedding-0.6B-GGUF:Q8_0
```

To print the local and Tailscale URLs:

```sh
./scripts/urls.sh
```

To check that the server is responding:

```sh
./scripts/health.sh
```

To test the embedding endpoint:

```sh
./scripts/test-embedding.sh
```

Embedding request example:

```sh
curl http://127.0.0.1:8001/v1/embeddings \
  -H "Content-Type: application/json" \
  -d '{
    "model": "Qwen/Qwen3-Embedding-0.6B-GGUF:Q8_0",
    "input": "hello local embeddings"
  }'
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

## Benchmark

```sh
./scripts/bench.sh
```

For a one-pass 16k prompt benchmark:

```sh
./scripts/bench-16k.sh
```

For a small tuning matrix:

```sh
./scripts/bench-matrix.sh
```

Best local benchmark result so far on an M4 16 GB MacBook:

- `pp2048`: about 337 tok/s
- `tg128`: about 37 tok/s
- Short API generation: about 38 tok/s

Ollama with the same GGUF is roughly comparable for raw generation speed, so
this setup is mainly about API compatibility and explicit configuration rather
than a major throughput win.

More details are in `notes/setup.md`.
