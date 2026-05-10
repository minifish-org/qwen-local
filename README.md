# qwen4b-local

Small local Qwen3-4B setup for a 16 GB Apple Silicon Mac.

Baseline:

- Runtime: llama.cpp
- Model: `Qwen/Qwen3-4B-GGUF:Q4_K_M`
- Context: 16384 tokens
- Backend: Metal
- Flash attention: on
- Reasoning: off by default

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

Server URL:

```text
http://127.0.0.1:8000/v1
```

Model ID:

```text
Qwen/Qwen3-4B-GGUF:Q4_K_M
```

## Benchmark

```sh
./scripts/bench.sh
```

For a one-pass 16k prompt benchmark:

```sh
./scripts/bench-16k.sh
```

Initial local result on an M4 16 GB MacBook:

- `pp2048`: about 209 tok/s
- `tg128`: about 25 tok/s
- Short API generation: about 38 tok/s

More details are in `notes/setup.md`.
