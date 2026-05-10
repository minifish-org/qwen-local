# qwen4b-local

Local setup for Qwen3-4B Q4_K_M on a 16 GB Apple Silicon Mac.

Chosen baseline:

- Runtime: llama.cpp
- Model: Qwen/Qwen3-4B-GGUF:Q4_K_M
- Context: 16384 tokens
- Backend: Metal
- Flash attention: enabled
- Reasoning/thinking: off by default
- Batch: 512
- UBatch: 256
- Default benchmark: 2048 prompt tokens, 128 generated tokens, 3 repetitions
- Long-context benchmark: 16384 prompt tokens, 128 generated tokens, 1 repetition

Run the server:

```sh
./scripts/run-server.sh
```

Run a benchmark:

```sh
./scripts/bench.sh
```

Run a one-pass 16k benchmark:

```sh
./scripts/bench-16k.sh
```

The first run downloads the GGUF through llama.cpp's Hugging Face loader.

## First Local Results

Hardware detected by llama.cpp:

- GPU: Apple M4
- Recommended Metal working set: 12713 MiB

Loaded model:

- Local cache path: `/Users/yusp/.cache/huggingface/hub/models--Qwen--Qwen3-4B-GGUF/snapshots/bc640142c66e1fdd12af0bd68f40445458f3869b/Qwen3-4B-Q4_K_M.gguf`
- File size reported by llama.cpp: 2.32 GiB
- Parameters: 4.02B
- Model train context: 40960
- Runtime context: 16384

Server memory projection:

- Total projected Metal self memory: 4833 MiB
- Model buffer on Metal: 2375 MiB
- KV buffer at 16k: 2304 MiB
- Compute buffer on Metal: 153 MiB
- Host mapped/buffered memory: about 325-346 MiB

Benchmark, `./scripts/bench.sh`:

| test | result |
| --- | ---: |
| pp2048 | 209.33 +/- 14.14 tok/s |
| tg128 | 25.09 +/- 2.54 tok/s |

Short API smoke test:

- `/v1/models`: OK
- `/health`: OK
- `/v1/chat/completions`: OK
- Short-generation timing: about 37.61 tok/s for a 23-token response

The model may hallucinate self-location facts; local client prompts should not
ask it to infer runtime identity unless that is supplied in the system prompt.
