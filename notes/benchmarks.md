# Benchmarks

Generated: 2026-05-10 15:00:13 +08

Model: `Qwen/Qwen3-4B-GGUF:Q4_K_M`
Prompt tokens: `2048`
Generation tokens: `128`
Repetitions: `2`

## Matrix

### b512-ub256-f16kv

```text
| model                          |       size |     params | backend    | threads | n_batch | n_ubatch | fa |            test |                  t/s |
| ------------------------------ | ---------: | ---------: | ---------- | ------: | ------: | -------: | -: | --------------: | -------------------: |
| qwen3 4B Q4_K - Medium         |   2.32 GiB |     4.02 B | MTL,BLAS   |       4 |     512 |      256 |  1 |          pp2048 |        337.33 ± 2.54 |
| qwen3 4B Q4_K - Medium         |   2.32 GiB |     4.02 B | MTL,BLAS   |       4 |     512 |      256 |  1 |           tg128 |         37.18 ± 0.23 |

build: 1e5ad35 (1)
```

## Ollama Comparison

An Ollama model was created from the same GGUF using `Modelfile.ollama`:

```text
qwen3-4b-q4-local
```

Short raw generation, 128 output tokens:

| service | output speed |
| --- | ---: |
| llama.cpp server | 35.95 / 36.00 / 36.39 tok/s |
| Ollama | 36.64 / 36.63 / 36.55 tok/s |

Long raw prompt, about 10k prompt tokens and 128 output tokens:

| service | prompt speed | output speed |
| --- | ---: | ---: |
| llama.cpp server, first run | 209.03 tok/s | 20.03 tok/s |
| llama.cpp server, second run | 19.03 tok/s | 16.03 tok/s |
| Ollama, first run | 145.90 tok/s | 17.36 tok/s |
| Ollama, second run | cache-hit/measurement-skewed | 16.65 tok/s |

Conclusion: raw generation speed is broadly comparable. This setup is useful
for OpenAI-compatible API access, explicit parameters, Tailscale use, and a
fixed Qwen3-4B GGUF baseline, not because it clearly outperforms Ollama.

### b1024-ub256-f16kv

```text
| model                          |       size |     params | backend    | threads | n_batch | n_ubatch | fa |            test |                  t/s |
| ------------------------------ | ---------: | ---------: | ---------- | ------: | ------: | -------: | -: | --------------: | -------------------: |
| qwen3 4B Q4_K - Medium         |   2.32 GiB |     4.02 B | MTL,BLAS   |       4 |    1024 |      256 |  1 |          pp2048 |        317.66 ± 4.78 |
| qwen3 4B Q4_K - Medium         |   2.32 GiB |     4.02 B | MTL,BLAS   |       4 |    1024 |      256 |  1 |           tg128 |         36.85 ± 0.05 |

build: 1e5ad35 (1)
```

### b1024-ub512-f16kv

```text
| model                          |       size |     params | backend    | threads | n_batch | fa |            test |                  t/s |
| ------------------------------ | ---------: | ---------: | ---------- | ------: | ------: | -: | --------------: | -------------------: |
| qwen3 4B Q4_K - Medium         |   2.32 GiB |     4.02 B | MTL,BLAS   |       4 |    1024 |  1 |          pp2048 |        296.77 ± 1.73 |
| qwen3 4B Q4_K - Medium         |   2.32 GiB |     4.02 B | MTL,BLAS   |       4 |    1024 |  1 |           tg128 |         35.45 ± 1.75 |

build: 1e5ad35 (1)
```

### b512-ub256-q8kv

```text
| model                          |       size |     params | backend    | threads | n_batch | n_ubatch | type_k | type_v | fa |            test |                  t/s |
| ------------------------------ | ---------: | ---------: | ---------- | ------: | ------: | -------: | -----: | -----: | -: | --------------: | -------------------: |
| qwen3 4B Q4_K - Medium         |   2.32 GiB |     4.02 B | MTL,BLAS   |       4 |     512 |      256 |   q8_0 |   q8_0 |  1 |          pp2048 |       229.33 ± 13.50 |
| qwen3 4B Q4_K - Medium         |   2.32 GiB |     4.02 B | MTL,BLAS   |       4 |     512 |      256 |   q8_0 |   q8_0 |  1 |           tg128 |         23.57 ± 1.54 |

build: 1e5ad35 (1)
```

### b1024-ub256-q8kv

```text
| model                          |       size |     params | backend    | threads | n_batch | n_ubatch | type_k | type_v | fa |            test |                  t/s |
| ------------------------------ | ---------: | ---------: | ---------- | ------: | ------: | -------: | -----: | -----: | -: | --------------: | -------------------: |
| qwen3 4B Q4_K - Medium         |   2.32 GiB |     4.02 B | MTL,BLAS   |       4 |    1024 |      256 |   q8_0 |   q8_0 |  1 |          pp2048 |        163.24 ± 1.91 |
| qwen3 4B Q4_K - Medium         |   2.32 GiB |     4.02 B | MTL,BLAS   |       4 |    1024 |      256 |   q8_0 |   q8_0 |  1 |           tg128 |         18.53 ± 0.78 |

build: 1e5ad35 (1)
```
