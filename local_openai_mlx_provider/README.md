# local-openai-mlx-provider

Small OpenAI-compatible API server backed by MLX models on Apple Silicon.

It exposes:

- `GET /v1/models`
- `POST /v1/chat/completions`
- `POST /v1/embeddings`

Chat and embedding inference share one serialized worker to keep memory usage
predictable on 16 GB Macs.

It intentionally does not implement `/v1/responses`, `/v1/rerank`, RAG, vector
storage, tool calling, agents, ASR, TTS, OCR, image generation, or VLM
workflows.

See the repository README for setup, launchd, and client examples.
