# local-openai-mlx-provider

Small OpenAI-compatible API server backed by MLX models on Apple Silicon.

It exposes:

- `GET /v1/models`
- `POST /v1/chat/completions`
- `POST /v1/embeddings`
- `POST /v1/audio/speech`

Chat and embedding inference share one serialized worker to keep memory usage
predictable on 16 GB Macs. TTS uses the same process and worker when the
service is launched through the repository launchd setup.

It intentionally does not implement `/v1/responses`, `/v1/rerank`, RAG, vector
storage, tool calling, agents, ASR, streaming audio, OCR, image generation, or
VLM workflows.

See the repository README for setup, launchd, and client examples.
