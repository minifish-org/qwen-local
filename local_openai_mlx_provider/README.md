# local-openai-mlx-provider

Small OpenAI-compatible API server backed by MLX models on Apple Silicon.

It exposes:

- `GET /v1/models`
- `POST /v1/chat/completions`
- `POST /v1/embeddings`
- `POST /v1/audio/speech`
- `POST /v1/audio/transcriptions`

Chat, embedding, TTS, and ASR inference share one serialized worker to keep
memory usage predictable on 16 GB Macs. TTS supports `local-tts`,
`local-tts-quality`, and `local-tts-voice-design`; only one TTS model is kept
loaded at a time. Loaded models are kept warm for `MODEL_KEEP_ALIVE` and then
unloaded when idle.

It intentionally does not implement `/v1/responses`, `/v1/rerank`, RAG, vector
storage, tool calling, agents, streaming audio, OCR, image generation, or VLM
workflows.

See the repository README for setup, launchd, and client examples.
