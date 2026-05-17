# qwen-local

OpenAI-compatible local Qwen service for a 16 GB Apple Silicon Mac.

This project is a thin FastAPI adapter around local MLX models. It runs chat,
embeddings, Kokoro text-to-speech, and Whisper speech-to-text in one process,
on one port, with no external API calls during inference after models are
cached.

## Defaults

- Runtime: MLX / mlx-lm
- Chat model: `mlx-community/Qwen3.5-4B-MLX-4bit`
- Embedding model: `mlx-community/Qwen3-Embedding-0.6B-4bit-DWQ`
- TTS backend: `kokoro-mlx`
- TTS model alias: `local-tts`
- TTS default voice: `af_heart` (`"default"` maps to this voice)
- TTS format: `wav` at 24000 Hz
- ASR backend: `mlx-whisper`
- ASR model: `mlx-community/whisper-small-mlx`
- ASR model alias: `local-asr`
- ASR format: `json`
- API URL: `http://127.0.0.1:8000/v1`
- Tailscale URL: `http://<mac-tailscale-ip>:8000/v1`
- Chat API model: `local-llm`
- Embedding API model: `local-embedding`
- Context: 8192 tokens
- Default temperature: 0
- Streaming chat: supported
- Thinking: disabled in the Qwen chat template
- Inference worker: single serialized worker for chat, embeddings, TTS, and ASR

## Setup

```sh
./scripts/setup.sh
```

Required Python dependencies are listed in
`local_openai_mlx_provider/requirements.txt`, including `mlx`, `mlx-lm`,
`mlx-embeddings`, `kokoro-mlx`, `mlx-whisper`, and `python-multipart`.

ASR also requires `ffmpeg` on the host so Whisper can read common audio formats:

```sh
brew install ffmpeg
```

The first chat, embedding, TTS, or ASR request may download model files from
Hugging Face. After that, inference runs locally from the model cache.

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

Test TTS:

```sh
./scripts/test-tts.sh
```

Test ASR with a local audio file:

```sh
./scripts/test-asr.sh
```

Generate local Kokoro speech:

```sh
curl http://127.0.0.1:8000/v1/audio/speech \
  -H "Authorization: Bearer local" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "local-tts",
    "input": "Hello, this is a local Kokoro text to speech test.",
    "voice": "default",
    "response_format": "wav",
    "speed": 1.0
  }' \
  --output speech.wav
```

Transcribe local audio:

```sh
curl http://127.0.0.1:8000/v1/audio/transcriptions \
  -H "Authorization: Bearer local" \
  -F model=local-asr \
  -F file=@speech.wav \
  -F response_format=json
```

Run an application-style OpenAI compatibility check:

```sh
./scripts/test-openai-app.sh
```

For apps that ask for a base API URL without the OpenAI suffix, use:

```text
http://127.0.0.1:8000
```

For OpenAI SDKs, use:

```text
http://127.0.0.1:8000/v1
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

speech_resp = client.audio.speech.create(
    model="local-tts",
    voice="default",
    input="Hello from local Kokoro text to speech.",
    response_format="wav",
    speed=1.0,
)

with open("speech.wav", "wb") as f:
    f.write(speech_resp.read())

with open("speech.wav", "rb") as f:
    transcription = client.audio.transcriptions.create(
        model="local-asr",
        file=f,
        response_format="json",
    )
print(transcription.text)
```

Any OpenAI-compatible client should point at `http://127.0.0.1:8000/v1` and use
`local-llm`, `local-embedding`, `local-tts`, or `local-asr` as the model name.

## Launchd

Install one user-level LaunchAgent so the same local API server starts when you
log in:

```sh
./scripts/install-launchd.sh
```

The LaunchAgent runs `scripts/run-server.sh`, which reads `config/model.env`.
Chat, embeddings, TTS, and ASR are served by that one process on the same port.

Remove it:

```sh
./scripts/uninstall-launchd.sh
```

Logs are written under `logs/`.

## Stability Notes

The service intentionally runs local inference through one shared worker. Chat,
embedding, TTS, and ASR requests are serialized so a 16 GB Mac does not try to
run multiple MLX generations at the same time. Concurrent clients will wait
their turn.

Defaults favor predictable local behavior: temperature is `0`, Qwen thinking is
disabled in the chat template, all local model capabilities share one process
and one port, and launchd writes stdout/stderr logs under `logs/`.

## TTS Limitations

- v0.1 supports `wav` output only.
- Audio speech generation is non-streaming.
- Voice cloning and reference audio upload are not implemented.
- Long text chunking for audiobook-style generation is not implemented.
- Kokoro English quality is the main target.
- Chinese support should be treated as experimental unless tested locally.

## ASR Limitations

- v0.1 supports transcription only, not translation.
- JSON output is the default; plain text output is also supported.
- Realtime ASR and streaming ASR are not implemented.
- Speaker diarization is not implemented.
- Long-audio chunking is not implemented.
- Whisper small is the default target for Mac M4 16 GB.
- Chinese and English should both be tested locally for your target audio.
