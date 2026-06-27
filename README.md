# qwen-local

OpenAI-compatible local Qwen service for a 16 GB Apple Silicon Mac.

This project is a thin FastAPI adapter around local MLX models. It runs chat,
embeddings, Qwen3 text-to-speech, and Whisper speech-to-text in one process,
on one port, with no external API calls during inference after models are
cached.

## Defaults

- Runtime: MLX / mlx-lm
- Chat model: `mlx-community/Qwen3.5-4B-MLX-4bit`
- Embedding model: `mlx-community/Qwen3-Embedding-0.6B-4bit-DWQ`
- TTS backend: `mlx-audio`
- TTS CustomVoice model: `mlx-community/Qwen3-TTS-12Hz-1.7B-CustomVoice-4bit`
- TTS VoiceDesign model: `mlx-community/Qwen3-TTS-12Hz-1.7B-VoiceDesign-4bit`
- TTS model aliases: `local-tts`, `local-tts-voice-design`
- TTS default voice: `vivian` (`"default"` maps to this voice)
- TTS default language: `auto`
- TTS format: `wav` at 24000 Hz
- ASR backend: `mlx-whisper`
- ASR model: `mlx-community/whisper-large-v3-turbo`
- ASR model alias: `local-asr`
- ASR format: `json`
- Translation backend: `ctranslate2`
- Translation source model: `facebook/nllb-200-3.3B`
- Translation model alias: `local-nllb-200-3.3b-ct2`
- Translation format: JSON via `POST /v1/translations`
- API URL: `http://127.0.0.1:8000/v1`
- Tailscale URL: `http://<mac-tailscale-ip>:8000/v1`
- Chat API model: `local-llm`
- Embedding API model: `local-embedding`
- Context: 8192 tokens
- Default temperature: 0
- Inference lock wait timeout: 60 seconds
- Model keep-alive: 5 minutes after the last request
- Streaming chat: supported
- Thinking: disabled in the Qwen chat template
- Inference worker: single serialized worker for chat, embeddings, TTS, ASR, and translation

## Setup

```sh
./scripts/setup.sh
```

Required Python dependencies are listed in
`local_openai_mlx_provider/requirements.txt`, including `mlx`, `mlx-lm`,
`mlx-embeddings`, `mlx-audio`, `mlx-whisper`, `python-multipart`,
`ctranslate2`, `transformers`, and `sentencepiece`.

ASR also requires `ffmpeg` on the host so Whisper can read common audio formats:

```sh
brew install ffmpeg
```

The first chat, embedding, TTS, ASR, or tokenizer request may download model
files from Hugging Face. After that, inference runs locally from the model
cache.

Translation uses a local CTranslate2 NLLB directory. Set
`TRANSLATION_MODEL_PATH` to an int8 conversion of `facebook/nllb-200-3.3B`.
The 3.3B model can be slow or memory-heavy on a 16 GB MacBook Air; the supported
fallback is an int8 CTranslate2 build or a distilled 1.3B NLLB model with the
same NLLB language codes.

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

The health response includes `loaded_models`, which shows whether chat,
embedding, TTS, or ASR models are currently loaded and when each one is
scheduled to unload.

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

Generate local Qwen3 speech:

```sh
curl http://127.0.0.1:8000/v1/audio/speech \
  -H "Authorization: Bearer local" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "local-tts",
    "input": "你好，这是一个本地 Qwen3 语音合成测试。Hello from local text to speech.",
    "voice": "default",
    "response_format": "wav",
    "speed": 1.0
  }' \
  --output speech.wav
```

Generate local Qwen3 speech with a designed voice:

```sh
curl http://127.0.0.1:8000/v1/audio/speech \
  -H "Authorization: Bearer local" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "local-tts-voice-design",
    "input": "你好，这是一个更自然的本地语音合成测试。",
    "instruct": "自然口语、温暖、不像播音腔，有正常的停顿。",
    "response_format": "wav",
    "speed": 1.0
  }' \
  --output speech-voice-design.wav
```

Transcribe local audio:

```sh
curl http://127.0.0.1:8000/v1/audio/transcriptions \
  -H "Authorization: Bearer local" \
  -F model=local-asr \
  -F file=@speech.wav \
  -F response_format=json
```

Translate local text:

```sh
curl http://127.0.0.1:8000/v1/translations \
  -H "Authorization: Bearer local" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "local-nllb-200-3.3b-ct2",
    "source_language": "jpn_Jpan",
    "target_language": "zho_Hans",
    "text": "こんにちは。今日は良い天気です。"
  }'
```

The translation endpoint supports these common NLLB language codes:
`eng_Latn`, `jpn_Jpan`, `kor_Hang`, `spa_Latn`, `fra_Latn`, `deu_Latn`,
`ita_Latn`, `por_Latn`, `zho_Hans`, and `zho_Hant`.

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
    input="你好，这是一个本地 Qwen3 语音合成测试。",
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
`local-llm`, `local-embedding`, `local-tts`, `local-tts-voice-design`,
`local-asr`, or `local-nllb-200-3.3b-ct2` as the model name.

## Model Lifecycle

Models are loaded on first use and then kept in memory for `MODEL_KEEP_ALIVE`,
which defaults to `5m`. When a model is idle past that duration, the service
releases its Python reference and clears the MLX Metal cache. The server process
continues running, so the next request loads the model again.

`MODEL_KEEP_ALIVE` supports seconds or duration strings such as `0`, `30s`,
`5m`, or `1h`. Any negative value, such as `-1`, keeps models loaded until the
service restarts.

Individual JSON requests can override the default with `keep_alive`:

```sh
curl http://127.0.0.1:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "local-llm",
    "messages": [{"role": "user", "content": "hello"}],
    "max_tokens": 32,
    "keep_alive": 0
  }'
```

Use `keep_alive: 0` to unload after the response, or `keep_alive: -1` to keep
that model loaded.

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
their turn. If a request waits more than `INFERENCE_LOCK_TIMEOUT_SECONDS` for
the worker, the server returns a `503` JSON error instead of queueing forever.

Defaults favor predictable local behavior: temperature is `0`, Qwen thinking is
disabled in the chat template, all local model capabilities share one process
and one port, and launchd writes stdout/stderr logs under `logs/`.

## TTS Limitations

- v0.1 supports `wav` output only.
- Audio speech generation is non-streaming.
- Qwen3-TTS CustomVoice built-in voices are supported through `voice`.
- Qwen3-TTS VoiceDesign is supported through `model=local-tts-voice-design`
  and a required `instruct` voice description.
- Reference audio upload and voice cloning are not exposed by this API.
- Long text chunking for audiobook-style generation is not implemented.
- The default voice is `vivian`; other known Qwen3 voices include `serena`,
  `uncle_fu`, `ryan`, `aiden`, `ono_anna`, `sohee`, `eric`, and `dylan`.
- Language is inferred as `zh` when CJK text is present, otherwise `en`.

## ASR Limitations

- v0.1 supports transcription only, not translation.
- JSON output is the default; plain text output is also supported.
- Realtime ASR and streaming ASR are not implemented.
- Speaker diarization is not implemented.
- Long-audio chunking is not implemented.
- Whisper small is the default target for Mac M4 16 GB.
- Chinese and English should both be tested locally for your target audio.
