from __future__ import annotations

import os
from dataclasses import dataclass


@dataclass(frozen=True)
class Settings:
    llm_model: str = "mlx-community/Qwen3.5-4B-MLX-4bit"
    embedding_model: str = "mlx-community/Qwen3-Embedding-0.6B-4bit-DWQ"
    tts_model: str = "kokoro"
    tts_backend: str = "kokoro-mlx"
    tts_default_voice: str = "af_heart"
    tts_response_format: str = "wav"
    tts_sample_rate: int = 24000
    asr_model: str = "mlx-community/whisper-small-mlx"
    asr_backend: str = "mlx-whisper"
    asr_default_language: str = ""
    asr_response_format: str = "json"
    host: str = "0.0.0.0"
    port: int = 8000
    max_context_tokens: int = 8192
    default_max_tokens: int = 512
    default_temperature: float = 0.0
    api_llm_model: str = "local-llm"
    api_embedding_model: str = "local-embedding"
    api_tts_model: str = "local-tts"
    api_asr_model: str = "local-asr"


def get_settings() -> Settings:
    return Settings(
        llm_model=os.getenv("LLM_MODEL", Settings.llm_model),
        embedding_model=os.getenv("EMBEDDING_MODEL", Settings.embedding_model),
        tts_model=os.getenv("TTS_MODEL", Settings.tts_model),
        tts_backend=os.getenv("TTS_BACKEND", Settings.tts_backend),
        tts_default_voice=os.getenv("TTS_DEFAULT_VOICE", Settings.tts_default_voice),
        tts_response_format=os.getenv(
            "TTS_RESPONSE_FORMAT", Settings.tts_response_format
        ),
        tts_sample_rate=int(os.getenv("TTS_SAMPLE_RATE", str(Settings.tts_sample_rate))),
        asr_model=os.getenv("ASR_MODEL", Settings.asr_model),
        asr_backend=os.getenv("ASR_BACKEND", Settings.asr_backend),
        asr_default_language=os.getenv(
            "ASR_DEFAULT_LANGUAGE", Settings.asr_default_language
        ),
        asr_response_format=os.getenv(
            "ASR_RESPONSE_FORMAT", Settings.asr_response_format
        ),
        host=os.getenv("HOST", Settings.host),
        port=int(os.getenv("PORT", str(Settings.port))),
        max_context_tokens=int(
            os.getenv("MAX_CONTEXT_TOKENS", str(Settings.max_context_tokens))
        ),
        default_max_tokens=int(
            os.getenv("DEFAULT_MAX_TOKENS", str(Settings.default_max_tokens))
        ),
        default_temperature=float(
            os.getenv("DEFAULT_TEMPERATURE", str(Settings.default_temperature))
        ),
        api_llm_model=os.getenv("API_LLM_MODEL", Settings.api_llm_model),
        api_embedding_model=os.getenv(
            "API_EMBEDDING_MODEL", Settings.api_embedding_model
        ),
        api_tts_model=os.getenv("API_TTS_MODEL", Settings.api_tts_model),
        api_asr_model=os.getenv("API_ASR_MODEL", Settings.api_asr_model),
    )
