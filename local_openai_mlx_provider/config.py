from __future__ import annotations

import os
from dataclasses import dataclass


@dataclass(frozen=True)
class Settings:
    llm_model: str = "mlx-community/Qwen3.5-4B-MLX-4bit"
    embedding_model: str = "mlx-community/Qwen3-Embedding-0.6B-4bit-DWQ"
    tts_model: str = "mlx-community/Qwen3-TTS-12Hz-1.7B-CustomVoice-4bit"
    tts_voice_design_model: str = (
        "mlx-community/Qwen3-TTS-12Hz-1.7B-VoiceDesign-4bit"
    )
    tts_backend: str = "mlx-audio"
    tts_default_voice: str = "vivian"
    tts_default_language: str = "auto"
    tts_response_format: str = "wav"
    tts_sample_rate: int = 24000
    asr_model: str = "mlx-community/whisper-large-v3-turbo"
    asr_backend: str = "mlx-whisper"
    asr_default_language: str = ""
    asr_response_format: str = "json"
    translation_model: str = "facebook/nllb-200-3.3B"
    translation_backend: str = "ctranslate2"
    translation_model_path: str = ""
    translation_tokenizer_model: str = "facebook/nllb-200-3.3B"
    translation_compute_type: str = "int8"
    translation_device: str = "cpu"
    translation_max_decoding_length: int = 512
    host: str = "0.0.0.0"
    port: int = 8000
    max_context_tokens: int = 8192
    default_max_tokens: int = 512
    default_temperature: float = 0.0
    inference_lock_timeout_seconds: float = 60.0
    model_keep_alive: str = "5m"
    api_llm_model: str = "local-llm"
    api_embedding_model: str = "local-embedding"
    api_tts_model: str = "local-tts"
    api_tts_voice_design_model: str = "local-tts-voice-design"
    api_asr_model: str = "local-asr"
    api_translation_model: str = "local-nllb-200-3.3b-ct2"


def get_settings() -> Settings:
    return Settings(
        llm_model=os.getenv("LLM_MODEL", Settings.llm_model),
        embedding_model=os.getenv("EMBEDDING_MODEL", Settings.embedding_model),
        tts_model=os.getenv("TTS_MODEL", Settings.tts_model),
        tts_voice_design_model=os.getenv(
            "TTS_VOICE_DESIGN_MODEL", Settings.tts_voice_design_model
        ),
        tts_backend=os.getenv("TTS_BACKEND", Settings.tts_backend),
        tts_default_voice=os.getenv("TTS_DEFAULT_VOICE", Settings.tts_default_voice),
        tts_default_language=os.getenv(
            "TTS_DEFAULT_LANGUAGE", Settings.tts_default_language
        ),
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
        translation_model=os.getenv(
            "TRANSLATION_MODEL", Settings.translation_model
        ),
        translation_backend=os.getenv(
            "TRANSLATION_BACKEND", Settings.translation_backend
        ),
        translation_model_path=os.getenv(
            "TRANSLATION_MODEL_PATH", Settings.translation_model_path
        ),
        translation_tokenizer_model=os.getenv(
            "TRANSLATION_TOKENIZER_MODEL", Settings.translation_tokenizer_model
        ),
        translation_compute_type=os.getenv(
            "TRANSLATION_COMPUTE_TYPE", Settings.translation_compute_type
        ),
        translation_device=os.getenv(
            "TRANSLATION_DEVICE", Settings.translation_device
        ),
        translation_max_decoding_length=int(
            os.getenv(
                "TRANSLATION_MAX_DECODING_LENGTH",
                str(Settings.translation_max_decoding_length),
            )
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
        inference_lock_timeout_seconds=float(
            os.getenv(
                "INFERENCE_LOCK_TIMEOUT_SECONDS",
                str(Settings.inference_lock_timeout_seconds),
            )
        ),
        model_keep_alive=os.getenv("MODEL_KEEP_ALIVE", Settings.model_keep_alive),
        api_llm_model=os.getenv("API_LLM_MODEL", Settings.api_llm_model),
        api_embedding_model=os.getenv(
            "API_EMBEDDING_MODEL", Settings.api_embedding_model
        ),
        api_tts_model=os.getenv("API_TTS_MODEL", Settings.api_tts_model),
        api_tts_voice_design_model=os.getenv(
            "API_TTS_VOICE_DESIGN_MODEL", Settings.api_tts_voice_design_model
        ),
        api_asr_model=os.getenv("API_ASR_MODEL", Settings.api_asr_model),
        api_translation_model=os.getenv(
            "API_TRANSLATION_MODEL", Settings.api_translation_model
        ),
    )
