from __future__ import annotations

import contextlib
import io
import logging
import os
import threading
import wave
from tempfile import TemporaryDirectory
from typing import Any

from .config import Settings
from .lifecycle import (
    mlx_memory_snapshot,
    release_mlx_memory,
    reset_mlx_peak_memory,
)

logger = logging.getLogger("uvicorn.error")


class LocalTTSProvider:
    def __init__(self, settings: Settings):
        self._settings = settings
        self._backend: KokoroMLXBackend | MLXAudioQwenTTSBackend | None = None
        self._backend_model_id: str | None = None
        self._backend_model_kind: str | None = None
        self._lock = threading.Lock()

    def _load(self, *, model_id: str, model_kind: str) -> None:
        if (
            self._backend is not None
            and self._backend_model_id == model_id
            and self._backend_model_kind == model_kind
        ):
            return

        with self._lock:
            if (
                self._backend is not None
                and self._backend_model_id == model_id
                and self._backend_model_kind == model_kind
            ):
                return

            if self._backend is not None:
                self._backend = None
                self._backend_model_id = None
                self._backend_model_kind = None
                release_mlx_memory()

            if model_kind == "kokoro":
                self._backend = KokoroMLXBackend(self._settings, model_id=model_id)
                self._backend_model_id = model_id
                self._backend_model_kind = model_kind
                return

            if model_kind in {"custom_voice", "voice_design"}:
                self._backend = MLXAudioQwenTTSBackend(
                    self._settings, model_id=model_id, model_kind=model_kind
                )
                self._backend_model_id = model_id
                self._backend_model_kind = model_kind
                return

            raise RuntimeError(f"unsupported TTS model kind: {model_kind}")

    def is_loaded(self) -> bool:
        return self._backend is not None

    def unload(self) -> None:
        with self._lock:
            self._backend = None
            self._backend_model_id = None
            self._backend_model_kind = None
        release_mlx_memory()

    def speech(
        self,
        *,
        model_id: str,
        model_kind: str,
        text: str,
        voice: str = "default",
        instruct: str | None = None,
        response_format: str = "wav",
        speed: float = 1.0,
    ) -> bytes:
        self._load(model_id=model_id, model_kind=model_kind)
        assert self._backend is not None
        return self._backend.speech(
            text=text,
            voice=voice,
            instruct=instruct,
            response_format=response_format,
            speed=speed,
        )


class KokoroMLXBackend:
    def __init__(self, settings: Settings, *, model_id: str):
        try:
            from kokoro_mlx import KokoroTTS
        except ImportError as exc:
            raise RuntimeError(
                "kokoro-mlx is not installed. Install requirements.txt first."
            ) from exc

        effective_model_id = None if model_id == "kokoro" else model_id
        if effective_model_id is None:
            self._tts = KokoroTTS.from_pretrained()
        else:
            self._tts = KokoroTTS.from_pretrained(effective_model_id)

        self._sample_rate = settings.tts_sample_rate
        self._default_voice = settings.tts_default_voice

    def speech(
        self,
        *,
        text: str,
        voice: str,
        instruct: str | None,
        response_format: str,
        speed: float,
    ) -> bytes:
        kokoro_voice = self._default_voice if voice == "default" else voice
        result = self._tts.generate(
            text,
            voice=kokoro_voice,
            speed=speed,
            sample_rate=self._sample_rate,
        )
        sample_rate = int(getattr(result, "sample_rate", self._sample_rate))
        audio = getattr(result, "audio", result)

        if response_format == "wav":
            return _wav_bytes(audio, sample_rate)

        raise ValueError(f"unsupported response_format: {response_format}")


class MLXAudioQwenTTSBackend:
    _SUPPORTED_VOICES = {
        "serena",
        "vivian",
        "uncle_fu",
        "ryan",
        "aiden",
        "ono_anna",
        "sohee",
        "eric",
        "dylan",
    }

    def __init__(self, settings: Settings, *, model_id: str, model_kind: str):
        try:
            from mlx_audio.tts.generate import generate_audio
            from mlx_audio.tts.utils import load_model
        except ImportError as exc:
            raise RuntimeError(
                "mlx-audio is not installed. Install requirements.txt first."
            ) from exc

        self._generate_audio = generate_audio
        self._model = load_model(model_id)
        self._model_kind = model_kind
        self._default_voice = settings.tts_default_voice
        self._default_language = settings.tts_default_language

    def speech(
        self,
        *,
        text: str,
        voice: str,
        instruct: str | None,
        response_format: str,
        speed: float,
    ) -> bytes:
        if response_format != "wav":
            raise ValueError(f"unsupported response_format: {response_format}")

        qwen_voice = self._default_voice if voice == "default" else voice
        if self._model_kind == "voice_design":
            if not instruct:
                raise ValueError("instruct is required for Qwen3-TTS VoiceDesign")
            qwen_voice = self._default_voice
        elif qwen_voice not in self._SUPPORTED_VOICES:
            supported = ", ".join(sorted(self._SUPPORTED_VOICES))
            raise ValueError(
                f"unsupported Qwen3-TTS voice: {qwen_voice}. Supported voices: {supported}"
            )

        lang_code = _language_code(text, self._default_language)

        reset_mlx_peak_memory()
        try:
            with TemporaryDirectory(prefix="qwen-local-tts-") as temp_dir:
                with contextlib.redirect_stdout(io.StringIO()):
                    self._generate_audio(
                        text=text,
                        model=self._model,
                        voice=qwen_voice,
                        instruct=instruct,
                        speed=speed,
                        lang_code=lang_code,
                        output_path=temp_dir,
                        file_prefix="speech",
                        audio_format=response_format,
                        join_audio=True,
                        verbose=False,
                        play=False,
                    )
                audio_path = os.path.join(temp_dir, f"speech.{response_format}")
                if not os.path.exists(audio_path):
                    raise RuntimeError(
                        "mlx-audio did not produce an output audio file."
                    )

                with open(audio_path, "rb") as audio_file:
                    return audio_file.read()
        finally:
            release_mlx_memory()
            _log_mlx_memory_after_tts_cleanup()


def _language_code(text: str, default_language: str) -> str:
    if default_language != "auto":
        return default_language

    if any("\u4e00" <= char <= "\u9fff" for char in text):
        return "zh"

    return "en"


def _log_mlx_memory_after_tts_cleanup() -> None:
    snapshot = mlx_memory_snapshot()
    if snapshot is None:
        return

    logger.info(
        "tts mlx memory after request cleanup: active=%.2fMB cache=%.2fMB peak=%.2fMB",
        _bytes_to_mb(snapshot["active_bytes"]),
        _bytes_to_mb(snapshot["cache_bytes"]),
        _bytes_to_mb(snapshot["peak_bytes"]),
    )


def _bytes_to_mb(value: int) -> float:
    return value / 1024 / 1024


def _wav_bytes(audio: Any, sample_rate: int) -> bytes:
    try:
        import numpy as np
    except ImportError as exc:
        raise RuntimeError("numpy is required to encode wav audio.") from exc

    samples = np.asarray(audio, dtype=np.float32)
    if samples.ndim > 1:
        samples = samples.reshape(-1)

    pcm = (np.clip(samples, -1.0, 1.0) * 32767.0).astype("<i2")

    buffer = io.BytesIO()
    with wave.open(buffer, "wb") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(sample_rate)
        wav.writeframes(pcm.tobytes())

    return buffer.getvalue()
