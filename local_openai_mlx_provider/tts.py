from __future__ import annotations

import contextlib
import io
import os
import threading
import wave
from tempfile import TemporaryDirectory
from typing import Any

from .config import Settings


class LocalTTSProvider:
    def __init__(self, settings: Settings):
        self._settings = settings
        self._backend: KokoroMLXBackend | MLXAudioQwenTTSBackend | None = None
        self._lock = threading.Lock()

    def _load(self) -> None:
        if self._backend is not None:
            return

        with self._lock:
            if self._backend is not None:
                return

            if self._settings.tts_backend == "kokoro-mlx":
                self._backend = KokoroMLXBackend(self._settings)
                return

            if self._settings.tts_backend == "mlx-audio":
                self._backend = MLXAudioQwenTTSBackend(self._settings)
                return

            raise RuntimeError(
                f"unsupported TTS backend: {self._settings.tts_backend}"
            )

    def speech(
        self,
        text: str,
        voice: str = "default",
        response_format: str = "wav",
        speed: float = 1.0,
    ) -> bytes:
        self._load()
        assert self._backend is not None
        return self._backend.speech(
            text=text,
            voice=voice,
            response_format=response_format,
            speed=speed,
        )


class KokoroMLXBackend:
    def __init__(self, settings: Settings):
        try:
            from kokoro_mlx import KokoroTTS
        except ImportError as exc:
            raise RuntimeError(
                "kokoro-mlx is not installed. Install requirements.txt first."
            ) from exc

        model_id = None if settings.tts_model == "kokoro" else settings.tts_model
        if model_id is None:
            self._tts = KokoroTTS.from_pretrained()
        else:
            self._tts = KokoroTTS.from_pretrained(model_id)

        self._sample_rate = settings.tts_sample_rate
        self._default_voice = settings.tts_default_voice

    def speech(
        self,
        *,
        text: str,
        voice: str,
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

    def __init__(self, settings: Settings):
        try:
            from mlx_audio.tts.generate import generate_audio
            from mlx_audio.tts.utils import load_model
        except ImportError as exc:
            raise RuntimeError(
                "mlx-audio is not installed. Install requirements.txt first."
            ) from exc

        self._generate_audio = generate_audio
        self._model = load_model(settings.tts_model)
        self._default_voice = settings.tts_default_voice
        self._default_language = settings.tts_default_language

    def speech(
        self,
        *,
        text: str,
        voice: str,
        response_format: str,
        speed: float,
    ) -> bytes:
        if response_format != "wav":
            raise ValueError(f"unsupported response_format: {response_format}")

        qwen_voice = self._default_voice if voice == "default" else voice
        if qwen_voice not in self._SUPPORTED_VOICES:
            supported = ", ".join(sorted(self._SUPPORTED_VOICES))
            raise ValueError(
                f"unsupported Qwen3-TTS voice: {qwen_voice}. Supported voices: {supported}"
            )

        lang_code = _language_code(text, self._default_language)

        with TemporaryDirectory(prefix="qwen-local-tts-") as temp_dir:
            with contextlib.redirect_stdout(io.StringIO()):
                self._generate_audio(
                    text=text,
                    model=self._model,
                    voice=qwen_voice,
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
                raise RuntimeError("mlx-audio did not produce an output audio file.")

            with open(audio_path, "rb") as audio_file:
                return audio_file.read()


def _language_code(text: str, default_language: str) -> str:
    if default_language != "auto":
        return default_language

    if any("\u4e00" <= char <= "\u9fff" for char in text):
        return "zh"

    return "en"


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
