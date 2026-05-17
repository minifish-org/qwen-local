from __future__ import annotations

import io
import threading
import wave
from typing import Any

from .config import Settings


class LocalTTSProvider:
    def __init__(self, settings: Settings):
        self._settings = settings
        self._backend: KokoroMLXBackend | None = None
        self._lock = threading.Lock()

    def _load(self) -> None:
        if self._backend is not None:
            return

        with self._lock:
            if self._backend is not None:
                return

            if self._settings.tts_backend != "kokoro-mlx":
                raise RuntimeError(
                    f"unsupported TTS backend: {self._settings.tts_backend}"
                )

            self._backend = KokoroMLXBackend(self._settings)

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
