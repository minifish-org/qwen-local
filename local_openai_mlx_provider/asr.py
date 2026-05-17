from __future__ import annotations

import threading

from .config import Settings


class LocalASRProvider:
    def __init__(self, settings: Settings):
        self._settings = settings
        self._backend: MLXWhisperBackend | None = None
        self._lock = threading.Lock()

    def _load(self) -> None:
        if self._backend is not None:
            return

        with self._lock:
            if self._backend is not None:
                return

            if self._settings.asr_backend != "mlx-whisper":
                raise RuntimeError(
                    f"unsupported ASR backend: {self._settings.asr_backend}"
                )

            self._backend = MLXWhisperBackend(self._settings)

    def transcribe(
        self,
        audio_path: str,
        language: str | None = None,
        prompt: str | None = None,
        response_format: str = "json",
        temperature: float = 0.0,
    ) -> str:
        self._load()
        assert self._backend is not None
        return self._backend.transcribe(
            audio_path=audio_path,
            language=language,
            prompt=prompt,
            response_format=response_format,
            temperature=temperature,
        )


class MLXWhisperBackend:
    def __init__(self, settings: Settings):
        try:
            import mlx_whisper
        except ImportError as exc:
            raise RuntimeError(
                "mlx-whisper is not installed. Install requirements.txt first."
            ) from exc

        self._mlx_whisper = mlx_whisper
        self._model = settings.asr_model
        self._default_language = settings.asr_default_language.strip() or None

    def transcribe(
        self,
        *,
        audio_path: str,
        language: str | None,
        prompt: str | None,
        response_format: str,
        temperature: float,
    ) -> str:
        if response_format not in {"json", "text"}:
            raise ValueError(f"unsupported response_format: {response_format}")

        effective_language = language or self._default_language
        kwargs = {
            "path_or_hf_repo": self._model,
            "temperature": temperature,
            "verbose": None,
            "task": "transcribe",
        }
        if effective_language:
            kwargs["language"] = effective_language
        if prompt:
            kwargs["initial_prompt"] = prompt

        result = self._mlx_whisper.transcribe(audio_path, **kwargs)
        return str(result.get("text", "")).strip()
