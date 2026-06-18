from __future__ import annotations

import threading
from typing import Iterable

from .config import Settings
from .lifecycle import release_mlx_memory


class EmbeddingRuntime:
    def __init__(self, settings: Settings):
        self._settings = settings
        self._model = None
        self._processor = None
        self._generate = None
        self._lock = threading.Lock()

    def _load(self) -> None:
        if self._model is not None and self._processor is not None:
            return

        with self._lock:
            if self._model is not None and self._processor is not None:
                return

            try:
                from mlx_embeddings import generate, load
            except ImportError as exc:
                raise RuntimeError(
                    "mlx-embeddings is not installed. Install requirements.txt first."
                ) from exc

            self._generate = generate
            self._model, self._processor = load(self._settings.embedding_model)

    def is_loaded(self) -> bool:
        return self._model is not None and self._processor is not None

    def unload(self) -> None:
        with self._lock:
            self._model = None
            self._processor = None
            self._generate = None
        release_mlx_memory()

    def embed(self, texts: Iterable[str]) -> list[list[float]]:
        self._load()
        assert self._model is not None
        assert self._processor is not None
        assert self._generate is not None

        output = self._generate(
            self._model,
            self._processor,
            list(texts),
            max_length=512,
            padding=True,
            truncation=True,
        )
        vectors = _extract_vectors(output)

        return _to_float_lists(vectors)


def _extract_vectors(output):
    if hasattr(output, "text_embeds") and output.text_embeds is not None:
        return output.text_embeds
    if hasattr(output, "pooler_output") and output.pooler_output is not None:
        return output.pooler_output
    return output


def _to_float_lists(vectors) -> list[list[float]]:
    if hasattr(vectors, "tolist"):
        raw = vectors.tolist()
    else:
        raw = vectors

    if not isinstance(raw, list):
        raw = list(raw)

    if raw and isinstance(raw[0], (int, float)):
        raw = [raw]

    return [[float(v) for v in row] for row in raw]
