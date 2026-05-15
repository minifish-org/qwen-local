from __future__ import annotations

import threading
from typing import Iterable

from .config import Settings


class EmbeddingRuntime:
    def __init__(self, settings: Settings):
        self._settings = settings
        self._model = None
        self._processor = None
        self._lock = threading.Lock()

    def _load(self) -> None:
        if self._model is not None and self._processor is not None:
            return

        with self._lock:
            if self._model is not None and self._processor is not None:
                return

            try:
                from mlx_embeddings import load
            except ImportError as exc:
                raise RuntimeError(
                    "mlx-embeddings is not installed. Install requirements.txt first."
                ) from exc

            self._model, self._processor = load(self._settings.embedding_model)

    def embed(self, texts: Iterable[str]) -> list[list[float]]:
        self._load()
        assert self._model is not None
        assert self._processor is not None

        items = [{"text": text} for text in texts]

        if hasattr(self._model, "process"):
            vectors = self._model.process(items, processor=self._processor)
        elif hasattr(self._model, "encode"):
            vectors = self._model.encode([item["text"] for item in items])
        else:
            raise RuntimeError(
                "Embedding model does not expose a supported process/encode API."
            )

        return _to_float_lists(vectors)


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

