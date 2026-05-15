from __future__ import annotations

import threading
from collections.abc import Iterator

from .config import Settings
from .openai_types import ChatMessage


class LLMRuntime:
    def __init__(self, settings: Settings):
        self._settings = settings
        self._model = None
        self._tokenizer = None
        self._lock = threading.Lock()

    def _load(self) -> None:
        if self._model is not None and self._tokenizer is not None:
            return

        with self._lock:
            if self._model is not None and self._tokenizer is not None:
                return

            try:
                from mlx_lm import load
            except ImportError as exc:
                raise RuntimeError(
                    "mlx-lm is not installed. Install requirements.txt first."
                ) from exc

            self._model, self._tokenizer = load(self._settings.llm_model)

    def complete(
        self,
        messages: list[ChatMessage],
        *,
        temperature: float,
        max_tokens: int,
    ) -> str:
        self._load()
        assert self._model is not None
        assert self._tokenizer is not None

        prompt = self._render_prompt(messages)

        try:
            from mlx_lm import stream_generate
            from mlx_lm.sample_utils import make_sampler
        except ImportError as exc:
            raise RuntimeError(
                "mlx-lm is not installed. Install requirements.txt first."
            ) from exc

        sampler = make_sampler(temp=temperature)
        chunks = stream_generate(
            self._model,
            self._tokenizer,
            prompt,
            max_tokens=max_tokens,
            sampler=sampler,
        )
        return "".join(getattr(chunk, "text", str(chunk)) for chunk in chunks).strip()

    def stream(
        self,
        messages: list[ChatMessage],
        *,
        temperature: float,
        max_tokens: int,
    ) -> Iterator[str]:
        self._load()
        assert self._model is not None
        assert self._tokenizer is not None

        prompt = self._render_prompt(messages)

        try:
            from mlx_lm import stream_generate
            from mlx_lm.sample_utils import make_sampler
        except ImportError as exc:
            raise RuntimeError(
                "mlx-lm is not installed. Install requirements.txt first."
            ) from exc

        sampler = make_sampler(temp=temperature)
        chunks = stream_generate(
            self._model,
            self._tokenizer,
            prompt,
            max_tokens=max_tokens,
            sampler=sampler,
        )
        for chunk in chunks:
            text = getattr(chunk, "text", str(chunk))
            if text:
                yield text

    def _render_prompt(self, messages: list[ChatMessage]) -> str:
        tokenizer = self._tokenizer
        rendered = [{"role": m.role, "content": m.content} for m in messages]

        apply_chat_template = getattr(tokenizer, "apply_chat_template", None)
        if callable(apply_chat_template):
            try:
                return str(
                    apply_chat_template(
                        rendered,
                        tokenize=False,
                        add_generation_prompt=True,
                        enable_thinking=False,
                    )
                )
            except TypeError:
                # Some tokenizers expose a narrower chat-template signature.
                return str(apply_chat_template(rendered, tokenize=False))

        parts: list[str] = []
        for message in messages:
            parts.append(f"{message.role}: {message.content}")
        parts.append("assistant:")
        return "\n".join(parts)
