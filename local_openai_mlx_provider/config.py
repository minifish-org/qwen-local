from __future__ import annotations

import os
from dataclasses import dataclass


@dataclass(frozen=True)
class Settings:
    llm_model: str = "mlx-community/Qwen3.5-4B-MLX-4bit"
    embedding_model: str = "mlx-community/Qwen3-Embedding-0.6B-4bit-DWQ"
    host: str = "127.0.0.1"
    port: int = 11435
    max_context_tokens: int = 8192
    default_max_tokens: int = 512


def get_settings() -> Settings:
    return Settings(
        llm_model=os.getenv("LLM_MODEL", Settings.llm_model),
        embedding_model=os.getenv("EMBEDDING_MODEL", Settings.embedding_model),
        host=os.getenv("HOST", Settings.host),
        port=int(os.getenv("PORT", str(Settings.port))),
        max_context_tokens=int(
            os.getenv("MAX_CONTEXT_TOKENS", str(Settings.max_context_tokens))
        ),
        default_max_tokens=int(
            os.getenv("DEFAULT_MAX_TOKENS", str(Settings.default_max_tokens))
        ),
    )

