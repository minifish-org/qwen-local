from __future__ import annotations

import time
import uuid
from typing import Any, Optional

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

from .config import get_settings
from .embedding import EmbeddingRuntime
from .llm import LLMRuntime
from .openai_types import (
    ChatCompletionRequest,
    EmbeddingsRequest,
    ModelList,
    ModelObject,
    error_payload,
)

settings = get_settings()
app = FastAPI(title="local-openai-mlx-provider", version="0.1.0")
llm_runtime = LLMRuntime(settings)
embedding_runtime = EmbeddingRuntime(settings)


@app.exception_handler(Exception)
async def generic_exception_handler(_: Request, exc: Exception) -> JSONResponse:
    return JSONResponse(
        status_code=500,
        content=error_payload(str(exc), code="internal_error"),
    )


@app.get("/health")
async def health() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/v1/models")
async def models() -> ModelList:
    return ModelList(
        data=[
            ModelObject(id="local-llm"),
            ModelObject(id="local-embedding"),
        ]
    )


@app.post("/v1/chat/completions")
async def chat_completions(req: ChatCompletionRequest) -> JSONResponse:
    if req.model != "local-llm":
        return _bad_request(f"unsupported chat model: {req.model}", param="model")

    if req.stream:
        return JSONResponse(
            status_code=400,
            content=error_payload(
                "streaming is not supported in v0.1",
                code="streaming_unsupported",
                param="stream",
            ),
        )

    if not req.messages:
        return _bad_request("messages must not be empty", param="messages")

    max_tokens = req.max_tokens or settings.default_max_tokens
    temperature = 0.7 if req.temperature is None else req.temperature
    content = llm_runtime.complete(
        req.messages,
        temperature=temperature,
        max_tokens=max_tokens,
    )

    payload: dict[str, Any] = {
        "id": f"chatcmpl-local-{uuid.uuid4().hex}",
        "object": "chat.completion",
        "created": int(time.time()),
        "model": req.model,
        "choices": [
            {
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": content,
                },
                "finish_reason": "stop",
            }
        ],
        "usage": {
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "total_tokens": 0,
        },
    }
    return JSONResponse(payload)


@app.post("/v1/embeddings")
async def embeddings(req: EmbeddingsRequest) -> JSONResponse:
    if req.model != "local-embedding":
        return _bad_request(f"unsupported embedding model: {req.model}", param="model")

    inputs = [req.input] if isinstance(req.input, str) else req.input
    if not inputs:
        return _bad_request("input must not be empty", param="input")

    vectors = embedding_runtime.embed(inputs)

    payload = {
        "object": "list",
        "model": req.model,
        "data": [
            {
                "object": "embedding",
                "index": i,
                "embedding": vector,
            }
            for i, vector in enumerate(vectors)
        ],
        "usage": {
            "prompt_tokens": 0,
            "total_tokens": 0,
        },
    }
    return JSONResponse(payload)


def _bad_request(message: str, *, param: Optional[str] = None) -> JSONResponse:
    return JSONResponse(
        status_code=400,
        content=error_payload(message, code="bad_request", param=param),
    )


def main() -> None:
    import uvicorn

    uvicorn.run(
        "local_openai_mlx_provider.server:app",
        host=settings.host,
        port=settings.port,
        reload=False,
    )


if __name__ == "__main__":
    main()
