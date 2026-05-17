from __future__ import annotations

import json
import threading
import time
import uuid
from collections.abc import Iterator
from typing import Any, Optional

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse, Response, StreamingResponse

from .config import get_settings
from .embedding import EmbeddingRuntime
from .llm import LLMRuntime
from .openai_types import (
    AudioSpeechRequest,
    ChatCompletionRequest,
    EmbeddingsRequest,
    ModelList,
    ModelObject,
    error_payload,
)
from .tts import LocalTTSProvider

settings = get_settings()
app = FastAPI(title="local-openai-mlx-provider", version="0.1.0")
llm_runtime = LLMRuntime(settings)
embedding_runtime = EmbeddingRuntime(settings)
tts_runtime = LocalTTSProvider(settings)
inference_lock = threading.Lock()


@app.exception_handler(RequestValidationError)
async def validation_exception_handler(
    _: Request, exc: RequestValidationError
) -> JSONResponse:
    first_error = exc.errors()[0] if exc.errors() else {}
    location = first_error.get("loc", [])
    param = str(location[-1]) if location else None
    message = str(first_error.get("msg", "invalid request"))
    return JSONResponse(
        status_code=400,
        content=error_payload(message, code="bad_request", param=param),
    )


@app.exception_handler(Exception)
async def generic_exception_handler(_: Request, exc: Exception) -> JSONResponse:
    return JSONResponse(
        status_code=500,
        content=error_payload(str(exc), code="internal_error"),
    )


@app.get("/health")
def health() -> dict[str, Any]:
    return {
        "status": "ok",
        "runtime": "mlx",
        "single_worker": True,
        "busy": inference_lock.locked(),
        "chat_model": settings.api_llm_model,
        "embedding_model": settings.api_embedding_model,
        "tts_model": settings.api_tts_model,
    }


@app.get("/")
def root() -> dict[str, Any]:
    return _service_info()


@app.get("/v1")
def v1_root() -> dict[str, Any]:
    return _service_info()


@app.get("/v1/models")
def models() -> ModelList:
    return ModelList(
        data=[
            ModelObject(id=settings.api_llm_model),
            ModelObject(id=settings.api_embedding_model),
            ModelObject(id=settings.api_tts_model),
        ]
    )


@app.post("/v1/chat/completions")
def chat_completions(req: ChatCompletionRequest):
    if req.model != settings.api_llm_model:
        return _bad_request(f"unsupported chat model: {req.model}", param="model")

    if not req.messages:
        return _bad_request("messages must not be empty", param="messages")

    max_tokens = req.max_tokens or settings.default_max_tokens
    temperature = settings.default_temperature if req.temperature is None else req.temperature

    if req.stream:
        return StreamingResponse(
            _chat_stream(req, temperature=temperature, max_tokens=max_tokens),
            media_type="text/event-stream",
        )

    with inference_lock:
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


def _chat_stream(
    req: ChatCompletionRequest, *, temperature: float, max_tokens: int
) -> Iterator[str]:
    completion_id = f"chatcmpl-local-{uuid.uuid4().hex}"
    created = int(time.time())

    first_chunk = {
        "id": completion_id,
        "object": "chat.completion.chunk",
        "created": created,
        "model": req.model,
        "choices": [{"index": 0, "delta": {"role": "assistant"}, "finish_reason": None}],
    }
    yield _sse(first_chunk)

    with inference_lock:
        for text in llm_runtime.stream(
            req.messages,
            temperature=temperature,
            max_tokens=max_tokens,
        ):
            payload = {
                "id": completion_id,
                "object": "chat.completion.chunk",
                "created": created,
                "model": req.model,
                "choices": [
                    {
                        "index": 0,
                        "delta": {"content": text},
                        "finish_reason": None,
                    }
                ],
            }
            yield _sse(payload)

    final_chunk = {
        "id": completion_id,
        "object": "chat.completion.chunk",
        "created": created,
        "model": req.model,
        "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}],
    }
    yield _sse(final_chunk)
    yield "data: [DONE]\n\n"


@app.post("/v1/embeddings")
def embeddings(req: EmbeddingsRequest) -> JSONResponse:
    if req.model != settings.api_embedding_model:
        return _bad_request(f"unsupported embedding model: {req.model}", param="model")

    inputs = [req.input] if isinstance(req.input, str) else req.input
    if not inputs:
        return _bad_request("input must not be empty", param="input")

    with inference_lock:
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


@app.post("/v1/audio/speech")
def audio_speech(req: AudioSpeechRequest):
    if req.model not in {settings.api_tts_model, settings.tts_model}:
        return _bad_request(f"unsupported TTS model: {req.model}", param="model")

    text = req.input.strip() if isinstance(req.input, str) else ""
    if not text:
        return _bad_request("input must be a non-empty string", param="input")

    response_format = req.response_format or settings.tts_response_format
    if response_format != "wav":
        return _bad_request(
            f"unsupported response_format: {response_format}. Supported formats: wav",
            param="response_format",
        )

    speed = 1.0 if req.speed is None else req.speed
    if not isinstance(speed, (int, float)) or speed <= 0:
        return _bad_request("speed must be a positive number", param="speed")

    voice = req.voice or settings.tts_default_voice

    with inference_lock:
        audio = tts_runtime.speech(
            text=text,
            voice=voice,
            response_format=response_format,
            speed=float(speed),
        )

    return Response(content=audio, media_type="audio/wav")


def _bad_request(message: str, *, param: Optional[str] = None) -> JSONResponse:
    return JSONResponse(
        status_code=400,
        content=error_payload(message, code="bad_request", param=param),
    )


def _sse(payload: dict[str, Any]) -> str:
    return f"data: {json.dumps(payload, ensure_ascii=False)}\n\n"


def _service_info() -> dict[str, Any]:
    return {
        "name": "qwen-local",
        "runtime": "mlx",
        "openai_base_url": f"http://127.0.0.1:{settings.port}/v1",
        "models": {
            "chat": settings.api_llm_model,
            "embedding": settings.api_embedding_model,
            "tts": settings.api_tts_model,
        },
    }


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
