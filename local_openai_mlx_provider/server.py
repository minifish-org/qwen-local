from __future__ import annotations

import json
import logging
import os
import threading
import time
import uuid
from collections.abc import Iterator
from tempfile import NamedTemporaryFile
from typing import Any, Optional

from fastapi import FastAPI, File, Form, Request, UploadFile
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse, Response, StreamingResponse

from .asr import LocalASRProvider
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
logger = logging.getLogger("uvicorn.error")
app = FastAPI(title="local-openai-mlx-provider", version="0.1.0")
llm_runtime = LLMRuntime(settings)
embedding_runtime = EmbeddingRuntime(settings)
tts_runtime = LocalTTSProvider(settings)
asr_runtime = LocalASRProvider(settings)
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
        "asr_model": settings.api_asr_model,
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
            ModelObject(id=settings.api_asr_model),
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
        lock_start = _acquire_inference_lock("chat_stream")
        if lock_start is None:
            return _busy_response("chat_stream")

        return StreamingResponse(
            _chat_stream(
                req,
                temperature=temperature,
                max_tokens=max_tokens,
                lock_start=lock_start,
            ),
            media_type="text/event-stream",
        )

    lock_start = _acquire_inference_lock("chat")
    if lock_start is None:
        return _busy_response("chat")

    try:
        content = llm_runtime.complete(
            req.messages,
            temperature=temperature,
            max_tokens=max_tokens,
        )
    finally:
        _release_inference_lock("chat", lock_start)

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
    req: ChatCompletionRequest,
    *,
    temperature: float,
    max_tokens: int,
    lock_start: float,
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
    try:
        yield _sse(first_chunk)

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
    except GeneratorExit:
        logger.warning("chat_stream client disconnected before completion")
        raise
    finally:
        _release_inference_lock("chat_stream", lock_start)


@app.post("/v1/embeddings")
def embeddings(req: EmbeddingsRequest) -> JSONResponse:
    if req.model != settings.api_embedding_model:
        return _bad_request(f"unsupported embedding model: {req.model}", param="model")

    inputs = [req.input] if isinstance(req.input, str) else req.input
    if not inputs:
        return _bad_request("input must not be empty", param="input")

    lock_start = _acquire_inference_lock("embeddings")
    if lock_start is None:
        return _busy_response("embeddings")

    try:
        vectors = embedding_runtime.embed(inputs)
    finally:
        _release_inference_lock("embeddings", lock_start)

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

    lock_start = _acquire_inference_lock("tts")
    if lock_start is None:
        return _busy_response("tts")

    try:
        audio = tts_runtime.speech(
            text=text,
            voice=voice,
            response_format=response_format,
            speed=float(speed),
        )
    finally:
        _release_inference_lock("tts", lock_start)

    return Response(content=audio, media_type="audio/wav")


@app.post("/v1/audio/transcriptions")
def audio_transcriptions(
    file: UploadFile = File(...),
    model: str = Form(...),
    language: Optional[str] = Form(None),
    prompt: Optional[str] = Form(None),
    response_format: Optional[str] = Form(None),
    temperature: Optional[float] = Form(None),
):
    if model not in {settings.api_asr_model, settings.asr_model}:
        return _bad_request(f"unsupported ASR model: {model}", param="model")

    effective_format = response_format or settings.asr_response_format
    if effective_format not in {"json", "text"}:
        return _bad_request(
            f"unsupported response_format: {effective_format}. Supported formats: json, text",
            param="response_format",
        )

    effective_temperature = 0.0 if temperature is None else temperature
    if not isinstance(effective_temperature, (int, float)) or effective_temperature < 0:
        return _bad_request(
            "temperature must be a non-negative number", param="temperature"
        )

    audio = file.file.read()
    if not audio:
        return _bad_request("file must be a non-empty audio file", param="file")

    suffix = _upload_suffix(file.filename)
    temp_path = ""
    try:
        with NamedTemporaryFile(delete=False, suffix=suffix) as temp_file:
            temp_file.write(audio)
            temp_path = temp_file.name

        lock_start = _acquire_inference_lock("asr")
        if lock_start is None:
            return _busy_response("asr")

        try:
            text = asr_runtime.transcribe(
                audio_path=temp_path,
                language=language,
                prompt=prompt,
                response_format=effective_format,
                temperature=float(effective_temperature),
            )
        finally:
            _release_inference_lock("asr", lock_start)
    finally:
        if temp_path:
            try:
                os.unlink(temp_path)
            except FileNotFoundError:
                pass

    if effective_format == "text":
        return Response(content=text, media_type="text/plain")

    return JSONResponse({"text": text})


def _bad_request(message: str, *, param: Optional[str] = None) -> JSONResponse:
    return _error_response(400, message, code="bad_request", param=param)


def _busy_response(operation: str) -> JSONResponse:
    return _error_response(
        503,
        "server is busy running another local inference request",
        code="server_busy",
        param=operation,
    )


def _error_response(
    status_code: int,
    message: str,
    *,
    code: Optional[str] = None,
    param: Optional[str] = None,
) -> JSONResponse:
    return JSONResponse(
        status_code=status_code,
        content=error_payload(message, code=code, param=param),
    )


def _acquire_inference_lock(operation: str) -> Optional[float]:
    timeout = settings.inference_lock_timeout_seconds
    start_wait = time.monotonic()
    acquired = inference_lock.acquire(timeout=timeout)
    waited = time.monotonic() - start_wait

    if not acquired:
        logger.warning(
            "%s timed out waiting for inference lock after %.2fs", operation, waited
        )
        return None

    logger.info("%s acquired inference lock after %.2fs", operation, waited)
    return time.monotonic()


def _release_inference_lock(operation: str, lock_start: float) -> None:
    elapsed = time.monotonic() - lock_start
    inference_lock.release()
    logger.info("%s released inference lock after %.2fs", operation, elapsed)


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
            "asr": settings.api_asr_model,
        },
    }


def _upload_suffix(filename: Optional[str]) -> str:
    if not filename:
        return ".audio"

    _, suffix = os.path.splitext(filename)
    return suffix or ".audio"


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
