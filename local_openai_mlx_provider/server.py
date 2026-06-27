from __future__ import annotations

import json
import logging
import math
import os
import subprocess
import threading
import time
import uuid
import wave
from array import array
from collections.abc import Iterator
from tempfile import NamedTemporaryFile
from typing import Any, Optional

from fastapi import FastAPI, File, Form, Request, UploadFile
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse, Response, StreamingResponse

from .asr import LocalASRProvider
from .config import get_settings
from .embedding import EmbeddingRuntime
from .lifecycle import (
    KeepAliveParseError,
    KeepAliveSeconds,
    RuntimeLifecycle,
    parse_keep_alive,
)
from .llm import LLMRuntime
from .openai_types import (
    AudioSpeechRequest,
    ChatMessage,
    ChatCompletionRequest,
    EmbeddingsRequest,
    ModelList,
    ModelObject,
    TranslationRequest,
    error_payload,
)
from .translation import NLLBTranslationRuntime, SUPPORTED_NLLB_LANGUAGE_CODES
from .tts import LocalTTSProvider

settings = get_settings()
logger = logging.getLogger("uvicorn.error")
app = FastAPI(title="local-openai-mlx-provider", version="0.1.0")
llm_runtime = LLMRuntime(settings)
embedding_runtime = EmbeddingRuntime(settings)
tts_runtime = LocalTTSProvider(settings)
asr_runtime = LocalASRProvider(settings)
translation_runtime = NLLBTranslationRuntime(settings)
inference_lock = threading.Lock()
llm_lifecycle = RuntimeLifecycle(
    name="chat",
    is_loaded=llm_runtime.is_loaded,
    unload=llm_runtime.unload,
    inference_lock=inference_lock,
    logger=logger,
)
embedding_lifecycle = RuntimeLifecycle(
    name="embedding",
    is_loaded=embedding_runtime.is_loaded,
    unload=embedding_runtime.unload,
    inference_lock=inference_lock,
    logger=logger,
)
tts_lifecycle = RuntimeLifecycle(
    name="tts",
    is_loaded=tts_runtime.is_loaded,
    unload=tts_runtime.unload,
    inference_lock=inference_lock,
    logger=logger,
)
asr_lifecycle = RuntimeLifecycle(
    name="asr",
    is_loaded=asr_runtime.is_loaded,
    unload=asr_runtime.unload,
    inference_lock=inference_lock,
    logger=logger,
)
translation_lifecycle = RuntimeLifecycle(
    name="translation",
    is_loaded=translation_runtime.is_loaded,
    unload=translation_runtime.unload,
    inference_lock=inference_lock,
    logger=logger,
)
_TRANSLATION_MODEL_ALIASES = {"local-translation"}
_TTS_MODEL_KIND_CUSTOM = "custom_voice"
_TTS_MODEL_KIND_VOICE_DESIGN = "voice_design"
_TTS_RESPONSE_FORMATS = ("wav", "mp3", "opus", "webm", "aac")
_TTS_MEDIA_TYPES = {
    "wav": "audio/wav",
    "mp3": "audio/mpeg",
    "opus": "audio/ogg; codecs=opus",
    "webm": "audio/webm; codecs=opus",
    "aac": "audio/aac",
}
_TTS_TRANSCODE_ARGS = {
    "mp3": ["-codec:a", "libmp3lame", "-b:a", "128k", "-f", "mp3"],
    "opus": ["-codec:a", "libopus", "-b:a", "48k", "-f", "opus"],
    "webm": ["-codec:a", "libopus", "-b:a", "48k", "-f", "webm"],
    "aac": ["-codec:a", "aac", "-b:a", "128k", "-f", "adts"],
}


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
        "tts_voice_design_model": settings.api_tts_voice_design_model,
        "asr_model": settings.api_asr_model,
        "translation_model": settings.api_translation_model,
        "model_keep_alive": settings.model_keep_alive,
        "loaded_models": _loaded_model_state(),
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
            ModelObject(id=settings.api_tts_voice_design_model),
            ModelObject(id=settings.api_asr_model),
            ModelObject(id=settings.api_translation_model),
        ]
    )


@app.post("/v1/chat/completions")
def chat_completions(req: ChatCompletionRequest):
    if req.model != settings.api_llm_model:
        return _bad_request(f"unsupported chat model: {req.model}", param="model")

    if not req.messages:
        return _bad_request("messages must not be empty", param="messages")

    keep_alive = _request_keep_alive(req.keep_alive)
    if isinstance(keep_alive, JSONResponse):
        return keep_alive

    max_tokens = req.max_tokens or settings.default_max_tokens
    temperature = settings.default_temperature if req.temperature is None else req.temperature

    if req.stream:
        lock_start = _acquire_inference_lock("chat_stream")
        if lock_start is None:
            return _busy_response("chat_stream")

        llm_lifecycle.begin_request()
        return StreamingResponse(
            _chat_stream(
                req,
                temperature=temperature,
                max_tokens=max_tokens,
                lock_start=lock_start,
                keep_alive=keep_alive,
            ),
            media_type="text/event-stream",
        )

    lock_start = _acquire_inference_lock("chat")
    if lock_start is None:
        return _busy_response("chat")

    llm_lifecycle.begin_request()
    try:
        content = llm_runtime.complete(
            req.messages,
            temperature=temperature,
            max_tokens=max_tokens,
        )
    finally:
        llm_lifecycle.finish_request(keep_alive)
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
    keep_alive: KeepAliveSeconds,
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
        llm_lifecycle.finish_request(keep_alive)
        _release_inference_lock("chat_stream", lock_start)


@app.post("/v1/embeddings")
def embeddings(req: EmbeddingsRequest) -> JSONResponse:
    if req.model != settings.api_embedding_model:
        return _bad_request(f"unsupported embedding model: {req.model}", param="model")

    inputs = [req.input] if isinstance(req.input, str) else req.input
    if not inputs:
        return _bad_request("input must not be empty", param="input")

    keep_alive = _request_keep_alive(req.keep_alive)
    if isinstance(keep_alive, JSONResponse):
        return keep_alive

    lock_start = _acquire_inference_lock("embeddings")
    if lock_start is None:
        return _busy_response("embeddings")

    embedding_lifecycle.begin_request()
    try:
        vectors = embedding_runtime.embed(inputs)
    finally:
        embedding_lifecycle.finish_request(keep_alive)
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
    resolved_tts_model = _resolve_tts_model(req.model)
    if resolved_tts_model is None:
        return _bad_request(f"unsupported TTS model: {req.model}", param="model")
    model_id, model_kind = resolved_tts_model

    text = req.input.strip() if isinstance(req.input, str) else ""
    if not text:
        return _bad_request("input must be a non-empty string", param="input")

    response_format = (req.response_format or settings.tts_response_format).lower()
    if response_format not in _TTS_RESPONSE_FORMATS:
        supported = ", ".join(_TTS_RESPONSE_FORMATS)
        return _bad_request(
            f"unsupported response_format: {response_format}. Supported formats: {supported}",
            param="response_format",
        )

    speed = 1.0 if req.speed is None else req.speed
    if not isinstance(speed, (int, float)) or speed <= 0:
        return _bad_request("speed must be a positive number", param="speed")

    keep_alive = _request_keep_alive(req.keep_alive)
    if isinstance(keep_alive, JSONResponse):
        return keep_alive

    voice = req.voice or settings.tts_default_voice
    instruct = req.instruct.strip() if req.instruct is not None else None
    if instruct == "":
        instruct = None

    if model_kind == _TTS_MODEL_KIND_VOICE_DESIGN and instruct is None:
        return _bad_request(
            "instruct is required for Qwen3-TTS VoiceDesign", param="instruct"
        )

    lock_start = _acquire_inference_lock("tts")
    if lock_start is None:
        return _busy_response("tts")

    tts_lifecycle.begin_request()
    try:
        try:
            audio = tts_runtime.speech(
                model_id=model_id,
                model_kind=model_kind,
                text=text,
                voice=voice,
                instruct=instruct,
                response_format="wav",
                speed=float(speed),
            )
        except ValueError as exc:
            param = "instruct" if model_kind == _TTS_MODEL_KIND_VOICE_DESIGN else "voice"
            return _bad_request(str(exc), param=param)
    finally:
        tts_lifecycle.finish_request(keep_alive)
        _release_inference_lock("tts", lock_start)

    if response_format != "wav":
        try:
            audio = _transcode_tts_audio(audio, response_format)
        except ValueError as exc:
            return _bad_request(str(exc), param="response_format")

    return Response(content=audio, media_type=_TTS_MEDIA_TYPES[response_format])


@app.post("/v1/audio/transcriptions")
def audio_transcriptions(
    file: UploadFile = File(...),
    model: str = Form(...),
    language: Optional[str] = Form(None),
    prompt: Optional[str] = Form(None),
    response_format: Optional[str] = Form(None),
    temperature: Optional[float] = Form(None),
    keep_alive: Optional[str] = Form(None),
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

    keep_alive_seconds = _request_keep_alive(keep_alive)
    if isinstance(keep_alive_seconds, JSONResponse):
        return keep_alive_seconds

    audio = file.file.read()
    if not audio:
        return _bad_request("file must be a non-empty audio file", param="file")

    suffix = _upload_suffix(file.filename)
    temp_path = ""
    asr_audio_path = ""
    try:
        with NamedTemporaryFile(delete=False, suffix=suffix) as temp_file:
            temp_file.write(audio)
            temp_path = temp_file.name

        try:
            asr_audio_path = _transcode_asr_audio_to_wav(temp_path)
        except ValueError as exc:
            return _bad_request(str(exc), param="file")

        try:
            _validate_asr_audio_has_signal(asr_audio_path)
        except ValueError as exc:
            return _bad_request(str(exc), param="file")

        lock_start = _acquire_inference_lock("asr")
        if lock_start is None:
            return _busy_response("asr")

        asr_lifecycle.begin_request()
        try:
            text = asr_runtime.transcribe(
                audio_path=asr_audio_path,
                language=language,
                prompt=prompt,
                response_format=effective_format,
                temperature=float(effective_temperature),
            )
        finally:
            asr_lifecycle.finish_request(keep_alive_seconds)
            _release_inference_lock("asr", lock_start)
    finally:
        for path in {temp_path, asr_audio_path}:
            if not path:
                continue
            try:
                os.unlink(path)
            except FileNotFoundError:
                pass

    if effective_format == "text":
        return Response(content=text, media_type="text/plain")

    return JSONResponse({"text": text})


@app.post("/v1/translations")
def translations(req: TranslationRequest) -> JSONResponse:
    if req.model not in {
        settings.api_translation_model,
        settings.translation_model,
        *_TRANSLATION_MODEL_ALIASES,
    }:
        return _bad_request(
            f"unsupported translation model: {req.model}", param="model"
        )

    text = req.text.strip() if isinstance(req.text, str) else ""
    if not text:
        return _bad_request("text must be a non-empty string", param="text")

    language_error = _validate_translation_language(
        req.source_language, field="source_language"
    )
    if language_error is not None:
        return language_error

    language_error = _validate_translation_language(
        req.target_language, field="target_language"
    )
    if language_error is not None:
        return language_error

    keep_alive = _request_keep_alive(req.keep_alive)
    if isinstance(keep_alive, JSONResponse):
        return keep_alive

    lock_start = _acquire_inference_lock("translation")
    if lock_start is None:
        return _busy_response("translation")

    translation_lifecycle.begin_request()
    try:
        try:
            translated_text = translation_runtime.translate(
                text=text,
                source_language=req.source_language,
                target_language=req.target_language,
            )
        except RuntimeError as exc:
            if not _is_missing_translation_model_path(exc):
                raise
            translated_text = _translate_with_llm(
                text=text,
                source_language=req.source_language,
                target_language=req.target_language,
            )
        except ValueError as exc:
            return _bad_request(str(exc))
    finally:
        translation_lifecycle.finish_request(keep_alive)
        _release_inference_lock("translation", lock_start)

    return JSONResponse(
        {
            "translated_text": translated_text,
            "source_language": req.source_language,
            "target_language": req.target_language,
            "model": req.model,
        }
    )


def _request_keep_alive(value: Any) -> KeepAliveSeconds | JSONResponse:
    try:
        return parse_keep_alive(value, settings.model_keep_alive)
    except KeepAliveParseError as exc:
        return _bad_request(str(exc), param="keep_alive")


def _loaded_model_state() -> dict[str, Any]:
    return {
        "chat": llm_lifecycle.state(),
        "embedding": embedding_lifecycle.state(),
        "tts": tts_lifecycle.state(),
        "asr": asr_lifecycle.state(),
        "translation": translation_lifecycle.state(),
    }


def _validate_translation_language(value: str, *, field: str) -> JSONResponse | None:
    if value in SUPPORTED_NLLB_LANGUAGE_CODES:
        return None

    supported = ", ".join(sorted(SUPPORTED_NLLB_LANGUAGE_CODES))
    return _bad_request(
        f"unsupported {field}: {value}. Supported NLLB language codes: {supported}",
        param=field,
    )


_TRANSLATION_LANGUAGE_NAMES = {
    "arb_Arab": "Arabic",
    "eng_Latn": "English",
    "hin_Deva": "Hindi",
    "ind_Latn": "Indonesian",
    "jpn_Jpan": "Japanese",
    "kor_Hang": "Korean",
    "rus_Cyrl": "Russian",
    "spa_Latn": "Spanish",
    "fra_Latn": "French",
    "deu_Latn": "German",
    "ita_Latn": "Italian",
    "por_Latn": "Portuguese",
    "tha_Thai": "Thai",
    "vie_Latn": "Vietnamese",
    "zho_Hans": "Simplified Chinese",
    "zho_Hant": "Traditional Chinese",
}


def _is_missing_translation_model_path(exc: RuntimeError) -> bool:
    return "TRANSLATION_MODEL_PATH must point to a local CTranslate2 NLLB model directory" in str(
        exc
    )


def _translate_with_llm(
    *, text: str, source_language: str, target_language: str
) -> str:
    source = _TRANSLATION_LANGUAGE_NAMES.get(source_language, source_language)
    target = _TRANSLATION_LANGUAGE_NAMES.get(target_language, target_language)
    messages = [
        ChatMessage(
            role="system",
            content=(
                "You are a precise translation engine. Translate the user's text "
                "from the source language to the target language. Return only the "
                "translation, without explanations or surrounding quotes."
            ),
        ),
        ChatMessage(
            role="user",
            content=f"Source language: {source}\nTarget language: {target}\nText:\n{text}",
        ),
    ]
    return llm_runtime.complete(
        messages,
        temperature=0.0,
        max_tokens=settings.translation_max_decoding_length,
    )


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
            "tts_voice_design": settings.api_tts_voice_design_model,
            "asr": settings.api_asr_model,
            "translation": settings.api_translation_model,
        },
    }


def _resolve_tts_model(model: str) -> tuple[str, str] | None:
    if model in {settings.api_tts_model, settings.tts_model}:
        return settings.tts_model, _TTS_MODEL_KIND_CUSTOM

    if model in {
        settings.api_tts_voice_design_model,
        settings.tts_voice_design_model,
    }:
        return settings.tts_voice_design_model, _TTS_MODEL_KIND_VOICE_DESIGN

    return None


def _transcode_tts_audio(audio: bytes, response_format: str) -> bytes:
    transcode_args = _TTS_TRANSCODE_ARGS.get(response_format)
    if transcode_args is None:
        raise ValueError(f"unsupported response_format: {response_format}")

    input_path = ""
    output_path = ""
    try:
        with NamedTemporaryFile(delete=False, suffix=".wav") as input_file:
            input_file.write(audio)
            input_path = input_file.name

        with NamedTemporaryFile(delete=False, suffix=f".{response_format}") as output_file:
            output_path = output_file.name

        subprocess.run(
            [
                "ffmpeg",
                "-hide_banner",
                "-loglevel",
                "error",
                "-y",
                "-i",
                input_path,
                "-vn",
                *transcode_args,
                output_path,
            ],
            check=True,
            capture_output=True,
            text=True,
        )

        with open(output_path, "rb") as output_file:
            return output_file.read()
    except FileNotFoundError as exc:
        raise ValueError(
            f"ffmpeg is required to encode TTS response_format: {response_format}"
        ) from exc
    except subprocess.CalledProcessError as exc:
        detail = (exc.stderr or exc.stdout or "").strip()
        if detail:
            detail = detail.splitlines()[-1]
            raise ValueError(
                f"failed to encode TTS audio with ffmpeg: {detail}"
            ) from exc
        raise ValueError("failed to encode TTS audio with ffmpeg") from exc
    finally:
        for path in {input_path, output_path}:
            if not path:
                continue
            try:
                os.unlink(path)
            except FileNotFoundError:
                pass


def _upload_suffix(filename: Optional[str]) -> str:
    if not filename:
        return ".audio"

    _, suffix = os.path.splitext(filename)
    return suffix or ".audio"


def _transcode_asr_audio_to_wav(audio_path: str) -> str:
    with NamedTemporaryFile(delete=False, suffix=".wav") as output_file:
        output_path = output_file.name

    try:
        subprocess.run(
            [
                "ffmpeg",
                "-hide_banner",
                "-loglevel",
                "error",
                "-y",
                "-i",
                audio_path,
                "-vn",
                "-ac",
                "1",
                "-ar",
                "16000",
                "-f",
                "wav",
                output_path,
            ],
            check=True,
            capture_output=True,
            text=True,
        )
    except FileNotFoundError as exc:
        try:
            os.unlink(output_path)
        except FileNotFoundError:
            pass
        raise ValueError("ffmpeg is required to decode uploaded audio for ASR") from exc
    except subprocess.CalledProcessError as exc:
        try:
            os.unlink(output_path)
        except FileNotFoundError:
            pass
        detail = (exc.stderr or exc.stdout or "").strip()
        if detail:
            detail = detail.splitlines()[-1]
            raise ValueError(
                f"failed to decode uploaded audio with ffmpeg: {detail}"
            ) from exc
        raise ValueError("failed to decode uploaded audio with ffmpeg") from exc

    return output_path


def _validate_asr_audio_has_signal(audio_path: str) -> None:
    try:
        with wave.open(audio_path, "rb") as wav:
            sample_width = wav.getsampwidth()
            frame_count = wav.getnframes()
            frame_rate = wav.getframerate()
            channel_count = wav.getnchannels()

            if sample_width != 2:
                raise ValueError("uploaded audio could not be normalized for ASR")
            if frame_rate <= 0 or frame_count <= 0:
                raise ValueError("audio is silent or too quiet for transcription")
            if frame_count / frame_rate < 0.25:
                raise ValueError("audio is too short for transcription")

            peak = 0
            sample_count = 0
            square_sum = 0.0
            while True:
                chunk = wav.readframes(8192)
                if not chunk:
                    break
                samples = array("h")
                samples.frombytes(chunk)
                for index, sample in enumerate(samples):
                    if channel_count > 1 and index % channel_count != 0:
                        continue
                    absolute = abs(sample)
                    peak = max(peak, absolute)
                    square_sum += float(sample) * float(sample)
                    sample_count += 1
    except wave.Error as exc:
        raise ValueError("uploaded audio could not be decoded after normalization") from exc

    if sample_count <= 0:
        raise ValueError("audio is silent or too quiet for transcription")

    rms = math.sqrt(square_sum / sample_count)
    if peak <= 1 or rms < 1:
        raise ValueError("audio is silent or too quiet for transcription")


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
