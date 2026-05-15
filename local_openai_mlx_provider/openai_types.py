from __future__ import annotations

from typing import Any, Literal, Optional, Union

from pydantic import BaseModel, Field


class OpenAIError(BaseModel):
    message: str
    type: str = "invalid_request_error"
    param: Optional[str] = None
    code: Optional[str] = None


class ErrorResponse(BaseModel):
    error: OpenAIError


class ChatMessage(BaseModel):
    role: Literal["system", "user", "assistant"]
    content: str


class ChatCompletionRequest(BaseModel):
    model: str
    messages: list[ChatMessage]
    temperature: Optional[float] = 0.7
    max_tokens: Optional[int] = None
    stream: Optional[bool] = False


class EmbeddingsRequest(BaseModel):
    model: str
    input: Union[str, list[str]]


class ModelObject(BaseModel):
    id: str
    object: str = "model"
    owned_by: str = "local"


class ModelList(BaseModel):
    object: str = "list"
    data: list[ModelObject]


def error_payload(
    message: str, *, code: Optional[str] = None, param: Optional[str] = None
) -> dict[str, Any]:
    return ErrorResponse(error=OpenAIError(message=message, code=code, param=param)).model_dump()
