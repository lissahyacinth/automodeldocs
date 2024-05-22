from __future__ import annotations

from typing import TypedDict


class OpenAIInputMessage(TypedDict):
    role: str
    content: str


class MessageResponse(TypedDict):
    role: str
    content: str


class ChoiceResponse(TypedDict):
    index: int
    message: MessageResponse


class OpenAIRawResponse(TypedDict):
    id: str
    object: str
    created: int
    model: str
    choices: list[ChoiceResponse]
