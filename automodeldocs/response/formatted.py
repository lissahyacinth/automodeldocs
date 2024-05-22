from __future__ import annotations

import logging
from dataclasses import dataclass

from automodeldocs.definitions import OpenAIRawResponse

# TODO: Have an error reply for backoff, and allow bigger backoffs.


@dataclass
class FormattedOpenAIResponse:
    role: str
    content: str

    @staticmethod
    def system_message(message: str) -> FormattedOpenAIResponse:
        return FormattedOpenAIResponse(role="system", content=message)

    @staticmethod
    def user_message(message: str) -> FormattedOpenAIResponse:
        return FormattedOpenAIResponse(role="user", content=message)

    @staticmethod
    def from_message(msg: OpenAIRawResponse) -> list[FormattedOpenAIResponse]:
        try:
            choices = msg["choices"]
            messages = [c["message"] for c in choices]
            return [FormattedOpenAIResponse(m["role"], m["content"]) for m in messages]
        except:
            logging.error(f"Could not parse Raw Response {msg=}")
            raise RuntimeError
