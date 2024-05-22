from __future__ import annotations

import json
import logging
import pathlib
from contextlib import contextmanager
from dataclasses import dataclass, field
from hashlib import sha256
from typing import Optional

from automodeldocs.definitions import OpenAIInputMessage
from automodeldocs.response.formatted import FormattedOpenAIResponse

logger = logging.getLogger(__name__)
logger.setLevel("WARN")


@contextmanager
def simple_cache():
    cache = SimpleFileCache.from_file()
    try:
        yield cache
    finally:
        cache.to_file()


def hash_dict(item: dict | list[dict] | list[OpenAIInputMessage]) -> int:
    hashed_item = int(sha256(json.dumps(item).encode()).hexdigest(), 16)
    return hashed_item


@dataclass
class SimpleFileCache:
    # InputHash => list[Role, Content]
    _cache: dict[str, list[list[str]]] = field(default_factory=dict)

    @classmethod
    def cache_file(cls) -> pathlib.Path:
        return pathlib.Path.home() / ".llm_cache.json"

    @classmethod
    def from_file(cls) -> SimpleFileCache:
        if cls.cache_file().exists():
            return SimpleFileCache(_cache=json.load(open(cls.cache_file(), "r")))
        return SimpleFileCache(_cache={})

    def to_file(self):
        json.dump(self._cache, open(self.cache_file(), "w"))

    @classmethod
    def hash_message(cls, messages: list[OpenAIInputMessage]) -> str:
        return str(hash_dict(messages))

    def try_retrieve(
        self, messages: list[OpenAIInputMessage]
    ) -> Optional[list[tuple[str, str]]]:
        message_hash = self.hash_message(messages)
        if message_hash in self._cache:
            res = self._cache[message_hash]
            return [(m[0], m[1]) for m in res]
        return None

    def add_item(
        self, messages: list[OpenAIInputMessage], value: list[FormattedOpenAIResponse]
    ) -> None:
        self._cache[self.hash_message(messages)] = [[v.role, v.content] for v in value]
