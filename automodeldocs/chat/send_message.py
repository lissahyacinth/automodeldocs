import logging
import uuid
from typing import Generic, TypeVar

import aiohttp
from dotenv import load_dotenv

from automodeldocs.chat.cache import SimpleFileCache, simple_cache
from automodeldocs.definitions import OpenAIInputMessage
from automodeldocs.response.formatted import FormattedOpenAIResponse

load_dotenv(r"D:\ShareableAI\automodeldocs\.env")

import openai
from tenacity import retry, stop_after_attempt, wait_exponential_jitter

from automodeldocs.chat.model import GPT_MODEL

T = TypeVar("T")

logger = logging.getLogger(__name__)


class CacheStatus(Generic[T]):
    def __init__(self, item: T, cached: bool):
        self.item = item
        self.cached = cached


async def reformat_json(text: str) -> list[FormattedOpenAIResponse]:
    return (
        await chat_completion_request(
            messages=[
                {
                    "role": "system",
                    "content": "You reformat text into valid JSON. You do not add comments or messages.",
                },
                {"role": "user", "content": text},
            ],
            model="gpt-3.5-turbo",
        )
    ).item


@retry(
    wait=wait_exponential_jitter(initial=60, max=2 * 180, exp_base=2),
    stop=stop_after_attempt(3),
    reraise=True,
)
async def chat_completion_request(
    messages: list[OpenAIInputMessage],
    functions: str | None = None,
    function_call: str | None = None,
    model: str = GPT_MODEL,
    use_cache: bool = True,
) -> CacheStatus[list[FormattedOpenAIResponse]]:
    cache: SimpleFileCache
    request_uuid = uuid.uuid4()
    with simple_cache() as cache:
        if use_cache and (
            (cached_response := cache.try_retrieve(messages)) is not None
        ):
            formatted_cached_response = [
                FormattedOpenAIResponse(r[0], r[1]) for r in cached_response
            ]
            return CacheStatus(formatted_cached_response, True)
        assert openai.api_key is not None
        headers = {
            "Content-Type": "application/json",
            "Authorization": "Bearer " + openai.api_key,
        }
        json_data = {"model": model, "messages": messages}
        if functions is not None:
            json_data.update({"functions": functions})
        if function_call is not None:
            json_data.update({"function_call": function_call})
        try:
            logger.info(f"[{request_uuid}] Started Chat Completion Request")
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    "https://api.openai.com/v1/chat/completions",
                    headers=headers,
                    json=json_data,
                ) as response:
                    formatted_response = FormattedOpenAIResponse.from_message(
                        await response.json()
                    )
                    cache.add_item(messages, formatted_response)
                    logging.info(
                        f"Replying with new response - {formatted_response[-1]}"
                    )
                    logger.info(f"[{request_uuid}] Finished Chat Completion Request")
                    return CacheStatus(formatted_response, False)
        except Exception as e:
            print("Unable to generate ChatCompletion response")
            print(f"Exception: {e}")
            raise e
