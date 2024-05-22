from typing import Optional, Coroutine, Any

from automodeldocs.chat.send_message import chat_completion_request, CacheStatus
from automodeldocs.describe.formatter import FormatResponsePrompt
from automodeldocs.describe.function_report_prompt import DescribeFunction
from automodeldocs.describe.function_scratch_prompt import ScratchFunctionPrompt
from automodeldocs.format_cache import (
    try_load_formatted_description_cache,
    save_formatted_description_to_cache,
)
from automodeldocs.response.formatted import FormattedOpenAIResponse
from automodeldocs.structures import (
    Improvement,
    message_from_system_str,
    message_from_user_str,
)


async def raw_llm_to_string(
    raw_response: Coroutine[Any, Any, CacheStatus[list[FormattedOpenAIResponse]]],
) -> str:
    return (await raw_response).item[-1].content.replace("# Report\n", "")


async def write_description(
    function_source: str,
    function_name: str,
    scratch: str,
    improvement: Optional[Improvement] = None,
) -> str:
    description_prompt = DescribeFunction(
        function_name=function_name,
        scratch=scratch,
        context=improvement.context if improvement is not None else None,
        code_source=function_source,
    )
    written_description = await raw_llm_to_string(
        chat_completion_request(
            [
                message_from_system_str(description_prompt.system_message()),
                message_from_user_str(description_prompt.user_message()),
            ]
            + description_prompt.evaluation_messages(improvement),
            model="gpt-3.5-turbo-16k",
            use_cache=False,
        )
    )
    return written_description


async def format_description(function_name: str, function_description: str) -> str:
    cached_formatted_description = try_load_formatted_description_cache(
        function_name, function_description
    )
    if cached_formatted_description is not None:
        return cached_formatted_description
    prompt = FormatResponsePrompt(function_name, function_description)
    # We cache at this stage regardless, but it's helpful to cache the description in a readable way
    formatted_description = await raw_llm_to_string(
        chat_completion_request(
            [
                message_from_system_str(prompt.system_message()),
                message_from_user_str(prompt.user_message()),
            ],
            model="gpt-4",
            use_cache=True,
        )
    )
    save_formatted_description_to_cache(
        function_name, function_description, formatted_description
    )
    return formatted_description


async def write_scratch(
    function_source: str,
    function_name: str,
    improvement: Optional[Improvement] = None,
) -> str:
    description_prompt = ScratchFunctionPrompt(
        function_name,
        improvement.context if improvement is not None else None,
        function_source,
    )
    scratch = await raw_llm_to_string(
        chat_completion_request(
            [
                message_from_system_str(description_prompt.system_message()),
                message_from_user_str(description_prompt.user_message()),
            ],
            model="gpt-3.5-turbo-16k",
            use_cache=False,
        )
    )
    return scratch
