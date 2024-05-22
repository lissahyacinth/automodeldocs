import pathlib
import asyncio

from function_discovery import parse_module


def describe_function(
    function_name: str, source_file: pathlib.Path, module_name: str
) -> str:
    target_module = parse_module(
        source_file, module_name=module_name, starting_path=None
    )
    target_function = target_module.resolve_name(function_name)
    return asyncio.run(describe_function(target_function))
