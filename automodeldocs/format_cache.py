import hashlib
import json
import pathlib

from automodeldocs.evaluator.parser import EvaluationResponse
from automodeldocs.structures import Improvement


def try_load_formatted_description_cache(
    function_name: str,
    function_description: str,
) -> str | None:
    fn_cache_location = _fn_cache_location(function_name, function_description)
    if not (fn_cache_location / "description.txt").exists():
        return None
    return open(fn_cache_location / "description.txt").read()


def save_formatted_description_to_cache(
    function_name: str, function_description: str, formatted_description: str
) -> None:
    fn_cache_location = _fn_cache_location(function_name, function_description)
    with open(fn_cache_location / "description.txt", "w") as f:
        f.write(formatted_description)


def _fn_cache_location(
    function_name: str,
    function_description: str,
) -> pathlib.Path:
    cache_dir = pathlib.Path.home() / ".function_description_cache"
    hasher = hashlib.md5()
    hasher.update(function_name.encode())
    hasher.update(function_description.encode())
    function_cache_root = cache_dir / function_name / str(hasher.hexdigest())
    function_cache_root.mkdir(exist_ok=True, parents=True)
    return function_cache_root
