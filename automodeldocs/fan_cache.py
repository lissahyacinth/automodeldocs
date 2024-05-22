import hashlib
import json
import pathlib

from automodeldocs.evaluator.parser import EvaluationResponse
from automodeldocs.structures import Improvement


def try_load_fan_cache(
    function_source: str,
    function_name: str,
    function_docs: str | None,
    improvement: Improvement | None = None,
) -> tuple[str, EvaluationResponse] | None:
    fan_cache_location = _fan_cache_location(
        function_source, function_name, function_docs, improvement
    )
    if not (fan_cache_location / "description.txt").exists():
        return None
    return (
        open(fan_cache_location / "description.txt").read(),
        EvaluationResponse.from_dict(
            json.load(open(fan_cache_location / "evaluation_response.json"))
        ),
    )


def save_fan_cache(
    function_source: str,
    function_name: str,
    function_docs: str | None,
    improvement: Improvement | None,
    description: str,
    evaluation_response: EvaluationResponse,
) -> None:
    fan_cache_location = _fan_cache_location(
        function_source, function_name, function_docs, improvement
    )
    fan_cache_location.mkdir(parents=True, exist_ok=True)
    with open(fan_cache_location / "description.txt", "w") as f:
        f.write(description)
    with open(fan_cache_location / "evaluation_response.json", "w") as f:
        json.dump(evaluation_response.to_dict(), f)


def _fan_cache_location(
    function_source: str,
    function_name: str,
    function_docs: str | None,
    improvement: Improvement | None = None,
) -> pathlib.Path:
    cache_dir = pathlib.Path.home() / ".function_cache"
    hasher = hashlib.md5()
    hasher.update(function_source.encode())
    if function_docs is not None:
        hasher.update(function_docs.encode())
    if improvement is not None:
        hasher.update(json.dumps(improvement.as_dict()).encode("utf-8"))
    function_cache_root = cache_dir / function_name / str(hasher.hexdigest())
    function_cache_root.mkdir(exist_ok=True, parents=True)
    return function_cache_root
