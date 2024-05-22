from typing import List, TypeVar, Iterable, Tuple, Generator

from termcolor import colored

T = TypeVar("T")


def flatten(lst: List[List[T]]) -> List[T]:
    return [item for sublist in lst for item in sublist]


def take_items(iterable: Iterable[T], n: int) -> List[T]:
    return [item for _, item in zip(range(n), iterable)]


def chunked_generator(
    iterable: Iterable[Tuple[str, T]], n: int
) -> Generator[List[Tuple[str, T]], None, None]:
    chunk: list[tuple[str, T]] = []
    for item in iterable:
        if len(chunk) < n:
            chunk.append(item)
        else:
            yield chunk
            chunk = [item]
    if chunk:
        yield chunk
