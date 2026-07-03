from collections.abc import Callable
from typing import Any, TypeVar

Algorithm = Callable[..., Any]
AlgorithmT = TypeVar("AlgorithmT", bound=Algorithm)

ALGORITHMS: dict[str, Algorithm] = {}


def register(function: AlgorithmT) -> AlgorithmT:
    """Register a public scaling algorithm under its function name."""
    ALGORITHMS[function.__name__] = function
    return function
