from collections.abc import Callable
from typing import TypeVar

from scalerack.image_io import ImageInput

Algorithm = Callable[..., ImageInput]
AlgorithmT = TypeVar("AlgorithmT", bound=Algorithm)

ALGORITHMS: dict[str, Algorithm] = {}


def register(function: AlgorithmT) -> AlgorithmT:
    """Register a public scaling algorithm under its function name."""
    ALGORITHMS[function.__name__] = function
    return function
