from collections.abc import Callable
from dataclasses import dataclass

from scalerack.image_io import ImageInput

AlgorithmFunction = Callable[..., ImageInput]


@dataclass(frozen=True)
class Algorithm:
    """A registered scaling function and its declared capabilities."""

    function: AlgorithmFunction
    slow: bool = False
    alpha: bool = True
    factor: float | None = None


ALGORITHMS: dict[str, Algorithm] = {}


def register(
    *,
    slow: bool = False,
    alpha: bool = True,
    factor: float | None = None,
) -> Callable[[AlgorithmFunction], AlgorithmFunction]:
    """Register a public scaling algorithm and its declared capabilities.

    Args:
        slow: Whether preview regeneration should be skipped unless explicitly requested.
        alpha: Whether the algorithm supports alpha channels.
        factor: The algorithm's fixed scale factor, if it has one.
    """

    def decorator(function: AlgorithmFunction) -> AlgorithmFunction:
        ALGORITHMS[function.__name__] = Algorithm(function, slow, alpha, factor)
        return function

    return decorator
