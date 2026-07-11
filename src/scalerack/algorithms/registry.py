from collections.abc import Callable
from dataclasses import dataclass
from functools import wraps
from typing import Concatenate, ParamSpec, Protocol, TypeVar, cast

from scalerack.image_io import ImageInput, as_image_input

P = ParamSpec("P")
ImageT = TypeVar("ImageT")


class AlgorithmFunction(Protocol):
    """A public algorithm preserving the input image representation."""

    def __call__(self, image: ImageT, *args: object, **kwargs: object) -> ImageT: ...


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
) -> Callable[
    [Callable[Concatenate[ImageInput, P], ImageInput]],
    Callable[Concatenate[ImageT, P], ImageT],
]:
    """Register a public scaling algorithm and its declared capabilities.

    Args:
        slow: Whether preview regeneration should be skipped unless explicitly requested.
        alpha: Whether the algorithm supports alpha channels.
        factor: The algorithm's fixed scale factor, if it has one.
    """

    def decorator(
        function: Callable[Concatenate[ImageInput, P], ImageInput],
    ) -> Callable[Concatenate[ImageT, P], ImageT]:
        @wraps(function)
        def public(image: ImageT, *args: P.args, **kwargs: P.kwargs) -> ImageT:
            result = function(as_image_input(image), *args, **kwargs)
            if isinstance(image, ImageInput):
                return cast(ImageT, result)
            return cast(ImageT, result.raw)

        ALGORITHMS[function.__name__] = Algorithm(
            cast(AlgorithmFunction, public), slow, alpha, factor
        )

        return cast(Callable[Concatenate[ImageT, P], ImageT], public)

    return decorator
