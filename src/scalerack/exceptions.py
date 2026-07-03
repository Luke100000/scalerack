class ScalerackError(Exception):
    """Base class for every error raised by scalerack."""


class UnknownAlgorithmError(ScalerackError, KeyError):
    """Raised when an algorithm name is not a known scalerack algorithm."""

    def __str__(self) -> str:
        # KeyError.__str__ wraps the message in repr quotes; show it plainly instead.
        return str(self.args[0]) if self.args else ""


class UnsupportedImageError(ScalerackError, TypeError):
    """Raised when an input image has an unsupported type, shape, mode, or dtype."""


class InvalidFactorError(ScalerackError, ValueError):
    """Raised when a sizing request violates the algorithm's constraints."""
