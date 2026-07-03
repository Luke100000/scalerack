import sys
from collections.abc import Sequence

from scalerack.exceptions import ScalerackError


def main(argv: Sequence[str] | None = None) -> int:
    """Console entry point; returns the process exit code.

    Imports the cyclopts app lazily so a missing ``cli`` extra yields an
    install hint instead of an ImportError traceback.
    """
    try:
        from scalerack.commands import app
    except ImportError:
        sys.stderr.write(
            "error: the scalerack CLI requires the 'cli' extra; "
            "install it with: pip install scalerack[cli]\n"
        )
        return 1
    try:
        app(argv)
    except (ScalerackError, ValueError, OSError) as error:
        sys.stderr.write(f"error: {error}\n")
        return 1
    return 0
