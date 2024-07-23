import os
import sys
from contextlib import contextmanager


@contextmanager
def suppress_output():
    # Save the original stdout and stderr
    original_stdout = sys.stdout
    original_stderr = sys.stderr

    # Open /dev/null to discard output
    devnull = open(os.devnull, 'w')

    try:
        # Redirect stdout and stderr to /dev/null
        sys.stdout = devnull
        sys.stderr = devnull
        yield
    finally:
        # Restore the original stdout and stderr
        sys.stdout = original_stdout
        sys.stderr = original_stderr
        devnull.close()


def _freeze(d):
    """Recursively freezes a dictionary, converting it and its nested dictionaries to immutable versions."""
    if isinstance(d, dict):
        return tuple({(k, _freeze(v)) for k, v in d.items()})
    elif isinstance(d, list):
        return tuple(_freeze(v) for v in d)
    elif isinstance(d, set):
        return frozenset(_freeze(v) for v in d)
    else:
        return d
