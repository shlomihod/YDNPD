import os
import sys
from contextlib import contextmanager, redirect_stdout, redirect_stderr


@contextmanager
def suppress_output():
    with open(os.devnull, "w") as devnull:
        old_stdout = os.dup(1)
        old_stderr = os.dup(2)
        os.dup2(devnull.fileno(), 1)
        os.dup2(devnull.fileno(), 2)
        try:
            yield
        finally:
            os.dup2(old_stdout, 1)
            os.dup2(old_stderr, 2)
            os.close(old_stdout)
            os.close(old_stderr)


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
