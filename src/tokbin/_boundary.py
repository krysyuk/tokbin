"""The public API boundary.

Everything that reaches the user either belongs to the :class:`TokbinError` hierarchy
or is wrapped in :class:`InternalError` with the cause chain preserved. Left untouched:

- ``KeyboardInterrupt`` and ``SystemExit``: they are not ``Exception``, an interrupt
  stays an interrupt;
- warnings turned into exceptions by the user (``-W error``): that is the user's
  choice, not a library bug.
"""

from __future__ import annotations

import functools
from collections.abc import Callable
from typing import ParamSpec, TypeVar

from tokbin.errors import InternalError, TokbinError

__all__ = ["public_api"]

P = ParamSpec("P")
R = TypeVar("R")


def public_api(func: Callable[P, R]) -> Callable[P, R]:
    """Wrap a public API function.

    Only suitable for functions that return a finished result. For generators, errors
    raised during iteration are not intercepted by this decorator.
    """
    where = f"{func.__module__}.{func.__qualname__}"

    @functools.wraps(func)
    def wrapper(*args: P.args, **kwargs: P.kwargs) -> R:
        try:
            return func(*args, **kwargs)
        except TokbinError as exc:
            if exc.where is None:
                exc.where = where
            raise
        except Warning:
            raise
        except Exception as exc:
            raise InternalError.wrap(exc, where=where) from exc

    return wrapper
