"""The public API boundary.

Everything that reaches the user either belongs to the :class:`TokbinError` hierarchy
or is wrapped in :class:`InternalError` with the cause chain preserved. Passed through
unchanged:

- ``KeyboardInterrupt`` and ``SystemExit``: they are not ``Exception``, an interrupt
  stays an interrupt;
- warnings turned into exceptions by the user (``-W error``): that is the user's
  choice, not a library bug;
- ``OSError`` (disk full, permission denied, missing path): a problem of the
  environment with its own errno and path, not a library bug;
- exceptions raised by user code that tokbin calls, such as the document generator
  (marked with :func:`mark_user_error`): they belong to the user;
- ``TypeError`` from calling a public function with wrong arguments: a mistake of the
  caller, reported by Python as usual.
"""

from __future__ import annotations

import contextlib
import functools
from collections.abc import Callable
from typing import ParamSpec, TypeVar

from tokbin.errors import InternalError, TokbinError

__all__ = ["is_user_error", "mark_user_error", "public_api"]

P = ParamSpec("P")
R = TypeVar("R")

_USER_ERROR_ATTR = "__tokbin_user_error__"


def mark_user_error(exc: BaseException) -> BaseException:
    """Mark an exception raised by user code so the boundary lets it through as is."""
    with contextlib.suppress(AttributeError, TypeError):  # exotic exception types
        setattr(exc, _USER_ERROR_ATTR, True)
    return exc


def is_user_error(exc: BaseException) -> bool:
    return bool(getattr(exc, _USER_ERROR_ATTR, False))


def _is_call_error(exc: Exception) -> bool:
    """A ``TypeError`` raised by the call itself: wrong arguments, a caller's mistake.

    Its traceback ends in the wrapper frame, because the function body never ran.
    """
    tb = exc.__traceback__
    return isinstance(exc, TypeError) and tb is not None and tb.tb_next is None


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
        except (Warning, OSError):
            raise
        except Exception as exc:
            if is_user_error(exc) or _is_call_error(exc):
                raise
            raise InternalError.wrap(exc, where=where) from exc

    return wrapper
