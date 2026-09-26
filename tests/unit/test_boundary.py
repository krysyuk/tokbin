from __future__ import annotations

import warnings

import pytest

from tokbin._boundary import public_api
from tokbin.codes import Code
from tokbin.errors import ConfigError, InternalError, TokbinWarning


def test_returns_result() -> None:
    @public_api
    def f(x: int) -> int:
        return x + 1

    assert f(1) == 2


def test_foreign_exception_becomes_internal_error_with_chain() -> None:
    @public_api
    def f() -> None:
        raise ValueError("boom")

    with pytest.raises(InternalError) as info:
        f()
    assert isinstance(info.value.__cause__, ValueError)
    assert info.value.where is not None
    assert info.value.where.endswith("f")


def test_tokbin_error_passes_through_and_gets_where() -> None:
    err = ConfigError(Code("TB-C999", "x"), why="w", fix="f")

    @public_api
    def f() -> None:
        raise err

    with pytest.raises(ConfigError) as info:
        f()
    assert info.value is err
    assert info.value.where is not None


def test_existing_where_is_kept() -> None:
    @public_api
    def f() -> None:
        raise ConfigError(Code("TB-C999", "x"), why="w", fix="f", where="inner")

    with pytest.raises(ConfigError) as info:
        f()
    assert info.value.where == "inner"


def test_keyboard_interrupt_is_not_wrapped() -> None:
    @public_api
    def f() -> None:
        raise KeyboardInterrupt

    with pytest.raises(KeyboardInterrupt):
        f()


def test_warning_as_error_is_not_wrapped() -> None:
    @public_api
    def f() -> None:
        warnings.warn("x", TokbinWarning, stacklevel=2)

    with warnings.catch_warnings():
        warnings.simplefilter("error")
        with pytest.raises(TokbinWarning):
            f()


def test_wraps_metadata() -> None:
    @public_api
    def documented() -> None:
        """Doc."""

    assert documented.__name__ == "documented"
    assert documented.__doc__ == "Doc."
