from __future__ import annotations

import contextlib
import logging
import os
import warnings
from collections.abc import Callable, Iterator
from dataclasses import dataclass
from typing import Any

import numpy as np
import pytest


@dataclass(frozen=True)
class _GlobalState:
    warning_filters: tuple[Any, ...]
    root_handlers: tuple[logging.Handler, ...]
    root_level: int
    tokbin_handlers: tuple[logging.Handler, ...]
    tokbin_level: int
    environ: dict[str, str]
    cwd: str
    printoptions: dict[str, Any]


def _snapshot() -> _GlobalState:
    root = logging.getLogger()
    tokbin_logger = logging.getLogger("tokbin")
    return _GlobalState(
        warning_filters=tuple(warnings.filters),
        root_handlers=tuple(root.handlers),
        root_level=root.level,
        tokbin_handlers=tuple(tokbin_logger.handlers),
        tokbin_level=tokbin_logger.level,
        environ=dict(os.environ),
        cwd=os.getcwd(),  # noqa: PTH109
        printoptions=np.get_printoptions(),
    )


@contextlib.contextmanager
def _no_side_effects() -> Iterator[None]:
    before = _snapshot()
    yield
    after = _snapshot()
    assert after == before, "the API call changed global process state"


@pytest.fixture
def no_side_effects() -> Callable[[], contextlib.AbstractContextManager[None]]:
    """Context manager: checks that the code inside does not change the environment (spec 9.1).

    The snapshot is taken within the test phase, so handlers that pytest installs
    for each phase do not affect the comparison.
    """
    return _no_side_effects


def _has_tokenizers() -> bool:
    import importlib.util

    return importlib.util.find_spec("tokenizers") is not None


def pytest_collection_modifyitems(config: pytest.Config, items: list[pytest.Item]) -> None:
    if _has_tokenizers():
        return
    skip = pytest.mark.skip(reason="core mode: tokenizers is not installed")
    for item in items:
        if "requires_tokenizers" in item.keywords:
            item.add_marker(skip)
