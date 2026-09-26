from __future__ import annotations

from collections.abc import Callable
from contextlib import AbstractContextManager

import pytest

import tokbin
from tokbin._version import get_version
from tokbin.cli.main import main


def test_version_comes_from_metadata() -> None:
    assert tokbin.__version__ == get_version()
    assert tokbin.__version__ != "0+unknown"


def test_unknown_attribute() -> None:
    with pytest.raises(AttributeError):
        _ = tokbin.does_not_exist  # type: ignore[attr-defined]


def test_cli_version(
    capsys: pytest.CaptureFixture[str],
    no_side_effects: Callable[[], AbstractContextManager[None]],
) -> None:
    with no_side_effects(), pytest.raises(SystemExit) as info:
        main(["--version"])
    assert info.value.code == 0
    assert capsys.readouterr().out.strip() == f"tokbin {tokbin.__version__}"
