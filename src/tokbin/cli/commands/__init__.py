"""CLI commands. Each module has ``NAME``, ``HELP``, ``add_arguments`` and ``run``."""

from __future__ import annotations

from types import ModuleType

from tokbin.cli.commands import (
    build,
    clean,
    doctor,
    info,
    ls,
    migrate,
    pack,
    rm,
    status,
    unpack,
    verify,
)

__all__ = ["COMMANDS"]

COMMANDS: tuple[ModuleType, ...] = (
    build,
    ls,
    info,
    status,
    verify,
    pack,
    unpack,
    migrate,
    clean,
    rm,
    doctor,
)
