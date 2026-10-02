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

__all__ = ["COMMANDS", "GROUPS"]

#: Commands by purpose, in the order of ``tokbin --help``.
GROUPS: tuple[tuple[str, tuple[ModuleType, ...]], ...] = (
    ("Write", (build, migrate)),
    ("Inspect", (ls, info, status, verify)),
    ("Transfer", (pack, unpack)),
    ("Maintenance", (clean, rm, doctor)),
)

COMMANDS: tuple[ModuleType, ...] = tuple(m for _, modules in GROUPS for m in modules)
