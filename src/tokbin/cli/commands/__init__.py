"""CLI commands. Each module has ``NAME``, ``HELP``, ``add_arguments`` and ``run``."""

from __future__ import annotations

from types import ModuleType

from tokbin.cli.commands import clean, doctor, info, ls, status, verify

__all__ = ["COMMANDS"]

COMMANDS: tuple[ModuleType, ...] = (ls, info, status, verify, clean, doctor)
