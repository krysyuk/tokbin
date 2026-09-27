"""What a command needs to produce output: streams, styles and global flags."""

from __future__ import annotations

import sys
from dataclasses import dataclass, field
from typing import TextIO

from tokbin.cli.progress import Progress
from tokbin.cli.render import Style, detect_style, fmt_bytes

__all__ = ["Context", "Outcome", "plural"]


def plural(n: int, word: str) -> str:
    """``1 warning``, ``2 warnings``."""
    return f"{n:,} {word}" if n == 1 else f"{n:,} {word}s"


@dataclass(frozen=True, slots=True, kw_only=True)
class Outcome:
    """The result of a command: exit code, JSON data and warnings shown to the user."""

    exit_code: int
    data: dict[str, object]
    n_warnings: int = 0


@dataclass(slots=True, kw_only=True)
class Context:
    out: TextIO
    err: TextIO
    style: Style
    err_style: Style
    json: bool
    strict: bool
    #: Text lines are suppressed in JSON mode: stdout carries exactly one document.
    _quiet: bool = field(default=False, repr=False)

    @classmethod
    def create(cls, *, json: bool, strict: bool, no_color: bool) -> Context:
        out, err = sys.stdout, sys.stderr
        return cls(
            out=out,
            err=err,
            style=detect_style(out, no_color=no_color),
            err_style=detect_style(err, no_color=no_color),
            json=json,
            strict=strict,
            _quiet=json,
        )

    def line(self, text: str = "") -> None:
        if not self._quiet:
            self.out.write(text + "\n")

    def lines(self, texts: list[str]) -> None:
        for text in texts:
            self.line(text)

    def progress(self) -> Progress | None:
        """A byte progress reporter on stderr; none in JSON mode."""
        if self.json:
            return None
        try:
            tty = self.err.isatty()
        except (AttributeError, ValueError):
            tty = False
        return Progress(self.err, tty=tty, unit=fmt_bytes)
