"""Terminal output without dependencies (spec 14.1).

- Color only on a terminal, never with ``NO_COLOR``, ``--no-color`` or ``TERM=dumb``;
  on Windows the console VT mode is enabled first, and without it there is no color.
- Every status has a symbol; color is never the only carrier of meaning.
- ``✔ ! ✖`` become ``[ok] [!] [x]`` when the output encoding is not UTF-8.
- The accent (commands and flags in help) is a light flesh tone: 24-bit where the
  terminal announces it, the nearest xterm-256 color elsewhere.
"""

from __future__ import annotations

import codecs
import importlib
import os
from collections.abc import Callable
from dataclasses import dataclass
from typing import Literal, TextIO

__all__ = [
    "Cell",
    "Mark",
    "Style",
    "detect_style",
    "fmt_bytes",
    "fmt_count",
    "fmt_int",
    "table",
]

Mark = Literal["ok", "warn", "error", "skip"]

_RESET = "\x1b[0m"
_CODES = {"green": "32", "yellow": "33", "red": "31", "dim": "2", "bold": "1"}
_ACCENT_TRUECOLOR = "1;38;2;242;196;170"  # bold #F2C4AA
_ACCENT_256 = "1;38;5;223"  # bold #FFD7AF, the nearest xterm-256 color

_UNICODE_MARKS: dict[Mark, str] = {"ok": "✔", "warn": "!", "error": "✖", "skip": "\u2013"}
_ASCII_MARKS: dict[Mark, str] = {"ok": "[ok]", "warn": "[!]", "error": "[x]", "skip": "[-]"}
_MARK_COLOR: dict[Mark, str] = {"ok": "green", "warn": "yellow", "error": "red", "skip": "dim"}


@dataclass(frozen=True, slots=True)
class Style:
    """How to decorate text for one output stream."""

    color: bool
    unicode: bool
    #: The terminal renders 24-bit colors; otherwise the accent uses the 256-color table.
    truecolor: bool = False

    def paint(self, text: str, color: str) -> str:
        if not self.color or not text:
            return text
        if color == "accent":
            code = _ACCENT_TRUECOLOR if self.truecolor else _ACCENT_256
        else:
            code = _CODES[color]
        return f"\x1b[{code}m{text}{_RESET}"

    def accent(self, text: str) -> str:
        """Commands and flags the user can type."""
        return self.paint(text, "accent")

    def green(self, text: str) -> str:
        return self.paint(text, "green")

    def yellow(self, text: str) -> str:
        return self.paint(text, "yellow")

    def red(self, text: str) -> str:
        return self.paint(text, "red")

    def dim(self, text: str) -> str:
        return self.paint(text, "dim")

    def bold(self, text: str) -> str:
        return self.paint(text, "bold")

    def symbol(self, kind: Mark) -> str:
        """The plain status symbol."""
        return (_UNICODE_MARKS if self.unicode else _ASCII_MARKS)[kind]

    def mark(self, kind: Mark) -> str:
        """The status symbol, colored."""
        return self.paint(self.symbol(kind), _MARK_COLOR[kind])

    def mark_cell(self, kind: Mark) -> Cell:
        return Cell(self.symbol(kind), paint=lambda t: self.paint(t, _MARK_COLOR[kind]))

    @property
    def dot(self) -> str:
        """Separator between facts on one line."""
        return "·" if self.unicode else "|"

    @property
    def branch(self) -> str:
        """Prefix of a detail line under a row."""
        return "└" if self.unicode else "`-"

    @property
    def arrow(self) -> str:
        return "→" if self.unicode else "->"

    def join(self, *parts: str) -> str:
        return f" {self.dot} ".join(p for p in parts if p)


def _is_utf8(stream: TextIO) -> bool:
    encoding = getattr(stream, "encoding", None)
    if not encoding:
        return False
    try:
        return codecs.lookup(encoding).name == "utf-8"
    except LookupError:
        return False


def _enable_windows_vt(stream: TextIO) -> bool:
    """Turn on ANSI escape processing of a Windows console; ``False`` if impossible."""
    if os.name != "nt":
        return True
    try:
        import ctypes

        # Windows-only modules, looked up dynamically so type checking works anywhere.
        msvcrt = importlib.import_module("msvcrt")
        kernel32 = ctypes.WinDLL("kernel32") if hasattr(ctypes, "WinDLL") else None
        if kernel32 is None:
            return False
        enable_vt = 0x0004  # ENABLE_VIRTUAL_TERMINAL_PROCESSING
        handle = msvcrt.get_osfhandle(stream.fileno())
        mode = ctypes.c_uint32()
        if not kernel32.GetConsoleMode(handle, ctypes.byref(mode)):
            return False
        if mode.value & enable_vt:
            return True
        return bool(kernel32.SetConsoleMode(handle, mode.value | enable_vt))
    except (AttributeError, ImportError, OSError, ValueError):
        return False


def _isatty(stream: TextIO) -> bool:
    try:
        return stream.isatty()
    except (AttributeError, ValueError):  # closed or exotic streams
        return False


def detect_style(stream: TextIO, *, no_color: bool) -> Style:
    """Choose decorations for ``stream`` from the environment and the flags."""
    color = (
        not no_color
        and not os.environ.get("NO_COLOR")
        and os.environ.get("TERM") != "dumb"
        and _isatty(stream)
        and _enable_windows_vt(stream)
    )
    truecolor = os.environ.get("COLORTERM", "").lower() in {"truecolor", "24bit"} or bool(
        os.environ.get("WT_SESSION")  # Windows Terminal
    )
    return Style(color=bool(color), unicode=_is_utf8(stream), truecolor=bool(color) and truecolor)


# --- numbers --------------------------------------------------------------------------


def fmt_int(n: int) -> str:
    """``847392014`` -> ``847,392,014``."""
    return f"{n:,}"


def fmt_count(n: int) -> str:
    """Short counts: ``412.0M``, ``1.20M``, ``890.4K``, ``57``."""
    if n < 1000:
        return str(n)
    value = float(n)
    for suffix in ("K", "M", "B", "T"):
        value /= 1000
        if value < 1000 or suffix == "T":
            # Rounding may carry into the next unit: 999_950 is 1.00M, not 1000.0K.
            if round(value, 1) >= 1000 and suffix != "T":
                continue
            return f"{value:.2f}{suffix}" if value < 10 else f"{value:.1f}{suffix}"
    raise AssertionError("unreachable")  # pragma: no cover


def fmt_bytes(n: int) -> str:
    """Binary units: ``720 B``, ``512.0 MiB``, ``1.5 GiB``."""
    if n < 1024:
        return f"{n} B"
    value = float(n)
    for unit in ("KiB", "MiB", "GiB", "TiB", "PiB"):
        value /= 1024
        if value < 1024 or unit == "PiB":
            if round(value, 1) >= 1024 and unit != "PiB":
                continue
            return f"{value:.1f} {unit}"
    raise AssertionError("unreachable")  # pragma: no cover


# --- tables ---------------------------------------------------------------------------

Painter = Callable[[str], str]


@dataclass(frozen=True, slots=True)
class Cell:
    """Plain text of a cell, its alignment and an optional decoration."""

    text: str
    align: Literal["<", ">"] = "<"
    paint: Painter | None = None
    #: A free-form last cell that does not take part in column widths.
    free: bool = False


def table(
    rows: list[list[Cell]], *, indent: str = "  ", gap: str = "   ", marked: bool = False
) -> list[str]:
    """Align columns by their plain text, then decorate: colors never break the layout.

    ``marked``: the first column holds status symbols and is followed by one space.
    """
    if not rows:
        return []
    widths = [0] * max(len(r) for r in rows)
    for row in rows:
        for i, cell in enumerate(row):
            if not cell.free:
                widths[i] = max(widths[i], len(cell.text))
    lines = []
    for row in rows:
        cells = []
        for i, cell in enumerate(row):
            last = i == len(row) - 1
            if cell.free:
                text = cell.text
            elif cell.align == ">":
                text = cell.text.rjust(widths[i])
            else:
                text = cell.text if last else cell.text.ljust(widths[i])
            cells.append(cell.paint(text) if cell.paint and cell.text else text)
        if marked and len(cells) > 1:
            line = cells[0] + " " + gap.join(cells[1:])
        else:
            line = gap.join(cells)
        lines.append((indent + line).rstrip())
    return lines
